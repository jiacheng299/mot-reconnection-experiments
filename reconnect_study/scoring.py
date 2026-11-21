"""Learn an interpretable score for candidate track reactivation."""

from collections import defaultdict
from dataclasses import dataclass
from math import log1p
from time import perf_counter

import numpy as np
import torch
from torch.nn import functional as F


BASE_FEATURES = ("gap_frames", "track_age_frames", "detector_score", "previous_score", "iou",
                 "center_distance", "log_area_ratio", "assignment_cost", "row_margin",
                 "column_margin", "row_alternatives", "column_alternatives")


def design(features, structured):
    values = [float(features[key]) for key in BASE_FEATURES]
    for index in (0, 1, 5, 6, 10, 11):
        values[index] = log1p(max(values[index], 0.0))
    if not structured:
        return values

    gap, age, confidence, previous, overlap, distance, area, _, row, column, n_row, n_col = values
    ambiguity = min(row, column)
    values.extend([ambiguity, abs(row - column), max(n_row, n_col),
                   gap * distance, gap * area, gap * overlap, gap * ambiguity,
                   confidence * ambiguity, previous * confidence, distance * ambiguity,
                   age * ambiguity])
    return values


@dataclass(frozen=True)
class Score:
    structured: bool
    offset: float
    weights: tuple
    center: tuple
    scale: tuple

    def __call__(self, features):
        vector = design(features, self.structured)
        return self.offset + sum(weight * (value - center) / scale
                                 for weight, value, center, scale
                                 in zip(self.weights, vector, self.center, self.scale, strict=True))


def _ranking_pairs(rows, rng):
    by_source = defaultdict(lambda: ([], []))
    for index, (source, _, label) in enumerate(rows):
        by_source[source][int(label)].append(index)
    wrong, right = [], []
    for negatives, positives in by_source.values():
        if negatives and positives:
            count = min(len(negatives) + len(positives), 512)
            wrong.extend(rng.choice(negatives, count))
            right.extend(rng.choice(positives, count))
    return wrong, right


def fit(rows, *, structured, device, iterations=200, seed=362, ranking_weight=None):
    """Fit logistic and within-source pairwise ranking losses on development rows."""
    started = perf_counter()
    dev = torch.device(device)
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)
    matrix = np.asarray([design(row[1], structured) for row in rows], dtype=np.float32)
    labels = [int(row[2]) for row in rows]
    if set(labels) != {0, 1}:
        raise ValueError("Both correct and wrong development reactivations are required")
    center = matrix.mean(axis=0)
    scale = np.maximum(matrix.std(axis=0), np.float32(1e-4))
    sample = torch.as_tensor((matrix - center) / scale, device=dev)
    target = torch.tensor(labels, dtype=torch.float32, device=dev)
    negative, positive = _ranking_pairs(rows, np.random.default_rng(seed))
    left = torch.tensor(negative, dtype=torch.long, device=dev)
    right = torch.tensor(positive, dtype=torch.long, device=dev)
    weights = torch.zeros(matrix.shape[1], device=dev, requires_grad=True)
    offset = torch.zeros((), device=dev, requires_grad=True)
    optimizer = torch.optim.Adam((weights, offset), lr=0.04)
    if ranking_weight is None:
        ranking_weight = 0.35 if structured else 0.0
    for _ in range(iterations):
        optimizer.zero_grad()
        logits = sample @ weights + offset
        objective = F.binary_cross_entropy_with_logits(logits, target) + 0.002 * weights.square().sum()
        if ranking_weight and len(left):
            objective = objective + ranking_weight * F.softplus(logits[left] - logits[right]).mean()
        objective.backward()
        optimizer.step()
    model = Score(structured, float(offset.detach().cpu()), tuple(weights.detach().cpu().tolist()),
                  tuple(center.tolist()), tuple(scale.tolist()))
    hardware = torch.cuda.get_device_name(dev) if dev.type == "cuda" else "CPU"
    peak = torch.cuda.max_memory_allocated(dev) if dev.type == "cuda" else None
    return model, {"source_count": len({row[0] for row in rows}), "rows": len(rows),
                   "positive": sum(labels), "ranking_pairs": len(left), "ranking_weight": ranking_weight,
                   "iterations": iterations, "seed": seed, "device": str(dev), "hardware": hardware,
                   "peak_cuda_allocated_bytes": peak, "seconds": perf_counter() - started,
                   "objective": float(objective.detach().cpu())}
