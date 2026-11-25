"""Fit and calibrate the fifteen-feature additive reconnect score."""

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from functools import partial
import math
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from reconnect_study.data_io import digest, read_manifest
from reconnect_study.policy import RecoveryPolicy
from reconnect_study.replay import load_sequence, replay
from reconnect_study.scoring import Score, design
from reconnect_study.study import TRACKERS, save
from reconnect_study.supplement_matched import ARCHIVE, IntervalGate, load, stable_record, summaries


ROOT = Path("runs/reconnect_supplement")
OUTPUT = ROOT / "round3" / "ablation"
MANIFEST = ARCHIVE / "manifest.json"
FEATURE_NAMES = ("min_margin", "absolute_margin_difference", "max_log_candidate_count")


def vector15(features):
    base = design(features, False)
    row, column, rows, columns = base[8:12]
    return [*base, min(row, column), abs(row - column), max(rows, columns)]


@dataclass(frozen=True)
class Additive15:
    offset: float
    weights: tuple
    center: tuple
    scale: tuple

    def __call__(self, features):
        return self.offset + sum(weight * (value - center) / scale for weight, value, center, scale
                                 in zip(self.weights, vector15(features), self.center, self.scale, strict=True))


def fit15(rows, device):
    dev = torch.device(device)
    torch.manual_seed(362)
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats(dev)
    raw = np.asarray([vector15(action["features"]) for action in rows], dtype=np.float32)
    center = raw.mean(axis=0)
    scale = np.maximum(raw.std(axis=0), np.float32(1e-4))
    sample = torch.as_tensor((raw - center) / scale, device=dev)
    labels = [int(action["correct"]) for action in rows]
    target = torch.tensor(labels, dtype=torch.float32, device=dev)
    weights = torch.zeros(raw.shape[1], device=dev, requires_grad=True)
    offset = torch.zeros((), device=dev, requires_grad=True)
    optimizer = torch.optim.Adam((weights, offset), lr=.04)
    ranking_weight = 0.0
    for _ in range(200):
        optimizer.zero_grad()
        logits = sample @ weights + offset
        objective = F.binary_cross_entropy_with_logits(logits, target) + .002 * weights.square().sum()
        objective.backward()
        optimizer.step()
    model = Additive15(float(offset.detach().cpu()), tuple(weights.detach().cpu().tolist()),
                       tuple(center.tolist()), tuple(scale.tolist()))
    return model, {
        "rows": len(rows), "positive": sum(labels), "ranking_weight": ranking_weight, "iterations": 200,
        "learning_rate": .04, "l2": .002, "seed": 362, "device": str(dev),
        "hardware": torch.cuda.get_device_name(dev) if dev.type == "cuda" else "CPU",
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(dev) if dev.type == "cuda" else None,
        "objective": float(objective.detach().cpu()),
    }


def train(key, device, sequences):
    base_path = ARCHIVE / f"{key}-config.json"
    dev_path = ARCHIVE / f"development-{key}.json"
    base, baseline = load(base_path), load(dev_path)
    source_rows = baseline["sequences"]
    if base["study_manifest_sha256"] != digest(MANIFEST) or base["baseline_sha256"] != digest(dev_path):
        raise ValueError("Development provenance differs from original frozen model")
    expected = {(row.name, row.frames, row.source_id) for row in sequences if row.split == "development"}
    if {(row["name"], row["frames"], row["source_id"]) for row in source_rows} != expected:
        raise ValueError("Development sequence roster changed")
    if any(row["tracker"] != base["tracker"] or row["split"] != "development" for row in source_rows):
        raise ValueError("Unexpected tracker or split for the development archive")
    actions = [action for row in source_rows for action in row["actions"] if action.get("evaluable", True)]
    model, fit = fit15(actions, device)
    features = [action["features"] for action in actions]
    if any(vector15(row) != design(row, True)[:15] for row in features):
        raise ValueError("Fifteen-feature vector differs from the structured prefix")
    fit["verified_feature_prefix_rows"] = len(features)
    for name, dimension in (("linear", 12), ("structured", 23)):
        frozen = base["training"][name]
        score = Score(**frozen["score"])
        if len(score.weights) != dimension or frozen["fit"]["ranking_weight"] != 0:
            raise ValueError("Old weights do not match the required ablation model")
        reproduced = np.unique(np.quantile([score(row) for row in features], base["quantiles"])).tolist()
        if reproduced != base["thresholds"][name]:
            raise ValueError("Old score implementation failed to reconstruct frozen thresholds")
        fit[f"verified_{dimension}_thresholds"] = reproduced
    return base, model, fit, [model(row) for row in features]


def calibrate_source(sequences, tracker_key, model, thresholds):
    input_rows = [(sequence, *load_sequence(sequence)) for sequence in sequences]
    tracker = TRACKERS[tracker_key]
    output, shared_check = {}, None
    remaining = sorted((threshold, name) for name, threshold in thresholds.items())
    physical = 0
    while remaining:
        cutoff, _ = remaining[len(remaining) // 2]
        gate = IntervalGate(model, cutoff, 12)
        results = [replay(*entry, gate, tracker=tracker) for entry in input_rows]
        physical += 1
        shared = [(value, name) for value, name in remaining if gate.lower < value <= gate.upper]
        if not shared:
            raise ValueError("Empty sharing interval")
        rows = [stable_record(record) for record, _ in results]
        output.update((name, rows) for _, name in shared)
        if shared_check is None and len(shared) > 1:
            shared_check = next(value for value, _ in shared if value != cutoff), results
        remaining = [(value, name) for value, name in remaining if name not in output]
    checked = 0
    if shared_check:
        cutoff, expected = shared_check
        fresh = RecoveryPolicy(model, cutoff, 12)
        for entry, (expected_row, expected_trajectory) in zip(input_rows, expected, strict=True):
            row, trajectory = replay(*entry, fresh, tracker=tracker)
            if stable_record(row) != stable_record(expected_row) or trajectory != expected_trajectory:
                raise ValueError(f"Independent equivalence failed for {entry[0].name}")
            checked += 1
    return output, physical, checked


def compare_match(first, second, target, tolerance):
    spread = abs(first - second)
    absolute = [abs(first - target), abs(second - target)]
    return {"spread": spread, "absolute_target_deviations": absolute,
            "spread_within_tolerance": spread <= tolerance + 1e-12,
            "target_deviations_within_tolerance": max(absolute) <= tolerance + 1e-12,
            "matched": max(spread, *absolute) <= tolerance + 1e-12}


def select(points, candidates, thresholds):
    selected = {}
    for name, point in points.items():
        if not point["matched"]:
            continue
        target, tolerance = point["target"], point["tolerance"]
        policy = min(candidates, key=lambda key: (round(abs(candidates[key]["mean_accepted"] - target), 12),
                                                 candidates[key]["mean_accepted"], -thresholds[key], key))
        new_mean = candidates[policy]["mean_accepted"]
        old = point["nearest"]
        comparisons = {
            "23_minus_15": compare_match(old["structured"]["mean_accepted"], new_mean, target, tolerance),
            "15_minus_12": compare_match(new_mean, old["linear"]["mean_accepted"], target, tolerance),
        }
        selected[name] = {
            "target": target, "tolerance": tolerance, "old": old,
            "new": {"policy": policy, "threshold": thresholds[policy], "mean_accepted": new_mean},
            "calibration_match": comparisons,
        }
    return selected


def sweep(executor, calibration, tracker, model, thresholds, rows, label):
    worker = partial(calibrate_source, tracker_key=tracker, model=model, thresholds=thresholds)
    physical, checked = 0, 0
    for index, (result, count, verification) in enumerate(executor.map(worker, calibration, chunksize=1), 1):
        for name, values in result.items():
            rows[name].extend(values)
        physical += count
        checked += verification
        if index % 10 == 0 or index == len(calibration):
            print(f"{label}: {index}/{len(calibration)} sources ({physical} closed-loop replays)", flush=True)
    return {"physical_source_replays": physical, "validated_shared_replays": checked}


def execute(key, workers, device):
    directory = OUTPUT / key
    oldroot = ROOT / "round2" / key
    old = oldroot / "B12"
    lockpath = old / "locked.json"
    locked = load(lockpath)
    provenance = {
        "manifest": (MANIFEST, digest(MANIFEST)),
        "old_config": (old / "config.json", locked["config_sha256"]),
        "old_calibration": (old / locked["calibration_file"], locked["calibration_sha256"]),
        "old_inputs": (oldroot / "input-splits.json", locked["inputs_sha256"]),
        "old_lock": (lockpath, digest(lockpath)),
        "old_test_integrity": (oldroot / "test-integrity.json", digest(oldroot / "test-integrity.json")),
        "development": (ARCHIVE / f"development-{key}.json", digest(ARCHIVE / f"development-{key}.json")),
    }
    for path, expected in provenance.values():
        if digest(path) != expected:
            raise ValueError(f"Input hash changed: {path}")
    old_config = load(old / "config.json")
    sequences = read_manifest(MANIFEST)
    base, model, fit, scores = train(key, device, sequences)
    if base["training"] != old_config["training"] or old_config["acceptance_cap"] != 12:
        raise ValueError("Prior round uses different weights or cap")
    cohorts = defaultdict(list)
    for row in sorted(sequences, key=lambda row: (row.source_id, row.name)):
        if row.split == "calibration":
            cohorts[row.source_id].append(row)
    points = locked["selection"]["points"]
    origin = np.unique(np.quantile(scores, base["quantiles"])).tolist()
    dense = np.unique(np.quantile(scores, np.arange(35, 98) / 100)).tolist()
    thresholds = {f"summary15_{index}": value for index, value in enumerate(origin)}
    source_files = {str(path): digest(path) for path in (
        Path(__file__), Path(__file__).with_name("scoring.py"), Path(__file__).with_name("tracker.py"),
        Path(__file__).with_name("supplement_matched.py"), Path(__file__).with_name("study.py"),
    )}
    definition = {
        "training": fit, "score": asdict(model), "added_features": FEATURE_NAMES,
        "tracker": base["tracker"], "acceptance_cap": 12, "study_manifest_sha256": digest(MANIFEST),
        "inputs": {name: {"path": str(path), "sha256": expected} for name, (path, expected) in provenance.items()},
        "source_sha256": source_files,
        "split_sizes": {name: len({row.source_id for row in sequences if row.split == name})
                        for name in ("development", "calibration", "test")},
    }
    save(directory / "fit.json", definition)
    print(f"{key}: verified/trained {fit['rows']} development proposals ({fit['positive']} positive) on {fit['hardware']}", flush=True)
    rows = defaultdict(list)
    calibration = list(cohorts.values())
    with ProcessPoolExecutor(max_workers=workers) as executor:
        audit = sweep(executor, calibration, base["tracker"], model, thresholds, rows, f"{key} original")
        aggregate = summaries(rows, 12)
        chosen = select(points, aggregate, thresholds)
        save(directory / "calibration-original.json", {
            "thresholds": thresholds, "by_video": rows, "aggregate": aggregate, "selection": chosen, "audit": audit,
            "fit_sha256": digest(directory / "fit.json"),
        })
        filename = "calibration-original.json"
        if any(not match["matched"] for point in chosen.values() for match in point["calibration_match"].values()):
            extra = {f"summary15_{index}": value for index, value in enumerate(
                sorted(set(dense) - set(origin)), start=len(origin))}
            audit = sweep(executor, calibration, base["tracker"], model, extra, rows, f"{key} dense")
            thresholds.update(extra)
            aggregate = summaries(rows, 12)
            chosen = select(points, aggregate, thresholds)
            filename = "calibration-dense.json"
            save(directory / filename, {
                "thresholds": thresholds, "by_video": rows, "aggregate": aggregate, "selection": chosen,
                "audit": audit, "fit_sha256": digest(directory / "fit.json"),
            })
    save(directory / "config.json", {**definition, "thresholds": thresholds})
    retained = sorted({item["new"]["policy"] for item in chosen.values()
                       if any(match["matched"] for match in item["calibration_match"].values())})
    save(directory / "locked.json", {
        "config_sha256": digest(directory / "config.json"),
        "calibration_file": filename, "calibration_sha256": digest(directory / filename),
        "test_policies": retained, "work_points": chosen, "reused_previously_analyzed_test": True,
    })
    print(f"{key}: locked {len(retained)} new policies", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracker", required=True, choices=("byte", "bot"))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    execute(args.tracker, args.workers, args.device)


if __name__ == "__main__":
    main()
