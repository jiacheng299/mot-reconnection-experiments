"""Bounded joint proposals and fixed geometric scoring for the C1 pilot."""

from math import prod

import numpy as np
from scipy.optimize import linear_sum_assignment

from scripts.p3_association import feasible_actions


def choose_group_action(group, scores, budget):
    """Return the best linear action, the legal proposals and a fallback flag."""
    if budget < 1:
        raise ValueError('The candidate budget must be positive')
    if prod(len(scores[track]) + 1 for track in group) <= budget:
        actions = list(feasible_actions(group, scores))
        selected, _ = max(actions, key=lambda pair: pair[1])
        return selected, [action for action, _ in actions], False

    detections = sorted({detection for track in group for detection in scores[track]})
    columns = {detection: index for index, detection in enumerate(detections)}
    matrix = np.full((len(group), len(detections) + len(group)), -np.inf)
    for row, track in enumerate(group):
        matrix[row, len(detections) + row] = 0.0
        for detection, score in scores[track].items():
            matrix[row, columns[detection]] = score
    all_miss = (None,) * len(group)

    def solve(weights):
        rows, cols = linear_sum_assignment(weights, maximize=True)
        if len(rows) != len(group) or not np.isfinite(weights[rows, cols]).all():
            raise ValueError('Solver returned an inadmissible assignment')
        return tuple(detections[c] if c < len(detections) else None for c in cols)

    try:
        best = solve(matrix)
    except ValueError:
        return all_miss, [all_miss], True

    candidates = [best]
    if budget > 1 and best != all_miss:
        candidates.append(all_miss)
    fallback = False
    for row, detection in enumerate(best):
        if len(candidates) >= budget:
            break
        if detection is None:
            continue
        alternate = matrix.copy()
        alternate[row, columns[detection]] = -np.inf
        try:
            proposal = solve(alternate)
        except ValueError:
            fallback = True
            break
        if proposal not in candidates:
            candidates.append(proposal)
    selected = max(candidates, key=lambda action: sum(
        scores[track][detection] for track, detection in zip(group, action) if detection is not None
    ))
    return selected, candidates, fallback
