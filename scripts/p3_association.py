"""Exact association on small competition groups for the P3 feasibility check."""

from collections import defaultdict
from collections.abc import Iterator, Mapping


CandidateScores = Mapping[int, Mapping[int, float]]
Action = tuple[int | None, ...]


def competition_groups(scores: CandidateScores) -> list[tuple[int, ...]]:
    """Group tracks connected by shared legal detection candidates."""
    tracks_by_detection = defaultdict(list)
    for track, candidates in scores.items():
        for detection in candidates:
            tracks_by_detection[detection].append(track)

    undiscovered = set(scores)
    visited_detections = set()
    groups = []
    for track in sorted(scores):
        if track not in undiscovered:
            continue
        undiscovered.remove(track)
        group = [track]
        # Traverse the track-detection graph without materializing every track-to-track edge.
        for member in group:
            for detection in scores[member]:
                if detection in visited_detections:
                    continue
                visited_detections.add(detection)
                for neighbor in tracks_by_detection[detection]:
                    if neighbor in undiscovered:
                        undiscovered.remove(neighbor)
                        group.append(neighbor)
        groups.append(tuple(sorted(group)))
    return groups


def feasible_actions(group: tuple[int, ...], scores: CandidateScores) -> Iterator[tuple[Action, float]]:
    """Enumerate legal actions and additive scores for a small group, in stable ID order."""
    ordered_candidates = [sorted(scores[track].items()) for track in group]
    used = set()

    def visit(action: Action, total: float) -> Iterator[tuple[Action, float]]:
        if len(action) == len(group):
            yield action, total
            return

        for detection, score in ordered_candidates[len(action)]:
            if detection in used:
                continue
            used.add(detection)
            yield from visit((*action, detection), total + score)
            used.remove(detection)
        # A miss has score zero and consumes no shared detection.
        yield from visit((*action, None), total)

    return visit((), 0.0)


def select_linear_actions(scores: CandidateScores) -> dict[int, int | None]:
    """Choose the best additive action in each small competition group."""
    selected = {}
    for group in competition_groups(scores):
        action, _ = max(feasible_actions(group, scores), key=lambda candidate: candidate[1])
        selected.update(zip(group, action, strict=True))
    return selected
