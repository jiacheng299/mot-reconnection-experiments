"""Label online reconnection actions and measure recovery on fixed detector gaps."""

from collections import defaultdict
from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment


def pair(boxes, people, minimum_iou=0.5):
    """Return box index -> true ID, prioritizing count before total overlap."""
    if not len(boxes) or not people:
        return {}

    identities = sorted(people)
    prediction = np.asarray([box[:4] for box in boxes], dtype=float)
    truth = np.asarray([people[identity] for identity in identities], dtype=float)
    start = np.maximum(prediction[:, None, :2], truth[None, :, :2])
    end = np.minimum(prediction[:, None, 2:], truth[None, :, 2:])
    intersection = np.maximum(end - start, 0).prod(axis=2)
    pred_area = np.maximum(prediction[:, 2:] - prediction[:, :2], 0).prod(axis=1)
    truth_area = np.maximum(truth[:, 2:] - truth[:, :2], 0).prod(axis=1)
    union = pred_area[:, None] + truth_area[None, :] - intersection
    overlap = np.divide(intersection, union, out=np.zeros_like(union), where=union > 0)
    valid = overlap >= minimum_iou
    reward = np.where(valid, 1 + min(overlap.shape) + overlap, 0)
    rows, columns = linear_sum_assignment(reward, maximize=True)
    return {int(row): identities[column] for row, column in zip(rows, columns)
            if valid[row, column]}


@dataclass(frozen=True)
class Gap:
    person: int
    before: int
    after: int


def _gaps(detector_ids, truth, max_gap):
    last_detection = {}
    gaps = []
    for frame, people in sorted(truth.items()):
        for person in sorted(detector_ids.get(frame, ())):
            previous = last_detection.get(person)
            last_detection[person] = frame
            if previous is None or not (2 <= frame - previous <= max_gap + 1):
                continue
            if all(person in truth.get(i, {}) for i in range(previous + 1, frame)):
                gaps.append(Gap(person, previous, frame))
    return tuple(gaps)


def natural_gaps(detections, truth, *, max_gap=5, minimum_iou=0.5):
    """A short detector interruption while the same identity remains labelled."""
    return prepare(detections, truth, max_gap=max_gap, minimum_iou=minimum_iou).gaps


@dataclass(frozen=True)
class Evidence:
    truth: dict
    detection_people: dict
    detection_rows: dict
    gaps: tuple
    minimum_iou: float
    ignored: dict


def prepare(detections, truth, *, max_gap=5, minimum_iou=0.5, ignored=None):
    matches = {frame: pair(boxes, truth.get(frame, {}), minimum_iou)
               for frame, boxes in detections.items()}
    by_box = {}
    for frame, matches_in_frame in matches.items():
        by_box[frame] = {tuple(detections[frame][row][:4]): person
                         for row, person in matches_in_frame.items()}
    return Evidence(truth, by_box, matches,
                    _gaps({frame: row.values() for frame, row in matches.items()},
                          truth, max_gap), minimum_iou, ignored or {})


@dataclass(frozen=True)
class ActionLabel:
    frame: int
    previous_frame: int
    track_id: int
    accepted: bool
    correct: bool
    verifiable_anchor: bool
    features: dict
    evaluable: bool


@dataclass(frozen=True)
class Outcome:
    gaps: int
    correct: int
    accepted: int
    wrong: int
    unverified_accepted: int
    recovery_outcomes: tuple
    actions: tuple
    ignored_accepted: int


def _intersects_ignore(box, regions):
    x, y, right, bottom = box
    area = (right - x) * (bottom - y)
    if area <= 0:
        return False
    for x1, y1, x2, y2 in regions:
        overlap = max(0, min(right, x2) - max(x, x1)) * max(0, min(bottom, y2) - max(y, y1))
        if overlap > 0.5 * area:
            return True
    return False


def evaluate(detections, truth, tracks, decisions, *, max_gap=5, minimum_iou=0.5):
    evidence = prepare(detections, truth, max_gap=max_gap, minimum_iou=minimum_iou)
    return evaluate_prepared(evidence, tracks, decisions)


def evaluate_prepared(evidence, tracks, decisions, *, duplicate_equivalence=False):
    """Recovery uses frozen gap anchors, risk uses all actual reactivation actions."""
    by_frame = defaultdict(list)
    for row in tracks:
        by_frame[row.frame].append(row)

    needed_frames = {frame for gap in evidence.gaps for frame in (gap.before, gap.after)}
    needed_frames.update(row.proposal.previous_frame for row in decisions)
    assignments = {}
    for frame in needed_frames:
        observations = by_frame[frame]
        matched = pair([row.box for row in observations], evidence.truth.get(frame, {}),
                       evidence.minimum_iou)
        assignments[frame] = {person: observations[index].track_id
                              for index, person in matched.items()}

    recovery = []
    for gap in evidence.gaps:
        anchor = assignments[gap.before].get(gap.person)
        current = assignments[gap.after].get(gap.person)
        outcome = "no_anchor" if anchor is None else "missed"
        if anchor is not None and current is not None:
            outcome = "correct" if current == anchor else "switched"
        recovery.append(outcome)

    people_by_track = {frame: {track: person for person, track in rows.items()}
                       for frame, rows in assignments.items()}
    actions = []
    for row in decisions:
        proposal = row.proposal
        person = people_by_track[proposal.previous_frame].get(proposal.track_id)
        index = getattr(proposal, "candidate_index", None)
        candidate = evidence.detection_rows.get(proposal.frame, {}).get(index)
        if index is None or (candidate is None and duplicate_equivalence):
            matches = {identity for box, identity in evidence.detection_people.get(proposal.frame, {}).items()
                       if np.allclose(box, proposal.candidate_box, rtol=0, atol=0.001)}
            if len(matches) == 1:
                candidate = matches.pop()
        ignore_anchor = person is None and _intersects_ignore(
            proposal.previous_box, evidence.ignored.get(proposal.previous_frame, ()))
        ignore_target = candidate is None and _intersects_ignore(
            proposal.candidate_box, evidence.ignored.get(proposal.frame, ()))
        actions.append(ActionLabel(proposal.frame, proposal.previous_frame, proposal.track_id,
                                   row.accepted, person is not None and person == candidate,
                                   person is not None, proposal.features,
                                   not (ignore_anchor or ignore_target)))

    accepted = [action for action in actions if action.accepted and action.evaluable]
    return Outcome(len(evidence.gaps), recovery.count("correct"),
                   len(accepted), sum(not action.correct for action in accepted),
                   sum(not action.verifiable_anchor for action in accepted),
                   tuple(recovery), tuple(actions),
                   sum(action.accepted and not action.evaluable for action in actions))
