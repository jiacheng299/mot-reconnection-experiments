"""Motion-only BoT-SORT with a first-order BoostTrack-v1 host-state adapter."""

from dataclasses import replace
from itertools import count
from types import SimpleNamespace

import numpy as np

from reconnect_study import tracker as host
from reconnect_study.round3_boost import (
    _measurement_xyhr, _track_confidence, boosttrack_similarity_v1,
)


def adapt_state(means, covariances):
    """Transform xywh to xyhr and propagate covariance by the local Jacobian."""
    widths, heights = means[:, 2], means[:, 3]
    if not np.isfinite(means).all() or np.any(widths <= 0) or np.any(heights <= 0):
        raise ValueError("BoT-SORT predicted a nonpositive or nonfinite box")
    ratio = widths / heights
    xyhr = np.column_stack((means[:, :2], heights, ratio))
    jacobian = np.zeros(covariances.shape)
    jacobian[:, 0, 0] = jacobian[:, 1, 1] = jacobian[:, 2, 3] = 1
    jacobian[:, 3, 2] = 1 / heights
    jacobian[:, 3, 3] = -ratio / heights
    adapted = jacobian @ covariances @ jacobian.transpose(0, 2, 1)
    variance = np.diagonal(adapted, axis1=1, axis2=2)
    if not np.isfinite(adapted).all() or np.any(variance <= 0):
        raise ValueError("BoT-SORT produced invalid adapted variances")
    return xyhr, variance


def matrix(frame, tracks, detections, observer):
    det = np.asarray([(*d.tlbr, d.score) for d in detections], dtype=float)
    track = np.asarray([(*t.tlbr, _track_confidence(t, frame)) for t in tracks], dtype=float)
    obs = np.asarray([_measurement_xyhr(d) for d in detections])
    means, variances = adapt_state(np.asarray([t.mean[:4] for t in tracks]),
                                    np.asarray([t.covariance[:4, :4] for t in tracks]))
    distance = (((obs[:, None, :] - means[None, :, :]) ** 2) / variances[None, :, :]).sum(2)
    similarity = boosttrack_similarity_v1(det, track, distance, track[:, 4], det[:, 4], iou_threshold=.3)
    if not np.isfinite(similarity).all():
        raise ValueError(f"Nonfinite similarity in frame {frame}")
    if observer is not None:
        observer({"frame": frame, "detections": det, "tracks": track, "mahalanobis_distance": distance,
                  "track_confidence": track[:, 4], "detection_confidence": det[:, 4],
                  "iou_threshold": .3, "similarity": similarity})
    return similarity


def run_botsort_boost(detections_by_frame, frame_count, *, policy=None, sequence="",
                      frame_rate=30, track_thresh=.6, track_buffer=30, match_thresh=.8,
                      mot20=False, track_low_thresh=.1, new_track_thresh=.7, proximity_thresh=.5,
                      appearance_thresh=.25, matrix_observer=None):
    tracker_class = host._vendor_tracker_class(host._VENDOR / "BoT-SORT" / "tracker", "bot_sort", "BoTSORT")
    identity = np.eye(2, 3)
    options = SimpleNamespace(track_high_thresh=track_thresh, track_low_thresh=track_low_thresh,
                              new_track_thresh=new_track_thresh, track_buffer=track_buffer,
                              match_thresh=match_thresh, mot20=mot20, proximity_thresh=proximity_thresh,
                              appearance_thresh=appearance_thresh, with_reid=False, encoder=None,
                              gmc=SimpleNamespace(apply=lambda frame, boxes: identity))
    tracker = tracker_class(options, frame_rate=frame_rate)
    tracker.next_local_track_id = count(1).__next__
    observations, decisions = [], []

    def gate(frame, tracks, detections, costs, matches, unmatched, tracked_state):
        similarity = matrix(frame, tracks, detections, matrix_observer) if tracks and detections else None
        retained, released = [], []
        for row, col in matches:
            track = tracks[row]
            if track.state == tracked_state:
                retained.append((row, col))
                continue
            proposal = host._proposal(sequence, frame, track, detections[col], costs, row, col, match_thresh)
            proposal = replace(proposal, features={**proposal.features,
                                                   "boosttrack_v1_similarity": float(similarity[col, row])})
            accepted = policy is None or bool(policy(proposal))
            decisions.append(host.ReconnectDecision(proposal, accepted))
            if accepted:
                retained.append((row, col))
            else:
                released.append(int(col))
        if not released:
            return matches, unmatched
        return retained, sorted([*unmatched, *released])

    tracker.reconnect_gate = gate
    for frame in range(1, frame_count + 1):
        boxes = np.array(detections_by_frame.get(frame, ()), dtype=float, copy=True).reshape(-1, 5)
        if np.any(~np.isfinite(boxes)) or np.any(boxes[:, 2:4] <= boxes[:, :2]):
            raise ValueError(f"Invalid detection coordinates or scores in frame {frame}")
        for track in tracker.update(boxes, frame):
            observations.append(host.TrackObservation(frame, track.track_id, host._box(track.tlbr), float(track.score)))
    return host.TrackingResult(tuple(observations), tuple(decisions))
