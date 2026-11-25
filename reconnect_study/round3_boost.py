"""BoostTrack v1 rich association similarity for a host-tracker adapter.

This preserves the released implementation, including the documented v1 shape
similarity denominator bug.  Detection boosting and assignment are intentionally
outside this component comparison.
"""

from __future__ import annotations

from dataclasses import replace
from itertools import count
from types import SimpleNamespace

import numpy as np

from reconnect_study import tracker as host


def iou_batch(detections, tracks):
    first = np.expand_dims(np.asarray(detections, dtype=float), 1)
    second = np.expand_dims(np.asarray(tracks, dtype=float), 0)
    x1 = np.maximum(first[..., 0], second[..., 0])
    y1 = np.maximum(first[..., 1], second[..., 1])
    x2 = np.minimum(first[..., 2], second[..., 2])
    y2 = np.minimum(first[..., 3], second[..., 3])
    intersection = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    union = ((first[..., 2] - first[..., 0]) * (first[..., 3] - first[..., 1])
             + (second[..., 2] - second[..., 0]) * (second[..., 3] - second[..., 1])
             - intersection)
    return intersection / union


def shape_similarity_v1(detections, tracks):
    detections, tracks = np.asarray(detections, dtype=float), np.asarray(tracks, dtype=float)
    if detections.size == 0 or tracks.size == 0:
        return np.zeros((0, 0))
    dw = (detections[:, 2] - detections[:, 0]).reshape(-1, 1)
    dh = (detections[:, 3] - detections[:, 1]).reshape(-1, 1)
    tw = (tracks[:, 2] - tracks[:, 0]).reshape(1, -1)
    th = (tracks[:, 3] - tracks[:, 1]).reshape(1, -1)
    # Author v1 uses max(widths) for both terms; do not "fix" this here.
    return np.exp(-(np.abs(dw - tw) / np.maximum(dw, tw)
                    + np.abs(dh - th) / np.maximum(dw, tw)))


def mahalanobis_similarity(distance, softmax_temp=1.0):
    distance = np.asarray(distance, dtype=float).copy()
    limit = 13.2767
    mask = distance > limit
    distance[mask] = limit
    logits = (limit - distance) / softmax_temp
    output = np.exp(logits) / np.exp(logits).sum(0).reshape(1, -1)
    return np.where(mask, 0, output)


def boosttrack_similarity_v1(detections, tracks, mahalanobis_distance,
                             track_confidence, detection_confidence, *,
                             iou_threshold, lambda_iou=0.5,
                             lambda_mhd=0.25, lambda_shape=0.25):
    """Return the released no-ReID BoostTrack association score matrix."""
    iou = iou_batch(detections, tracks)
    confidence = np.multiply(np.asarray(detection_confidence).reshape(-1, 1),
                             np.asarray(track_confidence).reshape(1, -1))
    confidence[iou < iou_threshold] = 0
    return (iou + lambda_iou * confidence * iou
            + lambda_mhd * mahalanobis_similarity(mahalanobis_distance)
            + lambda_shape * confidence * shape_similarity_v1(detections, tracks))


def _author_state_from_host(track):
    """Adapt ByteTrack's xyah state/covariance to BoostTrack's xyhr order."""
    order = np.array([0, 1, 3, 2])
    mean = np.asarray(track.mean, dtype=float)[order]
    variance = np.diag(np.asarray(track.covariance, dtype=float))[order]
    return mean, variance


def _measurement_xyhr(detection):
    x1, y1, x2, y2 = np.asarray(detection.tlbr, dtype=float)
    width, height = x2 - x1, y2 - y1
    return np.array([x1 + width / 2, y1 + height / 2, height,
                     width / (height + 1e-6)])


def _track_confidence(track, frame, coef=0.9):
    age = frame - track.start_frame
    time_since_update = frame - track.end_frame
    return coef ** (7 - age) if age < 7 else coef ** (time_since_update - 1)


def _matrix(frame, tracks, detections, iou_threshold, observer=None):
    if not tracks or not detections:
        return np.zeros((len(detections), len(tracks)))
    det_boxes = np.asarray([(*d.tlbr, d.score) for d in detections], dtype=float).reshape(-1, 5)
    track_boxes = np.asarray([(*t.tlbr, _track_confidence(t, frame)) for t in tracks],
                             dtype=float).reshape(-1, 5)
    measurements = np.asarray([_measurement_xyhr(d) for d in detections], dtype=float)
    states = [_author_state_from_host(t) for t in tracks]
    means = np.asarray([item[0] for item in states], dtype=float)
    variances = np.asarray([item[1] for item in states], dtype=float)
    distance = (((measurements[:, None, :] - means[None, :, :]) ** 2)
                / variances[None, :, :]).sum(axis=2)
    similarity = boosttrack_similarity_v1(det_boxes, track_boxes, distance, track_boxes[:, 4],
                                           det_boxes[:, 4], iou_threshold=iou_threshold)
    if observer is not None:
        observer({"frame": frame, "detections": det_boxes.copy(), "tracks": track_boxes.copy(),
                  "mahalanobis_distance": distance.copy(),
                  "track_confidence": track_boxes[:, 4].copy(),
                  "detection_confidence": det_boxes[:, 4].copy(),
                  "iou_threshold": iou_threshold, "similarity": similarity.copy()})
    return similarity


def run_bytetrack_boost(detections_by_frame, frame_count, *, policy=None, sequence="",
                        frame_rate=30, track_thresh=.5, track_buffer=30,
                        match_thresh=.8, mot20=False, matrix_observer=None):
    """Run ByteTrack with v1 BoostTrack similarity only as the old-ID gate score.

    ByteTrack still determines candidates and updates motion.  The BoostTrack
    Kalman inputs are therefore explicitly adapted from the host state.
    """
    tracker_class = host._vendor_tracker_class(
        host._VENDOR / "ByteTrack" / "yolox" / "tracker", "byte_tracker", "BYTETracker")
    tracker = tracker_class(SimpleNamespace(track_thresh=track_thresh, track_buffer=track_buffer,
                                            match_thresh=match_thresh, mot20=mot20),
                            frame_rate=frame_rate)
    tracker.next_local_track_id = count(1).__next__
    observations, decisions = [], []

    def gate(frame, tracks, detections, costs, matches, unmatched, tracked_state):
        similarities = _matrix(frame, tracks, detections, 0.3, matrix_observer)
        retained, released = [], []
        for row, col in matches:
            track = tracks[row]
            if track.state == tracked_state:
                retained.append((row, col))
                continue
            proposal = host._proposal(sequence, frame, track, detections[col], costs,
                                      row, col, match_thresh)
            features = dict(proposal.features)
            features["boosttrack_v1_similarity"] = float(similarities[col, row])
            proposal = replace(proposal, features=features)
            accepted = policy is None or bool(policy(proposal))
            decisions.append(host.ReconnectDecision(proposal, accepted))
            (retained if accepted else released).append((row, col) if accepted else int(col))
        if not released:
            return retained, unmatched
        return retained, sorted([*unmatched, *released])

    tracker.reconnect_gate = gate
    for frame in range(1, frame_count + 1):
        boxes = np.array(detections_by_frame.get(frame, ()), dtype=float, copy=True).reshape(-1, 5)
        for track in tracker.update(boxes, (1, 1), (1, 1)):
            observations.append(host.TrackObservation(frame, track.track_id,
                                                       host._box(track.tlbr), float(track.score)))
    return host.TrackingResult(tuple(observations), tuple(decisions))
