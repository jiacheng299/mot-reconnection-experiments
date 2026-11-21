"""Replay vendored trackers with a gate on finalized post-gap assignments."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import count
from pathlib import Path
import sys
from threading import Lock
from types import ModuleType, SimpleNamespace
from typing import Callable, Mapping, Sequence

import numpy as np


Box = tuple[float, float, float, float]
Detection = tuple[float, float, float, float, float]
_VENDOR = Path(__file__).resolve().parents[1] / "vendor"
_LOAD_LOCK = Lock()


@dataclass(frozen=True)
class ReconnectProposal:
    sequence: str
    frame: int
    track_id: int
    previous_frame: int
    previous_box: Box
    candidate_box: Box
    predicted_box: Box
    detector_score: float
    candidate_index: int
    features: Mapping[str, float]


@dataclass(frozen=True)
class ReconnectDecision:
    proposal: ReconnectProposal
    accepted: bool


@dataclass(frozen=True)
class TrackObservation:
    frame: int
    track_id: int
    box: Box
    score: float


@dataclass(frozen=True)
class TrackingResult:
    tracks: tuple[TrackObservation, ...]
    decisions: tuple[ReconnectDecision, ...]


Policy = Callable[[ReconnectProposal], bool]


def _replace_once(source: str, old: str, new: str) -> str:
    if source.count(old) != 1:
        raise RuntimeError(f"Vendored ByteTrack no longer matches replay hook: {old!r}")
    return source.replace(old, new, 1)


def _tracker_source(source: str, vendor_module: str) -> str:
    """Install the gate and raw-observation log without patching checkout files."""
    if vendor_module == "byte_tracker":
        source = _replace_once(source, "import torch\nimport torch.nn.functional as F\n", "")
        source = _replace_once(source, "from yolox.tracker import matching", "from . import matching")
        detection_indices = "np.flatnonzero(remain_inds)"
    else:
        source = _replace_once(source, "from tracker import matching", "from . import matching")
        source = _replace_once(source, "from tracker.gmc import GMC\n", "")
        source = _replace_once(source, "from tracker.basetrack import", "from .basetrack import")
        source = _replace_once(source, "from tracker.kalman_filter import", "from .kalman_filter import")
        source = _replace_once(source, "from fast_reid.fast_reid_interfece import FastReIDInterface\n", "")
        source = _replace_once(source, "        BaseTrack.clear_count()\n", "")
        source = _replace_once(
            source,
            "        if args.with_reid:\n"
            "            self.encoder = FastReIDInterface(args.fast_reid_config, args.fast_reid_weights, args.device)\n"
            "\n        self.gmc = GMC(method=args.cmc_method, verbose=[args.name, args.ablation])\n",
            "        self.encoder = args.encoder\n"
            "        self.gmc = args.gmc\n",
        )
        detection_indices = "np.flatnonzero(lowest_inds)[remain_inds] if detections else ()"
    source = _replace_once(
        source,
        "        self.start_frame = frame_id\n",
        "        self.start_frame = frame_id\n"
        "        self.last_observed_box = self.tlwh_to_tlbr(self._tlwh)\n",
    )
    source = _replace_once(
        source,
        "        if new_id:\n",
        "        self.last_observed_box = new_track.tlbr\n"
        "        if new_id:\n",
    )
    source = _replace_once(
        source,
        "        self.tracklet_len += 1\n",
        "        self.tracklet_len += 1\n"
        "        self.last_observed_box = new_track.tlbr\n",
    )
    source = _replace_once(
        source,
        "        ''' Add newly detected tracklets to tracked_stracks'''\n",
        f"        for detection, original_index in zip(detections, {detection_indices}):\n"
        "            detection.input_index = int(original_index)\n"
        "\n        ''' Add newly detected tracklets to tracked_stracks'''\n",
    )
    first_assignment = (
        "        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.args.match_thresh)\n"
    )
    source = _replace_once(
        source,
        first_assignment,
        first_assignment
        + "        matches, u_detection = self.reconnect_gate(\n"
        + "            self.frame_id, strack_pool, detections, dists, matches, u_detection, TrackState.Tracked\n"
        + "        )\n",
    )
    return _replace_once(
        source,
        "            track.activate(self.kalman_filter, self.frame_id)\n",
        "            track.next_id = self.next_local_track_id\n"
        "            track.activate(self.kalman_filter, self.frame_id)\n",
    )


def _vendor_tracker_class(vendor_path: Path, vendor_module: str, class_name: str):
    runtime_package = f"_reconnect_study_{vendor_module}"
    with _LOAD_LOCK:
        tracker_module = f"{runtime_package}.{vendor_module}"
        if tracker_module in sys.modules:
            return getattr(sys.modules[tracker_module], class_name)

        package = ModuleType(runtime_package)
        package.__path__ = [str(vendor_path)]
        sys.modules[runtime_package] = package
        loaded = []
        try:
            for name in ("basetrack", "kalman_filter", "matching", vendor_module):
                path = vendor_path / f"{name}.py"
                source = path.read_text(encoding="utf-8").replace("dtype=np.float)", "dtype=float)")
                if name == "matching":
                    upstream_package = "yolox.tracker" if vendor_module == "byte_tracker" else "tracker"
                    source = _replace_once(
                        source, f"from {upstream_package} import kalman_filter", "from . import kalman_filter"
                    )
                if name == vendor_module:
                    source = _tracker_source(source, vendor_module)
                module = ModuleType(f"{runtime_package}.{name}")
                module.__file__ = str(path)
                module.__package__ = runtime_package
                sys.modules[module.__name__] = module
                loaded.append(module.__name__)
                setattr(package, name, module)
                exec(compile(source, str(path), "exec"), module.__dict__)
        except Exception:
            for name in loaded:
                sys.modules.pop(name, None)
            sys.modules.pop(runtime_package, None)
            raise
        return getattr(sys.modules[tracker_module], class_name)


def _box(values) -> Box:
    x1, y1, x2, y2 = values
    return float(x1), float(y1), float(x2), float(y2)


def _comparison(costs: np.ndarray, selected: int, cost: float, threshold: float) -> tuple[float, float]:
    alternatives = np.delete(costs, selected)
    eligible = alternatives[alternatives <= threshold]
    if eligible.size == 0:
        return 1.0, 0.0
    return float(eligible.min() - cost), float(eligible.size)


def _proposal(sequence, frame, track, detection, costs, row, col, threshold) -> ReconnectProposal:
    previous_box = _box(track.last_observed_box)
    candidate_box = _box(detection.tlbr)
    predicted_box = _box(track.tlbr)
    candidate = np.asarray(candidate_box).reshape(2, 2)
    predicted = np.asarray(predicted_box).reshape(2, 2)
    candidate_size = candidate[1] - candidate[0]
    predicted_size = predicted[1] - predicted[0]
    overlap = np.maximum(0.0, np.minimum(candidate[1], predicted[1]) - np.maximum(candidate[0], predicted[0]))
    intersection = float(overlap.prod())
    candidate_area = float(candidate_size.prod())
    predicted_area = float(np.maximum(predicted_size, 1e-12).prod())
    union = candidate_area + predicted_area - intersection
    cost = float(costs[row, col])
    row_margin, row_alternatives = _comparison(costs[row, :], col, cost, threshold)
    column_margin, column_alternatives = _comparison(costs[:, col], row, cost, threshold)
    features = {
        "gap_frames": float(frame - track.end_frame - 1),
        "track_age_frames": float(track.end_frame - track.start_frame + 1),
        "detector_score": float(detection.score),
        "previous_score": float(track.score),
        "iou": intersection / max(union, 1e-12),
        "center_distance": float(np.linalg.norm(candidate.mean(axis=0) - predicted.mean(axis=0)))
        / max(float(np.linalg.norm(predicted_size)), 1e-12),
        "log_area_ratio": float(abs(np.log(candidate_area / predicted_area))),
        "assignment_cost": cost,
        "row_margin": row_margin,
        "column_margin": column_margin,
        "row_alternatives": row_alternatives,
        "column_alternatives": column_alternatives,
    }
    return ReconnectProposal(
        sequence, frame, track.track_id, track.end_frame, previous_box, candidate_box,
        predicted_box, float(detection.score), detection.input_index, features,
    )


def run_bytetrack(
    detections_by_frame: Mapping[int, Sequence[Detection]],
    frame_count: int,
    *,
    policy: Policy | None = None,
    sequence: str = "",
    frame_rate: float = 30,
    track_thresh: float = 0.5,
    track_buffer: int = 30,
    match_thresh: float = 0.8,
    mot20: bool = False,
) -> TrackingResult:
    """Run fresh tracker state on 1-based frames and input pixel-space xyxy boxes.

    The callback sees only the old-track matches from ByteTrack's high-score
    assignment. A rejected detection re-enters the normal unmatched-detection
    pipeline. An alternative old-track assignment is deferred to a later frame.
    Decisions retain the raw pixel boxes before and after the gap for evaluation.
    """
    tracker_class = _vendor_tracker_class(_VENDOR / "ByteTrack" / "yolox" / "tracker", "byte_tracker", "BYTETracker")
    options = SimpleNamespace(
        track_thresh=track_thresh, track_buffer=track_buffer, match_thresh=match_thresh, mot20=mot20,
    )
    tracker = tracker_class(options, frame_rate=frame_rate)

    def update(boxes):
        # Equal source and input dimensions leave the pixel boxes unscaled.
        return tracker.update(boxes, (1, 1), (1, 1))

    return _replay(tracker, update, detections_by_frame, frame_count, policy, sequence, match_thresh)


def run_botsort(
    detections_by_frame: Mapping[int, Sequence[Detection]],
    frame_count: int,
    *,
    policy: Policy | None = None,
    sequence: str = "",
    frame_rate: float = 30,
    track_thresh: float = 0.6,
    track_buffer: int = 30,
    match_thresh: float = 0.8,
    mot20: bool = False,
    track_low_thresh: float = 0.1,
    new_track_thresh: float = 0.7,
    proximity_thresh: float = 0.5,
    appearance_thresh: float = 0.25,
    appearance_by_frame: Mapping[int, Sequence[Sequence[float]]] | None = None,
    camera_warps_by_frame: Mapping[int, Sequence[Sequence[float]]] | None = None,
) -> TrackingResult:
    """Run BoT-SORT with optional cached causal affine warps and ReID vectors.

    A supplied warp maps frame t-1 to frame t. Missing warps use the identity.
    Appearance rows follow the original input detection order, including boxes
    filtered out by low-confidence thresholds. Supply same-dimensional vectors
    for every high-confidence detection for appearance-based matching.
    """
    tracker_class = _vendor_tracker_class(_VENDOR / "BoT-SORT" / "tracker", "bot_sort", "BoTSORT")
    camera_warps = camera_warps_by_frame if camera_warps_by_frame is not None else {}
    identity = np.eye(2, 3)

    def camera_warp(frame, _):
        warp = np.asarray(camera_warps.get(frame, identity), dtype=float)
        if warp.shape != (2, 3) or not np.isfinite(warp).all():
            raise ValueError(f"Invalid camera warp in frame {frame}")
        return warp

    def appearance(frame, high_boxes):
        if len(high_boxes) == 0:
            return ()
        original_scores = boxes[:, 4]
        high = (original_scores > track_low_thresh) & (original_scores > track_thresh)
        vectors = np.asarray(appearance_by_frame.get(frame, ()), dtype=float)
        if vectors.ndim != 2 or vectors.shape[0] != len(boxes):
            raise ValueError(f"Expected one appearance row per detection in frame {frame}")
        selected = vectors[high]
        if not np.isfinite(selected).all() or np.any(np.linalg.norm(selected, axis=1) == 0):
            raise ValueError(f"Invalid high-confidence appearance vectors in frame {frame}")
        return selected

    options = SimpleNamespace(
        track_high_thresh=track_thresh, track_low_thresh=track_low_thresh, new_track_thresh=new_track_thresh,
        track_buffer=track_buffer, match_thresh=match_thresh, mot20=mot20,
        proximity_thresh=proximity_thresh, appearance_thresh=appearance_thresh,
        with_reid=appearance_by_frame is not None, encoder=SimpleNamespace(inference=appearance),
        gmc=SimpleNamespace(apply=camera_warp),
    )
    tracker = tracker_class(options, frame_rate=frame_rate)
    boxes = np.empty((0, 5))

    def update(detections):
        nonlocal boxes
        boxes = detections
        # The vendored tracker forwards the input to the cached warp and feature providers.
        return tracker.update(boxes, tracker.frame_id + 1)

    return _replay(tracker, update, detections_by_frame, frame_count, policy, sequence, match_thresh)


def _replay(tracker, update, detections_by_frame, frame_count, policy, sequence, match_thresh) -> TrackingResult:
    tracker.next_local_track_id = count(1).__next__
    observations = []
    decisions = []

    def gate(frame, tracks, detections, costs, matches, unmatched, tracked_state):
        retained = []
        released = []
        for row, col in matches:
            track = tracks[row]
            if track.state == tracked_state:
                retained.append((row, col))
                continue
            proposal = _proposal(sequence, frame, track, detections[col], costs, row, col, match_thresh)
            accepted = policy is None or bool(policy(proposal))
            decisions.append(ReconnectDecision(proposal, accepted))
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
        for track in update(boxes):
            observations.append(TrackObservation(frame, track.track_id, _box(track.tlbr), float(track.score)))
    return TrackingResult(tuple(observations), tuple(decisions))
