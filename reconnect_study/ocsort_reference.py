"""Evaluate frozen OC-SORT as an external recovery and overall tracking reference."""

from __future__ import annotations

import argparse
from collections import Counter
from functools import lru_cache
from importlib import import_module
from importlib.metadata import version
import json
from pathlib import Path
from subprocess import check_output
import sys
from threading import Lock
from time import perf_counter
from types import ModuleType

import numpy as np

from reconnect_study.data_io import digest, read_manifest
from reconnect_study.events import evaluate_prepared
from reconnect_study.overall import OverallEvaluator
from reconnect_study.replay import load_sequence
from reconnect_study.tracker import TrackObservation, TrackingResult


OCSORT_COMMIT = "8462e7e729a93ccd3bd995c0a79a890336cb3a0b"
_VENDOR = Path(__file__).resolve().parents[1] / "vendor" / "OC_SORT"
_PACKAGE = "_reconnect_study_ocsort_vendor"
_RUN_LOCK = Lock()
_OPTIONS = {
    "det_thresh": 0.6,
    "max_age": 30,
    "min_hits": 3,
    "iou_threshold": 0.3,
    "delta_t": 3,
    "asso_func": "iou",
    "inertia": 0.2,
    "use_byte": False,
}


@lru_cache(maxsize=1)
def _vendor_class():
    command = ["git", "-C", str(_VENDOR)]
    commit = check_output([*command, "rev-parse", "HEAD"], text=True).strip()
    changes = check_output([*command, "status", "--porcelain", "--", "trackers/ocsort_tracker"], text=True)
    if commit != OCSORT_COMMIT or changes:
        raise ValueError("Expected a clean checkout of the pinned OC-SORT tracker modules")

    module = ModuleType(_PACKAGE)
    module.__path__ = [str(_VENDOR / "trackers")]
    sys.modules[_PACKAGE] = module
    return import_module(f"{_PACKAGE}.ocsort_tracker.ocsort").OCSort


def run_ocsort(detections_by_frame, frame_count):
    """Run the frozen motion-only vendor tracker on sequential xyxy score rows."""
    if frame_count < 1:
        raise ValueError("OC-SORT needs a positive frame count")
    observations = []
    # Upstream stores its sequence-local ID counter on a Kalman track class.
    with _RUN_LOCK:
        tracker = _vendor_class()(**_OPTIONS)
        for frame in range(1, frame_count + 1):
            boxes = np.array(detections_by_frame.get(frame, ()), dtype=float, copy=True).reshape(-1, 5)
            if not np.isfinite(boxes).all() or np.any(boxes[:, 2:4] <= boxes[:, :2]):
                raise ValueError(f"Invalid OC-SORT detection rows in frame {frame}")
            # Applied to already unscaled pixels, including empty frames.
            for x1, y1, x2, y2, track_id in tracker.update(boxes, (1, 1), (1, 1)):
                box = float(x1), float(y1), float(x2), float(y2)
                observations.append(TrackObservation(frame, int(track_id), box, 1.0))
    return TrackingResult(tuple(observations), ())


def _recovery_summary(outcomes):
    gaps = sum(outcomes.values())
    correct = outcomes["correct"]
    return {"gaps": gaps, "correct": correct, "correct_per_gap": correct / gaps if gaps else None,
            "outcomes": dict(outcomes)}


def evaluate_reference(sequences, *, protocol="personpath22-native-keyframe"):
    """Replay the fixed input once per video and use the study's natural-gap denominator."""
    overall = OverallEvaluator(protocol)
    all_outcomes = Counter()
    records = {}
    for sequence in sequences:
        detections, evidence = load_sequence(sequence)
        start = perf_counter()
        tracking = run_ocsort(detections, sequence.frames)
        elapsed = perf_counter() - start
        overall.add(sequence, evidence, tracking)
        event = evaluate_prepared(evidence, tracking.tracks, tracking.decisions)
        outcomes = Counter(event.recovery_outcomes)
        records[sequence.name] = {"source_id": sequence.source_id, "frames": sequence.frames,
                                  "track_rows": len(tracking.tracks), "tracker_seconds": elapsed,
                                  **_recovery_summary(outcomes)}
        all_outcomes.update(outcomes)
    return {"tracker": "OC-SORT", "vendor_commit": OCSORT_COMMIT, "tracker_options": dict(_OPTIONS),
            "variant": "Vendor private-detection update on a shared cached detector output, without appearance, "
                       "camera motion compensation, low-score BYTE association or output box filtering.",
            "parameter_units": {"max_age": "input frames", "min_hits": "consecutive observations", "delta_t": "input frames"},
            "filterpy_version": version("filterpy"), "commitment_risk_available": False,
            "recovery": {"sequences": records, "aggregate": _recovery_summary(all_outcomes)},
            "overall": overall.summary()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--split", default="test")
    parser.add_argument("--sequence", action="append")
    parser.add_argument("--protocol", default="personpath22-native-keyframe")
    args = parser.parse_args()

    selected = [sequence for sequence in read_manifest(args.manifest)
                if sequence.split == args.split and (not args.sequence or sequence.name in args.sequence)]
    missing = set(args.sequence or ()) - {sequence.name for sequence in selected}
    if missing:
        parser.error(f"Unknown or excluded sequence(s): {', '.join(sorted(missing))}")
    if not selected:
        parser.error("No sequences selected")
    result = evaluate_reference(selected, protocol=args.protocol)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {"manifest_sha256": digest(args.manifest), "split": args.split,
              "sequence_count": len(selected), "source_count": len({sequence.source_id for sequence in selected}),
              **result}
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(selected)} sequence results to {args.output} (SHA-256 {digest(args.output)})")


if __name__ == "__main__":
    main()
