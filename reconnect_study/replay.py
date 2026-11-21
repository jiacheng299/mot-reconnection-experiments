"""Replay complete public sequences and save event-level study evidence."""

import argparse
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
from time import perf_counter

from reconnect_study.data_io import boxes, digest, ignored_boxes, read_manifest
from reconnect_study.events import evaluate_prepared, prepare
from reconnect_study.tracker import run_botsort, run_bytetrack


def load_sequence(sequence):
    det = boxes(sequence.detections, truth=False, origin=sequence.detection_origin)
    gt = boxes(sequence.gt, truth=True, origin=sequence.gt_origin,
               minimum_visibility=sequence.minimum_visibility)
    if max(max(det, default=0), max(gt, default=0)) > sequence.frames:
        raise ValueError(f"Input exceeds declared length: {sequence.name}")
    ignored = ignored_boxes(sequence.gt, sequence.ignored_classes, origin=sequence.gt_origin)
    return det, prepare(det, gt, ignored=ignored)


def replay(sequence, detector, evidence, policy=None, *, include_records=False,
           tracker=run_bytetrack):
    start = perf_counter()
    tracking = tracker(detector, sequence.frames, policy=policy, sequence=sequence.name,
                       frame_rate=sequence.fps)
    tracker_seconds = perf_counter() - start
    outcome = evaluate_prepared(evidence, tracking.tracks, tracking.decisions)
    summary = {"name": sequence.name, "source_id": sequence.source_id,
               "frames": sequence.frames, "split": sequence.split,
               "gaps": outcome.gaps, "correct": outcome.correct, "accepted": outcome.accepted,
               "wrong": outcome.wrong, "unverified_accepted": outcome.unverified_accepted,
               "ignored_accepted": outcome.ignored_accepted,
               "recovery_outcomes": dict(Counter(outcome.recovery_outcomes)),
               "track_rows": len(tracking.tracks), "tracker_seconds": tracker_seconds}
    if include_records:
        summary["actions"] = [asdict(action) for action in outcome.actions]
        summary["gap_records"] = [
            {**asdict(gap), "outcome": status}
            for gap, status in zip(evidence.gaps, outcome.recovery_outcomes, strict=True)
        ]
    return summary, tracking


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--split", action="append")
    parser.add_argument("--tracker", choices=("bytetrack", "botsort-motion-only"), default="bytetrack")
    args = parser.parse_args()
    tracker = run_bytetrack if args.tracker == "bytetrack" else run_botsort
    records = []
    for sequence in read_manifest(args.manifest):
        if args.split and sequence.split not in args.split:
            continue
        detector, evidence = load_sequence(sequence)
        record, _ = replay(sequence, detector, evidence, include_records=True, tracker=tracker)
        record["tracker"] = args.tracker
        records.append(record)
        print({key: record[key] for key in ("name", "frames", "gaps", "correct", "accepted", "wrong")},
              flush=True)
    if not records:
        raise SystemExit("No sequences selected")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output = {"manifest_sha256": digest(args.manifest), "tracker": args.tracker, "sequences": records}
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
