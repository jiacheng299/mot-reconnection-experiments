"""Compute overall tracking metrics with TrackEval, independently of gap labels."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from importlib.metadata import version
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from trackeval.datasets import MotChallenge2DBox, PersonPath22
from trackeval.metrics import CLEAR, HOTA, Identity

from reconnect_study.data_io import boxes, read_manifest


_PERSONPATH_NATIVE = "personpath22-native-keyframe"
_PREPROCESSING = {
    "dancetrack": "person-only ground truth, all tracker predictions, box IoU",
    "mot17": "TrackEval MOT17 pedestrian and distractor preprocessing on raw ground truth",
    "mot20": "TrackEval MOT20 pedestrian and distractor preprocessing on raw ground truth",
    _PERSONPATH_NATIVE: ("Project PersonPath22 native 5 Hz annotated keyframes. TrackEval PersonPath22 "
                         "distractor classes and crowd-ignore boxes. Dense benchmark scores use a separate frame grid."),
}


def write_mot_rows(path, tracks, *, origin=1):
    """Write TrackEval-compatible xywh rows from zero-origin xyxy observations."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        for row in sorted(tracks, key=lambda record: record.frame):
            x1, y1, x2, y2 = row.box
            writer.writerow((row.frame, row.track_id, x1 + origin, y1 + origin, x2 - x1, y2 - y1,
                             row.score, -1, -1, -1))


def _validate_tracks(sequence, tracks):
    seen = set()
    for row in tracks:
        key = (row.frame, row.track_id)
        if not 1 <= row.frame <= sequence.frames or row.track_id < 0 or key in seen:
            raise ValueError(f"Invalid or duplicate tracker observation in {sequence.name}: {key}")
        seen.add(key)
        x1, y1, x2, y2 = row.box
        if not np.isfinite((x1, y1, x2, y2, row.score)).all() or x2 <= x1 or y2 <= y1:
            raise ValueError(f"Invalid tracker box or score in {sequence.name}: {key}")


def _contiguous_ids(identities):
    mapping = {identity: index for index, identity in enumerate(sorted({i for frame in identities for i in frame}))}
    return [np.array([mapping[identity] for identity in frame], dtype=int) for frame in identities], len(mapping)


def _dance_data(sequence, evidence, tracks):
    # A visibility filter can be useful for gap labels but changes the benchmark's GT denominator.
    truth = (evidence.truth if evidence is not None and sequence.minimum_visibility is None
             else boxes(sequence.gt, truth=True, origin=sequence.gt_origin))
    if any(frame < 1 or frame > sequence.frames for frame in truth):
        raise ValueError(f"Ground truth exceeds the declared frame range in {sequence.name}")

    outputs = defaultdict(list)
    for row in tracks:
        outputs[row.frame].append(row)

    gt_ids, tracker_ids, similarities = [], [], []
    for frame in range(1, sequence.frames + 1):
        people = truth.get(frame, {})
        predictions = outputs[frame]
        gt_ids.append(tuple(people.keys()))
        tracker_ids.append(tuple(row.track_id for row in predictions))
        gt_boxes = np.asarray(tuple(people.values()), dtype=float).reshape(-1, 4)
        tracker_boxes = np.asarray([row.box for row in predictions], dtype=float).reshape(-1, 4)
        similarities.append(MotChallenge2DBox._calculate_box_ious(gt_boxes, tracker_boxes, box_format="x0y0x1y1"))

    num_gt_dets = sum(map(len, gt_ids))
    num_tracker_dets = sum(map(len, tracker_ids))
    gt_ids, num_gt_ids = _contiguous_ids(gt_ids)
    tracker_ids, num_tracker_ids = _contiguous_ids(tracker_ids)
    return {"seq": sequence.name, "num_timesteps": sequence.frames,
            "num_gt_dets": num_gt_dets, "num_tracker_dets": num_tracker_dets,
            "num_gt_ids": num_gt_ids, "num_tracker_ids": num_tracker_ids,
            "gt_ids": gt_ids, "tracker_ids": tracker_ids, "similarity_scores": similarities}


def _mot_data(sequence, tracks, protocol):
    # Preserve raw annotations for TrackEval's joint assignment to pedestrians and distractors.
    if protocol == _PERSONPATH_NATIVE:
        dataset_type = PersonPath22
        benchmark = "person_path_22"
    else:
        dataset_type = MotChallenge2DBox
        benchmark = protocol.upper()
    with TemporaryDirectory(prefix="reconnect-trackeval-") as temporary:
        root = Path(temporary)
        raw_gt = Path(sequence.gt).resolve()
        key = "sequence"
        write_mot_rows(root / "replay" / "data" / f"{key}.txt", tracks, origin=sequence.gt_origin)
        dataset = dataset_type({
            "GT_FOLDER": str(raw_gt.parent), "GT_LOC_FORMAT": "{gt_folder}/" + raw_gt.name,
            "TRACKERS_FOLDER": str(root), "TRACKERS_TO_EVAL": ["replay"], "CLASSES_TO_EVAL": ["pedestrian"],
            "BENCHMARK": benchmark, "SKIP_SPLIT_FOL": True, "SEQ_INFO": {key: sequence.frames},
            "DO_PREPROC": True, "PRINT_CONFIG": False,
        })
        raw = dataset.get_raw_seq_data("replay", key)
        raw["seq"] = sequence.name
        return dataset.get_preprocessed_seq_data(raw, "pedestrian")


def _summary(hota, clear, identity):
    return {"HOTA": 100 * float(np.mean(hota["HOTA"])), "AssA": 100 * float(np.mean(hota["AssA"])),
            "IDF1": 100 * float(identity["IDF1"]), "IDSW": int(clear["IDSW"])}


class OverallEvaluator:
    """Collect raw official metric outputs and combine their sufficient statistics."""

    def __init__(self, protocol="dancetrack"):
        protocol = protocol.lower()
        if protocol not in _PREPROCESSING:
            raise ValueError(f"Unsupported overall evaluation protocol: {protocol}")
        self.protocol = protocol
        self._metrics = (HOTA(), CLEAR({"PRINT_CONFIG": False}), Identity({"PRINT_CONFIG": False}))
        self._results = {}
        self._source_ids = {}

    def add(self, sequence, evidence, tracking):
        if sequence.name in self._results:
            raise ValueError(f"Duplicate overall evaluation sequence: {sequence.name}")
        if sequence.frames < 1:
            raise ValueError(f"Overall evaluation requires positive frame count: {sequence.name}")
        if self.protocol == _PERSONPATH_NATIVE and sequence.fps != 5.0:
            raise ValueError(f"PersonPath22 native-keyframe evaluation requires a 5 Hz replay: {sequence.name}")
        _validate_tracks(sequence, tracking.tracks)
        data = (_dance_data(sequence, evidence, tracking.tracks) if self.protocol == "dancetrack"
                else _mot_data(sequence, tracking.tracks, self.protocol))
        results = tuple(metric.eval_sequence(data) for metric in self._metrics)
        self._results[sequence.name] = results
        self._source_ids[sequence.name] = sequence.source_id
        return _summary(*results)

    def summary(self):
        if not self._results:
            raise ValueError("No overall evaluation sequences were added")
        combined = [metric.combine_sequences({name: results[index] for name, results in self._results.items()})
                    for index, metric in enumerate(self._metrics)]
        sequences = {name: _summary(*results) for name, results in self._results.items()}
        idsw_by_source = defaultdict(int)
        for name, scores in sequences.items():
            # Each tracker run has a separate identity namespace, even when a source has multiple clips.
            idsw_by_source[self._source_ids[name]] += scores["IDSW"]
        return {"protocol": self.protocol, "preprocessing": _PREPROCESSING[self.protocol],
                "trackeval_version": version("trackeval"),
                "unit": {"HOTA": "percent", "AssA": "percent", "IDF1": "percent", "IDSW": "count"},
                "sequences": sequences, "idsw_by_source": dict(idsw_by_source),
                "aggregate": _summary(*combined)}


def evaluate_overall(rows, *, protocol="dancetrack"):
    """Evaluate an iterable of (Sequence, Evidence, TrackingResult) rows."""
    evaluator = OverallEvaluator(protocol)
    for sequence, evidence, tracking in rows:
        evaluator.add(sequence, evidence, tracking)
    return evaluator.summary()


def main():
    from reconnect_study.tracker import run_botsort, run_bytetrack

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--split", action="append", help="Only evaluate the given manifest split")
    parser.add_argument("--sequence", action="append", help="Only evaluate the given sequence name")
    parser.add_argument("--protocol", choices=_PREPROCESSING, default="dancetrack")
    parser.add_argument("--tracker", choices=("bytetrack", "botsort-motion"), default="bytetrack")
    args = parser.parse_args()

    evaluator = OverallEvaluator(args.protocol)
    tracker = run_bytetrack if args.tracker == "bytetrack" else run_botsort
    selected = [sequence for sequence in read_manifest(args.manifest)
                if (not args.split or sequence.split in args.split)
                and (not args.sequence or sequence.name in args.sequence)]
    missing = set(args.sequence or ()) - {sequence.name for sequence in selected}
    if missing:
        parser.error(f"Unknown or excluded sequence(s): {', '.join(sorted(missing))}")
    if not selected:
        parser.error("No sequences selected")
    for sequence in selected:
        detections = boxes(sequence.detections, truth=False, origin=sequence.detection_origin)
        if max(detections, default=0) > sequence.frames:
            raise ValueError(f"Detection input exceeds the frame count: {sequence.name}")
        tracking = tracker(detections, sequence.frames, sequence=sequence.name, frame_rate=sequence.fps)
        measured = evaluator.add(sequence, None, tracking)
        print(sequence.name, measured, flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {"tracker": args.tracker, "policy": "accept-all", **evaluator.summary()}
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
