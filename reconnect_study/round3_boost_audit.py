"""Numerically audit the BoostTrack v1 host-state adapter on development sources."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from reconnect_study.data_io import digest, read_manifest
from reconnect_study.replay import load_sequence, replay
from reconnect_study.round3_boost import run_bytetrack_boost
from reconnect_study.round3_boost_bot import run_botsort_boost
from reconnect_study.study import save
from reconnect_study.tracker import run_botsort


def load_author(upstream):
    sys.path.insert(0, str(upstream))
    try:
        settings_spec = importlib.util.spec_from_file_location("default_settings", upstream / "default_settings.py")
        settings = importlib.util.module_from_spec(settings_spec)
        settings_spec.loader.exec_module(settings)
        sys.modules["default_settings"] = settings
        spec = importlib.util.spec_from_file_location("round3_author_assoc", upstream / "tracker" / "assoc.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.pop(0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path,
                        default=Path("runs/reconnect_study/personpath22/manifest.json"))
    parser.add_argument("--baseline", type=Path,
                        default=Path("runs/reconnect_study/personpath22/development-byte.json"))
    parser.add_argument("--upstream", type=Path,
                        default=Path("runs/reconnect_supplement/round3/boost/upstream/BoostTrack"))
    parser.add_argument("--output", type=Path,
                        default=Path("runs/reconnect_supplement/round3/boost/development-numeric-audit.json"))
    parser.add_argument("--host", choices=("byte", "bot"), default="byte")
    args = parser.parse_args()
    author = load_author(args.upstream.resolve())
    baseline = json.loads(args.baseline.read_bytes())
    archived = {row["name"]: row for row in baseline["sequences"]}
    sequences = [row for row in read_manifest(args.manifest) if row.split == "development"]
    maximum_error, matrices, elements, rank_checks, action_checks, track_checks = 0.0, 0, 0, 0, 0, 0
    per_sequence = []
    for sequence in sequences:
        detector, evidence = load_sequence(sequence)

        def observe(payload):
            nonlocal maximum_error, matrices, elements, rank_checks
            expected = author.associate(
                payload["detections"], payload["tracks"], payload["iou_threshold"],
                mahalanobis_distance=payload["mahalanobis_distance"],
                track_confidence=payload["track_confidence"],
                detection_confidence=payload["detection_confidence"],
                lambda_iou=.5, lambda_mhd=.25, lambda_shape=.25,
            )[3]
            difference = np.max(np.abs(expected - payload["similarity"]), initial=0.0)
            maximum_error = max(maximum_error, float(difference))
            if not np.array_equal(np.argsort(expected, axis=0),
                                  np.argsort(payload["similarity"], axis=0)):
                raise ValueError(f"Author/adapter rank mismatch in {sequence.name} frame {payload['frame']}")
            matrices += 1
            elements += expected.size
            rank_checks += expected.shape[1]

        adapter = run_botsort_boost if args.host == "bot" else run_bytetrack_boost
        tracker = lambda detections, frames, **kwargs: adapter(
            detections, frames, matrix_observer=observe, **kwargs)
        record, adapted = replay(sequence, detector, evidence, None, tracker=tracker, include_records=True)
        if args.host == "bot":
            _, native = replay(sequence, detector, evidence, None, tracker=run_botsort)
            if native.tracks != adapted.tracks:
                raise ValueError(f"Ungated host output changed in {sequence.name}")
            track_checks += len(native.tracks)
        old_actions, new_actions = archived[sequence.name]["actions"], record["actions"]
        if len(old_actions) != len(new_actions):
            raise ValueError(f"Action count changed in {sequence.name}")
        for old, new in zip(old_actions, new_actions, strict=True):
            stripped = {key: value for key, value in new["features"].items()
                        if key != "boosttrack_v1_similarity"}
            old_key = (old["frame"], old["previous_frame"], old["track_id"],
                       old["accepted"], old["features"])
            new_key = (new["frame"], new["previous_frame"], new["track_id"],
                       new["accepted"], stripped)
            if old_key != new_key:
                raise ValueError(f"Frozen proposal changed in {sequence.name}")
            action_checks += 1
        per_sequence.append({"name": sequence.name, "frames": sequence.frames,
                             "actions": len(new_actions),
                             "boost_score_min": min((a["features"]["boosttrack_v1_similarity"]
                                                     for a in new_actions), default=None),
                             "boost_score_max": max((a["features"]["boosttrack_v1_similarity"]
                                                     for a in new_actions), default=None)})
        print(sequence.name, len(new_actions), "aligned actions", flush=True)
    commit = subprocess.run(["git", "-C", str(args.upstream), "rev-parse", "HEAD"],
                            check=True, capture_output=True, text=True).stdout.strip()
    if not matrices or not action_checks or maximum_error > 1e-12:
        raise ValueError(f"Insufficient coverage or author/adapter numerical agreement: {maximum_error}")
    save(args.output, {
        "admission": "BoostTrack-v1 host-state adapter; not the full BoostTrack tracker",
        "upstream_url": "https://github.com/vukasin-stanojevic/BoostTrack",
        "upstream_commit": commit,
        "author_mode": ["--no_reid", "--btpp_arg_iou_boost", "--btpp_arg_no_sb", "--btpp_arg_no_vt"],
        "shape_similarity": "v1 released implementation; height term also divides by max(dw, tw)",
        "state_adaptation": ("BoT-SORT xywh mapped to xyhr by width/height and J Sigma J^T; first-order approximation"
                             if args.host == "bot" else
                             "ByteTrack xyah mean/covariance reordered to author xyhr; host motion retained"),
        **({"unaltered_host_track_rows_checked": track_checks} if args.host == "bot" else {}),
        "manifest_sha256": digest(args.manifest), "baseline_sha256": digest(args.baseline),
        "development_sources": len(sequences), "matrices_checked": matrices,
        "matrix_elements_checked": elements, "rank_columns_checked": rank_checks,
        "maximum_absolute_error": maximum_error, "frozen_actions_checked": action_checks,
        "sequences": per_sequence,
    })


if __name__ == "__main__":
    main()
