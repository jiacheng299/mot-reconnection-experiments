"""Freeze development policies and replay their calibration or test split."""

import argparse
from collections import defaultdict
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from reconnect_study.data_io import digest, read_manifest
from reconnect_study.pilot import counts
from reconnect_study.policy import CONTROLS, RecoveryPolicy, policy_specs
from reconnect_study.replay import load_sequence, replay
from reconnect_study.scoring import fit
from reconnect_study.tracker import run_botsort, run_bytetrack


TRACKERS = {"bytetrack": run_bytetrack, "botsort-motion-only": run_botsort}
MODELS = {"linear": (False, 0.0), "structured": (True, 0.0),
          "structured_rank": (True, 0.35)}


def save(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def freeze(manifest_path, baseline_path, output, device, tracker, quantiles, acceptance_cap):
    sequences = read_manifest(manifest_path)
    development = {sequence.name: sequence for sequence in sequences if sequence.split == "development"}
    replay_data = json.loads(baseline_path.read_bytes())
    if replay_data.get("manifest_sha256", digest(manifest_path)) != digest(manifest_path):
        raise ValueError("Development labels refer to a different input manifest")
    baseline = replay_data["sequences"]
    if (not development or len(development) != len(baseline)
            or set(development) != {entry["name"] for entry in baseline}):
        raise ValueError("Baseline must use exactly the manifest development videos")
    if any(row.get("tracker", "bytetrack") != tracker for row in baseline):
        raise ValueError("Development replay tracker differs from the selected tracker")
    for row in baseline:
        source = development[row["name"]]
        if (source.source_id, source.frames, source.split) != (row["source_id"], row["frames"], row["split"]):
            raise ValueError(f"Development replay differs from manifest for {source.name}")
    rows = [(development[entry["name"]].source_id, action["features"], action["correct"])
            for entry in baseline for action in entry["actions"] if action.get("evaluable", True)]
    training, scorers = {}, dict(CONTROLS)
    for name, (structured, rank) in MODELS.items():
        model, report = fit(rows, structured=structured, ranking_weight=rank, device=device)
        training[name] = {"score": asdict(model), "fit": report}
        scorers[name] = model
    thresholds = {name: np.unique(np.quantile([scorer(row[1]) for row in rows], quantiles)).tolist()
                  for name, scorer in scorers.items()}
    save(output, {"tracker": tracker, "acceptance_cap": acceptance_cap, "training": training,
                  "quantiles": quantiles, "thresholds": thresholds,
                  "development_sources": sorted({item.source_id for item in development.values()}),
                  "baseline_sha256": digest(baseline_path), "study_manifest_sha256": digest(manifest_path)})


def run_split(manifest_path, config_path, split, output, required_policies=None,
              overall_protocol=None, source_filter=None, trace_directory=None, transfer=False):
    config = json.loads(config_path.read_bytes())
    if not transfer and config["study_manifest_sha256"] != digest(manifest_path):
        raise ValueError("Frozen configuration was built for a different source manifest")
    if transfer and required_policies is None:
        raise ValueError("Transfer requires selected frozen policies")
    sequences = [sequence for sequence in read_manifest(manifest_path) if sequence.split == split
                 and (source_filter is None or sequence.source_id in source_filter)]
    if not sequences:
        raise ValueError(f"No {split} sources")
    tracker = TRACKERS[config["tracker"]]
    policies = {"unconstrained_tracker": (None, None), **policy_specs(config)}
    if required_policies is not None and config["acceptance_cap"] is not None:
        policies["cap_only"] = (None, None)
    if required_policies is not None:
        policies = {name: policies[name] for name in required_policies}
    metrics = {}
    if overall_protocol:
        from reconnect_study.overall import OverallEvaluator

        metrics = {name: OverallEvaluator(overall_protocol) for name in policies}
    all_rows = defaultdict(list)
    source_gates = {}
    for sequence in sorted(sequences, key=lambda item: (item.source_id, item.name)):
        detector, evidence = load_sequence(sequence)
        for name, (score, threshold) in policies.items():
            gate = None
            if name != "unconstrained_tracker":
                key = sequence.source_id, name
                if key not in source_gates:
                    source_gates[key] = RecoveryPolicy(score, threshold, config["acceptance_cap"])
                gate = source_gates[key]
            record, tracking = replay(sequence, detector, evidence, gate, tracker=tracker,
                                      include_records=trace_directory is not None)
            if trace_directory is not None:
                trace_path = trace_directory / name / f"{sequence.name}.json"
                save(trace_path, {"tracks": [asdict(track) for track in tracking.tracks],
                                  "decisions": [asdict(decision) for decision in tracking.decisions],
                                  "actions": record.pop("actions"),
                                  "gap_records": record.pop("gap_records")})
                record["trace_sha256"] = digest(trace_path)
            all_rows[name].append(record)
            if metrics:
                metrics[name].add(sequence, evidence, tracking)
        print(sequence.name, len(policies), "policies", flush=True)
    save(output, {"tracker": config["tracker"], "split": split, "config_sha256": digest(config_path),
                  "benchmark_manifest_sha256": digest(manifest_path), "transfer": transfer,
                  "acceptance_cap": config["acceptance_cap"], "policies": all_rows,
                  "aggregate": {name: counts(rows) for name, rows in all_rows.items()},
                  "overall": {name: evaluator.summary() for name, evaluator in metrics.items()}})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)
    configure = subcommands.add_parser("freeze")
    configure.add_argument("manifest", type=Path)
    configure.add_argument("baseline", type=Path)
    configure.add_argument("output", type=Path)
    configure.add_argument("--device", default="cuda")
    configure.add_argument("--tracker", choices=tuple(TRACKERS), default="bytetrack")
    configure.add_argument("--quantiles", type=float, nargs="+", default=(0.35, 0.6, 0.8, 0.93))
    configure.add_argument("--acceptance-cap", type=int)
    evaluate = subcommands.add_parser("replay")
    evaluate.add_argument("manifest", type=Path)
    evaluate.add_argument("config", type=Path)
    evaluate.add_argument("split")
    evaluate.add_argument("output", type=Path)
    evaluate.add_argument("--policy", action="append")
    evaluate.add_argument("--locked-policies", type=Path)
    evaluate.add_argument("--overall-protocol")
    finite = evaluate.add_mutually_exclusive_group()
    finite.add_argument("--finite-calibration-piece", type=Path)
    finite.add_argument("--finite-complement-piece", type=Path)
    evaluate.add_argument("--trace-directory", type=Path)
    evaluate.add_argument("--transfer", action="store_true")
    args = parser.parse_args()
    if args.command == "freeze":
        if args.acceptance_cap is not None and args.acceptance_cap < 1:
            parser.error("--acceptance-cap must be positive")
        freeze(args.manifest, args.baseline, args.output, args.device, args.tracker,
               args.quantiles, args.acceptance_cap)
    else:
        if args.policy and args.locked_policies:
            parser.error("--policy and --locked-policies cannot be combined")
        selected = args.policy
        if args.locked_policies:
            locked = json.loads(args.locked_policies.read_bytes())
            if locked["config_sha256"] != digest(args.config):
                raise ValueError("Locked selection uses a different configuration")
            selected = locked["test_policies"]
        source_filter = None
        finite_design = args.finite_calibration_piece or args.finite_complement_piece
        if finite_design:
            design = json.loads(finite_design.read_bytes())
            if design["benchmark_manifest_sha256"] != digest(args.manifest):
                raise ValueError("Finite-pool design uses a different manifest")
            source_filter = set(design["calibration_sample"] if args.finite_calibration_piece
                                else design["held_out_complement"])
        run_split(args.manifest, args.config, args.split, args.output, selected,
                  args.overall_protocol, source_filter, args.trace_directory, args.transfer)


if __name__ == "__main__":
    main()
