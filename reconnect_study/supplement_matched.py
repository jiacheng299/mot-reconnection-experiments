"""Calibrate and lock acceptance-matched PersonPath22 policies."""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import json
import math
from pathlib import Path

import numpy as np

from reconnect_study.data_io import digest, read_manifest
from reconnect_study.policy import RecoveryPolicy, policy_specs
from reconnect_study.replay import load_sequence, replay
from reconnect_study.study import TRACKERS, save


ARCHIVE = Path("runs/reconnect_study/personpath22")
OUTPUT = Path("runs/reconnect_supplement/round2")
FAMILIES = ("assignment", "linear", "structured")
BASELINES = ("unconstrained_tracker", "cap_only")
CAPS = (6, 12, 24)


def load(path):
    return json.loads(path.read_bytes())


class IntervalGate(RecoveryPolicy):
    """Track the interval (lower, upper] that yields this replay's decisions."""

    def __init__(self, score, cutoff, cap):
        super().__init__(score, cutoff, cap)
        self.lower = -math.inf
        self.upper = math.inf

    def __call__(self, proposal):
        if self.accepted >= self.cap:
            return False
        score = self.score(proposal.features)
        if not math.isfinite(score):
            return False
        if score < self.cutoff:
            self.lower = max(self.lower, score)
            return False
        self.upper = min(self.upper, score)
        self.accepted += 1
        return True


def stable_record(record):
    return {key: value for key, value in record.items() if key != "tracker_seconds"}


def run_source(sequences, config, valid_names, *, validate_shared=True):
    tracker = TRACKERS[config["tracker"]]
    cap = config["acceptance_cap"]
    inputs = [(sequence, *load_sequence(sequence)) for sequence in sequences]
    available = policy_specs(config)
    names = set(valid_names)
    records = {}
    executions = 0
    validated = 0
    for name in BASELINES:
        if name not in names:
            continue
        gate = None if name == "unconstrained_tracker" else RecoveryPolicy(None, None, cap)
        records[name] = [stable_record(replay(*entry, gate, tracker=tracker)[0]) for entry in inputs]
        executions += 1
    for family in FAMILIES:
        remaining = sorted((value[1], name) for name, value in available.items()
                           if name in names and name.rsplit("_", 1)[0] == family)
        verification = None
        while remaining:
            threshold, name = remaining[len(remaining) // 2]
            score = available[name][0]
            gate = IntervalGate(score, threshold, cap)
            results = [replay(*entry, gate, tracker=tracker) for entry in inputs]
            executions += 1
            shared = [(cutoff, item) for cutoff, item in remaining if gate.lower < cutoff <= gate.upper]
            if not shared:
                raise ValueError(f"Interval excludes its replayed cutoff: {threshold}")
            rows = [stable_record(record) for record, _ in results]
            records.update((item, rows) for _, item in shared)
            if verification is None and len(shared) > 1:
                cutoff, _ = next(item for item in shared if item[0] != threshold)
                verification = score, cutoff, results
            taken = {item for _, item in shared}
            remaining = [(cutoff, item) for cutoff, item in remaining if item not in taken]
        if validate_shared and verification:
            score, cutoff, expected = verification
            fresh = RecoveryPolicy(score, cutoff, cap)
            for entry, (expected_row, expected_tracking) in zip(inputs, expected, strict=True):
                row, tracking = replay(*entry, fresh, tracker=tracker)
                if stable_record(row) != stable_record(expected_row) or tracking != expected_tracking:
                    raise ValueError(f"Shared trajectory differs from independent replay on {entry[0].name}")
            validated += len(inputs)
    return records, executions, validated


def resample_grid(config, development_path, development_sequences):
    development = load(development_path)
    rows = development["sequences"]
    if any(row["tracker"] != config["tracker"] for row in rows):
        raise ValueError("Development replay uses a different tracker")
    if digest(development_path) != config["baseline_sha256"]:
        raise ValueError("Development replay hash differs from frozen training input")
    if set(config["development_sources"]) != {row["source_id"] for row in development["sequences"]}:
        raise ValueError("Development replay uses a different source cohort")
    actual = {(row["name"], row["source_id"], row["frames"], row["split"]) for row in rows}
    expected = {(row.name, row.source_id, row.frames, row.split) for row in development_sequences}
    if actual != expected or len(rows) != len(expected):
        raise ValueError("Development archive sequence fields differ from the manifest")
    features = [action["features"] for row in rows
                for action in row["actions"] if action.get("evaluable", True)]
    specs = policy_specs(config)
    thresholds = {}
    for family in FAMILIES:
        score = specs[f"{family}_0"][0]
        values = [score(row) for row in features]
        original = config["thresholds"][family]
        reconstructed = np.unique(np.quantile(values, config["quantiles"])).tolist()
        if not np.array_equal(reconstructed, original):
            raise ValueError(f"Cannot reproduce frozen development thresholds for {family}")
        dense = np.unique(np.quantile(values, np.arange(35, 98) / 100)).tolist()
        thresholds[family] = original + sorted(set(dense) - set(original))
    return thresholds


def configure(base, cap, thresholds=None):
    return {
        **base, "acceptance_cap": cap,
        "thresholds": {key: base["thresholds"][key] for key in FAMILIES}
        if thresholds is None else thresholds,
    }


def summaries(policies, cap):
    output = {}
    for name, rows in policies.items():
        source_counts = defaultdict(Counter)
        for row in rows:
            source_counts[row["source_id"]].update(
                {key: row[key] for key in ("accepted", "wrong", "ignored_accepted", "unverified_accepted",
                                          "gaps", "correct", "frames")}
            )
        accepted = [row["accepted"] for row in source_counts.values()]
        runtime = [row["accepted"] + row["ignored_accepted"] for row in source_counts.values()]
        total = sum(source_counts.values(), Counter())
        row = {
            **dict(total), "sources": len(source_counts),
            "mean_accepted": total["accepted"] / len(source_counts),
            "active_source_fraction": sum(count > 0 for count in accepted) / len(source_counts),
            "accepted_quartiles": np.quantile(accepted, [.25, .5, .75]).tolist(),
            "runtime_accepted_quartiles": np.quantile(runtime, [.25, .5, .75]).tolist(),
            "runtime_accepted": sum(runtime),
            "reached_cap_sources": None if name == "unconstrained_tracker"
            else sum(count == cap for count in runtime),
            "correct_accepted": total["accepted"] - total["wrong"],
            "wrong_per_acceptance": total["wrong"] / total["accepted"] if total["accepted"] else None,
            "correct_per_gap": total["correct"] / total["gaps"] if total["gaps"] else None,
        }
        if name != "unconstrained_tracker" and any(count > cap for count in runtime):
            raise ValueError(f"Cap violated under {name}")
        output[name] = row
    return output


def match(config, aggregate, targets=None):
    specs = policy_specs(config)
    candidates = {
        family: [(name, aggregate[name]["mean_accepted"], specs[name][1])
                 for name in specs if name.rsplit("_", 1)[0] == family]
        for family in FAMILIES
    }
    coverage = {key: [min(item[1] for item in rows), max(item[1] for item in rows)]
                for key, rows in candidates.items()}
    overlap = [max(item[0] for item in coverage.values()), min(item[1] for item in coverage.values())]
    if targets is None:
        targets = {}
        if overlap[0] <= overlap[1]:
            for label, fraction in (("low", .25), ("mid", .50), ("high", .75)):
                target = overlap[0] + fraction * (overlap[1] - overlap[0])
                if target not in targets.values():
                    targets[label] = target
    points = {}
    for label, target in targets.items():
        chosen = {}
        for key, rows in candidates.items():
            name, mean, threshold = min(rows, key=lambda item: (
                round(abs(item[1] - target), 12), item[1], -item[2], item[0]))
            chosen[key] = {"policy": name, "threshold": threshold, "mean_accepted": mean,
                           "deviation": mean - target}
        means = [row["mean_accepted"] for row in chosen.values()]
        spread = max(means) - min(means)
        tolerance = max(.10, .05 * target)
        points[label] = {
            "target": target, "tolerance": tolerance, "spread": spread,
            "matched": max(spread, *(abs(item["deviation"]) for item in chosen.values()))
            <= tolerance + 1e-12,
            "nearest": chosen,
        }
    return {"family_coverage": coverage, "common_interval": overlap,
            "targets": targets, "points": points}


def sweep(executor, cohorts, config, label, previous=None):
    names = set(policy_specs(config)) | set(BASELINES)
    if previous is not None:
        names -= previous.keys()
    policies = {name: [] for name in sorted(names)}
    independent = 0
    shared_checks = 0
    if not names:
        return dict(previous), {"physical_source_replays": 0, "validated_shared_replays": 0}
    worker = partial(run_source, config=config, valid_names=sorted(names))
    for index, (rows, count, checks) in enumerate(executor.map(worker, cohorts, chunksize=1), 1):
        independent += count
        shared_checks += checks
        for name, entries in rows.items():
            policies[name].extend(entries)
        if index % 10 == 0 or index == len(cohorts):
            print(f"{label}: {index}/{len(cohorts)} sources; {independent} physical replays", flush=True)
    return {**(previous or {}), **policies}, {"physical_source_replays": independent,
                                           "validated_shared_replays": shared_checks}


def calibrate(key, root, workers):
    base_path = ARCHIVE / f"{key}-config.json"
    manifest = ARCHIVE / "manifest.json"
    development = ARCHIVE / f"development-{key}.json"
    base = load(base_path)
    if base["study_manifest_sha256"] != digest(manifest):
        raise ValueError("Manifest hash changed")
    cohorts = defaultdict(list)
    for row in sorted(read_manifest(manifest), key=lambda item: (item.source_id, item.name)):
        cohorts[row.split, row.source_id].append(row)
    calibration = [rows for (split, _), rows in cohorts.items() if split == "calibration"]
    manifest_rows = load(manifest)["sequences"]
    save(root / "input-splits.json", {
        "manifest_sha256": digest(manifest), "source_config_sha256": digest(base_path),
        "development_sha256": digest(development),
        "code_sha256": {name: digest(Path(__file__).with_name(name)) for name in (
            "supplement_matched.py", "study.py", "tracker.py", "events.py", "policy.py",
            "overall.py", "effect_sizes.py", "replay.py", "scoring.py", "data_io.py")},
        "sequences": [row for row in manifest_rows if row["split"] in ("development", "calibration", "test")],
        "test_scope": "Previously analyzed official test set; exploratory sensitivity analysis.",
    })
    thresholds = resample_grid(base, development, [
        row for (split, _), rows in cohorts.items() if split == "development" for row in rows
    ])
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for cap in CAPS:
            directory = root / f"B{cap}"
            config = configure(base, cap)
            save(directory / "config-original.json", config)
            policies, audit = sweep(executor, calibration, config, f"{key} B={cap} original")
            aggregate = summaries(policies, cap)
            selected = match(config, aggregate)
            save(directory / "calibration-original.json", {
                "by_video": policies, "aggregate": aggregate, "selection": selected, "audit": audit,
                "config_sha256": digest(directory / "config-original.json"),
                "manifest_sha256": digest(manifest),
            })
            output = directory / "calibration-original.json"
            if not selected["points"] or any(not point["matched"] for point in selected["points"].values()):
                config = configure(base, cap, thresholds)
                save(directory / "config-dense.json", config)
                policies, audit = sweep(executor, calibration, config, f"{key} B={cap} dense", policies)
                aggregate = summaries(policies, cap)
                selected = match(config, aggregate, selected["targets"])
                output = directory / "calibration-dense.json"
                save(output, {
                    "by_video": policies, "aggregate": aggregate, "selection": selected, "audit": audit,
                    "config_sha256": digest(directory / "config-dense.json"),
                    "manifest_sha256": digest(manifest),
                    "original_calibration_sha256": digest(directory / "calibration-original.json"),
                })
            save(directory / "config.json", config)
            retained = {row["policy"] for point in selected["points"].values() if point["matched"]
                        for row in point["nearest"].values()}
            lock = {
                "acceptance_cap": cap,
                "inputs_sha256": digest(root / "input-splits.json"),
                "config_sha256": digest(directory / "config.json"),
                "calibration_file": output.name, "calibration_sha256": digest(output),
                "test_policies": [*BASELINES, *sorted(retained)],
                "selection": selected, "previously_analyzed_test_reused": True,
            }
            save(directory / "locked.json", lock)
            print(f"{key} B={cap} locked: {len(retained)} gated policies", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracker", choices=("byte", "bot"), required=True)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    calibrate(args.tracker, OUTPUT / args.tracker, args.workers)


if __name__ == "__main__":
    main()
