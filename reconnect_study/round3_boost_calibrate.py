"""Calibrate and lock the ByteTrack BoostTrack-v1 adapter at round-two targets."""

from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import json
from pathlib import Path

import numpy as np

from reconnect_study.data_io import digest, read_manifest
from reconnect_study.replay import load_sequence, replay
from reconnect_study.round3_boost import run_bytetrack_boost
from reconnect_study.study import save
from reconnect_study.supplement_matched import IntervalGate, stable_record, summaries


ARCHIVE = Path("runs/reconnect_study/personpath22")
ROUND2 = Path("runs/reconnect_supplement/round2/byte/B12")
OUTPUT = Path("runs/reconnect_supplement/round3/boost")
CAP = 12


def score(features):
    return features["boosttrack_v1_similarity"]


def tracker(detections, frames, **kwargs):
    return run_bytetrack_boost(detections, frames, **kwargs)


def development_scores(sequences, runner=tracker):
    values = []
    for sequence in sequences:
        detector, evidence = load_sequence(sequence)
        record, _ = replay(sequence, detector, evidence, None, tracker=runner, include_records=True)
        values.extend(action["features"]["boosttrack_v1_similarity"] for action in record["actions"]
                      if action.get("evaluable", True))
    return np.asarray(values, dtype=float)


def run_source(sequences, thresholds, runner=tracker):
    inputs = [(sequence, *load_sequence(sequence)) for sequence in sequences]
    remaining = sorted((threshold, name) for name, threshold in thresholds.items())
    records, executions = {}, 0
    while remaining:
        threshold, name = remaining[len(remaining) // 2]
        gate = IntervalGate(score, threshold, CAP)
        rows = [replay(*entry, gate, tracker=runner)[0] for entry in inputs]
        executions += 1
        shared = [(cutoff, item) for cutoff, item in remaining if gate.lower < cutoff <= gate.upper]
        if not shared:
            raise ValueError(f"Interval excludes replayed cutoff {threshold}")
        stable = [stable_record(row) for row in rows]
        records.update((item, stable) for _, item in shared)
        taken = {item for _, item in shared}
        remaining = [(cutoff, item) for cutoff, item in remaining if item not in taken]
    return records, executions


def select(aggregate, thresholds, targets):
    points = {}
    candidates = [(name, row["mean_accepted"], thresholds[name])
                  for name, row in aggregate.items()]
    for label, target in targets.items():
        name, mean, threshold = min(candidates, key=lambda item: (
            round(abs(item[1] - target), 12), item[1], -item[2], item[0]))
        tolerance = max(.10, .05 * target)
        points[label] = {"target": target, "tolerance": tolerance, "policy": name,
                         "threshold": threshold, "mean_accepted": mean,
                         "deviation": mean - target,
                         "matched": abs(mean - target) <= tolerance + 1e-12}
    return points


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    manifest = ARCHIVE / "manifest.json"
    sequences = read_manifest(manifest)
    development = [row for row in sequences if row.split == "development"]
    cohorts = defaultdict(list)
    for row in sequences:
        if row.split == "calibration":
            cohorts[row.source_id].append(row)
    values = development_scores(development)
    quantiles = np.arange(35, 98) / 100
    threshold_values = np.unique(np.quantile(values, quantiles)).tolist()
    thresholds = {f"boost_{index}": value for index, value in enumerate(threshold_values)}
    round2_lock = json.loads((ROUND2 / "locked.json").read_bytes())
    targets = {label: point["target"] for label, point in round2_lock["selection"]["points"].items()
               if label in ("mid", "high") and point["matched"]}
    config = {
        "tracker": "bytetrack", "score": "boosttrack_v1_host_state_adapter", "acceptance_cap": CAP,
        "quantiles": quantiles.tolist(), "thresholds": thresholds,
        "development_sources": sorted({row.source_id for row in development}),
        "development_scores": len(values), "study_manifest_sha256": digest(manifest),
        "round2_lock_sha256": digest(ROUND2 / "locked.json"),
        "numeric_audit_sha256": digest(OUTPUT / "development-numeric-audit.json"),
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    save(OUTPUT / "config.json", config)
    policies = {name: [] for name in thresholds}
    executions = 0
    worker = partial(run_source, thresholds=thresholds)
    cohort_rows = [cohorts[key] for key in sorted(cohorts)]
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        for index, (rows, count) in enumerate(executor.map(worker, cohort_rows, chunksize=1), 1):
            for name, entries in rows.items():
                policies[name].extend(entries)
            executions += count
            if index % 10 == 0 or index == len(cohort_rows):
                print(f"boost calibration: {index}/{len(cohort_rows)} sources; {executions} physical replays",
                      flush=True)
    aggregate = summaries(policies, CAP)
    selected = select(aggregate, thresholds, targets)
    calibration = {"tracker": "bytetrack", "split": "calibration", "acceptance_cap": CAP,
                   "config_sha256": digest(OUTPUT / "config.json"), "by_video": policies,
                   "aggregate": aggregate, "selection": selected,
                   "audit": {"physical_source_replays": executions, "threshold_count": len(thresholds)}}
    save(OUTPUT / "calibration.json", calibration)
    lock = {
        "acceptance_cap": CAP, "config_sha256": digest(OUTPUT / "config.json"),
        "calibration_sha256": digest(OUTPUT / "calibration.json"),
        "round2_lock_sha256": digest(ROUND2 / "locked.json"), "selection": selected,
        "test_policies": sorted({point["policy"] for point in selected.values() if point["matched"]}),
        "previously_analyzed_test_reused": True,
    }
    save(OUTPUT / "locked.json", lock)
    print(json.dumps({"selection": selected, "test_policies": lock["test_policies"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
