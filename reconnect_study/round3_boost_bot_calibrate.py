"""Calibrate the BoT-SORT first-order BoostTrack adapter using development and calibration only."""

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np

from reconnect_study.data_io import digest, read_manifest
from reconnect_study.round3_ablation import compare_match
from reconnect_study.round3_boost_bot import run_botsort_boost
from reconnect_study.round3_boost_calibrate import ARCHIVE, CAP, OUTPUT, development_scores, run_source, select
from reconnect_study.study import save
from reconnect_study.supplement_matched import load, summaries


ROUND2 = OUTPUT.parent.parent / "round2" / "bot" / "B12"
BOT_OUTPUT = OUTPUT / "bot"


def sweep(executor, cohorts, thresholds, by_policy):
    runner = partial(run_source, thresholds=thresholds, runner=run_botsort_boost)
    count = 0
    for index, (rows, replays) in enumerate(executor.map(runner, cohorts, chunksize=1), 1):
        for policy, entries in rows.items():
            by_policy[policy].extend(entries)
        count += replays
        if index % 10 == 0 or index == len(cohorts):
            print(f"BoT BoostTrack: {index}/{len(cohorts)} sources, {count} replays", flush=True)
    return count


def select_points(aggregate, thresholds, targets, references):
    selected = select(aggregate, thresholds, targets)
    for point, item in selected.items():
        match = compare_match(references[point], item["mean_accepted"], item["target"], item["tolerance"])
        item["calibration_comparison"] = match
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    manifest = ARCHIVE / "manifest.json"
    audit_path = BOT_OUTPUT / "development-numeric-audit.json"
    audit = load(audit_path)
    if (audit["manifest_sha256"] != digest(manifest) or audit["maximum_absolute_error"] > 1e-12
            or not audit.get("unaltered_host_track_rows_checked")):
        raise ValueError("BoT-SORT development audit failed or uses different inputs")
    sequences = read_manifest(manifest)
    development = [row for row in sequences if row.split == "development"]
    cohorts = defaultdict(list)
    for row in sequences:
        if row.split == "calibration":
            cohorts[row.source_id].append(row)
    if len({row.source_id for row in development}) != 16 or len(cohorts) != 91:
        raise ValueError("Unexpected development/calibration cohorts")
    old_lock_path = ROUND2 / "locked.json"
    old_lock = load(old_lock_path)
    old_points = old_lock["selection"]["points"]
    targets = {point: entry["target"] for point, entry in old_points.items() if entry["matched"]}
    old_config = next(path for path in ROUND2.glob("config*.json")
                      if digest(path) == old_lock["config_sha256"])
    quantiles = load(old_config)["quantiles"]
    references = {point: old_points[point]["nearest"]["structured"]["mean_accepted"] for point in targets}
    values = development_scores(development, run_botsort_boost)
    original = np.unique(np.quantile(values, quantiles)).tolist()
    dense_grid = (np.arange(35, 98) / 100).tolist()
    thresholds = {f"boost_bot_{index}": value for index, value in enumerate(original)}
    policies = defaultdict(list)
    grouped = [cohorts[key] for key in sorted(cohorts)]
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        replays = sweep(executor, grouped, thresholds, policies)
        aggregate = summaries(policies, CAP)
        selected = select_points(aggregate, thresholds, targets, references)
        if any(not entry["calibration_comparison"]["matched"] for entry in selected.values()):
            save(BOT_OUTPUT / "calibration-original.json", {"thresholds": thresholds, "by_video": policies,
                 "aggregate": aggregate, "selection": selected, "physical_source_replays": replays})
            dense = np.unique(np.quantile(values, dense_grid)).tolist()
            extra = {f"boost_bot_{index}": value for index, value in
                     enumerate(sorted(set(dense) - set(original)), len(original))}
            replays += sweep(executor, grouped, extra, policies)
            thresholds.update(extra)
            aggregate = summaries(policies, CAP)
            selected = select_points(aggregate, thresholds, targets, references)
    config_path = BOT_OUTPUT / "config.json"
    selected_path = BOT_OUTPUT / "calibration.json"
    code = tuple(Path("reconnect_study") / (name + ".py") for name in (
        "round3_boost", "round3_boost_bot", "round3_boost_calibrate", "round3_boost_bot_calibrate",
        "round3_boost_audit", "tracker", "study", "replay", "round3_boost_eval",
    ))
    save(config_path, {"tracker": "botsort-motion-only", "score": "boosttrack_v1_jacobian_host_state_adapter",
         "acceptance_cap": CAP, "study_manifest_sha256": digest(manifest),
         "numeric_audit_sha256": digest(audit_path), "round2_lock_sha256": digest(old_lock_path),
         "source_sha256": {str(path): digest(path) for path in code},
         "quantiles": quantiles, "dense_quantiles": dense_grid, "thresholds": thresholds,
         "development_scores": len(values), "development_sources": sorted({row.source_id for row in development})})
    save(selected_path, {"tracker": "botsort-motion-only", "split": "calibration", "acceptance_cap": CAP,
         "config_sha256": digest(config_path), "by_video": policies, "aggregate": aggregate,
         "selection": selected, "audit": {"physical_source_replays": replays, "threshold_count": len(thresholds)}})
    save(BOT_OUTPUT / "locked.json", {"acceptance_cap": CAP, "config_sha256": digest(config_path),
         "calibration_sha256": digest(selected_path), "round2_lock_sha256": digest(old_lock_path),
         "selection": selected, "test_policies": sorted({entry["policy"] for entry in selected.values()
         if entry["calibration_comparison"]["matched"]}), "previously_analyzed_test_reused": True})
    print(selected, flush=True)


if __name__ == "__main__":
    main()

