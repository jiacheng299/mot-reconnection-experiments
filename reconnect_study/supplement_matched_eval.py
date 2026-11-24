"""Audit locked matched-acceptance test replays and compute paired effects.

The lock is read-only input: this module never selects or changes a policy.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path

import numpy as np

from reconnect_study.data_io import digest
from reconnect_study.effect_sizes import compare
from reconnect_study.study import save


COUNT_KEYS = ("accepted", "wrong", "unverified_accepted", "ignored_accepted", "correct", "gaps")


def validate_lock_files(lock, lock_path, config_path):
    lock_path, config_path = Path(lock_path), Path(config_path)
    calibration = lock_path.parent / lock["calibration_file"]
    inputs = lock_path.parent.parent / "input-splits.json"
    checks = ((config_path, lock["config_sha256"], "configuration"),
              (calibration, lock["calibration_sha256"], "calibration"),
              (inputs, lock["inputs_sha256"], "input split"))
    for path, expected, label in checks:
        if not path.is_file() or digest(path) != expected:
            raise ValueError(f"Locked {label} file is missing or has changed: {path}")


def _source_counts(rows):
    grouped = defaultdict(lambda: {key: 0 for key in COUNT_KEYS})
    for row in rows:
        source = row["source_id"]
        values = {key: int(row.get(key, 0)) for key in COUNT_KEYS}
        if any(value < 0 for value in values.values()):
            raise ValueError(f"Negative count for source {source}")
        if values["unverified_accepted"] > values["wrong"]:
            raise ValueError("unverified_accepted must be a subset of wrong")
        if values["wrong"] > values["accepted"]:
            raise ValueError("wrong cannot exceed accepted")
        for key, value in values.items():
            grouped[source][key] += value
    if not grouped:
        raise ValueError("Policy has no source rows")
    return dict(grouped)


def summarize_policy(rows, *, cap, overall=None):
    by_source = _source_counts(rows)
    totals = {key: sum(source[key] for source in by_source.values()) for key in COUNT_KEYS}
    accepted = np.asarray([source["accepted"] for source in by_source.values()], dtype=float)
    gate = np.asarray([source["accepted"] + source["ignored_accepted"]
                       for source in by_source.values()], dtype=int)
    return {
        "source_count": len(by_source),
        **totals,
        "correct_accepted": totals["accepted"] - totals["wrong"],
        "unverified_is_wrong_subset": True,
        "wrong_per_acceptance": totals["wrong"] / totals["accepted"] if totals["accepted"] else None,
        "natural_gap_recovery_rate": totals["correct"] / totals["gaps"] if totals["gaps"] else None,
        "gate_actions": int(gate.sum()),
        "sources_reaching_cap": None if cap is None else int((gate >= cap).sum()),
        "sources_reaching_cap_fraction": None if cap is None else float((gate >= cap).mean()),
        "cap_interpretation": ("unconstrained; no cap applies" if cap is None
                               else "accepted + ignored_accepted >= B within source"),
        "active_sources": int((accepted > 0).sum()),
        "active_source_fraction": float((accepted > 0).mean()),
        "accepted_per_source": {
            "mean": float(accepted.mean()),
            "quartiles": np.quantile(accepted, [0.25, 0.5, 0.75]).tolist(),
            "minimum": int(accepted.min()),
            "maximum": int(accepted.max()),
            "counts": {source: counts["accepted"] for source, counts in sorted(by_source.items())},
        },
        "gate_actions_per_source": {
            "counts": {source: counts["accepted"] + counts["ignored_accepted"]
                       for source, counts in sorted(by_source.items())}
        },
        "overall": None if overall is None else overall.get("aggregate", overall),
    }


def evaluate_payload(payload, lock, *, replicas=10000, seed=362):
    cap = int(payload["acceptance_cap"])
    if cap != int(lock["acceptance_cap"]):
        raise ValueError("Replay and lock use different acceptance cap")
    policies = payload["policies"]
    selected = set(lock["test_policies"])
    if not selected.issubset(policies):
        raise ValueError(f"Replay misses locked policies: {sorted(selected - set(policies))}")
    summaries = {name: summarize_policy(policies[name],
                                         cap=None if name == "unconstrained_tracker" else cap,
                                         overall=payload.get("overall", {}).get(name))
                 for name in sorted(selected)}
    matched = []
    for point_id, work_point in lock["selection"]["points"].items():
        nearest = work_point["nearest"]
        mapping = {family: detail["policy"] for family, detail in nearest.items()}
        if not work_point["matched"]:
            matched.append({"id": point_id, **work_point, "policies": mapping,
                            "evaluation_status": "not_run_unmatched_at_calibration",
                            "test_actual_accepts_per_source": None,
                            "test_max_pairwise_spread": None,
                            "test_absolute_target_deviation": None,
                            "spread_only_within_tolerance": None,
                            "target_deviations_within_tolerance": None,
                            "test_within_frozen_tolerance": None,
                            "comparisons": {}})
            continue
        missing = set(mapping.values()) - selected
        if missing:
            raise ValueError(f"Matched point policies absent from test lock: {sorted(missing)}")
        structured = mapping["structured"]
        comparisons = {}
        for family, reference in sorted(mapping.items()):
            if family == "structured":
                continue
            comparisons[family] = compare(policies[reference], policies[structured],
                                          replicas=replicas, seed=seed)
        actual = {family: summaries[policy]["accepted_per_source"]["mean"]
                  for family, policy in mapping.items()}
        spread = max(actual.values()) - min(actual.values())
        target = float(work_point["target"])
        tolerance = float(work_point["tolerance"])
        target_deviations = {family: abs(value - target) for family, value in actual.items()}
        spread_ok = spread <= tolerance
        target_ok = max(target_deviations.values()) <= tolerance
        matched.append({"id": point_id, **work_point, "policies": mapping,
                        "evaluation_status": "evaluated",
                        "test_actual_accepts_per_source": actual,
                        "test_max_pairwise_spread": spread,
                        "test_absolute_target_deviation": target_deviations,
                        "spread_only_within_tolerance": spread_ok,
                        "target_deviations_within_tolerance": target_ok,
                        "test_within_frozen_tolerance": spread_ok and target_ok,
                        "comparisons": comparisons})
    return {
        "tracker": payload["tracker"],
        "split": payload["split"],
        "acceptance_cap": cap,
        "resampling": {"unit": "source", "replicas": replicas, "seed": seed,
                       "interval": "paired empirical 95% interval"},
        "metric_definitions": {
            "action_error": "wrong / accepted; unverified_accepted is a subset of wrong",
            "natural_gap_recovery": "correct / gaps",
            "cap_trigger": "accepted + ignored_accepted >= B within source",
            "HOTA": "TrackEval aggregate under the replay's frozen overall protocol",
        },
        "policies": summaries,
        "matched_sets": matched,
    }


def report_tables(report):
    policy_rows = []
    for policy, row in report["policies"].items():
        overall = row.get("overall") or {}
        policy_rows.append({
            "tracker": report["tracker"], "acceptance_cap": report["acceptance_cap"],
            "policy": policy, "source_count": row["source_count"],
            "accepted": row["accepted"], "correct_accepted": row["correct_accepted"],
            "wrong": row["wrong"], "unverified_accepted": row["unverified_accepted"],
            "ignored_accepted": row["ignored_accepted"], "wrong_per_acceptance": row["wrong_per_acceptance"],
            "correct": row["correct"], "gaps": row["gaps"],
            "natural_gap_recovery_rate": row["natural_gap_recovery_rate"],
            "active_sources": row["active_sources"],
            "sources_reaching_cap": row["sources_reaching_cap"],
            "accepted_mean": row["accepted_per_source"]["mean"],
            "accepted_q25": row["accepted_per_source"]["quartiles"][0],
            "accepted_median": row["accepted_per_source"]["quartiles"][1],
            "accepted_q75": row["accepted_per_source"]["quartiles"][2],
            **{key: overall.get(key) for key in ("HOTA", "AssA", "IDF1", "IDSW")},
        })
    comparison_rows = []
    for point in report["matched_sets"]:
        for reference, comparison in point["comparisons"].items():
            for metric, effect in comparison["effects"].items():
                interval = effect["empirical_95_percent_interval"]
                comparison_rows.append({
                    "tracker": report["tracker"], "acceptance_cap": report["acceptance_cap"],
                    "point_id": point["id"], "target": point["target"],
                    "tolerance": point["tolerance"], "reference_family": reference,
                    "metric": metric, "delta_percentage_points": effect["delta_percentage_points"],
                    "ci95_low": None if interval is None else interval[0],
                    "ci95_high": None if interval is None else interval[1],
                    "valid_replicas": effect["valid_replicas"],
                    "test_max_pairwise_spread": point["test_max_pairwise_spread"],
                    "spread_only_within_tolerance": point["spread_only_within_tolerance"],
                    "target_deviations_within_tolerance": point["target_deviations_within_tolerance"],
                    "test_within_frozen_tolerance": point["test_within_frozen_tolerance"],
                })
    return policy_rows, comparison_rows


def _write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("lock", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--replicas", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=362)
    parser.add_argument("--run-manifest", type=Path,
                        help="Run the locked test replay before evaluating this results path")
    parser.add_argument("--run-config", type=Path)
    parser.add_argument("--trace-directory", type=Path)
    parser.add_argument("--overall-protocol", default="personpath22-native-keyframe")
    args = parser.parse_args()
    lock = json.loads(args.lock.read_bytes())
    if args.run_manifest:
        if not args.run_config or not args.trace_directory:
            parser.error("--run-manifest requires --run-config and --trace-directory")
        validate_lock_files(lock, args.lock, args.run_config)
        if args.results.exists():
            raise ValueError(f"Refusing to overwrite existing test replay: {args.results}")
        from reconnect_study.study import run_split
        run_split(args.run_manifest, args.run_config, "test", args.results,
                  lock["test_policies"], args.overall_protocol, None,
                  args.trace_directory, False)
    payload = json.loads(args.results.read_bytes())
    report = evaluate_payload(payload, lock, replicas=args.replicas, seed=args.seed)
    report["test_sha256"] = digest(args.results)
    report["lock_sha256"] = digest(args.lock)
    save(args.output, report)
    policies, comparisons = report_tables(report)
    _write_csv(args.output.with_suffix(".policies.csv"), policies)
    _write_csv(args.output.with_suffix(".comparisons.csv"), comparisons)


if __name__ == "__main__":
    main()
