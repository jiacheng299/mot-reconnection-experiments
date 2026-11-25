"""Audit both ablation reports and consolidate their tables and source links."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
from pathlib import Path

import numpy as np
from trackeval.metrics import HOTA

from reconnect_study.data_io import digest, read_manifest
from reconnect_study.round3_ablation import MANIFEST, OUTPUT
from reconnect_study.round3_ablation_eval import bound, config_path_in_checkout, write_csv
from reconnect_study.study import save
from reconnect_study.supplement_matched import load
from reconnect_study.supplement_matched_verify import grouped, rates, score_trace


def verify_report(report, combined):
    roles = (("23_minus_15", "structured", "summary15"), ("15_minus_12", "summary15", "linear"))
    checks = 0
    choices = np.random.default_rng(362).integers(74, size=(10_000, 74))
    for point in report["work_points"].values():
        for name, changed, reference in roles:
            comparison = point["comparisons"][name]["test"]
            if not comparison:
                continue
            left, right = (grouped(combined[point["policies"][family]["policy"]])
                           for family in (changed, reference))
            if len(left) != 74 or not np.array_equal(left[:, 3], right[:, 3]):
                raise ValueError("Held-out cohort differs across paired policies")
            delta = 100 * (rates(left.sum(axis=0)) - rates(right.sum(axis=0)))
            samples = 100 * (rates(left[choices].sum(axis=1)) - rates(right[choices].sum(axis=1)))
            for index, measure in enumerate(("wrong_per_acceptance", "gap_recovery")):
                values = samples[:, index][np.isfinite(samples[:, index])]
                entry = comparison["paired_effect"]["effects"][measure]
                np.testing.assert_allclose(
                    [delta[index], *np.quantile(values, [.025, .975])],
                    [entry["delta_percentage_points"], *entry["empirical_95_percent_interval"]],
                    atol=1e-12, rtol=0,
                )
                if len(values) != entry["valid_replicas"]:
                    raise ValueError("Paired bootstrap replica count differs")
                checks += 1
    return checks


def finalize(workers):
    sequences = {row.name: row for row in read_manifest(MANIFEST) if row.split == "test"}
    audits = {}
    manifests = {}
    for tracker in ("byte", "bot"):
        directory = OUTPUT / tracker
        new_test_path = directory / "test.json"
        report_path = directory / "report.json"
        report = load(report_path)
        manifest = load(directory / "manifest.json")
        bound(directory / "locked.json", report["lock_sha256"])
        bound(new_test_path, report["test_sha256"])
        new = load(new_test_path)
        old = load(config_path_in_checkout(manifest["old_test_path"]))
        combined = {**old["policies"], **new["policies"]}
        audit = {"paired_intervals_verified": verify_report(report, combined),
                 "new_trace_count": 0, "independent_HOTA": {}}
        with ProcessPoolExecutor(max_workers=workers) as executor:
            for policy, rows in new["policies"].items():
                entries = []
                for row in rows:
                    trace_path = directory / "traces" / policy / f"{row['name']}.json"
                    bound(trace_path, row["trace_sha256"])
                    entries.append((trace_path, sequences[row["name"]]))
                scores = dict(zip([row["name"] for row in rows],
                                  executor.map(score_trace, entries, chunksize=5), strict=True))
                for name, score in scores.items():
                    np.testing.assert_allclose(
                        100 * score["HOTA"].mean(), new["overall"][policy]["sequences"][name]["HOTA"],
                        atol=1e-12, rtol=0,
                    )
                value = 100 * HOTA().combine_sequences(scores)["HOTA"].mean()
                np.testing.assert_allclose(value, new["overall"][policy]["aggregate"]["HOTA"],
                                           atol=1e-12, rtol=0)
                audit["independent_HOTA"][policy] = float(value)
                audit["new_trace_count"] += len(entries)
        audits[tracker] = audit
        manifests[tracker] = {**manifest, "new_test_sha256": digest(new_test_path),
                              "report_sha256": digest(report_path)}
        print(tracker, audit, flush=True)
    for name in ("effects.csv", "results.csv"):
        rows = []
        for tracker in ("byte", "bot"):
            source = OUTPUT / tracker / name
            if source.exists():
                with source.open(encoding="utf-8-sig", newline="") as file:
                    rows.extend(csv.DictReader(file))
        if rows:
            write_csv(OUTPUT / name, rows)
    save(OUTPUT / "manifest.json", manifests)
    save(OUTPUT / "metric-audit.json", audits)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    finalize(parser.parse_args().workers)
