"""Recalculate ByteTrack's held-out effects and HOTA from existing traces."""

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from trackeval.metrics import HOTA

from reconnect_study.data_io import digest, read_manifest
from reconnect_study.overall import _mot_data
from reconnect_study.study import save
from reconnect_study.supplement_matched import ARCHIVE, CAPS, OUTPUT, load
from reconnect_study.tracker import TrackObservation


PROTOCOL = "personpath22-native-keyframe"


def score_trace(entry):
    trace, sequence = entry
    data = load(trace)
    tracks = tuple(TrackObservation(**row) for row in data["tracks"])
    return HOTA().eval_sequence(_mot_data(sequence, tracks, PROTOCOL))


def grouped(rows):
    result = defaultdict(lambda: np.zeros(4, dtype=np.int64))
    for row in rows:
        result[row["source_id"]] += [row["wrong"], row["accepted"], row["correct"], row["gaps"]]
    return np.array([result[key] for key in sorted(result)])


def rates(matrix):
    numerators = matrix[..., ::2]
    denominators = matrix[..., 1::2]
    return np.divide(numerators, denominators,
                     out=np.full(numerators.shape, np.nan), where=denominators > 0)


def verify_effects(report, test):
    annotated = {name: grouped(rows) for name, rows in test["policies"].items()}
    checks = 0
    for point in report["matched_sets"]:
        for family, comparison in point["comparisons"].items():
            target = annotated[point["policies"]["structured"]]
            reference = annotated[point["policies"][family]]
            if len(target) != 74 or not np.array_equal(target[:, 3], reference[:, 3]):
                raise ValueError("Fixed paired video cohort changed")
            choices = np.random.default_rng(362).integers(74, size=(10_000, 74))
            deltas = 100 * (rates(target.sum(axis=0)) - rates(reference.sum(axis=0)))
            intervals = 100 * (rates(target[choices].sum(axis=1))
                               - rates(reference[choices].sum(axis=1)))
            for index, name in enumerate(("wrong_per_acceptance", "gap_recovery")):
                values = intervals[:, index]
                values = values[np.isfinite(values)]
                interval = np.quantile(values, [.025, .975])
                archived = comparison["effects"][name]
                np.testing.assert_allclose(
                    [deltas[index], *interval],
                    [archived["delta_percentage_points"], *archived["empirical_95_percent_interval"]],
                    atol=1e-12, rtol=0,
                )
                if archived["valid_replicas"] != len(values):
                    raise ValueError(f"Unexpected valid replicate count for {name}")
                checks += 1
    return checks


def verify(root, workers):
    manifest = ARCHIVE / "manifest.json"
    sequences = {row.name: row for row in read_manifest(manifest) if row.split == "test"}
    audit = load(root / "test-integrity.json")
    tests = {}
    unique = {}
    checks = {"source_count": len(sequences), "by_cap": {}}
    for cap in CAPS:
        directory = root / f"B{cap}"
        test = load(directory / "test.json")
        report = load(directory / "report.json")
        key = f"B{cap}"
        if any(digest(directory / path) != audit[key][field] for path, field in (
            ("test.json", "test_sha256"), ("report.json", "report_sha256"),
        )):
            raise ValueError(f"Test data changed after integrity audit for {key}")
        checks["by_cap"][key] = {"intervals_verified": verify_effects(report, test)}
        tests[cap] = test, report
        for policy, rows in test["policies"].items():
            for row in rows:
                sequence = sequences[row["name"]]
                trace = directory / "traces" / policy / f"{sequence.name}.json"
                unique.setdefault((sequence.name, row["trace_sha256"]), (trace, sequence))
    hota = HOTA()
    results = {}
    with ProcessPoolExecutor(max_workers=workers) as executor:
        scores = executor.map(score_trace, unique.values(), chunksize=5)
        for index, (key, score) in enumerate(zip(unique, scores, strict=True), 1):
            results[key] = score
            if index % 200 == 0 or index == len(unique):
                print(f"HOTA verified from tracks: {index}/{len(unique)} distinct traces", flush=True)
    checks["distinct_trace_evaluations"] = len(unique)
    for cap, (test, report) in tests.items():
        intervals = checks["by_cap"][f"B{cap}"]
        intervals["HOTA_policies"] = {}
        for policy, rows in test["policies"].items():
            scores = {row["name"]: results[(row["name"], row["trace_sha256"])] for row in rows}
            overall = test["overall"][policy]
            for name, score in scores.items():
                np.testing.assert_allclose(100 * score["HOTA"].mean(),
                                           overall["sequences"][name]["HOTA"], atol=1e-12, rtol=0)
            recalculated = 100 * hota.combine_sequences(scores)["HOTA"].mean()
            np.testing.assert_allclose(recalculated, overall["aggregate"]["HOTA"], atol=1e-12, rtol=0)
            np.testing.assert_allclose(recalculated, report["policies"][policy]["overall"]["HOTA"],
                                       atol=1e-12, rtol=0)
            intervals["HOTA_policies"][policy] = float(recalculated)
    save(root / "independent-metric-audit.json", checks)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    verify(OUTPUT / "byte", args.workers)


if __name__ == "__main__":
    main()
