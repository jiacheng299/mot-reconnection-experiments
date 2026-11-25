"""Independently recalculate round-three BoostTrack HOTA and paired intervals."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import csv

import numpy as np
from trackeval.metrics import HOTA

from reconnect_study.data_io import digest, read_manifest
from reconnect_study.round3_ablation_eval import bound
from reconnect_study.round3_boost_calibrate import ARCHIVE, OUTPUT
from reconnect_study.study import save
from reconnect_study.supplement_matched import load
from reconnect_study.supplement_matched_verify import grouped, rates, score_trace


def verify(workers):
    test_sequences = {s.name: s for s in read_manifest(ARCHIVE / "manifest.json") if s.split == "test"}
    with (OUTPUT.parent / "paired-effects.csv").open(encoding="utf-8-sig") as file:
        rows = list(csv.DictReader(file))
    choices = np.random.default_rng(362).integers(74, size=(10_000, 74))
    out = {}
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for key, tracker, root in (("byte", "ByteTrack", OUTPUT), ("bot", "BoT-SORT (motion)", OUTPUT / "bot")):
            result = {"HOTA": {}, "trace_count": 0, "intervals_verified": 0}
            summary = load(OUTPUT / f"{key}-summary-input.json")
            if not any(point["calibration_matched"] for point in summary.values()):
                out[key] = result
                continue
            test_path = root / "test.json"
            test = load(test_path)
            bound(root / "locked.json", test["lock_sha256"])
            result["test_sha256"] = digest(test_path)
            old = load(OUTPUT.parent.parent / "round2" / key / "B12" / "test.json")
            for policy, per_video in test["policies"].items():
                entries = [(root / "traces" / policy / f"{row['name']}.json", test_sequences[row["name"]])
                           for row in per_video]
                for row, (path, _) in zip(per_video, entries, strict=True):
                    bound(path, row["trace_sha256"])
                scores = dict(zip((sequence.name for _, sequence in entries),
                                  executor.map(score_trace, entries, chunksize=5), strict=True))
                overall = test["overall"][policy]
                for name, score in scores.items():
                    np.testing.assert_allclose(100 * score["HOTA"].mean(), overall["sequences"][name]["HOTA"],
                                               rtol=0, atol=1e-12)
                hota = 100 * HOTA().combine_sequences(scores)["HOTA"].mean()
                np.testing.assert_allclose(hota, overall["aggregate"]["HOTA"], rtol=0, atol=1e-12)
                result["HOTA"][policy] = float(hota)
                result["trace_count"] += len(scores)
            for row in rows:
                if row["tracker"] != tracker or row["right"] != "BoostTrack" or row["calibration_matched"] != "True":
                    continue
                left, right = (grouped(data["policies"][row[policy]]) for data, policy in
                               ((old, "left_policy"), (test, "right_policy")))
                if len(left) != 74 or not np.array_equal(left[:, 3], right[:, 3]):
                    raise ValueError("Paired sources or gaps changed")
                point = 100 * (rates(left.sum(0)) - rates(right.sum(0)))
                samples = 100 * (rates(left[choices].sum(1)) - rates(right[choices].sum(1)))
                for index, kind in enumerate(("failure", "recovery")):
                    values = samples[np.isfinite(samples[:, index]), index]
                    np.testing.assert_allclose(
                        [point[index], *np.quantile(values, [.025, .975])],
                        [float(row[kind + suffix]) for suffix in ("_pp", "_lo", "_hi")], rtol=0, atol=1e-12)
                    if len(values) != int(row[kind + "_valid_replicas"]):
                        raise ValueError("Bootstrap replicate count changed")
                    result["intervals_verified"] += 1
            out[key] = result
            print(tracker, result, flush=True)
    save(OUTPUT / "independent-metric-audit.json", out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=6)
    verify(parser.parse_args().workers)
