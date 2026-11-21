"""Paired source resampling for descriptive tracker comparisons."""

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from reconnect_study.data_io import digest
from reconnect_study.study import save


KEYS = ("gaps", "correct", "accepted", "wrong")


def grouped(rows):
    sources = defaultdict(lambda: np.zeros(len(KEYS), dtype=np.int64))
    for row in rows:
        sources[row["source_id"]] += np.asarray([row[key] for key in KEYS])
    return sources


def _rates(matrix):
    denominators = matrix[..., (0, 2)]
    numerators = matrix[..., (1, 3)]
    return np.divide(numerators, denominators, out=np.full_like(numerators, np.nan, dtype=float),
                     where=denominators > 0)


def compare(reference, changed, *, replicas=10000, seed=362):
    first, second = grouped(reference), grouped(changed)
    if not first or set(first) != set(second):
        raise ValueError("Both policies need the same source cohort")
    cohort = sorted(first)
    a = np.asarray([first[name] for name in cohort])
    b = np.asarray([second[name] for name in cohort])
    if not np.array_equal(a[:, 0], b[:, 0]):
        raise ValueError("Policies use different fixed gap denominators")
    choice = np.random.default_rng(seed).integers(len(cohort), size=(replicas, len(cohort)))
    effect = _rates(b.sum(axis=0)) - _rates(a.sum(axis=0))
    samples = _rates(b[choice].sum(axis=1)) - _rates(a[choice].sum(axis=1))
    output = {}
    for i, measure in enumerate(("gap_recovery", "wrong_per_acceptance")):
        finite = samples[np.isfinite(samples[:, i]), i]
        output[measure] = {
            "delta_percentage_points": None if not np.isfinite(effect[i]) else 100 * effect[i],
            "empirical_95_percent_interval": (100 * np.quantile(finite, [.025, .975])).tolist()
            if len(finite) else None,
            "valid_replicas": len(finite),
        }
    return {"source_count": len(cohort), "resampling_replicas": replicas, "resampling_seed": seed,
            "interpretation": "Paired empirical source resampling interval; separate from design-based risk bounds.",
            "effects": output}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("lock", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    results = json.loads(args.results.read_bytes())
    lock = json.loads(args.lock.read_bytes())
    policies = results["policies"]
    settings = {}
    for ceiling, names in lock["budgets"].items():
        main_policy = names["structured"]
        if main_policy is None:
            continue
        references = {"unconstrained_tracker", "cap_only"}
        references.update(name for name in names.values() if name and name != main_policy)
        settings[ceiling] = {"interaction_policy": main_policy,
                             "comparisons": {name: compare(policies[name], policies[main_policy])
                                             for name in sorted(references) if name in policies}}
    save(args.output, {"test_sha256": digest(args.results), "selection_sha256": digest(args.lock),
                       "tracker": results["tracker"], "comparisons": settings})


if __name__ == "__main__":
    main()
