"""Certify a fixed family of closed-loop reconnect policy replays.

The JSON manifest declares ``cohort`` and exactly one replay input. Use
``policies`` for inline {policy: [sequence summary, ...]}, ``policy_rows`` for a
path to the same mapping or a study split replay envelope, or ``replays`` for
per-sequence {policy: summary} files. ``include_policies`` can enumerate the
frozen family to be tested. Paths are relative to the manifest. Each summary supplies ``name``,
``source_id``, ``accepted``, ``wrong``, ``correct`` and ``gaps`` from replay.

Event-weighted certification requires a documented source-level action cap
enforced both during calibration replay and future deployment. The CLI checks
counts and declarations. It cannot independently establish the data collection
protocol, policy freezing, cap enforcement in future videos or iid sampling.

Fixed finite-pool proof
-----------------------
The fixed source pool and the policy family are decided independently of one
uniform subset draw, with no replacement. For a candidate risk r and cap B,
write the per-source event-weighted betting factor as
f_s = 1 + t(r A_s - W_s)/(B(1-r)), t in [0,1]. Under the full-pool risk null,
the factors are nonnegative and their population mean is at most one.
The expectation of the product over a uniform size-n sample is the n-th
elementary symmetric mean of those factors. Maclaurin's inequality bounds it
by the n-th power of the population mean, hence at most one. Fixed mixtures,
Markov's inequality and Bonferroni give simultaneous finite-pool bounds.
For active-video risk, condition on the number of active sampled sources.
They form a uniform sample from the active population; set A_s=1 and
W_s equal to the observed within-source wrong fraction. The same proof applies.
For KL, apply Maclaurin instead to the nonnegative exponential factors in the
Chernoff argument. These results average over the prespecified random draw.
They do not model a future video population or constrain the realized complement.
"""

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
from hashlib import sha256
import json
from math import comb, isfinite
from pathlib import Path

from reconnect_study.data_io import archived_path
from reconnect_study.risk import RiskTest, VideoCounts, certify_policies


COUNT_KEYS = ("accepted", "wrong", "correct", "gaps")


def certify_manifest(
    manifest_path: Path,
    *,
    alpha: float,
    delta: float,
    min_active_coverage: float = 0.0,
    event_delta_fraction: float = 0.5,
    test: RiskTest = "kl",
) -> dict:
    """Summarize risk and select the best eligible policy separately by method."""
    if not 0 < alpha < 1 or not 0 < delta < 1:
        raise ValueError("alpha and delta must be strictly between 0 and 1")
    if not 0 <= min_active_coverage <= 1:
        raise ValueError("min_active_coverage must be between 0 and 1")
    if not 0 < event_delta_fraction < 1:
        raise ValueError("event_delta_fraction must be strictly between 0 and 1")

    manifest = _object(json.loads(manifest_path.read_bytes()), "manifest")
    finite = "finite_population_design" in manifest
    cohort = _cohort(manifest.get("cohort"), finite=finite)
    cap = _cap(manifest.get("event_weighted_cap"))
    rows, replay_metadata = _load_policies(manifest, manifest_path.parent)
    if cap is not None and replay_metadata is not None:
        if replay_metadata.get("acceptance_cap") != cap["max_accepted"]:
            raise ValueError("the recorded replay cap differs from the declared event-weighted cap")
        if "unconstrained_tracker" in rows:
            raise ValueError("unconstrained_tracker bypasses the cap. Define the tested family with include_policies")
    counts, utility, sequence_count = _aggregate(rows)

    event_delta = delta * event_delta_fraction if cap is not None else 0.0
    methods_to_delta = {"active_video": delta - event_delta}
    if cap is not None:
        methods_to_delta["event_weighted"] = event_delta
    design = _verify_finite_population(
        manifest, manifest_path.parent, rows, replay_metadata, alpha=alpha, delta=delta,
        event_delta_fraction=event_delta_fraction if cap is not None else 0.0, test=test,
    ) if finite else None

    methods = {}
    for method, method_delta in methods_to_delta.items():
        certificates = certify_policies(
            counts, alpha=alpha, delta=method_delta, method=method,
            acceptance_cap=cap["max_accepted"] if method == "event_weighted" else None, test=test,
        )
        policy_results = {}
        for policy, certificate in certificates.items():
            certified = certificate.certified and cohort["status"] != "exploratory"
            meets_coverage = utility[policy]["active_source_coverage"] >= min_active_coverage
            policy_results[policy] = {
                **asdict(certificate),
                "passes_statistical_test": certificate.certified,
                "certified": certified,
                "eligible_for_selection": certified and meets_coverage,
            }
        candidates = [policy for policy, result in policy_results.items() if result["eligible_for_selection"]]
        selected = min(candidates, key=lambda policy: (
            -utility[policy]["correct"],
            -utility[policy]["active_source_coverage"],
            policy_results[policy]["risk_upper_bound"],
            policy,
        ), default=None)
        methods[method] = {"delta": method_delta, "selected_policy": selected, "policies": policy_results}

    return {
        "alpha": alpha,
        "delta": delta,
        "confidence": 1 - delta,
        "test": test,
        "cohort": cohort,
        "finite_population_design": design,
        "replay_metadata": replay_metadata,
        "source_count": len(next(iter(counts.values()))),
        "sequence_count": sequence_count,
        "min_empirical_active_source_coverage": min_active_coverage,
        "selection_objective": "Maximize observed correctly recovered gaps over the shared cohort.",
        "risk_label_scope": "Wrong among accepted reconnects counted as evaluable by the replay labeling protocol.",
        "event_weighted_cap": cap,
        "event_weighted_status": "computed" if cap else "no_enforced_cap_declared",
        "utility": utility,
        "methods": methods,
    }


def _object(value, label):
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _nonempty_string(value, label):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a nonempty string")
    return value


def _cohort(value, *, finite=False):
    cohort = _object(value, "cohort")
    _nonempty_string(cohort.get("description"), "cohort.description")
    iid = cohort.get("iid_source_sampling")
    frozen = cohort.get("policies_frozen_before_calibration")
    if any(value is not None and type(value) is not bool for value in (iid, frozen)):
        raise ValueError("cohort sampling and freeze declarations must be boolean or omitted")
    caveats = cohort.get("caveats", [])
    if not isinstance(caveats, list) or any(not isinstance(note, str) for note in caveats):
        raise ValueError("cohort.caveats must be a list of strings")

    if finite:
        status = "finite_population_design_verified"
        scope = "Fixed finite source pool only. The random complement is held out for descriptive evaluation."
        assumption = "Coverage is over one prospectively drawn uniform subset of the frozen pool and fixed policies."
        verification = "The tool verifies artifact consistency. The operator must preserve prospective draw and policy-freeze timing."
    else:
        status = "assumptions_declared" if iid is True and frozen is True else "exploratory"
        scope = "Stated deployment source distribution, subject to the sampling and policy-freeze assumptions."
        assumption = "Bounds require iid source groups from the stated deployment distribution and policies frozen before calibration."
        verification = "The tool checks declarations only. Exploratory diagnostics do not certify deployment risk."

    return {
        **cohort,
        "status": status,
        "scope": scope,
        "caveats": [
            *caveats,
            assumption,
            "Counts within source groups and results across policies may be dependent.",
            "Empirical active-source coverage and recovery utility have no lower confidence guarantee here.",
            verification,
            "Choose the risk test and policy family before calibration or allocate confidence across all tried tests.",
        ],
    }


def _verify_finite_population(manifest, directory, rows, replay_metadata, *, alpha, delta,
                              event_delta_fraction, test):
    design, design_hash = _load_hashed_json(manifest, directory, "finite_population_design")
    if design.get("draw_method") != "uniform-combination-rank-v1":
        raise ValueError("finite population design requires the uniform-combination-rank-v1 draw")
    population = _source_list(design, "population")
    sample = _source_list(design, "calibration_sample")
    complement = _source_list(design, "held_out_complement", allow_empty=True)
    if set(sample) & set(complement) or set(sample) | set(complement) != set(population):
        raise ValueError("calibration sample and complement must partition the finite population without overlap")
    rank_text = _nonempty_string(design.get("random_rank_decimal"), "random rank")
    if not rank_text.isascii() or not rank_text.isdecimal():
        raise ValueError("random rank must be a nonnegative decimal integer")
    rank = int(rank_text)
    subset_count = comb(len(population), len(sample))
    if not 0 <= rank < subset_count or set(_unrank(sorted(population), len(sample), rank)) != set(sample):
        raise ValueError("random rank does not reproduce the declared calibration sample")

    expected_test = {"betting": "fixed-betting-mixture", "kl": "bounded-kl"}.get(test)
    if (design.get("risk_target"), design.get("delta"), design.get("event_delta_fraction"), design.get("method")) != (
        alpha, delta, event_delta_fraction, expected_test,
    ):
        raise ValueError("risk target, confidence allocation or test differs from the frozen finite design")

    config, config_hash = _load_hashed_json(manifest, directory, "frozen_config")
    benchmark, benchmark_hash = _load_hashed_json(manifest, directory, "benchmark_manifest")
    if config_hash != design.get("config_sha256"):
        raise ValueError("frozen configuration hash differs from the finite design")
    if benchmark_hash != design.get("benchmark_manifest_sha256") or benchmark_hash != config.get("study_manifest_sha256"):
        raise ValueError("benchmark manifest hash differs from the finite design or frozen configuration")
    if replay_metadata is None or replay_metadata.get("config_sha256") != config_hash:
        raise ValueError("need a replay envelope with the finite design's frozen configuration hash")
    if replay_metadata.get("split") not in ("calibration", "finite_calibration"):
        raise ValueError("finite population certification requires a calibration replay envelope")
    if not design.get("tracker") or any(item.get("tracker") != design["tracker"] for item in (config, replay_metadata)):
        raise ValueError("replay tracker differs from the finite design")
    if any(item.get("acceptance_cap") != design.get("acceptance_cap") for item in (config, replay_metadata)):
        raise ValueError("replay or configuration cap differs from the finite design")

    _verify_frozen_family(config, rows)
    _verify_pool_replays(benchmark, config, rows, set(population), set(sample))
    return {
        "draw_method": design["draw_method"],
        "random_rank_decimal": rank_text,
        "population_size": len(population),
        "calibration_size": len(sample),
        "held_out_size": len(complement),
        "calibration_source_ids": sorted(sample),
        "held_out_source_ids": sorted(complement),
        "target": "fixed_finite_source_pool",
        "design_sha256": design_hash,
        "benchmark_manifest_sha256": benchmark_hash,
        "config_sha256": config_hash,
    }


def _load_hashed_json(manifest, directory, key):
    path = archived_path(directory, _nonempty_string(manifest.get(key), key))
    data = path.read_bytes()
    return _object(json.loads(data), key), sha256(data).hexdigest()


def _source_list(design, key, *, allow_empty=False):
    values = design.get(key)
    if (not isinstance(values, list) or (not values and not allow_empty)
            or any(not isinstance(value, str) or not value.strip() for value in values)
            or len(set(values)) != len(values)):
        raise ValueError(f"{key} must be a list of distinct nonempty source IDs")
    return values


def _unrank(population, sample_size, rank):
    sample = []
    start, remaining = 0, sample_size
    while remaining:
        block = comb(len(population) - start - 1, remaining - 1)
        if rank < block:
            sample.append(population[start])
            remaining -= 1
        else:
            rank -= block
        start += 1
    return sample


def _verify_frozen_family(config, rows):
    names = set()
    for scorer, thresholds in _object(config.get("thresholds"), "configuration thresholds").items():
        if (not isinstance(thresholds, list)
                or any(type(value) not in (float, int) or not isfinite(value) for value in thresholds)
                or len(set(thresholds)) != len(thresholds)):
            raise ValueError("frozen configuration thresholds must be finite and distinct within each scorer")
        names.update(f"{scorer}_{index}" for index in range(len(thresholds)))
    if set(rows) != names:
        raise ValueError("finite-pool replay must contain exactly the frozen threshold family")


def _verify_pool_replays(benchmark, config, rows, population, sample):
    sequences = benchmark.get("sequences")
    if not isinstance(sequences, list) or not sequences:
        raise ValueError("benchmark manifest requires sequences")
    source_splits = defaultdict(set)
    expected = {}
    for sequence in sequences:
        sequence = _object(sequence, "benchmark sequence")
        source = _nonempty_string(sequence.get("source_id"), "benchmark source ID")
        split = _nonempty_string(sequence.get("split"), "benchmark split")
        source_splits[source].add(split)
        if source in sample:
            name = _nonempty_string(sequence.get("name"), "benchmark sequence name")
            if name in expected:
                raise ValueError(f"benchmark manifest contains duplicate sequence {name!r}")
            expected[name] = source

    development = set(_source_list(config, "development_sources", allow_empty=True))
    development.update(source for source, splits in source_splits.items() if "development" in splits)
    if population & development:
        raise ValueError("finite population must not overlap with development sources")
    if not population <= source_splits.keys():
        raise ValueError("finite population contains unknown benchmark sources")
    observed = {row["name"]: row["source_id"] for row in next(iter(rows.values()))}
    if observed != expected:
        raise ValueError("policy replays must cover exactly all benchmark sequences in the sampled source groups")


def _cap(value):
    if value is None:
        return None
    cap = _object(value, "event_weighted_cap")
    _nonempty_string(cap.get("enforcement"), "event_weighted_cap.enforcement")
    if cap.get("enforced_in_replay_and_deployment") is not True:
        raise ValueError("event_weighted_cap.enforced_in_replay_and_deployment must be true")
    maximum = cap.get("max_accepted")
    if type(maximum) is not int or maximum < 1:
        raise ValueError("event_weighted_cap.max_accepted must be a positive integer")
    return cap


def _load_policies(manifest, directory):
    inputs = [key for key in ("policies", "policy_rows", "replays") if key in manifest]
    if len(inputs) != 1:
        raise ValueError("provide exactly one of policies, policy_rows or replays")
    key = inputs[0]
    data = manifest[key]
    metadata = None
    if key == "policy_rows":
        path = archived_path(directory, _nonempty_string(data, "policy_rows"))
        data = _object(json.loads(path.read_bytes()), str(path))
        if isinstance(data.get("policies"), dict):
            metadata = {field: data.get(field) for field in ("tracker", "split", "config_sha256", "acceptance_cap")}
            data = data["policies"]
    if key == "replays":
        data = _read_sequence_maps(data, directory)
    rows = _object(data, "policies")
    if "include_policies" in manifest:
        include = manifest["include_policies"]
        if (not isinstance(include, list) or not include or any(not isinstance(name, str) for name in include)
                or len(set(include)) != len(include)):
            raise ValueError("include_policies must be a nonempty list of unique policy names")
        missing = set(include) - rows.keys()
        if missing:
            raise ValueError(f"include_policies names unknown policies: {sorted(missing)}")
        rows = {policy: rows[policy] for policy in include}
    return rows, metadata


def _read_sequence_maps(filenames, directory):
    if not isinstance(filenames, list) or not filenames:
        raise ValueError("replays must be a nonempty list of per-sequence JSON paths")
    rows = defaultdict(list)
    for filename in filenames:
        path = archived_path(directory, _nonempty_string(filename, "replay path"))
        for policy, row in _object(json.loads(path.read_bytes()), str(path)).items():
            rows[policy].append(row)
    return rows


def _aggregate(policies):
    if not policies:
        raise ValueError("at least one policy is required")
    counts, utility = {}, {}
    reference = None
    for policy, rows in policies.items():
        _nonempty_string(policy, "policy name")
        if not isinstance(rows, list) or not rows:
            raise ValueError(f"policy {policy!r} requires a nonempty list of replay rows")
        sequences = {}
        grouped = defaultdict(Counter)
        for raw_row in rows:
            row = _object(raw_row, f"policy {policy!r} replay row")
            name = _nonempty_string(row.get("name"), "sequence name")
            if name in sequences:
                raise ValueError(f"policy {policy!r} has duplicate sequence {name!r}")
            source = _nonempty_string(row.get("source_id"), "source ID")
            totals = {key: row.get(key) for key in COUNT_KEYS}
            if any(type(value) is not int or value < 0 for value in totals.values()):
                raise ValueError(f"policy {policy!r}, sequence {name!r} has invalid counts")
            if totals["wrong"] > totals["accepted"] or totals["correct"] > totals["gaps"]:
                raise ValueError(f"policy {policy!r}, sequence {name!r} has inconsistent counts")
            sequences[name] = (source, totals["gaps"])
            grouped[source].update(totals)

        if reference is not None and sequences != reference:
            raise ValueError("all policies must replay the same sequence cohort, source grouping and gap counts")
        reference = sequences
        counts[policy] = [VideoCounts(source, totals["wrong"], totals["accepted"])
                          for source, totals in sorted(grouped.items())]
        total = {key: sum(row[key] for row in grouped.values()) for key in COUNT_KEYS}
        active_sources = sum(row.accepted > 0 for row in counts[policy])
        utility[policy] = {
            **total,
            "correct_per_gap": total["correct"] / total["gaps"] if total["gaps"] else None,
            "active_source_coverage": active_sources / len(grouped),
            "source_counts": [asdict(row) for row in counts[policy]],
        }
    return counts, utility, len(reference)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="certification input manifest JSON")
    parser.add_argument("--output", required=True, type=Path, help="output certification JSON")
    parser.add_argument("--alpha", required=True, type=float, help="target population wrong-reconnect risk")
    parser.add_argument("--delta", required=True, type=float, help="total confidence error budget")
    parser.add_argument("--min-active-coverage", default=0.0, type=float,
                        help="minimum empirical fraction of sources with an accepted reconnect")
    parser.add_argument("--event-delta-fraction", default=0.5, type=float,
                        help="fraction of delta for event-weighted risk when an enforced cap is declared")
    parser.add_argument("--test", choices=("kl", "betting"), default="kl",
                        help="risk test fixed before observing calibration data")
    args = parser.parse_args()
    try:
        result = certify_manifest(args.input, alpha=args.alpha, delta=args.delta,
                                  min_active_coverage=args.min_active_coverage,
                                  event_delta_fraction=args.event_delta_fraction, test=args.test)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    except (ValueError, OSError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
