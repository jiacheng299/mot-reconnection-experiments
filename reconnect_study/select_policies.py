"""Select fixed test policies using calibration replays only."""

import argparse
from collections import defaultdict
import json
from pathlib import Path

from reconnect_study.data_io import digest
from reconnect_study.study import save


METHODS = ("assignment", "margin", "linear", "structured", "structured_rank")


def choose(calibration, minimum_coverage):
    if calibration["split"] != "calibration":
        raise ValueError("Policy selection requires calibration replays")
    candidates = {}
    for name, rows in calibration["policies"].items():
        if name == "unconstrained_tracker":
            continue
        active = defaultdict(int)
        for row in rows:
            active[row["source_id"]] += row["accepted"]
        totals = calibration["aggregate"][name]
        candidates[name] = {**totals,
                            "active_source_coverage": sum(value > 0 for value in active.values()) / len(active)}

    reference = calibration["aggregate"]["unconstrained_tracker"]["wrong_per_acceptance"]
    primary_budget = 0.2
    comparison_budget = max(primary_budget, reference - 0.05) if reference is not None else primary_budget
    selections = {}
    for budget in sorted({primary_budget, comparison_budget}):
        policies = {}
        for method in METHODS:
            options = [(name, values) for name, values in candidates.items()
                       if name.rsplit("_", 1)[0] == method
                       and values["wrong_per_acceptance"] is not None
                       and values["wrong_per_acceptance"] <= budget
                       and values["active_source_coverage"] >= minimum_coverage]
            policies[method] = max(options, key=lambda item: (
                item[1]["correct"], item[1]["accepted"] - item[1]["wrong"],
                -item[1]["wrong_per_acceptance"], item[0]), default=(None, None))[0]
        selections[str(budget)] = policies
    test_policies = {"unconstrained_tracker"}
    if calibration["acceptance_cap"] is not None:
        test_policies.add("cap_only")
    for values in selections.values():
        test_policies.update(name for name in values.values() if name is not None)
    return {"tracker": calibration["tracker"], "config_sha256": calibration["config_sha256"],
            "budgets": selections, "references": candidates, "test_policies": sorted(test_policies),
            "minimum_active_coverage": minimum_coverage,
            "selection_rule": "For each score and empirical budget, maximize correct natural-gap recoveries "
                              "then correct accepted reactivations within calibration."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("calibration", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--minimum-coverage", type=float, default=0.20)
    args = parser.parse_args()
    calibration = json.loads(args.calibration.read_bytes())
    result = choose(calibration, args.minimum_coverage)
    result["calibration_sha256"] = digest(args.calibration)
    save(args.output, result)
    print(result["budgets"])


if __name__ == "__main__":
    main()
