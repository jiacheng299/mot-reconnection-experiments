"""Runtime gate for a frozen reconnect score and a source-level budget."""

import json
from math import isfinite
from pathlib import Path
from typing import Callable

from reconnect_study.scoring import Score


def margin(features):
    return min(features["row_margin"], features["column_margin"])


def assignment(features):
    return -features["assignment_cost"]


CONTROLS = {"assignment": assignment, "margin": margin}


class RecoveryPolicy:
    """One independent instance per source, carried across clips of that source."""

    def __init__(self, score: Callable | None, cutoff: float | None, acceptance_cap: int | None):
        if (score is None) != (cutoff is None):
            raise ValueError("Supply both a score and a cutoff, or neither")
        if cutoff is not None and not isfinite(cutoff):
            raise ValueError("Cutoff must be finite")
        if acceptance_cap is not None and acceptance_cap < 1:
            raise ValueError("Acceptance cap must be positive")
        self.score = score
        self.cutoff = cutoff
        self.cap = acceptance_cap
        self.accepted = 0

    def __call__(self, proposal):
        if self.cap is not None and self.accepted >= self.cap:
            return False
        if self.score is not None:
            result = self.score(proposal.features)
            if not isfinite(result) or result < self.cutoff:
                return False
        self.accepted += 1
        return True


def policy_specs(config):
    """Expand a frozen threshold grid without inspecting replay outcomes."""
    scores = {**CONTROLS, **{name: Score(**value["score"]) for name, value in config["training"].items()}}
    return {f"{method}_{index}": (scores[method], threshold)
            for method, values in config["thresholds"].items()
            for index, threshold in enumerate(values)}


def load_policy(config_path: Path, name: str):
    """Load one policy from a saved study configuration for one new source."""
    config = json.loads(config_path.read_bytes())
    score, threshold = policy_specs(config)[name]
    return RecoveryPolicy(score, threshold, config["acceptance_cap"])
