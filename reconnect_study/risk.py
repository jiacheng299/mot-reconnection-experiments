"""Finite-sample risk certification for frozen MOT reconnect policies.

Use one row per independently sampled source video, replayed from the start
under the corresponding policy. ``source_id`` is a grouping key. Unique IDs
alone do not establish that calibration videos are iid or that deployment uses
the same distribution. Counts from the same video may be arbitrarily dependent.
"""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from math import exp, fsum, log, log1p
from numbers import Integral
from typing import Literal

from scipy.special import logsumexp, rel_entr
from scipy.stats import beta, binom


RiskMethod = Literal["event_weighted", "active_video", "active_video_exact"]
RiskTest = Literal["kl", "betting"]


@dataclass(frozen=True)
class VideoCounts:
    """Counts for one complete replay, including videos with zero reconnects.

    ``sampled_wrong`` is optional for the exact test. Supply the outcome of one
    independently uniformly sampled accepted action per active video. The draw
    must be independent of action correctness and must not be retried.
    """

    source_id: str
    wrong: int
    accepted: int
    sampled_wrong: bool | None = None


@dataclass(frozen=True)
class PolicyCertificate:
    """Certification decision and simultaneous one-sided upper confidence bound."""

    method: RiskMethod
    certified: bool
    p_value: float
    adjusted_p_value: float
    risk_upper_bound: float
    empirical_risk: float | None
    source_count: int
    active_source_count: int
    wrong: int
    accepted: int
    sampled_wrong: int | None = None
    test: RiskTest | Literal["exact_binomial"] = "kl"


def certify_policies(
    counts_by_policy: Mapping[str, Sequence[VideoCounts]],
    *,
    alpha: float,
    delta: float,
    acceptance_cap: int | None = None,
    method: RiskMethod = "event_weighted",
    test: RiskTest = "kl",
) -> dict[str, PolicyCertificate]:
    """Certify population risk at most ``alpha`` with confidence ``1-delta``.

    Fix the policy family, outcome definition, method and source population
    independently of this calibration sample. Each policy must be replayed on
    the same iid source videos. Replay outcomes can be dependent across policies.

    Event-weighted risk is E[wrong] / E[accepted] when E[accepted] > 0.
    ``acceptance_cap`` must be a deterministic upper bound that applies to all
    future replays too. A maximum observed on these videos is inadequate.

    Active-video risk is E[wrong / accepted | accepted > 0]. The deterministic
    test uses all observed per-video fractions. The optional exact binomial
    test uses only the independently sampled labels described in VideoCounts.
    Its inference conditions on the number of active videos. In both active
    modes, ``empirical_risk`` averages the complete observed video fractions.

    Bonferroni makes the bounds simultaneous across the supplied policies for
    this method and permits post-certification utility selection among them.
    The default ``kl`` test uses a bounded-loss Chernoff bound. The ``betting``
    test inverts a fixed equal mixture of half- and full-stake product e-values.
    Choose the test before seeing calibration data. Calls comparing methods or
    tests need their own multiplicity allocation.
    The result is a population statement over the stated source distribution.
    A policy with no observed accepted actions never gets certified.
    """
    if not 0 < alpha < 1:
        raise ValueError("alpha must be strictly between 0 and 1")
    if not 0 < delta < 1:
        raise ValueError("delta must be strictly between 0 and 1")
    if method not in ("event_weighted", "active_video", "active_video_exact"):
        raise ValueError(f"unknown risk method {method!r}")
    if test not in ("kl", "betting"):
        raise ValueError(f"unknown risk test {test!r}")
    if method == "active_video_exact" and test != "kl":
        raise ValueError("active_video_exact uses a binomial test instead of betting")
    if method == "event_weighted" and acceptance_cap is None:
        raise ValueError("event-weighted certification requires an acceptance cap")
    if acceptance_cap is not None and (not _is_count(acceptance_cap) or acceptance_cap < 1):
        raise ValueError("acceptance cap must be a positive integer")

    certificates = {}
    source_ids = None
    for policy, inputs in counts_by_policy.items():
        rows = tuple(inputs)
        ids = _validate_rows(rows, acceptance_cap, method)
        if source_ids is not None and ids != source_ids:
            raise ValueError("all policies must contain the same source IDs")
        source_ids = ids
        certificates[policy] = _certify_policy(
            rows, alpha, delta, len(counts_by_policy), acceptance_cap, method, test
        )
    return certificates


def _is_count(value: object) -> bool:
    return isinstance(value, Integral) and not isinstance(value, bool) and value >= 0


def _validate_rows(
    rows: Sequence[VideoCounts], acceptance_cap: int | None, method: RiskMethod
) -> set[str]:
    source_ids = set()
    for row in rows:
        if not isinstance(row.source_id, str) or not row.source_id.strip():
            raise ValueError("source IDs must be nonempty strings")
        if row.source_id in source_ids:
            raise ValueError(f"duplicate source ID {row.source_id!r}")
        source_ids.add(row.source_id)
        if not _is_count(row.wrong) or not _is_count(row.accepted) or row.wrong > row.accepted:
            raise ValueError(f"invalid reconnect counts for source {row.source_id!r}")
        if acceptance_cap is not None and row.accepted > acceptance_cap:
            raise ValueError(f"acceptance cap exceeded for source {row.source_id!r}")
        if method == "active_video_exact" and row.accepted:
            if type(row.sampled_wrong) is not bool:
                raise ValueError(f"active source {row.source_id!r} requires a boolean sampled_wrong")
            if (row.sampled_wrong and row.wrong == 0) or (
                not row.sampled_wrong and row.wrong == row.accepted
            ):
                raise ValueError(f"sampled_wrong contradicts counts for source {row.source_id!r}")
        if method == "active_video_exact" and not row.accepted and row.sampled_wrong is not None:
            raise ValueError(f"inactive source {row.source_id!r} cannot supply sampled_wrong")
    return source_ids


def _certify_policy(
    rows: Sequence[VideoCounts],
    alpha: float,
    delta: float,
    family_size: int,
    acceptance_cap: int | None,
    method: RiskMethod,
    test: RiskTest,
) -> PolicyCertificate:
    source_count = len(rows)
    active = [row for row in rows if row.accepted]
    active_count = len(active)
    wrong = sum(int(row.wrong) for row in active)
    accepted = sum(int(row.accepted) for row in active)
    empirical_risk = None
    sampled_wrong = None
    p_value = upper = 1.0
    per_policy_delta = delta / family_size

    if accepted and method == "event_weighted":
        assert acceptance_cap is not None
        empirical_risk = wrong / accepted
        if test == "betting":
            normalized = [(int(row.accepted) / acceptance_cap, (int(row.accepted) - int(row.wrong)) / acceptance_cap)
                          for row in rows]
            p_value, upper = _betting_test(alpha, per_policy_delta, empirical_risk, normalized)
        else:
            exposure = accepted / (source_count * int(acceptance_cap))
            p_value, upper = _bounded_test(alpha, per_policy_delta, empirical_risk, source_count, exposure)
    elif accepted:
        empirical_risk = fsum(row.wrong / row.accepted for row in active) / active_count
        if method == "active_video_exact":
            sampled_wrong = sum(row.sampled_wrong is True for row in active)
            p_value = float(binom.cdf(sampled_wrong, active_count, alpha))
            if sampled_wrong < active_count and per_policy_delta > 0:
                upper = float(beta.isf(per_policy_delta, sampled_wrong + 1, active_count - sampled_wrong))
        elif test == "betting":
            normalized = [(1.0, (int(row.accepted) - int(row.wrong)) / int(row.accepted)) for row in active]
            p_value, upper = _betting_test(alpha, per_policy_delta, empirical_risk, normalized)
        else:
            p_value, upper = _bounded_test(alpha, per_policy_delta, empirical_risk, active_count, 1.0)

    adjusted = min(1.0, family_size * p_value)
    return PolicyCertificate(
        method=method,
        certified=accepted > 0 and upper <= alpha,
        p_value=p_value,
        adjusted_p_value=adjusted,
        risk_upper_bound=upper,
        empirical_risk=empirical_risk,
        source_count=source_count,
        active_source_count=active_count,
        wrong=wrong,
        accepted=accepted,
        sampled_wrong=sampled_wrong,
        test="exact_binomial" if method == "active_video_exact" else test,
    )


def _bounded_test(
    alpha: float, delta: float, empirical_risk: float, count: int, exposure: float
) -> tuple[float, float]:
    def log_p_value(risk: float) -> float:
        if risk <= empirical_risk:
            return 0.0
        # For the event-weighted null, Y_i = risk + (W_i - risk*A_i)/cap is in
        # [0,1], with E[Y_i] >= risk. For the active test, Y_i is simply W_i/A_i.
        mean = (1 - exposure) * risk + exposure * empirical_risk
        divergence = rel_entr(mean, risk) + rel_entr(1 - mean, 1 - risk)
        return -count * max(0.0, float(divergence))

    return _test_bounds(log_p_value, alpha, delta, empirical_risk)


def _betting_test(
    alpha: float, delta: float, empirical_risk: float, normalized: Sequence[tuple[float, float]]
) -> tuple[float, float]:
    """Use (accepted/cap, correct/cap) pairs, retaining within-source dependence."""
    def log_p_value(risk: float) -> float:
        if risk <= empirical_risk:
            return 0.0
        component_logs = []
        for stake in (0.5, 1.0):
            # Each factor is 1 + stake*(risk*A-W)/(cap*(1-risk)). Its mean is
            # at most 1 under the risk null, so independence makes its product
            # an e-value. The equivalent form below avoids subtracting r*A-W
            # when the tested risk is near one and almost every action is wrong.
            offsets = [stake * (correct / (1 - risk) - accepted) for accepted, correct in normalized]
            log_product = fsum(log1p(offset) for offset in offsets) if -1.0 not in offsets else float("-inf")
            component_logs.append(log_product)
        log_e_value = logsumexp(component_logs) - log(2)
        return min(0.0, -float(log_e_value))

    return _test_bounds(log_p_value, alpha, delta, empirical_risk)


def _test_bounds(
    log_p_value: Callable[[float], float], alpha: float, delta: float, empirical_risk: float
) -> tuple[float, float]:
    upper = _invert_log_p_value(log_p_value, log(delta), empirical_risk) if delta > 0 else 1.0
    return exp(log_p_value(alpha)), upper


def _invert_log_p_value(
    log_p_value: Callable[[float], float], log_delta: float, empirical_risk: float
) -> float:
    # Along the zero-at-empirical-risk path, joint convexity makes KL grow.
    # Each betting factor is nondecreasing as risk grows because correct >= 0.
    lower, upper = empirical_risk, 1.0
    for _ in range(54):
        middle = (lower + upper) / 2
        if middle in (lower, upper):
            break
        if log_p_value(middle) <= log_delta:
            upper = middle
        else:
            lower = middle
    return upper
