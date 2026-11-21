"""Conservative first-stage association changes for BoT-SORT."""

import inspect
from types import MethodType

import numpy as np
from scipy.spatial.distance import cdist


RISK_THRESHOLDS = {f'RecoveryRisk-{value:.2f}': value for value in (0.28, 0.37, 0.46, 0.55)}
PROFILES = {
    'BoT-SORT': None,
    'Consensus-lite': (0.10, 0.35),
    'Consensus': (0.20, 0.60),
    'Consensus-wide': (0.30, 0.85),
    'LowAppearance': 'appearance',
    'RecentRecovery': 'recovery',
    'RecoveryOnly': 'recovery-only',
    **{label: label for label in RISK_THRESHOLDS},
}


def rerank_cost(baseline, geometry, appearance, threshold, margin, weight):
    """Penalize weak corroboration among close, competing admissible pairs."""
    if not baseline.size:
        return baseline
    legal = np.isfinite(baseline) & (baseline < threshold)
    legal_cost = np.where(legal, baseline, np.inf)
    near = (legal & (baseline <= legal_cost.min(axis=0) + margin)
            & (baseline <= legal_cost.min(axis=1)[:, None] + margin))
    active = near & (near.sum(axis=0) > 1) & (near.sum(axis=1)[:, None] > 1)
    if not active.any():
        return baseline
    alternate = np.maximum(geometry[active], appearance[active])
    revised = baseline.copy()
    penalty = weight * np.maximum(0, alternate - baseline[active])
    revised[active] = np.minimum(baseline[active] + penalty, np.nextafter(threshold, -np.inf))
    return revised


def low_confidence_cost(geometry, appearance, scores, lost, *, adjust_active=True):
    """Protect active assignments and allow strongly corroborated recent recoveries."""
    if not geometry.size:
        return geometry
    lost = np.asarray(lost, dtype=bool)
    active = ~lost[:, None]
    cost = geometry.copy()
    if adjust_active:
        supported = active & (geometry <= 0.60) & (appearance <= 0.20)
        cost[supported] = 0.65 * geometry[supported] + 0.35 * appearance[supported]
    active_claims = (active & (cost < 0.5)).any(axis=0)
    if not lost.any():
        return cost
    candidates = (lost[:, None] & (geometry < 0.48) & (appearance <= 0.13)
                  & (np.asarray(scores) >= 0.20) & ~active_claims)
    joint = np.where(candidates, 0.65 * geometry + 0.35 * appearance, np.inf)
    near_row = candidates & (joint <= joint.min(axis=1)[:, None] + 0.06)
    near_col = candidates & (joint <= joint.min(axis=0) + 0.06)
    unique = near_row & near_col & (near_row.sum(axis=1)[:, None] == 1) & (near_col.sum(axis=0) == 1)
    cost[lost, :] = 1.0
    cost[unique] = joint[unique]
    return cost


def recovery_risk_cost(geometry, appearance, scores, age, threshold):
    """Trade geometry for stronger appearance, with a competitive reject option."""
    if not geometry.size:
        return geometry
    age = np.asarray(age)
    lost = age > 0
    score = np.asarray(scores)
    claimed = ((~lost[:, None]) & (geometry <= 0.5)).any(axis=0)
    eligible = (lost[:, None] & ~claimed & (geometry <= 0.72)
                & (appearance <= 0.24) & (score > 0.1))
    raw = 0.48 * geometry + 1.2 * appearance + 0.14 * (0.6 - score) + 0.025 * (age[:, None] - 2)
    risk = np.where(eligible, raw, np.inf)
    by_row = eligible & (risk <= risk.min(axis=1)[:, None] + 0.055)
    by_col = eligible & (risk <= risk.min(axis=0) + 0.055)
    acceptable = (by_row & by_col & (by_row.sum(axis=1)[:, None] == 1)
                  & (by_col.sum(axis=0) == 1) & (risk <= threshold))
    cost = geometry.copy()
    cost[lost, :] = 1.0
    cost[acceptable] = 0.49 * risk[acceptable] / threshold
    return cost


def monitor_assignments(tracker, profile, observer, low_embeddings):
    """Use the pinned BoT-SORT implementation with explicit association hooks."""
    import textwrap

    original = type(tracker).update
    namespace = dict(original.__globals__)
    source = textwrap.dedent(inspect.getsource(original))
    anchor = 'matching.linear_assignment(dists, thresh=self.args.match_thresh)'
    if source.count(anchor) != 1:
        raise ValueError('BoT-SORT first assignment hook has changed')
    source = source.replace(
        anchor, 'self.first_assignment(dists, ious_dists, raw_emb_dists, strack_pool, detections)'
    )
    if isinstance(profile, str):
        pool = 'r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == TrackState.Tracked]'
        second = 'matches, u_track, u_detection_second = matching.linear_assignment(dists, thresh=0.5)'
        if source.count(pool) != 1 or source.count(second) != 1:
            raise ValueError('BoT-SORT second association hook has changed')
        source = source.replace(pool, 'r_tracked_stracks = self.low_pool(strack_pool, u_track)')
        source = source.replace(second, 'matches, u_track, u_detection_second = '
                                'self.second_assignment(dists, r_tracked_stracks, detections_second, img)')
    exec(compile(source, inspect.getfile(original), 'exec'), namespace)
    tracker.update = MethodType(namespace['update'], tracker)
    solver = namespace['matching'].linear_assignment

    def first_assignment(costs, geometry, appearance, tracks, detections):
        limit = tracker.args.match_thresh
        revised = rerank_cost(costs, geometry, appearance, limit, *profile) if isinstance(profile, tuple) else costs
        result = solver(revised, thresh=limit)
        observer(costs, revised, limit, tracks, detections, result[0], 'high')
        return result

    tracker.first_assignment = first_assignment

    if isinstance(profile, str):
        track_state = namespace['TrackState']
        risk_limit = RISK_THRESHOLDS.get(profile)

        def low_pool(pool, unmatched):
            return [pool[i] for i in unmatched
                    if pool[i].state == track_state.Tracked or
                    (profile != 'appearance' and pool[i].state == track_state.Lost
                     and tracker.frame_id - pool[i].end_frame <= 4)]

        def second_assignment(geometry, tracks, detections, image):
            lost = [track.state == track_state.Lost for track in tracks]
            revised = geometry
            if geometry.size and (geometry <= (0.72 if risk_limit is not None else 0.60)).any():
                boxes = np.asarray([det.tlbr for det in detections])
                features = low_embeddings(image, boxes)
                profiles = np.asarray([track.smooth_feat for track in tracks])
                appearance = np.maximum(0, cdist(profiles, features, 'cosine') / 2)
                scores = [det.score for det in detections]
                if risk_limit is None:
                    revised = low_confidence_cost(geometry, appearance, scores,
                                                  lost, adjust_active=profile != 'recovery-only')
                else:
                    age = [tracker.frame_id - t.end_frame if is_lost else 0
                           for t, is_lost in zip(tracks, lost, strict=True)]
                    revised = recovery_risk_cost(geometry, appearance, scores, age, risk_limit)
            elif any(lost):
                revised = geometry.copy()
                revised[lost, :] = 1.0
            result = solver(revised, thresh=0.5)
            observer(geometry, revised, 0.5, tracks, detections, result[0], 'low')
            return result

        tracker.low_pool = low_pool
        tracker.second_assignment = second_assignment
