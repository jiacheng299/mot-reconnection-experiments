"""Evaluate the fixed-geometry C1 pilot on the MOT17-10 shared detection cache."""

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
from time import perf_counter_ns

import numpy as np

from scripts.detect_mot17_dev import DATA, ROOT, SEQUENCE, file_sha256, sequence_info
from scripts.p3_association import competition_groups
from scripts.p3_c1 import choose_group_action
from scripts.run_p1 import evaluate, load_tracker, read_detections


METHOD = 'C1-fixed-geometry'
OPTIONS = {'low_confidence': 0.1, 'high_confidence': 0.5, 'birth_confidence': 0.6,
           'strong_box_iou': 0.3, 'weak_box_iou': 0.5, 'lost_buffer_frames': 30,
           'max_candidates_per_group': 16}


def select_frame_actions(scores, confidences=None, ranker=None, context=None, *, candidate_limit=16):
    matched = {}
    groups = []
    batched_ranker = getattr(ranker, 'score_groups', None)
    for tracks in competition_groups(scores):
        action, proposals, fallback = choose_group_action(tracks, scores, candidate_limit)
        if ranker is not None and not fallback and batched_ranker is None:
            if hasattr(ranker, 'score_candidates'):
                estimates = ranker.score_candidates(tracks, proposals, scores, confidences, context)
                action, _ = max(zip(proposals, estimates, strict=True), key=lambda item: item[1])
            else:
                action = max(proposals, key=lambda candidate: ranker(tracks, candidate, scores, confidences))
        for proposal in proposals:
            consumed = [detection for detection in proposal if detection is not None]
            if len(consumed) != len(set(consumed)) or any(
                detection is not None and detection not in scores[track]
                for track, detection in zip(tracks, proposal, strict=True)
            ):
                raise ValueError('Generated an illegal group proposal')
        if action not in proposals:
            raise ValueError('Selected proposal was not in the candidate set')
        matched.update(zip(tracks, action, strict=True))
        groups.append({'tracks': tracks, 'candidates': proposals, 'selected': action, 'fallback': fallback})
    if batched_ranker is not None:
        eligible = [g for g in groups if not g['fallback'] and len(g['candidates']) > 1]
        if eligible:
            estimates = batched_ranker(eligible, scores, confidences, context)
            for group, group_scores in zip(eligible, estimates, strict=True):
                group['selected'], _ = max(zip(group['candidates'], group_scores, strict=True), key=lambda item: item[1])
                matched.update(zip(group['tracks'], group['selected'], strict=True))
    used = [detection for detection in matched.values() if detection is not None]
    if len(used) != len(set(used)):
        raise ValueError('A detection was assigned twice across groups')
    return matched, groups, set(used)


def prepare_frame(active, detections):
    from yolox.tracker.byte_tracker import STrack, TrackState, matching

    boxes = {i: STrack(STrack.tlbr_to_tlwh(np.array(box[:4])), box[4])
             for i, box in enumerate(detections, start=1)
             if box[4] >= OPTIONS['low_confidence']}
    tracks = list(active.values())
    STrack.multi_predict([track for track in tracks if track.is_activated])
    ious = 1.0 - matching.iou_distance(tracks, list(boxes.values()))
    scores = {}
    for track, overlaps in zip(tracks, ious, strict=True):
        allowed = {}
        for (index, detection), overlap in zip(boxes.items(), overlaps, strict=True):
            if detection.score >= OPTIONS['high_confidence']:
                gate = OPTIONS['strong_box_iou']
            elif track.state == TrackState.Tracked and track.is_activated:
                gate = OPTIONS['weak_box_iou']
            else:
                continue
            if overlap >= gate:
                allowed[index] = float(overlap)
        scores[track.track_id] = allowed

    return boxes, scores


def commit_frame(active, boxes, assigned, frame, kalman):
    from yolox.tracker.basetrack import TrackState

    consumed = {detection for detection in assigned.values() if detection is not None}
    births = []
    ended = []
    lost = []
    reactivated = []
    for track_id, detection in assigned.items():
        track = active[track_id]
        if detection is not None:
            if track.state == TrackState.Lost:
                track.re_activate(boxes[detection], frame, new_id=False)
                reactivated.append(track_id)
            else:
                track.update(boxes[detection], frame)
        elif not track.is_activated or frame - track.end_frame > OPTIONS['lost_buffer_frames']:
            track.mark_removed()
            del active[track_id]
            ended.append(track_id)
        elif track.state == TrackState.Tracked:
            track.mark_lost()
            lost.append(track_id)

    for detection, box in boxes.items():
        if detection not in consumed and box.score >= OPTIONS['birth_confidence']:
            box.activate(kalman, frame)
            active[box.track_id] = box
            births.append((box.track_id, detection))
            consumed.add(detection)

    return births, ended, lost, reactivated


def track_sequence(run, detections, length, sequence=SEQUENCE, method=METHOD, ranker=None, *,
                   candidate_limit=16, audit_path=None, timings=None):
    load_tracker(run)
    from yolox.tracker.basetrack import BaseTrack
    from yolox.tracker.byte_tracker import KalmanFilter

    BaseTrack._count = 0
    context_for_frame = None
    if hasattr(ranker, 'score_candidates'):
        from scripts.p4_joint import tracking_context

        context_for_frame = tracking_context
    kalman = KalmanFilter()
    active = {}
    counters = Counter()
    group_sizes = Counter()
    output = run / 'trackers' / method / 'data' / f'{sequence}.txt'
    output.parent.mkdir(parents=True, exist_ok=True)
    audit_path = audit_path or run / 'association.jsonl'
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w', newline='') as poses, audit_path.open('w', encoding='utf-8') as audit:
        writer = csv.writer(poses, lineterminator='\n')
        for frame in range(1, length + 1):
            start = perf_counter_ns() if timings is not None else 0
            boxes, scores = prepare_frame(active, detections.get(frame, ()))
            context = context_for_frame(active, frame) if context_for_frame else None
            assigned, groups, consumed = select_frame_actions(
                scores, {i: box.score for i, box in boxes.items()}, ranker, context, candidate_limit=candidate_limit
            )
            births, ended, lost, reactivated = commit_frame(active, boxes, assigned, frame, kalman)
            if timings is not None:
                timings.append((perf_counter_ns() - start, sum(len(g['candidates']) for g in groups), len(groups)))

            for track in active.values():
                if not track.is_activated or track.frame_id != frame:
                    continue
                x, y, width, height = track.tlwh
                if height > 0 and width * height > 10 and width / height <= 1.6:
                    writer.writerow((frame, track.track_id, f'{x:.4f}', f'{y:.4f}',
                                     f'{width:.4f}', f'{height:.4f}', f'{track.score:.6f}', -1, -1, -1))
                    counters['output_boxes'] += 1

            counters['frames'] += 1
            counters['matches'] += len(consumed)
            counters['misses'] += sum(detection is None for detection in assigned.values())
            counters['births'] += len(births)
            counters['terminations'] += len(ended)
            counters['reactivations'] += len(reactivated)
            counters['competition_frames'] += any(len(group['tracks']) > 1 for group in groups)
            counters['fallback_groups'] += sum(group['fallback'] for group in groups)
            counters['candidate_actions'] += sum(len(group['candidates']) for group in groups)
            group_sizes.update(len(group['tracks']) for group in groups)
            audit.write(json.dumps({'frame': frame, 'legal_scores': scores, 'groups': groups,
                                    'matches': assigned, 'births': births, 'newly_lost': lost,
                                    'reactivations': reactivated, 'terminations': ended}, allow_nan=False) + '\n')

    return {**counters, 'group_sizes': dict(sorted(group_sizes.items())), 'duplicate_detection_assignments': 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--sequence', choices=(SEQUENCE, 'MOT17-05-FRCNN'), default=SEQUENCE)
    parser.add_argument('--cache', type=Path)
    scorers = parser.add_mutually_exclusive_group()
    scorers.add_argument('--ranker', type=Path)
    scorers.add_argument('--immediate-ranker', type=Path)
    scorers.add_argument('--td-ranker', type=Path)
    scorers.add_argument('--joint-ranker', type=Path)
    parser.add_argument('--rollout-only', action='store_true', help='record options and skip TrackEval')
    args = parser.parse_args()
    sequence, run = args.sequence, args.run.resolve()
    cache = args.cache or ROOT / 'runs' / 'p2-002' / 'detections' / f'{sequence}.txt'
    manifest = json.loads(cache.with_suffix('.json').read_text(encoding='utf-8'))
    length = sequence_info(sequence).getint('seqLength')
    digest = file_sha256(cache)
    if manifest['detections_sha256'] != digest or manifest['sequence'] != sequence or manifest['frames'] != length:
        raise ValueError(f'Shared detection cache does not match the {sequence} manifest')
    detections = read_detections(cache, length)
    count = sum(map(len, detections.values()))
    if count != manifest['count']:
        raise ValueError('Unexpected number of cached detections')
    run.mkdir(parents=True, exist_ok=False)
    print(f'Using shared cache SHA-256 {digest}; {length} frames and {count} detections', flush=True)
    ranker = None
    method = METHOD
    checkpoint = args.ranker or args.immediate_ranker or args.td_ranker or args.joint_ranker
    if args.joint_ranker:
        from scripts.p4_joint import load_joint_ranker

        ranker, method = load_joint_ranker(args.joint_ranker)
    elif args.td_ranker:
        from scripts.p4_td import load_td_ranker

        ranker, method = load_td_ranker(args.td_ranker)
    elif checkpoint:
        from scripts.p4_supervision import load_ranker

        ranker = load_ranker(checkpoint, immediate=args.immediate_ranker is not None)
        method = 'P4-immediate-reward' if args.immediate_ranker else 'P4-current-supervision'
    summary = track_sequence(run, detections, length, sequence, method, ranker)
    record = {
        'sequence': sequence, 'detections_sha256': digest, 'cached_detections': count,
        'configuration': OPTIONS, 'summary': summary,
    }
    if checkpoint:
        record['ranker_sha256'] = file_sha256(checkpoint)
    (run / 'association-summary.json').write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(summary), flush=True)
    if not args.rollout_only:
        evaluate(run, length, sequence, DATA / 'gt', tracker=method)


if __name__ == '__main__':
    main()
