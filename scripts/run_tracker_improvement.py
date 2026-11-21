"""Evaluate first-assignment changes with shared causal BoT-SORT inputs."""

import argparse
from collections import Counter
from contextlib import ExitStack
import csv
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import torch

from scripts.detect_mot17_dev import ROOT, file_sha256, load_image
from scripts.p4_supervision import load_pedestrian_truth, matched_people
from scripts.p5_report import read_metrics
from scripts.p6_cases import match_outputs
from scripts.p6_data import FROZEN, sequences, verify_cache
from scripts.p6_eval import APPEARANCE, score_local, track_count, verify_freeze
from scripts.run_p1 import read_detections
from scripts.tracker_improvement import PROFILES, monitor_assignments


def audit_frame(snapshot, previous, current, current_truth, raw, counters):
    if snapshot is None:
        return
    costs, limit, ids, high, matches = snapshot
    if not previous or not current_truth:
        return
    labels = matched_people(high, current_truth)
    high_by_person = {person: det - 1 for det, person in labels.items()}
    raw_people = set(matched_people(raw, current_truth).values())
    for row, track in enumerate(ids):
        person = previous.get(track)
        if person not in current_truth:
            continue
        det = high_by_person.get(person)
        chosen = matches.get(row)
        counters['first_stage_events'] += 1
        if det is None:
            if person in raw_people:
                counters['no_labelled_high_box'] += 1
                if current.get(person) == track:
                    counters['low_only_identity_retained'] += 1
            else:
                counters['no_labelled_raw_box'] += 1
        elif costs[row, det] > limit:
            counters['threshold_excluded'] += 1
        elif chosen == det:
            counters['correct_high_match'] += 1
        else:
            counters['admissible_high_missed'] += 1
            if current.get(person) != track:
                counters['admissible_high_missed_at_output'] += 1
            if chosen is not None:
                counters['admissible_high_wrong_assignment'] += 1


def run_sequence(root, name, info, images, cache, truth_path, profiles):
    sys.path.insert(0, str(ROOT / 'scripts'))
    import score_appearance_mot17 as legacy

    truth = load_pedestrian_truth(truth_path)
    length = len(images)
    raw = read_detections(cache, length)
    tracker_config = {}
    trackers = {}
    for label in profiles:
        torch.manual_seed(0)
        tracker, options, min_area = legacy.make_tracker('BoT-SORT', root / 'engine', info)
        tracker_config[label] = options
        trackers[label] = SimpleNamespace(tracker=tracker, minimum_area=min_area, next_id=0, counts=Counter())
    baseline = trackers['BoT-SORT'].tracker
    from tracker.basetrack import BaseTrack, TrackState

    shared = SimpleNamespace(features=None, warp=None, snapshot=None)
    baseline_inference, baseline_warp = baseline.encoder.inference, baseline.gmc.apply

    def first_features(image, boxes):
        values = baseline_inference(image, boxes)
        shared.features = values.copy()
        shared.boxes = np.asarray(boxes).copy()
        return values

    def reuse_features(image, boxes):
        if not np.array_equal(shared.boxes, np.asarray(boxes)):
            raise ValueError('Candidate detector boxes differ between variants')
        return shared.features.copy()

    def first_warp(image, boxes):
        shared.warp = baseline_warp(image, boxes)
        return shared.warp

    def low_embeddings(image, boxes):
        if shared.low is None:
            shared.low = boxes.copy(), baseline_inference(image, boxes)
        other, features = shared.low
        if not np.array_equal(boxes, other):
            raise ValueError('Low-confidence boxes differ between variants')
        return features

    baseline.encoder.inference = first_features
    baseline.gmc.apply = first_warp
    for label, state in trackers.items():
        if label != 'BoT-SORT':
            state.tracker.encoder.inference = reuse_features
            state.tracker.gmc.apply = lambda image, boxes: shared.warp.copy()

        def observe(costs, revised, limit, tracks, detections, matches, stage, *, state=state, label=label):
            state.counts[f'{stage}_adjusted_edges'] += np.count_nonzero(costs != revised)
            state.counts[f'{stage}_adjusted_frames'] += int(np.any(costs != revised))
            if stage == 'low':
                state.counts['low_recovered_lost_tracks'] += sum(
                    tracks[row].state == TrackState.Lost for row, _ in matches
                )
            if label == 'BoT-SORT':
                high = [(*tuple(det.tlbr + 1), det.score) for det in detections]
                shared.snapshot = costs, limit, [track.track_id for track in tracks], high, dict(matches)

        monitor_assignments(state.tracker, PROFILES[label], observe, low_embeddings)

    local = root / 'trackers'
    paths = {label: local / label / 'data' / f'{name}.txt' for label in trackers}
    previous, reasons = {}, Counter()
    with ExitStack() as stack, torch.inference_mode():
        writers = {}
        for label, path in paths.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            handle = stack.enter_context(path.with_suffix('.tmp').open('w', newline=''))
            writers[label] = csv.writer(handle, lineterminator='\n')
        for frame, image_path in enumerate(images, 1):
            image = load_image(image_path)
            shared.low = None
            boxes = np.asarray(raw[frame], dtype=float).reshape(-1, 5).copy()
            boxes[:, :4] -= 1
            for label, state in trackers.items():
                BaseTrack._count = state.next_id
                tracks = state.tracker.update(boxes, image)
                state.next_id = BaseTrack._count
                # Upstream only appends to the constructor's class fallback before
                # assigning a per-instance feature deque. Retaining it has no effect.
                BaseTrack.features.clear()
                outputs = []
                for track in tracks:
                    x, y, width, height = map(float, track.tlwh)
                    if height <= 0 or width * height <= state.minimum_area or width / height > 1.6:
                        continue
                    x, y = x + 1, y + 1
                    writers[label].writerow((frame, track.track_id, f'{x:.4f}', f'{y:.4f}',
                                             f'{width:.4f}', f'{height:.4f}', f'{track.score:.6f}', -1, -1, -1))
                    if label == 'BoT-SORT':
                        outputs.append((track.track_id, (x, y, x + width, y + height, 1.0)))
                if label == 'BoT-SORT':
                    current = {person: row[0] for person, row in match_outputs(outputs, truth[frame]).items()}
                    audit_frame(shared.snapshot, previous, current, truth[frame], raw[frame], reasons)
                    previous = {track: person for person, track in current.items()}
            if frame % 300 == 0 or frame == length:
                print(f'{name}: {frame}/{length}', flush=True)

    results = {}
    for label, path in paths.items():
        path.with_suffix('.tmp').replace(path)
        results[label] = {'rows': track_count(path, length), 'sha256': file_sha256(path),
                          **{key: int(value) for key, value in trackers[label].counts.items()}}
    return {'frames': length, 'options': tracker_config, 'tracks': results, 'baseline_audit': dict(reasons)}


def evaluate(domain, root, profiles, only=None, *, heldout=False):
    sys.path.insert(0, str(ROOT / 'scripts'))
    import score_appearance_mot17 as legacy
    from scripts.p5_mot_detector import DATA

    if heldout and domain != 'MOT17':
        raise ValueError('This annotated held-out evaluation uses MOT17')
    verify_freeze()
    legacy.verify_source('BoT-SORT')
    if file_sha256(legacy.WEIGHTS) != APPEARANCE:
        raise ValueError('Unexpected appearance weights')
    torch.set_num_threads(1)
    root.mkdir(parents=True, exist_ok=True)
    phase = 'final' if heldout else 'dev'
    run_config = {'domain': domain, 'evaluation': 'heldout' if heldout else 'development',
                  'sequences_requested': only,
                  'profiles': {label: PROFILES[label] for label in profiles},
                  'runner_sha256': file_sha256(Path(__file__)),
                  'association_sha256': file_sha256(ROOT / 'scripts' / 'tracker_improvement.py'),
                  'appearance_sha256': APPEARANCE, 'p5_freeze_sha256': file_sha256(FROZEN),
                  'upstream': legacy.COMMITS['BoT-SORT'], 'online_only': True}
    (root / 'run-config.json').write_text(json.dumps(run_config, indent=2) + '\n', encoding='utf-8')
    source = ROOT / 'runs' / f'p6-{domain}-{phase}'
    cache_root = (ROOT / 'runs' / 'p6-detections' / domain if heldout else
                  ROOT / 'runs' / ('p5-dance-detections' if domain == 'DanceTrack' else 'p5-mot-detections') / 'val')
    lengths, manifests = {}, {}
    for folder, info, images in sequences(domain, development=not heldout):
        name = folder.name
        if only is not None and name not in only:
            continue
        raw = cache_root / f'{name}.txt'
        cache = verify_cache(raw, domain, name, len(images), not heldout)
        truth_folder = folder if heldout or domain == 'DanceTrack' else DATA / 'gt' / name
        result = run_sequence(root, name, info, images, raw, truth_folder / 'gt' / 'gt.txt', profiles)
        archived = source / 'trackers' / 'BoT-SORT' / 'data' / f'{name}.txt'
        pinned = json.loads((source / 'manifests' / 'BoT-SORT' / f'{name}.json').read_bytes())
        if (result['tracks']['BoT-SORT']['sha256'] != file_sha256(archived)
                or file_sha256(archived) != pinned['output_sha256']
                or cache['detections_sha256'] != pinned['detections_sha256']):
            raise ValueError(f'Replayed baseline differs from the verified P6 result on {name}')
        lengths[name] = len(images)
        manifests[name] = {'cache': cache, **result}
        (root / 'sequences.json').write_text(json.dumps(manifests, indent=2) + '\n', encoding='utf-8')
    if not lengths or (only is not None and set(lengths) != set(only)):
        raise ValueError('Requested sequences must be present in the selected split')
    result = {}
    for label in profiles:
        score_local(root, domain, not heldout, label, lengths)
        result[label] = read_metrics(root / 'trackeval' / label / 'pedestrian_summary.txt')
    (root / 'metrics.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('domain', choices=('MOT17', 'DanceTrack'))
    parser.add_argument('output', type=Path)
    parser.add_argument('--sequence', nargs='+')
    parser.add_argument('--heldout', action='store_true')
    parser.add_argument('--profiles', nargs='+', choices=PROFILES, default=list(PROFILES)[1:])
    args = parser.parse_args()
    evaluate(args.domain, args.output, list(dict.fromkeys(['BoT-SORT', *args.profiles])),
             args.sequence, heldout=args.heldout)
