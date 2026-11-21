"""Audit predefined recovery events shared by every tracker on the selected split."""

import argparse
from collections import Counter
import json
from pathlib import Path

from scripts.detect_mot17_dev import ROOT, file_sha256
from scripts.p4_supervision import load_pedestrian_truth, matched_people
from scripts.p6_cases import match_outputs, output_boxes
from scripts.p6_data import sequences, verify_cache
from scripts.run_p1 import read_detections


def recovery_events(truth, scores):
    """Use only annotations and frozen detector evidence to choose events."""
    events = []
    for frame, values in sorted(scores.items()):
        for identity, score in sorted(values.items()):
            if frame < 2 or identity not in truth.get(frame, {}):
                continue
            for before in range(frame - 1, max(0, frame - 5), -1):
                if identity not in truth.get(before, {}):
                    break
                prior = scores.get(before, {}).get(identity)
                if prior is None:
                    continue
                gap = frame - before - 1
                if prior > 0.60 and (gap or score <= 0.60):
                    group = ('gap_low' if score <= 0.60 else 'gap_high') if gap else 'consecutive_low'
                    events.append({'identity': identity, 'before': before, 'return': frame,
                                   'gap': gap, 'score': score, 'group': group})
                break
    return events


def continuation(event, assigned):
    identity = event['identity']
    before = assigned.get(event['before'], {}).get(identity)
    now = assigned.get(event['return'], {}).get(identity)
    if before is None:
        return 'no_anchor'
    if now is None:
        return 'missed'
    return 'correct' if now == before else 'switched'


def audit(domain, roots, output, *, heldout=False):
    from scripts.p5_mot_detector import DATA

    roots = {root: json.loads((root / 'sequences.json').read_bytes()) for root in roots}
    coverage = set(next(iter(roots.values())))
    if not coverage or any(set(rows) != coverage for rows in roots.values()):
        raise ValueError('All runs must cover the same development sequence list')
    if heldout and domain != 'MOT17':
        raise ValueError('This annotated held-out evaluation uses MOT17')
    phase = 'final' if heldout else 'dev'
    baseline = ROOT / 'runs' / f'p6-{domain}-{phase}'
    cache_root = (ROOT / 'runs' / 'p6-detections' / domain if heldout else
                  ROOT / 'runs' / ('p5-dance-detections' if domain == 'DanceTrack' else 'p5-mot-detections') / 'val')
    events, counts, sources = [], {}, {}
    for folder, _, images in sequences(domain, development=not heldout):
        name = folder.name
        if name not in coverage:
            continue
        truth_folder = folder if heldout or domain == 'DanceTrack' else DATA / 'gt' / name
        truth = load_pedestrian_truth(truth_folder / 'gt' / 'gt.txt')
        cache = cache_root / f'{name}.txt'
        cache_record = verify_cache(cache, domain, name, len(images), not heldout)
        raw = read_detections(cache, len(images))
        scores = {f: {person: boxes[index - 1][4] for index, person in matched_people(boxes, truth[f]).items()}
                  for f, boxes in raw.items()}
        rows = recovery_events(truth, scores)
        methods = {}
        for root, run in roots.items():
            for method, record in run[name]['tracks'].items():
                path = root / 'trackers' / method / 'data' / f'{name}.txt'
                if file_sha256(path) != record['sha256'] or (method in methods
                        and file_sha256(methods[method]) != record['sha256']):
                    raise ValueError(f'Changed or inconsistent track output: {method}/{name}')
                methods[method] = path
        for method in ('BoT-SORT', 'OC-SORT', 'TrackTrack'):
            reference = baseline / 'trackers' / method / 'data' / f'{name}.txt'
            manifest = json.loads((baseline / 'manifests' / method / f'{name}.json').read_bytes())
            if (cache_record['detections_sha256'] != manifest['detections_sha256']
                    or file_sha256(reference) != manifest['output_sha256']
                    or (method in methods and file_sha256(methods[method]) != manifest['output_sha256'])):
                raise ValueError(f'Archived comparison differs: {method}/{name}')
            methods[method] = reference
        assigned = {}
        frames = sorted({frame for event in rows for frame in (event['before'], event['return'])})
        for method, path in methods.items():
            predictions = output_boxes(path)
            assigned[method] = {
                frame: {person: row[0] for person, row in match_outputs(predictions[frame], truth[frame]).items()}
                for frame in frames
            }
        sources[name] = {method: file_sha256(path) for method, path in methods.items()}
        for row in rows:
            outcomes = {method: continuation(row, by_frame) for method, by_frame in assigned.items()}
            events.append({'sequence': name, **row, 'outcomes': outcomes})
            for method, category in outcomes.items():
                counts.setdefault(row['group'], {}).setdefault(method, Counter())[category] += 1
    if sources.keys() != coverage:
        raise ValueError('A run referenced a sequence outside the selected split')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({'domain': domain, 'split': phase, 'sequence_output_hashes': sources,
                                  'counts': counts, 'events': events}, indent=2) + '\n', encoding='utf-8')
    for group, methods in counts.items():
        print(group, {name: (row['correct'], row.total()) for name, row in methods.items()}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('domain', choices=('MOT17', 'DanceTrack'))
    parser.add_argument('output', type=Path)
    parser.add_argument('runs', type=Path, nargs='+')
    parser.add_argument('--heldout', action='store_true')
    args = parser.parse_args()
    audit(args.domain, args.runs, args.output, heldout=args.heldout)
