"""Verify online prefix invariance and track feasibility for the P4 joint network."""

import argparse
import builtins
from collections import Counter
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from scripts.detect_mot17_dev import SEQUENCE
from scripts.p4_joint import load_joint_ranker
from scripts.run_p1 import read_detections
from scripts.score_p3_c1 import track_sequence


class OrderedFrames:
    def __init__(self, cached, stop=None):
        self.cached = cached
        self.stop = stop
        self.last = 0

    def get(self, frame, default):
        if frame != self.last + 1:
            raise AssertionError(f'Out-of-order detection access: {frame} after {self.last}')
        self.last = frame
        if self.stop is not None and frame > self.stop:
            return default
        return self.cached.get(frame, default)


def prefix_invariance(detections, ranker, method, prefix=54, length=100, sequence=SEQUENCE):
    path_open, builtin_open = Path.open, builtins.open

    def guard(open_file):
        def checked(path, *args, **kwargs):
            if not isinstance(path, int) and 'gt' in Path(path).parts:
                raise AssertionError(f'Tracker opened ground truth: {path}')
            return open_file(path, *args, **kwargs)
        return checked

    def before(path, delimiter=b','):
        return [line for line in path.read_bytes().splitlines() if int(line.split(delimiter)[0]) <= prefix]

    with TemporaryDirectory() as directory:
        original, changed = Path(directory) / 'original', Path(directory) / 'changed'
        with patch.object(Path, 'open', guard(path_open)), patch('builtins.open', guard(builtin_open)):
            a = OrderedFrames(detections)
            track_sequence(original, a, length, sequence, method, ranker)
            b = OrderedFrames(detections, stop=prefix)
            track_sequence(changed, b, length, sequence, method, ranker)
        relative = Path('trackers') / method / 'data' / f'{sequence}.txt'
        assert a.last == b.last == length
        assert before(original / relative) == before(changed / relative)
        left = (original / 'association.jsonl').read_bytes().splitlines()
        right = (changed / 'association.jsonl').read_bytes().splitlines()
        assert left[:prefix] == right[:prefix] and left[prefix:] != right[prefix:]
    return {'verified_prefix': prefix, 'verified_access_order': length, 'ground_truth_opens': 0}


def audit_run(run, method):
    counts = Counter()
    with (run / 'association.jsonl').open(encoding='utf-8') as file:
        for frame, line in enumerate(file, 1):
            row = json.loads(line)
            assert row['frame'] == frame
            used = [det for det in row['matches'].values() if det is not None] + [det for _, det in row['births']]
            assert len(used) == len(set(used))
            for group in row['groups']:
                tracks, actions = group['tracks'], group['candidates']
                assert 0 < len(actions) <= 16 and group['selected'] in actions
                counts['fallback_groups'] += group['fallback']
                counts['max_candidates'] = max(counts['max_candidates'], len(actions))
                for act in actions:
                    boxes = [det for det in act if det is not None]
                    assert len(boxes) == len(set(boxes))
                    assert all(det is None or str(det) in row['legal_scores'][str(track)]
                               for track, det in zip(tracks, act, strict=True))
    assert frame == 654 and counts['fallback_groups'] == 0
    poses = run / 'trackers' / method / 'data' / f'{SEQUENCE}.txt'
    ids = [tuple(line.split(',')[:2]) for line in poses.read_text().splitlines()]
    assert len(ids) == len(set(ids))
    return {'frames': frame, 'unique_output_pairs': len(ids), **counts}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs', type=Path)
    args = parser.parse_args()
    root = args.runs
    detections = read_detections(root / 'p2-002' / 'detections' / f'{SEQUENCE}.txt', 654)
    for key, gamma in (('000', '0.00'), ('090', '0.90'), ('099', '0.99')):
        leg = f'P4-TD-g{gamma}'
        joint = f'P4-joint-g{gamma}'
        print(json.dumps({'method': leg, **audit_run(root / f'p4-td-g{key}-dev', leg)}))
        ranker, method = load_joint_ranker(root / f'p4-joint-g{key}.json')
        assert method == joint
        print(json.dumps({'method': joint, **audit_run(root / f'p4-joint-g{key}-dev', joint),
                          **prefix_invariance(detections, ranker, method)}))
