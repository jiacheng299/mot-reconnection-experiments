"""Freeze diagnostic bins on the original calibration cohort, then summarize archived test traces."""

import argparse
from bisect import bisect_left
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path

import numpy as np

from reconnect_study.data_io import digest, read_manifest
from reconnect_study.replay import load_sequence, replay
from reconnect_study.study import TRACKERS, save


MANIFEST = Path('runs/reconnect_study/personpath22/manifest.json')
ROOT = Path('runs/reconnect_supplement/round3/diagnostics')


def freeze_bins():
    sequences = [s for s in read_manifest(MANIFEST) if s.split == 'calibration']
    margins = defaultdict(list)
    lengths = []
    for sequence in sequences:
        det, evidence = load_sequence(sequence)
        lengths.extend(g.after - g.before - 1 for g in evidence.gaps)
        for tracker in ('bytetrack', 'botsort-motion-only'):
            record, _ = replay(sequence, det, evidence, tracker=TRACKERS[tracker], include_records=True)
            for action in record['actions']:
                if action['evaluable'] and action['accepted']:
                    features = action['features']
                    margins[tracker].append(min(features['row_margin'], features['column_margin']))
    edges = np.quantile(lengths, [1/3, 2/3], method='inverted_cdf')
    gap_edges = sorted({int(x) for x in edges if x < max(lengths)})
    result = {'source_ids': sorted({s.source_id for s in sequences}),
              'manifest_sha256': digest(MANIFEST), 'script_sha256': digest(Path(__file__)),
              'gap_edges': gap_edges, 'gap_counts': dict(sorted(Counter(lengths).items())),
              'margin_edges': {t: sorted(set(np.quantile(v, [1/3, 2/3]).tolist()))
                               for t, v in margins.items()},
              'unconstrained_evaluable_actions': {t: len(v) for t, v in margins.items()},
              'bin_rule': 'Upper edges inclusive; last bin is open at its lower bound.',
              'margin_reference': 'Host native unconstrained evaluable accepted proposals on calibration.'}
    if len(result['source_ids']) != 91 or len(lengths) != 4536:
        raise ValueError('Unexpected calibration cohort')
    save(ROOT/'calibration-bins.json', result)


def summarize(tracker, directories):
    bins = json.loads((ROOT/'calibration-bins.json').read_bytes())
    expected = {s.name for s in read_manifest(MANIFEST) if s.split == 'test'}
    rows = []
    for directory in directories:
        stats = defaultdict(Counter)
        files = sorted(directory.glob('*.json'))
        if {f.stem for f in files} != expected:
            raise ValueError(f'Trace coverage mismatch: {directory}')
        for file in files:
            record = json.loads(file.read_bytes())
            active = set()
            for g in record['gap_records']:
                key = 'natural_gap_length', bisect_left(bins['gap_edges'], g['after'] - g['before'] - 1)
                active.add(key)
                stats[key].update(denominator=1, numerator=int(g['outcome'] == 'correct'))
            for a in record['actions']:
                if a['evaluable'] and a['accepted']:
                    f = a['features']
                    key = 'accepted_competition_margin', bisect_left(bins['margin_edges'][tracker], min(f['row_margin'], f['column_margin']))
                    active.add(key)
                    stats[key].update(denominator=1, numerator=int(not a['correct']))
            for key in active:
                stats[key]['videos_with_observations'] += 1
        for kind, edges in [('natural_gap_length', bins['gap_edges']),
                            ('accepted_competition_margin', bins['margin_edges'][tracker])]:
            for i in range(len(edges)+1):
                count = stats[kind, i]
                rows.append({'tracker':tracker, 'policy':directory.name, 'trace_dir':str(directory),
                             'kind':kind, 'lower_exclusive':edges[i-1] if i else None,
                             'upper_inclusive':edges[i] if i<len(edges) else None,
                             'numerator':count['numerator'], 'denominator':count['denominator'],
                             'fraction':count['numerator']/count['denominator'] if count['denominator'] else None,
                             'videos_with_observations':count['videos_with_observations']})
        if sum(v['denominator'] for (kind,_),v in stats.items() if kind=='natural_gap_length')!=3155:
            raise ValueError(f'Unexpected gap denominator in {directory}')
    with (ROOT/f'{tracker}-strata.csv').open('w',newline='',encoding='utf-8-sig') as file:
        writer=csv.DictWriter(file,fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tracker',choices=('bytetrack','botsort-motion-only'))
    parser.add_argument('--trace-dir',type=Path,nargs='+')
    args = parser.parse_args()
    if args.tracker and args.trace_dir:
        summarize(args.tracker, args.trace_dir)
    elif not args.tracker and not args.trace_dir:
        freeze_bins()
    else:
        parser.error('Supply both --tracker and --trace-dir for test summaries')
