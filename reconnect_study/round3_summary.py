"""Rebuild the round-three policy and paired-effect tables from closed-loop per-source outputs."""

import csv
import json
from functools import lru_cache
from pathlib import Path

from reconnect_study.effect_sizes import compare
from reconnect_study.round3_ablation_eval import bound, config_path_in_checkout


STUDY = Path('runs/reconnect_supplement')
ROOT = STUDY / 'round3'


@lru_cache(maxsize=12)
def load(path):
    return json.loads(config_path_in_checkout(str(path)).read_bytes())


def write_rows(path, rows):
    with path.open('w', encoding='utf-8-sig', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=list(dict.fromkeys(k for row in rows for k in row)))
        writer.writeheader()
        writer.writerows(rows)


def summarize(spec):
    data = load(spec['file'])
    policy = spec['policy']
    rows = data['policies'][policy]
    if len(rows) != 74 or len({r['source_id'] for r in rows}) != 74 or data['split'] != 'test':
        raise ValueError(f'Unexpected test sources for {spec}')
    keys = ('accepted', 'wrong', 'correct', 'gaps', 'ignored_accepted', 'unverified_accepted')
    total = {k:sum(r[k] for r in rows) for k in keys}
    if total['gaps'] != 3155 or total['wrong'] < total['unverified_accepted']:
        raise ValueError(f'Unexpected totals for {spec}')
    if policy != 'unconstrained_tracker' and any(r['accepted']+r['ignored_accepted']>12 for r in rows):
        raise ValueError(f'Exceeded cap in {spec}')
    return {**total, 'failure_fraction': total['wrong']/total['accepted'] if total['accepted'] else None,
            'confirmed_correct': total['accepted']-total['wrong'], 'active_sources':sum(r['accepted']>0 for r in rows),
            'recovery_fraction':total['correct']/total['gaps'], **data['overall'][policy]['aggregate']}


def reference(file, detail):
    return {'file': config_path_in_checkout(str(file)).as_posix(), 'policy':detail['policy'], 'threshold':detail.get('threshold')}


def build():
    results, effects = [], []
    for key in ('byte', 'bot'):
        tracker = {'byte':'ByteTrack', 'bot':'BoT-SORT (motion)'}[key]
        archive = STUDY/'round2'/key/'B12'
        new_dir = ROOT/'ablation'/key
        bound(archive/'test.json', load(archive.parent/'test-integrity.json')['B12']['test_sha256'])
        bound(new_dir/'test.json', load(new_dir/'report.json')['test_sha256'])
        external_path = ROOT/'boost'/f'{key}-summary-input.json'
        boost = load(external_path) if external_path.exists() else {}
        for point, entry in load(new_dir/'locked.json')['work_points'].items():
            old = entry['old']
            target, tol = entry['target'], entry['tolerance']
            common = {'tracker':tracker, 'setting':point, 'target':target, 'tolerance':tol}
            choices = {label:reference(archive/'test.json', old[family]) for label, family in
                       [('12','linear'), ('23','structured'), ('native','assignment')]}
            choices['unlimited'] = reference(archive/'test.json', {'policy':'unconstrained_tracker'})
            pairs = [('23','15',entry['calibration_match']['23_minus_15']['matched']),
                     ('15','12',entry['calibration_match']['15_minus_12']['matched'])]
            if entry['new']['policy'] in load(new_dir/'locked.json')['test_policies']:
                choices['15'] = reference(new_dir/'test.json',entry['new'])
            if point in boost:
                ext = boost[point]
                pairs.append(('23','BoostTrack',ext['calibration_matched']))
                if ext['calibration_matched']:
                    choices['BoostTrack'] = reference(ext['file'],ext)
            values = {label:summarize(spec) for label,spec in choices.items()}
            for label,spec in choices.items():
                adapter = {'byte': 'ByteTrack state reordering', 'bot': 'BoT-SORT first-order covariance approximation'}
                results.append({**common,'score':label,**spec,**values[label],
                                'state_adapter': adapter[key] if label == 'BoostTrack' else ''})
            for left,right,matched in pairs:
                base = {**common,'left':left,'right':right,'calibration_matched':matched}
                if not matched:
                    effects.append(base)
                    continue
                a,b = values[left]['accepted'], values[right]['accepted']
                spread = abs(a-b)/74
                offset=max(abs(a/74-target),abs(b/74-target))
                first,second=choices[left],choices[right]
                est=compare(load(second['file'])['policies'][second['policy']],
                            load(first['file'])['policies'][first['policy']],replicas=10000,seed=362)
                row={**base,'left_policy':first['policy'],'right_policy':second['policy'],
                     'left_accepted':a,'right_accepted':b,'test_spread':spread,
                     'test_spread_within_tolerance':spread<=tol,'test_max_target_deviation':offset,
                     'test_both_at_target':offset<=tol,'test_joint_match':spread<=tol and offset<=tol,
                     'HOTA_delta_pp':values[left]['HOTA']-values[right]['HOTA']}
                for name,metric in [('failure','wrong_per_acceptance'),('recovery','gap_recovery')]:
                    effect=est['effects'][metric]
                    row[name+'_pp']=effect['delta_percentage_points']
                    row[name+'_lo'],row[name+'_hi']=effect['empirical_95_percent_interval']
                    row[name+'_valid_replicas']=effect['valid_replicas']
                effects.append(row)
    anchor={r['score']:(r['wrong'],r['accepted']) for r in results if r['tracker']=='ByteTrack' and r['setting']=='high'}
    if any(anchor[k]!=expected for k,expected in {'native':(158,390),'12':(114,393),'23':(98,381)}.items()):
        raise ValueError('Frozen ByteTrack high anchor mismatch')
    write_rows(ROOT/'policy-results.csv',results)
    write_rows(ROOT/'paired-effects.csv',effects)


if __name__ == '__main__':
    build()
