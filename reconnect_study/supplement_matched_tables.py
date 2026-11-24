"""Combine the six frozen PersonPath22 sensitivity reports into publication tables."""

import argparse
import csv
import json
from pathlib import Path


def write_table(destination, rows):
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with destination.open('w', newline='', encoding='utf-8-sig') as file:
        writer = csv.DictWriter(file, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def collect(report):
    header = {'tracker': report['tracker'], 'cap': report['acceptance_cap']}
    policies = []
    for policy, value in report['policies'].items():
        info = {key: value[key] for key in (
            'source_count', 'accepted', 'wrong', 'correct_accepted', 'unverified_accepted',
            'ignored_accepted', 'gate_actions', 'active_sources', 'gaps', 'correct',
            'wrong_per_acceptance', 'natural_gap_recovery_rate')}
        percentiles = value['accepted_per_source']['quartiles']
        policies.append({**header, 'policy_id': policy, **info,
                         'evaluated_accepts_per_source': value['accepted_per_source']['mean'],
                         'accepts_q25': percentiles[0], 'accepts_median': percentiles[1],
                         'accepts_q75': percentiles[2],
                         'sources_reaching_cap': value['sources_reaching_cap']
                         if policy != 'unconstrained_tracker' else None, **value['overall']})
    comparisons = []
    for point in report['matched_sets']:
        common = {**header, 'point': point['id'], 'target_accepts_per_source': point['target'],
                  'tolerance': point['tolerance'], 'calibration_matched': point['matched']}
        if not point['matched']:
            comparisons.append(common)
            continue
        structured = point['policies']['structured']
        for control, comparison in point['comparisons'].items():
            control_policy = point['policies'][control]
            s, c = [report['policies'][p] for p in (structured, control_policy)]
            gap = comparison['effects']['gap_recovery']
            error = comparison['effects']['wrong_per_acceptance']
            comparisons.append({**common, 'structured': structured, 'control': control_policy,
                                'structured_threshold': point['nearest']['structured']['threshold'],
                                'control_threshold': point['nearest'][control]['threshold'],
                                'calibration_spread': point['spread'],
                                'test_spread': point['test_max_pairwise_spread'],
                                'test_spread_within_tolerance': point['test_max_pairwise_spread'] <= point['tolerance'],
                                'test_max_target_deviation': max(point['test_absolute_target_deviation'].values()),
                                'test_all_conditions_met': point['test_within_frozen_tolerance'],
                                'accepted_structured': s['accepted'], 'accepted_control': c['accepted'],
                                'accepted_delta': s['accepted'] - c['accepted'],
                                'wrong_structured': s['wrong'], 'wrong_control': c['wrong'],
                                'gaps': s['gaps'], 'recovered_structured': s['correct'],
                                'recovered_control': c['correct'],
                                'failure_delta_pp': error['delta_percentage_points'],
                                'failure_ci_low': error['empirical_95_percent_interval'][0],
                                'failure_ci_high': error['empirical_95_percent_interval'][1],
                                'recovery_delta_pp': gap['delta_percentage_points'],
                                'recovery_ci_low': gap['empirical_95_percent_interval'][0],
                                'recovery_ci_high': gap['empirical_95_percent_interval'][1],
                                'HOTA_structured': s['overall']['HOTA'], 'HOTA_control': c['overall']['HOTA'],
                                'HOTA_delta_pp': s['overall']['HOTA'] - c['overall']['HOTA']})
    return policies, comparisons


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('reports', type=Path, nargs=6)
    parser.add_argument('--output', type=Path, default=Path('runs/reconnect_supplement/round2'))
    args = parser.parse_args()
    summaries, contrasts, observed = [], [], set()
    for file in args.reports:
        report = json.loads(file.read_bytes())
        key = report['tracker'], report['acceptance_cap']
        if key in observed or report['split'] != 'test':
            raise ValueError(f'Unexpected report {file}')
        observed.add(key)
        policies, comparisons = collect(report)
        summaries.extend(policies)
        contrasts.extend(comparisons)
    if observed != {(t, b) for t in ('bytetrack', 'botsort-motion-only') for b in (6, 12, 24)}:
        raise ValueError(f'Unexpected study reports: {observed}')
    write_table(args.output / 'test-policy-summary.csv', summaries)
    write_table(args.output / 'test-paired-contrasts.csv', contrasts)


if __name__ == '__main__':
    main()
