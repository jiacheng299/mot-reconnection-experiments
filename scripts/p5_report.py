"""Audit the P5 evidence and produce tables and the E5 accuracy/latency figure."""

import argparse
from collections import Counter, defaultdict
import csv
from hashlib import sha256
from importlib import metadata
import json
from pathlib import Path
import platform
from statistics import mean, stdev

from scripts.detect_mot17_dev import file_sha256
from scripts.audit_p4_online import prefix_invariance
from scripts.p5_dance_data import official_names, sequences, verify_cache
from scripts.p5_dance_eval import cached_detections, set_gate
from scripts.p5_mot_detector import TRAIN as MOT_TRAIN, verify_cache as verify_mot
from scripts.p5_train import EPOCHS, load_ranker


KEY_METRICS = ('HOTA', 'DetA', 'AssA', 'IDF1', 'IDSW')
BUDGETS = (1, 4, 8, 16, 32)


def read_json(path):
    return json.loads(path.read_bytes())


def require(condition, text):
    if not condition:
        raise ValueError(text)


def cached_inputs(cache):
    result = {}
    for split in ('train', 'val'):
        records = {}
        for folder, info in sequences(split):
            name = folder.name
            records[name] = verify_cache(cache / split / f'{name}.txt', split, name, info.getint('seqLength'))
        result[split] = records
    require(result['train'].keys().isdisjoint(result['val'].keys()), 'DanceTrack splits overlap')
    return result


def audit_dance_result(row, expected, checkpoint=None):
    folder = Path(row['run'])
    manifest = read_json(folder / 'input-manifest.json')
    require(manifest['coverage'] == {name: meta['frames'] for name, meta in expected.items()}, 'Wrong evaluation coverage')
    require(manifest['detections_sha256'] == {name: meta['detections_sha256'] for name, meta in expected.items()},
            'Evaluation files did not use the shared cache')
    if checkpoint:
        require(manifest['checkpoint_sha256'] == file_sha256(checkpoint), 'Evaluation used another checkpoint')
    summary, = (folder / 'trackeval').glob('*/pedestrian_summary.txt')
    header, values = summary.read_text().splitlines()
    metrics = dict(zip(header.split(), map(float, values.split()), strict=True))
    require(metrics == row['metrics'], 'Recorded metrics differ from TrackEval')
    method = manifest['method']
    group_sizes = Counter()
    for sequence in expected:
        path = folder / 'trackers' / method / 'data' / f'{sequence}.txt'
        keys = [tuple(line.split(',')[:2]) for line in path.read_text().splitlines()]
        require(len(keys) == len(set(keys)), f'Repeated output ID in {path.name}')
        if method != 'ByteTrack':
            trace = read_json(folder / 'associations' / f'{sequence}.json')
            require(trace['frames'] == expected[sequence]['frames'] and trace['duplicate_detection_assignments'] == 0,
                    f'Invalid association log {sequence}')
            group_sizes.update(trace['group_sizes'])
    return group_sizes


def audit_models(official, rows, caches, collected):
    update_counts = set()
    for row in rows:
        chosen = Path(row['checkpoint'])
        key = '-'.join([row['variant'], row['objective'], f"{row['gamma']:.2f}", f"s{row['seed']}"])
        candidates = read_json(official / 'selections' / f'{key}.json')
        require(len(candidates['checkpoints']) == EPOCHS and candidates['winner'] == row, 'Checkpoint selection mismatch')
        rank = lambda item: (-item['metrics']['HOTA'], item['association_ms'], item['epoch'])
        require(min(candidates['checkpoints'], key=rank) == row, 'Wrong checkpoint selection order')
        for entry in candidates['checkpoints']:
            path = Path(entry['checkpoint'])
            audit_dance_result(entry, caches['val'], path)
            update_counts.add((entry['epoch'], read_json(path)['history'][-1]['steps']))
        checkpoint = read_json(chosen)
        require(checkpoint['dataset'] == 'DanceTrack' and checkpoint['split'] == 'train'
                and set(checkpoint['replay']) == set(collected), 'Checkpoint did not use full official training')
        for name, meta in checkpoint['replay'].items():
            require(meta['source_sha256'] == collected[name]['sha256'], 'Model trained on another replay')
    require(len(update_counts) == EPOCHS, 'Model variants had different update budgets')


def grouped(rows, key):
    groups = defaultdict(list)
    for row in rows:
        groups[key(row)].append(row)
    return groups


def aggregate(rows):
    return {key: {'mean': mean(row['metrics'][key] for row in rows),
                  'sd': stdev(row['metrics'][key] for row in rows)} for key in KEY_METRICS}


def write_csv(path, rows):
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open('w', newline='', encoding='utf-8-sig') as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def table(label, groups):
    lines = [f'## {label}', '',
             '| 方法 | HOTA | DetA | AssA | IDF1 | IDSW |',
             '|---|---:|---:|---:|---:|---:|']
    for name, metrics in groups.items():
        values = [f'{metrics[key]["mean"]:.3f} ± {metrics[key]["sd"]:.3f}' if isinstance(metrics[key], dict)
                  else f'{metrics[key]:.3f}' for key in KEY_METRICS]
        lines.append('| ' + ' | '.join([name, *values]) + ' |')
    return '\n'.join(lines) + '\n'


def ablation_decision(e23, full):
    full_hota = mean(row['metrics']['HOTA'] for row in full)
    full_association = mean(row['metrics']['AssA'] for row in full)
    variants = grouped(e23, lambda row: row['variant'])
    decisions = {}
    eligible = []
    for name in ('no-neighborhood', 'sum-mix', 'no-pair'):
        below = {row['seed']: row['metrics'] for row in variants[name]}
        deltas = [row['metrics']['HOTA'] - below[row['seed']]['HOTA'] for row in full]
        without_association = mean(row['metrics']['AssA'] for row in variants[name])
        decisions[name] = {'full_minus_ablation_hota': mean(deltas),
                           'seeds_with_full_hota_gain': sum(delta > 0 for delta in deltas),
                           'full_minus_ablation_assa': full_association - without_association,
                           'evidence_to_retain': mean(deltas) >= 0.25 and sum(delta > 0 for delta in deltas) >= 2
                           and full_association >= without_association}
        if mean(deltas) <= 0 and without_association >= full_association:
            eligible.append((mean(deltas), -without_association, name))
    return {'full_mean_hota': full_hota, 'full_mean_assa': full_association, 'modules': decisions,
            'simplification_to_check': min(eligible)[2] if eligible else None}


def choose_structure(root, decisions, mot, gamma):
    candidate = decisions['simplification_to_check']
    if candidate is None:
        return {'deployment': 'C3', 'dance_candidate': None, 'mot_confirmation': None}, mot
    reduced = read_json(root / 'p5-mot-ablation' / 'mot17-e2.json')
    require(len(reduced) == 3 and all(row['variant'] == candidate and row['gamma'] == gamma for row in reduced)
            and {row['seed'] for row in reduced} == {0, 1, 2}, 'MOT17 simplification does not confirm the DanceTrack candidate')
    full = {row['seed']: row['winner']['metrics'] for row in mot if row['variant'] == 'C3'}
    differences = {key: mean(row['winner']['metrics'][key] - full[row['seed']][key] for row in reduced)
                   for key in ('HOTA', 'AssA')}
    deploy = all(delta >= 0 for delta in differences.values())
    return {'deployment': candidate if deploy else 'C3', 'dance_candidate': candidate,
            'mot_confirmation': differences}, mot + reduced


def temporal_effect(modes, full):
    anchors = {row['seed']: row['metrics'] for row in full}
    controls = grouped([row for row in modes if row['gamma'] == 0], lambda row: row['objective'])
    rows = {}
    for name, variants in controls.items():
        hota = [anchors[row['seed']]['HOTA'] - row['metrics']['HOTA'] for row in variants]
        assa = [anchors[row['seed']]['AssA'] - row['metrics']['AssA'] for row in variants]
        rows[name] = {'hota_delta': hota, 'assa_delta': assa, 'mean_hota_delta': mean(hota),
                      'mean_assa_delta': mean(assa),
                      'criterion_met': mean(hota) >= 0.25 and sum(value > 0 for value in hota) >= 2 and mean(assa) >= 0}
    return {'controls': rows, 'passes_both': all(row['criterion_met'] for row in rows.values())}


def read_metrics(path):
    names, values = path.read_text().splitlines()
    return dict(zip(names.split(), map(float, values.split()), strict=True))


def e5_figure(search, latency, out):
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'],
                         'pdf.fonttype': 42, 'svg.fonttype': 'none', 'font.size': 8,
                         'axes.spines.right': False, 'axes.spines.top': False,
                         'axes.linewidth': .8, 'legend.frameon': False})
    groups = grouped(search, lambda row: 'manual-pair' if row['method'] == 'hand'
                     else 'geometry' if row['method'] == 'C1' else 'learned')
    figure, axes = plt.subplots(1, 3, figsize=(7.15, 2.55), layout='constrained', gridspec_kw={'width_ratios': [1.3, 1, 1]})
    colors = {'geometry': '#74767A', 'manual-pair': '#A96539', 'learned': '#216B8F'}
    labels = {'geometry': 'Geometry', 'manual-pair': 'Manual pair', 'learned': 'Learned joint'}
    symbols = {'geometry': 'o', 'manual-pair': 's', 'learned': 'D'}
    for label in labels:
        by_k = {row['candidate_limit']: row['metrics']['HOTA'] for row in groups[label]}
        color = colors[label]
        axes[0].plot(BUDGETS, [by_k[k] for k in BUDGETS], '-',
                     marker=symbols[label], color=color, lw=1.3, ms=4, label=labels[label])
        for axis, field in zip(axes[1:], ('mean_ms', 'p95_ms'), strict=True):
            repetitions = {k: [r[field] for r in latency if r['scope'] == 'association' and r['scorer'] == label
                               and r['budget'] == k] for k in BUDGETS}
            averages = [mean(repetitions[k]) for k in BUDGETS]
            axis.errorbar(BUDGETS, averages,
                          yerr=[[value - min(repetitions[k]) for k, value in zip(BUDGETS, averages)],
                                [max(repetitions[k]) - value for k, value in zip(BUDGETS, averages)]],
                          marker=symbols[label], color=color, lw=1.2, ms=4, capsize=2)
    axes[0].legend(loc='best', fontsize=7)
    for axis, tag, ylabel in zip(axes, 'abc', ('HOTA', 'Mean association time (ms)', 'P95 association time (ms)'), strict=True):
        axis.set_xscale('log', base=2)
        axis.set_xticks(BUDGETS, labels=BUDGETS)
        axis.set_xlabel('Candidate limit K')
        axis.set_ylabel(ylabel)
        axis.set_title(tag, loc='left', fontweight='bold')
        axis.grid(axis='y', color='.90', linewidth=.45)
    figure.savefig(out / 'E5-search-budget.pdf', bbox_inches='tight')
    figure.savefig(out / 'E5-search-budget.svg', bbox_inches='tight')
    figure.savefig(out / 'E5-search-budget.png', dpi=600, bbox_inches='tight')
    plt.close(figure)


def report(root, output):
    official = root / 'p5-dance-formal'
    base = read_json(official / 'gate-selection.json')
    main = read_json(official / 'tables-e2-e4.json')
    sweep = read_json(official / 'table-e5-search.json')
    timing = read_json(root / 'p5-dance-latency.json')
    local = read_json(root / 'p5-dance-diagnostics' / 'scene-diagnostics.json')
    mot = read_json(root / 'p5-mot-formal' / 'mot17-e2.json')
    mot_diagnostics = read_json(root / 'p5-mot-diagnostics' / 'scene-diagnostics.json')
    caches = cached_inputs(root / 'p5-dance-detections')
    replay = {name: read_json(official / 'replay' / f'{name}.json') for name in official_names('train')}
    for name, meta in replay.items():
        require(meta['cache_sha256'] == caches['train'][name]['detections_sha256'] and
                meta['sha256'] == file_sha256(official / 'replay' / f'{name}.jsonl.gz'), 'Replay hash changed')
    expected_methods = {(variant, seed) for variant in ('C2', 'C2-wide', 'no-neighborhood', 'sum-mix', 'no-pair')
                        for seed in (0, 1, 2)}
    require({(r['variant'], r['seed']) for r in main['e23']} == expected_methods, 'E2/E3 incomplete')
    expected_e4 = {(seed, method, gamma) for seed in (0, 1, 2)
                   for method, gamma in (('supervised', 0), ('td', 0), ('td', 0.9), ('td', 0.99))}
    require({(row['seed'], row['objective'], row['gamma']) for row in main['e4']} == expected_e4, 'E4 incomplete')
    audit_models(official, main['e4'] + main['e23'], caches, replay)
    for row in base['c1_search'] + [base['c0']]:
        audit_dance_result(row, caches['val'])
    e5_keys = {(k, scorer) for k in BUDGETS for scorer in ('geometry', 'manual-pair', 'learned')}
    measured = {(r['budget'], r['scorer'], r['scope'], r['repeat']) for r in timing['results']}
    require(measured == {(k, label, scope, rep) for k, label in e5_keys
                         for scope in ('association', 'end_to_end') for rep in range(3)}, 'Timing grid is incomplete')
    require(len(sweep['closed_loop']) == len(e5_keys), 'End-to-end accuracy sweep is incomplete')
    require(len({row['metrics']['HOTA'] for row in sweep['closed_loop'] if row['candidate_limit'] == 1}) == 1,
            'K=1 should expose the identical geometry candidate to every scorer')
    for row in sweep['closed_loop']:
        audit_dance_result(row, caches['val'])
    require(set(local['inputs']) == set(caches['val']) and bool(mot_diagnostics['rows']), 'E6 diagnostics incomplete')
    require(local['inputs'] == {name: record['detections_sha256'] for name, record in caches['val'].items()},
            'Wrong inputs to frozen-state audit')
    for source, trace in ((root / 'p5-dance-diagnostics', local), (root / 'p5-mot-diagnostics', mot_diagnostics)):
        require(set(trace['reference_trace_sha256']) == set(trace['inputs']), 'Missing reference-state archives')
        for name, sha in trace['reference_trace_sha256'].items():
            require(file_sha256(source / f'{name}.jsonl.gz') == sha, 'Reference-state archive digest mismatch')
    require({(row['variant'], row['seed']) for row in mot} ==
            {(variant, seed) for variant in ('C2', 'C2-wide', 'C3') for seed in (0, 1, 2)}, 'MOT17 E2 missing')
    mot_sequence = 'MOT17-10-FRCNN'
    mot_cache = verify_mot(root / 'p5-mot-detections' / 'val' / f'{mot_sequence}.txt',
                           'val', mot_sequence, 654)
    mot_baselines = {}
    for run, method in (('p5-mot-c0', 'ByteTrack'), ('p5-mot-c1', 'C1')):
        baseline_dir = root / run
        require(read_json(baseline_dir / 'input-manifest.json')['detection_cache_sha256'] == mot_cache['detections_sha256'],
                'MOT17 baseline cache mismatch')
        mot_baselines[method] = read_metrics(baseline_dir / 'trackeval' / method / 'pedestrian_summary.txt')
    for row in mot:
        checkpoint = read_json(Path(row['winner']['checkpoint']))
        require(set(checkpoint['replay']) == set(MOT_TRAIN), 'MOT17 training membership wrong')
        require(len(row['choices']) == EPOCHS, 'Missing MOT17 checkpoints')
        for entry in row['choices']:
            path = Path(entry['summary'])
            record = read_json(path.parents[2] / 'input-manifest.json')
            require(read_metrics(path) == entry['metrics'] and record['model_sha256'] == file_sha256(Path(entry['checkpoint']))
                    and record['detection_cache_sha256'] == mot_cache['detections_sha256'], 'MOT17 evaluation input mismatch')
    require(mot_diagnostics['inputs'] == {mot_sequence: mot_cache['detections_sha256']},
            'MOT17 visibility diagnostic used different detections')
    choice = main['selection']
    full = [row for row in main['e4'] if row['objective'] == 'td' and row['gamma'] == choice['gamma']]
    set_gate(base['gate_selection']['gate'])
    first = next(iter(caches['val']))
    source = root / 'p5-dance-detections' / 'val' / f'{first}.txt'
    _, detections = cached_detections(source, 'val', first, caches['val'][first]['frames'])
    causal = []
    for row in full:
        ranker, method = load_ranker(Path(row['checkpoint']))
        causal.append({'method': method, 'sequence': first,
                       **prefix_invariance(detections, ranker, method, sequence=first)})
    breakdown = [*main['e23'], *full]
    distribution = {row['method']: audit_dance_result(row, caches['val']) for row in breakdown}
    decisions = ablation_decision(main['e23'], full)
    structure, mot_all = choose_structure(root, decisions, mot, choice['gamma'])
    for row in mot_all[len(mot):]:
        entry = row['winner']
        path = Path(entry['summary'])
        model = Path(entry['checkpoint'])
        manifest = read_json(path.parents[2] / 'input-manifest.json')
        require(read_metrics(path) == entry['metrics'] and manifest['model_sha256'] == file_sha256(model)
                and manifest['detection_cache_sha256'] == mot_cache['detections_sha256']
                and set(read_json(model)['replay']) == set(MOT_TRAIN), 'MOT17 simplification input mismatch')
    chosen_dance = [row for row in breakdown if row['variant'] == structure['deployment']]
    comparisons = {}
    full_by_seed = {row['seed']: row['metrics'] for row in full}
    for name in ('C2', 'C2-wide'):
        comparator = grouped(breakdown, lambda row: row['variant'])[name]
        comparisons[name] = {key: {
            'paired_deltas': [full_by_seed[row['seed']][key] - row['metrics'][key] for row in comparator],
            'mean_delta': mean(full_by_seed[row['seed']][key] - row['metrics'][key] for row in comparator),
        } for key in ('HOTA', 'AssA', 'IDF1')}
    learning_logs = []
    for row in main['e4'] + main['e23']:
        record = read_json(Path(row['checkpoint']).with_name(f'epoch-{EPOCHS}.json'))
        for epoch in record['history']:
            learning_logs.append({'variant': row['variant'], 'objective': row['objective'], 'gamma': row['gamma'],
                                  'seed': row['seed'], 'parameter_count': record['parameter_count'], **epoch})
    output.mkdir(parents=True, exist_ok=True)
    e5_figure(sweep['closed_loop'], timing['results'], output)
    manifest = {'training': {'DanceTrack': sorted(caches['train']), 'MOT17': list(MOT_TRAIN)},
                'selection_split': {'DanceTrack': sorted(caches['val']), 'MOT17': ['MOT17-10-FRCNN']},
                'development_gates': {'DanceTrack': base['gate_selection']['gate'], 'MOT17': 'reference'},
                'gamma': choice['gamma'], 'candidate_limit': 16,
                'architecture': structure,
                'appearance_model': None, 'online_only': True, 'training_steps_matched': True,
                'environment': {'python': platform.python_version(),
                                **{name: metadata.version(name) for name in ('trackeval', 'torch', 'numpy', 'scipy')}},
                'evaluation_source_record_sha256': sha256(metadata.distribution('trackeval').read_text('RECORD').encode()).hexdigest(),
                'scripts_sha256': {path.name: file_sha256(path) for path in sorted(Path(__file__).parent.glob('*.py'))},
                'detector_sha256': {'DanceTrack': next(iter(caches['train'].values()))['weights_sha256'],
                                    'MOT17': read_json(root / 'p5-mot-detector' / 'd-mot.json')['weights_sha256']},
                'dance_checkpoints': {row['seed']: file_sha256(Path(row['checkpoint'])) for row in chosen_dance},
                'mot_checkpoints': {row['seed']: file_sha256(Path(row['winner']['checkpoint']))
                                    for row in mot_all if row['variant'] == structure['deployment']},
                'ablation_evidence': decisions,
                'long_term_evidence': temporal_effect(main['e4'], full)}
    summary = {'dataset_frames': {split: sum(row['frames'] for row in records.values()) for split, records in caches.items()},
               'replay_rows': sum(meta['rows'] for meta in replay.values()), 'frozen': manifest,
               'online_prefix_audit': causal,
               'E2_E3': {label: aggregate(rows) for label, rows in grouped(breakdown, lambda r: r['variant']).items()},
               'E4': {label: aggregate(rows) for label, rows in grouped(main['e4'],
                                                                      lambda r: f"{r['objective']}-{r['gamma']:.2f}").items()},
               'MOT17_E2': {label: aggregate([row['winner'] for row in rows]) for label, rows in
                            grouped(mot_all, lambda r: r['variant']).items()},
               'group_distributions': distribution, 'C3_paired_deltas': comparisons}
    write_csv(output / 'E2-E4-DanceTrack-seeds.csv',
              [{'method': row['variant'], 'objective': row['objective'], 'gamma': row['gamma'], 'seed': row['seed'],
                'epoch': row['epoch'], **row['metrics'], 'run': row['run']}
               for row in main['e4'] + main['e23']])
    write_csv(output / 'E5-timing-repeats.csv', timing['results'])
    write_csv(output / 'E3-E4-learning-history.csv', learning_logs)
    write_csv(output / 'E6-DanceTrack-fixed-states.csv', local['rows'])
    write_csv(output / 'E6-MOT17-visibility.csv', mot_diagnostics['rows'])
    sections = [
        '# P5 · 机制结果表',
        '各学习方法的条目为三个种子的均值 ± 样本标准差。HOTA、DetA、AssA 和 IDF1 按百分制列出，IDSW 单位为次。',
        table('E2 与 E3 · DanceTrack val', {'C0 ByteTrack': base['c0']['metrics'],
                                          'C1 固定 IoU': base['gate_selection']['metrics'], **summary['E2_E3']}),
        table('E4 · DanceTrack val 的四种训练目标', summary['E4']),
        table('E2 · MOT17-10 独立复查', {**mot_baselines, **summary['MOT17_E2']}),
        '## E5 · 候选预算',
        '| 方法 | K | HOTA | 平均关联延迟 ms | P95 关联延迟 ms | P99 关联延迟 ms | 每组实际候选数 | 端到端 FPS |',
        '|---|---:|---:|---:|---:|---:|---:|---:|',
    ]
    for row in sweep['closed_loop']:
        scorer = 'geometry' if row['method'] == 'C1' else 'manual-pair' if row['method'] == 'hand' else 'learned'
        k = row['candidate_limit']
        records = [r for r in timing['results'] if r['budget'] == k and r['scorer'] == scorer]
        pure = [r for r in records if r['scope'] == 'association']
        live = [r for r in records if r['scope'] == 'end_to_end']
        columns = [mean(r[key] for r in pure) for key in ('mean_ms', 'p95_ms', 'p99_ms', 'candidates_per_group')]
        columns.append(mean(r['effective_fps'] for r in live))
        sections.append('| ' + ' | '.join([scorer, str(k), f"{row['metrics']['HOTA']:.3f}",
                                           *(f'{n:.3f}' for n in columns)]) + ' |')
    sections.extend(['', '延迟表为三次重复的统计量平均。端到端包含输入图像解码、检测推理、后处理与关联，计时不含轨迹文件写入。',
                     '图中延迟误差棒给出三次重复的最小值和最大值。候选诊断表以冻结 ByteTrack 状态为参考。',
                     '续接正确率的分母是已有身份参照且当前帧存在合法正确检测框的轨迹。无正确框的轨迹单列计数。'])
    (output / 'P5-实测表.md').write_text('\n'.join(sections) + '\n', encoding='utf-8')
    (output / 'P5-evidence.json').write_text(json.dumps(summary, ensure_ascii=False, allow_nan=False, indent=2) + '\n',
                                           encoding='utf-8')
    (output / 'P6-frozen-config.json').write_text(json.dumps(manifest, ensure_ascii=False, allow_nan=False, indent=2) + '\n',
                                                 encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('runs', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    report(args.runs, args.output)
