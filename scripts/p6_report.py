"""Verify final tracking runs and write the E1/E7/E8 handoff."""

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
from statistics import mean, stdev
from zipfile import ZipFile

from scripts.detect_mot17_dev import ROOT, file_sha256
from scripts.p6_data import FROZEN, final_names, verify_cache
from scripts.p6_eval import ALL_METHODS, BASELINES, CGH_METHODS, resolve_checkpoints, track_count, verify_freeze
from scripts.p5_dance_data import sequences as development_sequences
from scripts.p5_report import audit_dance_result, read_metrics


METRICS = ('HOTA', 'DetA', 'AssA', 'IDF1', 'MOTA', 'IDSW', 'CLR_FP', 'CLR_FN')
MISSING = {
    'CAMELTrack': {'reason': '发布检查点缺可核验的训练序列清单。本协议的开发集公平性未通过。',
                   'source': 'https://github.com/TrackingLaboratory/CAMELTrack#-reproducing-the-paper-results-two-stage-training'},
    'HyperMOT': {'reason': '作者公开仓库未提供关联所需训练检查点。',
                 'source': 'https://github.com/SkyeSong38/HyperMOT/releases'},
}


def verify_domain(runs, domain, config):
    root = runs / f'p6-{domain}-final'
    expected = set(final_names(domain))
    thresholds = resolve_checkpoints(domain, config)
    out = {}
    for method in ALL_METHODS:
        p = root / 'manifests' / f'{method}.json'
        manifest = json.loads(p.read_bytes())
        if manifest['domain'] != domain or manifest['development'] or set(manifest['coverage']) != expected:
            raise ValueError(f'{domain}/{method}: invalid final coverage')
        detector_counts = 0
        groups = Counter()
        fallback = groups_total = frames_total = 0
        for name, count in manifest['coverage'].items():
            cache_path = runs / 'p6-detections' / domain / f'{name}.txt'
            cache = verify_cache(cache_path, domain, name, count)
            record = manifest['runs'][name]
            path = root / 'trackers' / method / 'data' / f'{name}.txt'
            if (record['detections_sha256'] != cache['detections_sha256'] or record['detector_sha256'] != cache['weights_sha256']
                    or record['p5_freeze_sha256'] != file_sha256(FROZEN) or not record['online_only']
                    or record['output_sha256'] != file_sha256(path) or record['output_rows'] != track_count(path, count)):
                raise ValueError(f'{domain}/{method}/{name}: failed provenance or output checks')
            frames_total += count
            detector_counts += cache['count']
            if method in CGH_METHODS:
                if record['checkpoint_sha256'] != file_sha256(thresholds[method[-1]]):
                    raise ValueError('C3 used another seed or domain checkpoint')
                summary = record['configuration']['summary']
                if summary['frames'] != count or summary['duplicate_detection_assignments'] != 0:
                    raise ValueError('Invalid candidate/track sequence')
                association = root / 'associations' / method / f'{name}.jsonl'
                if file_sha256(association) != record['configuration']['audit_sha256']:
                    raise ValueError('Association output changed after inference')
                counts = {int(size): n for size, n in summary['group_sizes'].items()}
                groups.update(counts)
                groups_total += sum(counts.values())
                fallback += summary['fallback_groups']
        evidence = {'frames': frames_total, 'detections': detector_counts, 'sequence_count': len(expected),
                    'manifest': str(p.resolve())}
        if method in CGH_METHODS:
            evidence['competition'] = {'size_histogram': dict(sorted(groups.items())), 'groups': groups_total,
                                       'large_groups_ge9': sum(n for size, n in groups.items() if size >= 9),
                                       'fallback_groups': fallback,
                                       'fallback_rate': fallback / groups_total if groups_total else None}
        if domain == 'DanceTrack':
            bundle = Path(manifest['submission']['file'])
            if file_sha256(bundle) != manifest['submission']['sha256']:
                raise ValueError('Submission archive changed')
            with ZipFile(bundle) as archive:
                if archive.testzip() is not None or set(archive.namelist()) != {f'tracker/{x}.txt' for x in expected}:
                    raise ValueError('Submission archive has unexpected entries')
                for name in expected:
                    path = root / 'trackers' / method / 'data' / f'{name}.txt'
                    if archive.read(f'tracker/{name}.txt') != path.read_bytes():
                        raise ValueError('Submission archive does not match the checked tracks')
            evidence.update({'submission_file': str(bundle), 'submission_sha256': file_sha256(bundle),
                             'official_result': 'deferred by user'})
        else:
            result = root / 'trackeval' / method / 'pedestrian_summary.txt'
            expected_tracks = root / 'trackeval' / method / 'pedestrian_detailed.csv'
            with expected_tracks.open(newline='') as file:
                detailed = list(csv.reader(file))
            names = {row[0] for row in detailed[1:]}
            if names != expected | {'COMBINED'}:
                raise ValueError(f'Unexpected TrackEval evidence for {domain}/{method}: {sorted(names)}')
            evidence['metrics'] = {key: value for key, value in read_metrics(result).items() if key in METRICS}
            evidence['trackeval_sha256'] = file_sha256(result)
        out[method] = evidence
    return out


def audit_cases(runs):
    root = runs / 'p6-E8-cases'
    file = root / 'index.json'
    cases = json.loads(file.read_bytes())
    counts = Counter(item['outcome'] for item in cases['cases'])
    if (cases['peer'] != 'TrackTrack' and 'TrackTrack' not in cases['development_scores']) or any(n > 3 for n in counts.values()):
        raise ValueError('E8 comparator or case counts changed')
    strongest = min(cases['development_scores'],
                    key=lambda method: (-cases['development_scores'][method]['HOTA'], method))
    with (root / 'all-candidates.jsonl').open(encoding='utf-8') as source:
        n = sum(1 for _ in source)
    if cases['peer'] != strongest or n != cases['candidate_frames']:
        raise ValueError('E8 case index did not match the frozen selection rules')
    for row in cases['cases']:
        if not (root / row['snapshot']).read_bytes().startswith(b'%PDF') or (root / row['video']).stat().st_size == 0:
            raise ValueError('Missing selected case rendering')
    if counts['improvement'] and counts['failure'] and not (root / 'Figure-2.pdf').is_file():
        raise ValueError('Missing manuscript case figure')
    return {'candidate_count': n, 'picked': dict(counts), 'peer': strongest, 'index': str(file.resolve())}


def verify_dance_development(runs, config):
    local = runs / 'p6-DanceTrack-dev'
    p5 = runs / 'p5-dance-formal'
    caches = {}
    for folder, info in development_sequences('val'):
        name = folder.name
        path = runs / 'p5-dance-detections' / 'val' / f'{name}.txt'
        caches[name] = verify_cache(path, 'DanceTrack', name, info.getint('seqLength'), True)
    expected = {name: cache['frames'] for name, cache in caches.items()}
    c0 = json.loads((p5 / 'gate-selection.json').read_bytes())['c0']
    audit_dance_result(c0, caches)
    out = {'ByteTrack': {'metrics': c0['metrics']}}
    for method in BASELINES[1:]:
        manifest = json.loads((local / 'manifests' / f'{method}.json').read_bytes())
        if not manifest['development'] or manifest['coverage'] != expected:
            raise ValueError(f'Incomplete shared DanceTrack development results for {method}')
        for name, record in manifest['runs'].items():
            output = local / 'trackers' / method / 'data' / f'{name}.txt'
            if (record['detector_sha256'] != caches[name]['weights_sha256'] or
                    record['detections_sha256'] != caches[name]['detections_sha256'] or
                    file_sha256(output) != record['output_sha256']):
                raise ValueError(f'DanceTrack development result changed: {method}/{name}')
        values = read_metrics(local / 'trackeval' / method / 'pedestrian_summary.txt')
        out[method] = {'metrics': values}
    c3 = json.loads((p5 / 'tables-e2-e4.json').read_bytes())['e4']
    chosen = [row for row in c3 if row['variant'] == 'C3' and row['objective'] == 'td' and row['gamma'] == config['gamma']]
    if sorted(row['seed'] for row in chosen) != [0, 1, 2]:
        raise ValueError('Missing the chosen C3 development results')
    for row in chosen:
        checkpoint = Path(row['checkpoint'])
        if file_sha256(checkpoint) != config['dance_checkpoints'][str(row['seed'])]:
            raise ValueError('Development result differs from P5 freeze')
        audit_dance_result(row, caches, checkpoint)
        out[f'CGHTrack-s{row["seed"]}'] = {'metrics': row['metrics']}
    return out


def local_table(label, rows):
    lines = [f'## {label}', '', '| 方法 | ' + ' | '.join(METRICS) + ' |',
             '|---|' + '---:|' * len(METRICS)]
    for method in BASELINES + ('CGHTrack', 'CAMELTrack', 'HyperMOT'):
        if method == 'CGHTrack':
            stats = [[rows[seed]['metrics'][metric] for seed in CGH_METHODS] for metric in METRICS]
            values = [f'{mean(s):.3f} ± {stdev(s):.3f}' for s in stats]
        elif method in MISSING:
            values = ['缺项'] * len(METRICS)
        else:
            values = [f'{rows[method]["metrics"][metric]:.3f}' for metric in METRICS]
        lines.append('| ' + ' | '.join([method, *values]) + ' |')
    return '\n'.join(lines) + '\n'


def generate(runs, out, local_only=False):
    config = verify_freeze()
    results = {domain: verify_domain(runs, domain, config) for domain in ('MOT17', 'MOT20')}
    cases = audit_cases(runs)
    dance_dev = None
    if not local_only:
        results['DanceTrack'] = verify_domain(runs, 'DanceTrack', config)
        dance_dev = verify_dance_development(runs, config)
    out.mkdir(parents=True, exist_ok=True)
    report = ['# P6 · E1 / E7 / E8 结果核验', '',
              '分数由完整 TrackEval 聚合得到。CGHTrack 报告三个种子的均值 ± 样本标准差。'
              'HOTA 综合衡量框检测和身份关联，DetA 衡量检测准确度，AssA 衡量关联准确度。'
              'IDF1 衡量预测身份与人工身份匹配的精确率和召回率，MOTA 汇总误检、漏检和身份切换。'
              'TrackEval 将这五项按 100 倍标度显示，越高越好，MOTA 可为负。'
              'IDSW、CLR_FP、CLR_FN 分别是身份切换、误检、漏检的次数，越低越好。', '',
              local_table('MOT17 local held-out · 两条视频', results['MOT17']),
              local_table('MOT20 train-transfer · 四条视频', results['MOT20']),
              '## 基线缺项记录', '']
    for method, row in MISSING.items():
        report.append(f'- {method}。{row["reason"]} [作者来源]({row["source"]})')
    report.extend([
        '', '## 主表配置与训练来源', '',
        'MOT17 和 MOT20 的每行共用 D-MOT 检测框与分数。D-MOT 由 COCO 初始化，'
        '在 MOT17-02、04、05、09 上微调检测头。DanceTrack 每行共用 D-Dance，'
        '由 COCO 初始化，在 DanceTrack 官方训练集训练。MOT20 复用 MOT17 的模型和参数。', '',
        '| 方法 | 外观模型训练来源 | 跟踪关联训练来源 | 推理与输出 |',
        '|---|---|---|---|',
        '| ByteTrack、OC-SORT | 无 | 无额外学习权重 | 在线跟踪 |',
        '| BoT-SORT、TrackTrack | 冻结 Market-1501 SBS-S50 | 无额外学习权重 | 在线跟踪与相邻帧相机补偿 |',
        '| CGHTrack | 无 | MOT 域使用 MOT17-02、04、05、09，Dance 域使用 DanceTrack 官方训练集 | 在线跟踪 |',
        '', '所有实测行均禁用离线插值与轨迹连接。TrackTrack 采用作者关联核心的单缓存适配。',
    ])
    if 'DanceTrack' in results:
        report.extend(['', local_table('DanceTrack val · 开发验证（25 条视频）', dance_dev),
                       '## DanceTrack official test', '',
                       '经用户确认暂缓官方计分。七组轨迹包均含 35 个序列文件，并通过内容与摘要核验。', '',
                       '| 方法或种子 | 覆盖 | CodaBench 分数 | 包 |', '|---|---:|---|---|'])
        for method, row in results['DanceTrack'].items():
            path = Path(row['submission_file'])
            relative = f'../p6-DanceTrack-final/submission-zips/{path.name}'
            report.append(f'| {method} | {row["sequence_count"]} | 暂缓 | [{path.name}]({relative}) |')
        report.extend(['', '登录后进入 My Submissions，选择 Test Phase 并上传一个 ZIP。'
                       '首包完成后保存提交编号与完整分数，再依次上传其余已冻结的包。'
                       '[竞赛入口](https://www.codabench.org/competitions/14885/) · '
                       '[官方提交说明](https://docs.codabench.org/v1.34/Participants/User_Participating-in-a-Competition/)'])
    report.extend(['', '## CGHTrack 各种子的实测分数', '',
                   '| 评价域 | 种子 | HOTA | AssA | IDF1 |', '|---|---:|---:|---:|---:|'])
    scored = [(domain, results[domain]) for domain in ('MOT17', 'MOT20')]
    if dance_dev is not None:
        scored.append(('DanceTrack val', dance_dev))
    for domain, rows in scored:
        for seed in CGH_METHODS:
            row = rows[seed]['metrics']
            report.append(f'| {domain} | {seed[-1]} | {row["HOTA"]:.3f} | {row["AssA"]:.3f} | {row["IDF1"]:.3f} |')
    report.extend(['', '## E7 竞争组统计', '',
                   '| 种子 | 竞争组实例 | ≥9 条轨迹 | 大组占比 | 回退组数 | 回退率 |',
                   '|---|---:|---:|---:|---:|---:|'])
    for method in CGH_METHODS:
        record = results['MOT20'][method]['competition']
        denominator = max(1, record['groups'])
        report.append(f'| {method} | {record["groups"]} | {record["large_groups_ge9"]} | '
                      f'{record["large_groups_ge9"] / denominator:.3%} | {record["fallback_groups"]} | '
                      f'{record["fallback_groups"] / denominator:.3%} |')
    report.extend(['', '## E8 案例', '',
                   f'比较了 ByteTrack、{cases["peer"]} 和 CGHTrack 种子 0。候选事件共 {cases["candidate_count"]} 条。'
                   f'入选改进 {cases["picked"].get("improvement", 0)} 例，失败 {cases["picked"].get("failure", 0)} 例。'])
    suffix = 'local' if local_only else 'handoff'
    (out / f'P6-{suffix}.md').write_text('\n'.join(report) + '\n', encoding='utf-8')
    (out / f'P6-{suffix}.json').write_text(json.dumps({
        'official_dance_status': 'deferred_user_approved', 'missing_baselines': MISSING,
        'domains': results, 'cases': cases, 'freeze_sha256': file_sha256(FROZEN),
        'dance_development': dance_dev,
    }, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps({'domains_checked': list(results), 'case_picks': cases['picked'],
                      'dance_official_scores': 'deferred_user_approved'}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--local-only', action='store_true')
    args = parser.parse_args()
    generate(ROOT / 'runs', ROOT / 'runs' / 'p6-acceptance', local_only=args.local_only)
