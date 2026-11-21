"""Audit identity continuations and render the preregistered P6 examples."""

import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory

import cv2
import numpy as np

from scripts.detect_mot17_dev import ROOT, file_sha256, load_image
from scripts.p4_supervision import load_pedestrian_truth, matched_people
from scripts.p5_diagnostics import visibility_rows
from scripts.p6_data import sequences, verify_cache
from scripts.run_p1 import read_detections


ORDER = ('detector', 'gate', 'search', 'value', 'output-match', 'ambiguous')
METHOD_COLOR = {'ByteTrack': '#1C79A6', 'CGHTrack-s0': '#9A412E'}
TRUTH_COLOR = '#FAE58C'


def output_boxes(path):
    records = defaultdict(list)
    with path.open(newline='') as file:
        for frame, track, x, y, width, height, *_ in csv.reader(file):
            x, y, width, height = map(float, (x, y, width, height))
            records[int(frame)].append((int(track), (x, y, x + width, y + height, 1.0)))
    return records


def match_outputs(outputs, truth):
    matched = matched_people([box for _, box in outputs], truth)
    return {person: outputs[det - 1] for det, person in matched.items()}


def iou(box, truth):
    x, y, r, b = box[:4]
    u, v, s, t = truth
    area = max(0, min(r, s) - max(x, u)) * max(0, min(b, t) - max(y, v))
    return area / max(1e-9, (r - x) * (b - y) + (s - u) * (t - v) - area)


def failure_cause(has_detection, correct, legal, searched, selected):
    if not has_detection:
        return 'detector'
    if not correct:
        return 'ambiguous'
    if not correct & legal:
        return 'gate'
    if not correct & searched:
        return 'search'
    if selected not in correct:
        return 'value'
    return 'output-match'


def baseline_from_dev():
    root = ROOT / 'runs' / 'p6-MOT17-dev'
    candidates = {}
    for method in ('OC-SORT', 'BoT-SORT', 'TrackTrack'):
        manifest = root / 'manifests' / f'{method}.json'
        file = root / 'trackeval' / method / 'pedestrian_summary.txt'
        values = file.read_text().splitlines()
        metrics = dict(zip(values[0].split(), map(float, values[1].split()), strict=True))
        if json.loads(manifest.read_bytes())['coverage'] != {'MOT17-10-FRCNN': 654}:
            raise ValueError(f'{method} missing the exact development sequence')
        candidates[method] = {'HOTA': metrics['HOTA'], 'source_sha256': file_sha256(file)}
    strongest = min(candidates, key=lambda method: (-candidates[method]['HOTA'], method))
    return strongest, candidates


def audit_candidates(final, methods):
    candidates, assets = [], {}
    for folder, info, images in sequences('MOT17'):
        sequence, length = folder.name, len(images)
        truth = load_pedestrian_truth(folder / 'gt' / 'gt.txt')
        visible = visibility_rows(folder / 'gt' / 'gt.txt')
        raw = ROOT / 'runs' / 'p6-detections' / 'MOT17' / f'{sequence}.txt'
        source = verify_cache(raw, 'MOT17', sequence, length)
        detections = read_detections(raw, length)
        tracksets = {method: output_boxes(final / 'trackers' / method / 'data' / f'{sequence}.txt')
                     for method in methods}
        identity = {method: {frame: match_outputs(rows[frame], truth[frame])
                             for frame in range(1, length + 1)} for method, rows in tracksets.items()}
        last = {}
        association = final / 'associations' / 'CGHTrack-s0' / f'{sequence}.jsonl'
        with association.open(encoding='utf-8') as file:
            for frame, line in enumerate(file, 1):
                state = json.loads(line)
                if frame != state['frame']:
                    raise ValueError(f'Unexpected association log frame at {sequence}/{frame}')
                now = {method: identity[method][frame] for method in methods}
                if frame < 5 or frame > length - 4:
                    last = now
                    continue
                eligible = sorted(set.intersection(*(set(last[method]) for method in methods))
                                  & truth[frame - 4].keys() & truth[frame].keys() & truth[frame + 4].keys())
                labelled = matched_people(detections[frame], truth[frame])
                for person in eligible:
                    before = {method: last[method][person][0] for method in methods}
                    after = {method: now[method].get(person, (None,))[0] for method in methods}
                    succeeded = {method: after[method] == before[method] for method in methods}
                    simple, other, cgh = [succeeded[method] for method in methods]
                    if (cgh and simple and other) or (not cgh and not simple and not other):
                        continue
                    track = before['CGHTrack-s0']
                    legal = {int(det) for det in state['legal_scores'].get(str(track), {})}
                    group = next((group for group in state['groups'] if track in group['tracks']), None)
                    if group:
                        idx = group['tracks'].index(track)
                        searched = {a[idx] for a in group['candidates'] if a[idx] is not None}
                    else:
                        searched = set()
                    selected = state['matches'].get(str(track))
                    correct = {det for det, identity_id in labelled.items() if identity_id == person}
                    has_box = any(box[4] >= 0.10 and iou(box, truth[frame][person]) >= 0.5 for box in detections[frame])
                    outcome = 'improvement' if cgh else 'failure'
                    visibility = visible.get(frame, {}).get(person)
                    overlap = max((iou(truth[frame][person], box) for identity_id, box in truth[frame].items()
                                   if person != identity_id), default=0)
                    context = ('occlusion' if visibility is not None and 0 <= visibility < 0.7
                               else 'overlap' if overlap >= 0.15 else 'uncrowded')
                    reason = (','.join(method for method in methods[:-1] if not succeeded[method])
                              if cgh else failure_cause(has_box, correct, legal, searched, selected))
                    candidates.append({'outcome': outcome, 'reason': reason, 'sequence': sequence, 'frame': frame,
                                       'identity': person, 'before': before, 'at_event': after,
                                       'context': context, 'visibility': visibility, 'max_pedestrian_iou': overlap,
                                       'group_size': len(group['tracks']) if group else 0,
                                       'legal_detection_ids': sorted(legal), 'candidate_detection_ids': sorted(searched),
                                       'correct_detection_ids': sorted(correct), 'selected_detection_id': selected,
                                       'detector_sha256': source['weights_sha256']})
                last = now
        assets[sequence] = (images, truth, identity, info.getint('frameRate'))
    return candidates, assets


def choose_cases(candidates):
    chosen = []
    for outcome in ('improvement', 'failure'):
        pools = [row for row in candidates if row['outcome'] == outcome and row['reason'] != 'ambiguous'
                 and row['context'] != 'uncrowded']
        keep, seen_classes, seen_videos = [], set(), set()
        while len(keep) < 3 and pools:
            row = min(pools, key=lambda item: (item['reason'] in seen_classes,
                                               item['sequence'] in seen_videos,
                                               item['sequence'], item['frame'], item['identity']))
            keep.append(row)
            seen_classes.add(row['reason'])
            seen_videos.add(row['sequence'])
            pools = [p for p in pools if p['sequence'] != row['sequence'] or abs(p['frame'] - row['frame']) > 12]
        chosen.extend(keep)
    return chosen


def panel(image, truth, box, crop, color):
    x1, y1, x2, y2 = crop
    pixel = image[y1:y2, x1:x2].copy()

    def draw(coords, border):
        left, top, right, bottom = [round(c - 1 - shift)
                                    for c, shift in zip(coords[:4], (x1, y1, x1, y1), strict=True)]
        rgb = tuple(int(border[j:j + 2], 16) for j in (1, 3, 5))
        cv2.rectangle(pixel, (left, top), (right, bottom), rgb[::-1], 2)
    if truth is not None:
        draw(truth, TRUTH_COLOR)
    if box is not None:
        draw(box, color)
    return pixel


def crop_for(assets, case):
    images, truth, _, _ = assets
    middle, person = case['frame'], case['identity']
    boxes = [truth[frame][person] for frame in range(middle - 4, middle + 5) if person in truth[frame]]
    image = load_image(images[middle - 1])
    height, width = image.shape[:2]
    x1, y1, x2, y2 = (min(b[0] for b in boxes), min(b[1] for b in boxes),
                      max(b[2] for b in boxes), max(b[3] for b in boxes))
    dx, dy = max(20, (x2 - x1) * 0.55), max(15, (y2 - y1) * 0.24)
    return max(0, int(x1 - dx)), max(0, int(y1 - dy)), min(width, int(x2 + dx)), min(height, int(y2 + dy))


def figure(rows, assets, methods, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'],
                         'font.size': 7, 'pdf.fonttype': 42})
    fig, axes = plt.subplots(3, len(rows) * 3, figsize=(7.15 if len(rows) == 2 else 4.25, 4.3),
                            squeeze=False, layout='constrained')
    for group, row in enumerate(rows):
        frames, annotations, identity, _ = assets[row['sequence']]
        person, middle = row['identity'], row['frame']
        crop = crop_for(assets[row['sequence']], row)
        for col, index in enumerate((middle - 4, middle, middle + 4)):
            image = load_image(frames[index - 1])
            for j, method in enumerate(methods):
                ax = axes[j][group * 3 + col]
                track, box = identity[method][index].get(person, (None, None))
                rendered = panel(image, annotations[index].get(person), box, crop, METHOD_COLOR.get(method, '#209261'))
                ax.imshow(cv2.cvtColor(rendered, cv2.COLOR_BGR2RGB))
                ax.set_axis_off()
                prefix = ('CGHTrack' if method.startswith('CGHTrack') else method) + ' · ' if col == 0 else ''
                ax.set_title(prefix + f't{index} · ID {track if track is not None else "—"}', fontsize=7, pad=3)
    fig.savefig(output.with_suffix('.pdf'), bbox_inches='tight')
    fig.savefig(output.with_suffix('.png'), dpi=600, bbox_inches='tight')
    plt.close(fig)


def render_video(case, asset, methods, path):
    images, truth, identity, fps = asset
    person, middle = case['identity'], case['frame']
    crop = crop_for(asset, case)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), min(12, fps), (768, 288))
    if not writer.isOpened():
        raise ValueError('Could not initialize MP4 encoder')
    try:
        for index in range(max(1, middle - 10), min(len(images), middle + 10) + 1):
            image = load_image(images[index - 1])
            rendered = []
            for method in methods:
                tid, box = identity[method][index].get(person, (None, None))
                local = panel(image, truth[index].get(person), box, crop, METHOD_COLOR.get(method, '#209261'))
                height, width = local.shape[:2]
                factor = 256 / max(height, width)
                tile_width, tile_height = max(1, round(width * factor)), max(1, round(height * factor))
                box_image = cv2.resize(local, (tile_width, tile_height), interpolation=cv2.INTER_AREA)
                tile = np.full((288, 256, 3), 245, dtype=np.uint8)
                y, x = 32 + (256 - tile_height) // 2, (256 - tile_width) // 2
                tile[y:y+tile_height, x:x+tile_width] = box_image
                name = 'CGHTrack' if method.startswith('CGHTrack') else method
                cv2.putText(tile, f'{name} | t{index} | ID {tid if tid is not None else "-"}', (6, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (30, 30, 30), 1, cv2.LINE_AA)
                rendered.append(tile)
            writer.write(np.concatenate(rendered, axis=1))
    finally:
        writer.release()
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.read()[0]:
            raise ValueError(f'Unreadable visual case video: {path}')
    finally:
        capture.release()


def video(case, asset, methods, path):
    # Some Windows OpenCV builds cannot open an output filename with Unicode components.
    with TemporaryDirectory() as scratch:
        temporary = Path(scratch) / 'visualization.mp4'
        render_video(case, asset, methods, temporary)
        shutil.copyfile(temporary, path)


def run(final, output):
    peer, dev_metrics = baseline_from_dev()
    methods = ('ByteTrack', peer, 'CGHTrack-s0')
    output.mkdir(parents=True, exist_ok=False)
    rows, assets = audit_candidates(final, methods)
    with (output / 'all-candidates.jsonl').open('x', encoding='utf-8') as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + '\n')
    cases = choose_cases(rows)
    for idx, case in enumerate(cases, 1):
        name = f'{idx:02d}-{case["outcome"]}'
        case['snapshot'] = f'{name}.pdf'
        case['video'] = f'{name}.mp4'
        figure([case], assets, methods, output / name)
        video(case, assets[case['sequence']], methods, output / case['video'])
    core = [next((case for case in cases if case['outcome'] == outcome), None)
            for outcome in ('improvement', 'failure')]
    if all(core):
        figure(core, assets, methods, output / 'Figure-2')
    counts = {outcome: dict(Counter(row['reason'] for row in rows if row['outcome'] == outcome))
              for outcome in ('improvement', 'failure')}
    (output / 'index.json').write_text(json.dumps({
        'peer': peer, 'development_scores': dev_metrics, 'methods': methods, 'candidate_counts': counts,
        'rule': 'All three methods matched the person on the previous frame. Compare ID on current frame.',
        'cases': cases, 'candidate_frames': len(rows), 'annotated_box': TRUTH_COLOR,
    }, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    lines = ['# E8 · MOT17 留出案例', '',
             f'方法为 ByteTrack、开发集 HOTA 最高的其他基线 {peer} 和 CGHTrack 种子 0。黄色框表示人工标注。'
             '蓝色、绿色和砖红色依次表示三个跟踪器的输出框。每例图按前、中、后排列，视频标签列出实际轨迹 ID。', '',
             '| 例 | 序列 | 事件帧 | 人工身份 | 结果与原因 | 定格 | 视频 |',
             '|---|---|---:|---:|---|---|---|']
    for idx, case in enumerate(cases, 1):
        lines.append(f'| {idx} | {case["sequence"]} | {case["frame"]} | {case["identity"]} | '
                     f'{case["outcome"]} / {case["reason"]} | [{case["snapshot"]}]({case["snapshot"]}) | '
                     f'[{case["video"]}]({case["video"]}) |')
    (output / 'index.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(json.dumps({'peer': peer, 'candidate_frames': len(rows), 'selected': len(cases), 'causes': counts}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('final', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    run(args.final, args.output)
