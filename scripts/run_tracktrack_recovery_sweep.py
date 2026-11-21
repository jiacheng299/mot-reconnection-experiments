"""Evaluate TrackTrack's low-score penalty while sharing its frozen image evidence."""

import argparse
from contextlib import ExitStack
import csv
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import torch

from scripts.detect_mot17_dev import ROOT, file_sha256, load_image
from scripts.p5_report import read_metrics
from scripts.p6_data import sequences, verify_cache
from scripts.p6_eval import APPEARANCE, score_local, track_count, verify_freeze
from scripts.run_p1 import read_detections


PENALTIES = {'TrackTrack': 0.20, **{f'TrackTrack-penalty-{value:.2f}': value
             for value in (0.12, 0.28, 0.36, 0.44, 0.52)}}


def track_sequence(legacy, root, name, info, images, cache):
    length = len(images)
    detections = read_detections(cache, length)
    trackers = {}
    legacy.SEQUENCE = name
    for label, penalty in PENALTIES.items():
        torch.manual_seed(0)
        adapter, options, min_area = legacy.make_tracker('TrackTrack', root / 'engine' / name / label, info)
        adapter.tracker.args.penalty_p = penalty
        trackers[label] = adapter, min_area
    base, _ = trackers['TrackTrack']
    embedding, camera = base.encoder.inference, base.camera.apply
    evidence = SimpleNamespace()

    def capture_features(image, boxes):
        evidence.boxes = boxes.copy()
        evidence.features = embedding(image, boxes)
        return evidence.features.copy()

    def reuse_features(image, boxes):
        if not np.array_equal(boxes, evidence.boxes):
            raise ValueError('Frozen TrackTrack detection boxes differ')
        return evidence.features.copy()

    def capture_warp(image, boxes):
        evidence.warp = camera(image, boxes)
        return evidence.warp.copy()

    base.encoder.inference = capture_features
    base.camera.apply = capture_warp
    for label, (adapter, _) in trackers.items():
        if label != 'TrackTrack':
            adapter.encoder.inference = reuse_features
            adapter.camera.apply = lambda image, boxes: evidence.warp.copy()
    paths = {label: root / 'trackers' / label / 'data' / f'{name}.txt' for label in trackers}
    with ExitStack() as stack, torch.inference_mode():
        writers = {}
        for label, path in paths.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            handle = stack.enter_context(path.with_suffix('.tmp').open('w', newline=''))
            writers[label] = csv.writer(handle, lineterminator='\n')
        for frame, file in enumerate(images, 1):
            image = load_image(file)
            boxes = np.asarray(detections[frame], dtype=float).reshape(-1, 5)
            boxes[:, :4] -= 1
            for label, (adapter, min_area) in trackers.items():
                for track in adapter.update(boxes, image):
                    x, y, w, h = map(float, track.tlwh)
                    if h <= 0 or w * h <= min_area or w / h > 1.6:
                        continue
                    writers[label].writerow((frame, track.track_id, f'{x + 1:.4f}', f'{y + 1:.4f}',
                                             f'{w:.4f}', f'{h:.4f}', f'{track.score:.6f}', -1, -1, -1))
            if frame % 300 == 0 or frame == length:
                print(f'{name}: {frame}/{length}', flush=True)
    for path in paths.values():
        path.with_suffix('.tmp').replace(path)
    return {label: {'sha256': file_sha256(path), 'rows': track_count(path, length)}
            for label, path in paths.items()}


def evaluate(root):
    sys.path.insert(0, str(ROOT / 'scripts'))
    import score_appearance_mot17 as legacy

    verify_freeze()
    legacy.verify_source('TrackTrack')
    if file_sha256(legacy.WEIGHTS) != APPEARANCE:
        raise ValueError('Unexpected appearance weights')
    torch.set_num_threads(1)
    root.mkdir(parents=True, exist_ok=True)
    (root / 'run-config.json').write_text(json.dumps({
        'penalty_p': PENALTIES, 'appearance_sha256': APPEARANCE,
        'source_sha256': file_sha256(Path(__file__)), 'upstream': legacy.COMMITS['TrackTrack'],
        'protocol': 'P6 MOT17 heldout, single shared detection cache',
    }, indent=2) + '\n', encoding='utf-8')
    source = ROOT / 'runs' / 'p6-MOT17-final'
    lengths, runs = {}, {}
    for folder, info, images in sequences('MOT17'):
        name = folder.name
        raw = ROOT / 'runs' / 'p6-detections' / 'MOT17' / f'{name}.txt'
        cache = verify_cache(raw, 'MOT17', name, len(images))
        lengths[name] = len(images)
        expected = json.loads((source / 'manifests' / 'TrackTrack' / f'{name}.json').read_bytes())
        tracks = track_sequence(legacy, root, name, info, images, raw)
        if (tracks['TrackTrack']['sha256'] != expected['output_sha256']
                or cache['detections_sha256'] != expected['detections_sha256']):
            raise ValueError(f'TrackTrack replay differs from P6 on {name}')
        runs[name] = {'frames': len(images), 'tracks': tracks, 'cache': cache}
        (root / 'sequences.json').write_text(json.dumps(runs, indent=2) + '\n', encoding='utf-8')
    metrics = {}
    for label in PENALTIES:
        score_local(root, 'MOT17', False, label, lengths)
        metrics[label] = read_metrics(root / 'trackeval' / label / 'pedestrian_summary.txt')
    (root / 'metrics.json').write_text(json.dumps(metrics, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    evaluate(parser.parse_args().output)
