"""Run appearance-based baselines against the pinned MOT17-10 cache."""

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import torch

from detect_mot17_dev import DATA, ROOT, SEQUENCE, file_sha256, load_image, sequence_info
from run_p1 import evaluate, read_detections
from score_mot17_dev import preview
from score_ocsort_dev import CACHE_SHA256


COMMITS = {'BoT-SORT': '251985436d6712aaf682aaaf5f71edb4987224bd',
           'TrackTrack': 'ee7f1c5fcbdcac48ed8bfab38d52c0006bf304da'}
WEIGHTS = DATA / 'market_sbs_S50.pth'
REID_REPO = ROOT / 'vendor' / 'BoT-SORT'
CAMERA_METHOD = 'sparseOptFlow'


def stage_reid(runtime):
    destination = runtime / 'fast_reid'
    shutil.copytree(REID_REPO / 'fast_reid' / 'fastreid', destination / 'fastreid')
    source = REID_REPO / 'fast_reid' / 'fast_reid_interfece.py'
    content = source.read_text(encoding='utf-8')
    content = re.sub(r'^from fast_reid\.fastreid\.engine import .*\n', '', content, flags=re.M)
    (destination / source.name).write_text(content, encoding='utf-8')


def make_tracker(method, run, info, has_secondary=False):
    runtime = run / 'runtime'
    if not runtime.exists():
        stage_reid(runtime)
    if str(runtime) not in sys.path:
        sys.path.insert(0, str(runtime))
    config = REID_REPO / 'fast_reid' / 'configs' / 'Market1501' / 'sbs_S50.yml'
    common = {'fast_reid_config': str(config), 'fast_reid_weights': str(WEIGHTS), 'device': 'cuda'}
    if method == 'BoT-SORT':
        destination = runtime / 'tracker'
        if not destination.exists():
            destination.mkdir()
            for name in ('bot_sort.py', 'matching.py', 'gmc.py', 'basetrack.py', 'kalman_filter.py'):
                content = (REID_REPO / 'tracker' / name).read_text(encoding='utf-8')
                content = re.sub(r'\bnp\.float_?\b', 'np.float64', content)
                (destination / name).write_text(content, encoding='utf-8')
        from tracker.bot_sort import BoTSORT

        options = dict(common, with_reid=True, track_high_thresh=0.6, track_low_thresh=0.1,
                       new_track_thresh=0.7, track_buffer=30, proximity_thresh=0.5,
                       appearance_thresh=0.25, match_thresh=0.8, mot20=False,
                       cmc_method=CAMERA_METHOD, name=info['name'], ablation=False)
        tracker = BoTSORT(SimpleNamespace(**options), frame_rate=info.getint('frameRate'))
        return tracker, options, 10

    upstream = ROOT / 'vendor' / 'TrackTrack' / '3. Tracker'
    destination = runtime / 'trackers'
    shutil.copytree(upstream / 'trackers', destination, ignore=shutil.ignore_patterns('cmc', '__pycache__'))
    camera = runtime / 'tracktrack_gmc.py'
    camera_source = (REID_REPO / 'tracker' / 'gmc.py').read_text(encoding='utf-8')
    camera.write_text(camera_source, encoding='utf-8')
    from fast_reid.fast_reid_interfece import FastReIDInterface
    from tracktrack_gmc import GMC
    from trackers import tracker as tracktrack

    options = dict(common, det_thr=0.6, init_thr=0.7, match_thr=0.7, min_len=3,
                   penalty_p=0.2, penalty_q=0.4, reduce_step=0.05, tai_thr=0.55,
                   max_time_lost=2 * info.getint('frameRate'), img_w=info.getint('imWidth'),
                   img_h=info.getint('imHeight'), input_score_floor=0.1,
                   cmc_method=CAMERA_METHOD, data_path='MOT17/train/',
                   detection_streams='NMS 0.80 + 0.95 from shared detector' if has_secondary else
                   'same shared cache in both slots')
    encoder = FastReIDInterface(str(config), str(WEIGHTS), 'cuda')
    return TrackTrackAdapter(tracktrack, options, encoder, GMC(CAMERA_METHOD)), options, 100


class TrackTrackAdapter:
    """Adapt cached detections and compute adjacent-frame camera compensation."""

    def __init__(self, upstream, options, encoder, camera):
        self.encoder = encoder
        self.camera = camera
        self.warp = np.eye(2, 3)
        # Public CMC files contain 326 rows for MOT17-10. Compute causal warps on all 654 frames.
        upstream.CMC = lambda _: self
        self.tracker = upstream.Tracker(SimpleNamespace(**options), SEQUENCE)

    def get_warp_matrix(self):
        return self.warp

    def update(self, boxes, image, secondary=None):
        boxes = boxes[boxes[:, 4] > 0.1]
        self.warp = self.camera.apply(image, boxes[:, :4])
        if secondary is not None:
            secondary = secondary[secondary[:, 4] > 0.1]
        if not len(boxes):
            if secondary is not None and len(secondary):
                raise ValueError('Secondary candidates present without any primary detections')
            return self.tracker.update_without_detections()
        features = self.encoder.inference(image, boxes[:, :4])
        rows = np.column_stack((boxes, np.zeros(len(boxes)), features))
        if secondary is None:
            secondary_rows = rows
        else:
            secondary_features = self.encoder.inference(image, secondary[:, :4])
            secondary_rows = np.column_stack((secondary, np.zeros(len(secondary)), secondary_features))
        tracks = self.tracker.update(rows, secondary_rows)
        return [SimpleNamespace(tlwh=t.x1y1wh, track_id=t.track_id, score=t.score) for t in tracks]


def verify_source(method):
    for name in {method, 'BoT-SORT'}:
        path = ROOT / 'vendor' / name
        git = ['git', '-C', str(path)]
        commit = subprocess.check_output([*git, 'rev-parse', 'HEAD'], text=True).strip()
        if commit != COMMITS[name] or subprocess.check_output([*git, 'status', '--porcelain'], text=True):
            raise ValueError(f'Expected the clean pinned {name} source')


def write_tracks(run, method, info, cache, images, secondary_cache=None):
    length = info.getint('seqLength')
    detections = read_detections(cache, length)
    torch.manual_seed(0)
    tracker, options, min_area = make_tracker(method, run, info, secondary_cache is not None)
    config = {
        'sequence': info['name'], 'frames': length, 'commit': COMMITS[method],
        'appearance_training': 'FastReID SBS-S50, Market-1501 with ImageNet initialization',
        'appearance_sha256': file_sha256(WEIGHTS), 'detections_sha256': file_sha256(cache),
        'options': options, 'min_box_area': min_area, 'max_width_height_ratio': 1.6,
    }
    secondary_detections = None
    if secondary_cache is not None:
        secondary_detections = read_detections(secondary_cache, length)
        config['secondary_detections_sha256'] = file_sha256(secondary_cache)
    output = run / 'trackers' / method / 'data' / f'{info["name"]}.txt'
    output.parent.mkdir(parents=True, exist_ok=True)
    staged = output.with_suffix('.tmp')
    tracks = defaultdict(list)
    with staged.open('w', newline='') as file, torch.inference_mode():
        writer = csv.writer(file, lineterminator='\n')
        for frame, image_path in enumerate(images, start=1):
            image = load_image(image_path)
            boxes = np.array(detections[frame], dtype=float).reshape(-1, 5)
            # MOT cache coordinates are one-based. Tracker code crops zero-based images.
            boxes[:, :4] -= 1
            if secondary_detections is None:
                updated = tracker.update(boxes, image)
            else:
                secondary = np.array(secondary_detections[frame], dtype=float).reshape(-1, 5)
                secondary[:, :4] -= 1
                updated = tracker.update(boxes, image, secondary)
            for track in updated:
                x, y, width, height = map(float, track.tlwh)
                if height <= 0 or width * height <= min_area or width / height > 1.6:
                    continue
                x, y = x + 1, y + 1
                writer.writerow((frame, track.track_id, f'{x:.4f}', f'{y:.4f}',
                                 f'{width:.4f}', f'{height:.4f}', f'{track.score:.6f}', -1, -1, -1))
                tracks[frame].append((track.track_id, x, y, width, height))
            if frame % 300 == 0 or frame == length:
                print(f'{frame}/{length} frames', flush=True)
    staged.replace(output)
    print(f'{sum(map(len, tracks.values()))} exported boxes', flush=True)
    return config, tracks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('method', choices=COMMITS)
    parser.add_argument('run', type=Path)
    parser.add_argument('--tracktrack-inputs', type=Path, help='diagnostic input directory with the two NMS views')
    args = parser.parse_args()
    method, run = args.method, args.run.resolve()
    if args.tracktrack_inputs and method != 'TrackTrack':
        parser.error('--tracktrack-inputs applies only to TrackTrack')
    verify_source(method)
    info = sequence_info()
    length = info.getint('seqLength')
    cache = ROOT / 'runs' / 'p2-002' / 'detections' / f'{SEQUENCE}.txt'
    if file_sha256(cache) != CACHE_SHA256:
        raise ValueError('Input must match the detection cache used for ByteTrack')
    print(f'Verified reference cache SHA-256 {CACHE_SHA256}' if args.tracktrack_inputs else
          f'Using shared cache SHA-256 {CACHE_SHA256}', flush=True)
    reference_cache = cache
    secondary = None
    if args.tracktrack_inputs:
        inputs = args.tracktrack_inputs.resolve()
        record = json.loads((inputs / 'inputs.json').read_text(encoding='utf-8'))
        cache = inputs / 'nms-0.80.txt'
        secondary = inputs / 'nms-0.95.txt'
        if (record['sequence'] != SEQUENCE or record['frames'] != length or
                record['views']['0.65']['sha256'] != CACHE_SHA256 or
                file_sha256(cache) != record['views']['0.80']['sha256'] or
                file_sha256(secondary) != record['views']['0.95']['sha256']):
            raise ValueError('Invalid diagnostic detector views')
        print('TrackTrack diagnostic uses the separately verified NMS 0.80 and 0.95 views', flush=True)
    run.mkdir(parents=True, exist_ok=False)
    directory = DATA / 'gt' / SEQUENCE / info['imDir']
    images = (directory / f'{frame:06d}{info["imExt"]}' for frame in range(1, length + 1))
    config, tracks = write_tracks(run, method, info, cache, images, secondary)
    (run / 'run-config.json').write_text(json.dumps(config, indent=2) + '\n', encoding='utf-8')
    evaluate(run, length, SEQUENCE, DATA / 'gt', tracker=method)
    preview(run, info, reference_cache, tracks, tracker=method)


if __name__ == '__main__':
    main()
