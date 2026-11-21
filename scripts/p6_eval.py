"""Evaluate frozen final trackers, and prepare official DanceTrack bundles locally."""

import argparse
import csv
from hashlib import sha256
from importlib import metadata
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np

from scripts.detect_mot17_dev import ROOT, file_sha256
from scripts.p5_dance_eval import set_gate
from scripts.p5_train import load_ranker
from scripts.p6_data import FINAL_ROOT, FROZEN, final_names, frozen_config, sequences, verify_cache
from scripts.run_p1 import load_tracker, predict, read_detections
from scripts.score_p3_c1 import OPTIONS, track_sequence


BASELINES = ('ByteTrack', 'OC-SORT', 'BoT-SORT', 'TrackTrack')
CGH_METHODS = tuple(f'CGHTrack-s{seed}' for seed in (0, 1, 2))
ALL_METHODS = BASELINES + CGH_METHODS
APPEARANCE = '989f336d0530fd15f413f9d6ced0e93882808f93520d6a9d4e406eaf62798092'
OC_COMMIT = '8462e7e729a93ccd3bd995c0a79a890336cb3a0b'
OC_OPTIONS = {'det_thresh': 0.6, 'max_age': 30, 'min_hits': 3, 'iou_threshold': 0.3,
              'delta_t': 3, 'asso_func': 'iou', 'inertia': 0.2, 'use_byte': False}


def verify_freeze():
    cfg = frozen_config()
    for name, digest in cfg['scripts_sha256'].items():
        if file_sha256(ROOT / 'scripts' / name) != digest:
            raise ValueError(f'Frozen P5 code differs: {name}')
    for name in ('torch', 'numpy', 'scipy', 'trackeval'):
        if metadata.version(name) != cfg['environment'][name]:
            raise ValueError(f'Frozen package version differs: {name}')
    if (platform.python_version() != cfg['environment']['python'] or
            sha256(metadata.distribution('trackeval').read_text('RECORD').encode()).hexdigest()
            != cfg['evaluation_source_record_sha256']):
        raise ValueError('Frozen TrackEval/Python runtime differs')
    if not cfg['online_only']:
        raise ValueError('An offline configuration cannot enter P6')
    return cfg


def resolve_checkpoints(domain, cfg):
    if domain == 'DanceTrack':
        p = ROOT / 'runs' / 'p5-dance-formal' / 'tables-e2-e4.json'
        rows = json.loads(p.read_bytes())['e4']
        saved = {str(row['seed']): Path(row['checkpoint']) for row in rows if row['objective'] == 'td'
                 and row['variant'] == cfg['architecture']['deployment'] and row['gamma'] == cfg['gamma']}
        expected = cfg['dance_checkpoints']
    else:
        p = ROOT / 'runs' / 'p5-mot-formal' / 'mot17-e2.json'
        rows = json.loads(p.read_bytes())
        saved = {str(row['seed']): Path(row['winner']['checkpoint']) for row in rows
                 if row['variant'] == cfg['architecture']['deployment'] and row['gamma'] == cfg['gamma']}
        expected = cfg['mot_checkpoints']
    if set(saved) != {'0', '1', '2'} or {seed: file_sha256(path) for seed, path in saved.items()} != expected:
        raise ValueError('Development-selected checkpoints do not match the frozen hashes')
    return saved


def oc_tracker():
    repo = ROOT / 'vendor' / 'OC_SORT'
    args = ['git', '-C', str(repo)]
    commit = subprocess.check_output([*args, 'rev-parse', 'HEAD'], text=True).strip()
    changes = subprocess.check_output([*args, 'status', '--porcelain', '--', 'trackers'], text=True)
    if commit != OC_COMMIT or changes:
        raise ValueError('OC-SORT implementation differs from the P2 pin')
    sys.path.insert(0, str(repo))
    from trackers.ocsort_tracker.ocsort import OCSort

    return OCSort(**OC_OPTIONS)


def write_oc(path, detections, info):
    tracker = oc_tracker()
    size = (info.getint('imHeight'), info.getint('imWidth'))
    with path.open('w', newline='') as file:
        writer = csv.writer(file, lineterminator='\n')
        for frame in range(1, info.getint('seqLength') + 1):
            boxes = np.asarray(detections[frame], dtype=float).reshape(-1, 5)
            for x, y, right, bottom, track_id in tracker.update(boxes, size, size):
                width, height = right - x, bottom - y
                if height > 0 and width * height > 10 and width / height <= 1.6:
                    writer.writerow((frame, int(track_id), f'{x:.4f}', f'{y:.4f}',
                                     f'{width:.4f}', f'{height:.4f}', 1, -1, -1, -1))
    return {'commit': OC_COMMIT, 'options': OC_OPTIONS}


def write_appearance(run, method, info, cache, images):
    sys.path.insert(0, str(ROOT / 'scripts'))
    import score_appearance_mot17 as legacy
    import torch

    torch.set_num_threads(1)
    legacy.verify_source(method)
    if file_sha256(legacy.WEIGHTS) != APPEARANCE:
        raise ValueError('Appearance model changed since P2')
    # The P2 adapter's frozen scope is one sequence. Isolate runtime staging and the
    # sequence identifier, while retaining its causal camera compensation and options.
    legacy.SEQUENCE = info['name']
    config, _ = legacy.write_tracks(run, method, info, cache, iter(images))
    return config


def track_count(path, length):
    seen = set()
    previous = 0
    with path.open(newline='') as file:
        for row in csv.reader(file):
            if len(row) != 10:
                raise ValueError(f'Invalid MOT row in {path}')
            frame, track_id = int(row[0]), int(row[1])
            if not max(1, previous) <= frame <= length or track_id < 1 or (frame, track_id) in seen:
                raise ValueError(f'Duplicate or invalid frame/track ID in {path}')
            if not all(np.isfinite(float(x)) for x in row[2:]) or any(float(x) <= 0 for x in row[4:6]):
                raise ValueError(f'Invalid output geometry in {path}')
            seen.add((frame, track_id))
            previous = frame
    return len(seen)


def score_local(root, domain, development, method, lengths):
    import trackeval
    from scripts import p5_dance_data as dance, p5_mot_detector as mot

    gt = (dance.DANCE / 'val' if domain == 'DanceTrack' else mot.DATA / 'gt') if development else FINAL_ROOT / domain
    evaluator = trackeval.Evaluator({'USE_PARALLEL': False, 'PLOT_CURVES': False})
    dataset = trackeval.datasets.MotChallenge2DBox({
        'GT_FOLDER': str(gt), 'TRACKERS_FOLDER': str(root / 'trackers'),
        'OUTPUT_FOLDER': str(root / 'trackeval'), 'TRACKERS_TO_EVAL': [method],
        'BENCHMARK': domain, 'SPLIT_TO_EVAL': 'val' if development else 'train',
        'SKIP_SPLIT_FOL': True, 'SEQ_INFO': lengths, 'DO_PREPROC': True, 'PRINT_RESULTS': True,
    })
    evaluator.evaluate([dataset], [trackeval.metrics.HOTA(), trackeval.metrics.CLEAR(), trackeval.metrics.Identity()])


def pack_dance(root, method, lengths):
    output = root / 'submission-zips'
    output.mkdir(parents=True, exist_ok=True)
    target = output / f'{method}.zip'
    staged = target.with_suffix('.tmp')
    with ZipFile(staged, 'w', ZIP_DEFLATED) as archive:
        for name in sorted(lengths):
            file = root / 'trackers' / method / 'data' / f'{name}.txt'
            track_count(file, lengths[name])
            archive.write(file, f'tracker/{name}.txt')
    with ZipFile(staged) as archive:
        if archive.testzip() is not None or set(archive.namelist()) != {f'tracker/{n}.txt' for n in final_names('DanceTrack')}:
            raise ValueError('DanceTrack package failed integrity checks')
    staged.replace(target)
    digest = file_sha256(target)
    target.with_suffix('.sha256').write_text(f'{digest}  {target.name}\n', encoding='ascii')
    return {'file': str(target), 'sha256': digest, 'sequences': len(lengths), 'result': 'awaiting official server score'}


def evaluate(root, domain, method, development=False):
    cfg = verify_freeze()
    checkpoint_path, ranker = None, None
    options_domain = 'DanceTrack' if domain == 'DanceTrack' else 'MOT17'
    if method in CGH_METHODS:
        set_gate(cfg['development_gates'][options_domain])
        checkpoint_path = resolve_checkpoints(domain, cfg)[method[-1]]
        ranker, _ = load_ranker(checkpoint_path)
    if development:
        subdir = 'p5-dance-detections' if domain == 'DanceTrack' else 'p5-mot-detections'
        cache_root = ROOT / 'runs' / subdir / 'val'
    else:
        cache_root = ROOT / 'runs' / 'p6-detections' / domain
    root.mkdir(parents=True, exist_ok=True)
    manifests = root / 'manifests' / method
    manifests.mkdir(parents=True, exist_ok=True)
    lengths, records = {}, {}
    code_hash = file_sha256(Path(__file__))
    checkpoint_sha = file_sha256(checkpoint_path) if checkpoint_path else None
    for folder, info, images in sequences(domain, development):
        name, length = folder.name, len(images)
        lengths[name] = length
        cache_path = cache_root / f'{name}.txt'
        cache = verify_cache(cache_path, domain, name, length, development)
        detections = read_detections(cache_path, length)
        if sum(map(len, detections.values())) != cache['count']:
            raise ValueError(f'Cache count mismatch for {name}')
        path = root / 'trackers' / method / 'data' / f'{name}.txt'
        path.parent.mkdir(parents=True, exist_ok=True)
        manifest = manifests / f'{name}.json'
        input_identity = {'domain': domain, 'method': method, 'frames': length, 'sequence': name,
                          'detector_sha256': cache['weights_sha256'], 'detections_sha256': cache['detections_sha256'],
                          'checkpoint_sha256': checkpoint_sha, 'p5_freeze_sha256': file_sha256(FROZEN),
                          'p6_evaluation_sha256': code_hash, 'online_only': True}
        if manifest.is_file():
            saved = json.loads(manifest.read_bytes())
            if any(saved[key] != value for key, value in input_identity.items()) or saved['output_sha256'] != file_sha256(path):
                raise ValueError(f'Cannot reuse a run with changed inputs: {name}/{method}')
            records[name] = saved
            continue
        if method == 'ByteTrack':
            byte = load_tracker(root)
            from yolox.tracker.basetrack import BaseTrack

            BaseTrack._count = 0
            predict(root, info, cache_path, tracker_class=byte)
            config = {'track_thresh': 0.5, 'match_thresh': 0.8, 'track_buffer': 30}
        elif method == 'OC-SORT':
            staged = path.with_suffix('.tmp')
            config = write_oc(staged, detections, info)
            staged.replace(path)
        elif method in ('BoT-SORT', 'TrackTrack'):
            staging = root / 'baseline-runs' / method / name
            config = write_appearance(staging, method, info, cache_path, images)
            source = staging / 'trackers' / method / 'data' / path.name
            shutil.copyfile(source, path)
        else:
            audit_path = root / 'associations' / method / f'{name}.jsonl'
            summary = track_sequence(root, detections, length, name, method, ranker,
                                     candidate_limit=cfg['candidate_limit'], audit_path=audit_path)
            config = {'association': OPTIONS, 'summary': summary, 'audit_sha256': file_sha256(audit_path)}
        records[name] = {**input_identity, 'output_rows': track_count(path, length),
                         'output_sha256': file_sha256(path), 'configuration': config}
        staged = manifest.with_suffix('.tmp')
        staged.write_text(json.dumps(records[name], indent=2) + '\n', encoding='utf-8')
        staged.replace(manifest)
        print(f'{domain} {method}: {len(records)} sequences, {name}, {length} frames', flush=True)
    submitted = None
    if domain == 'DanceTrack' and not development:
        submitted = pack_dance(root, method, lengths)
    else:
        score_local(root, domain, development, method, lengths)
    (root / 'manifests' / f'{method}.json').write_text(json.dumps({
        'domain': domain, 'development': development, 'coverage': lengths, 'submission': submitted, 'runs': records,
    }, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('domain', choices=('MOT17', 'MOT20', 'DanceTrack'))
    parser.add_argument('method', choices=ALL_METHODS)
    parser.add_argument('output', type=Path)
    parser.add_argument('--development', action='store_true')
    args = parser.parse_args()
    evaluate(args.output.resolve(), args.domain, args.method, args.development)
