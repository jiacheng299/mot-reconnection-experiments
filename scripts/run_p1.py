"""Run the MOT17-05-FRCNN engineering check using the author's ByteTrack."""

import argparse
from collections import defaultdict
from configparser import ConfigParser
import csv
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import cv2
import numpy as np
import trackeval


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data' / 'p0-p1'
SEQUENCE = 'MOT17-05-FRCNN'
BYTETRACK_COMMIT = 'd1bf0191adff59bc8fcfeaa0b33d3d1642552a99'


def load_tracker(run):
    repo = ROOT / 'vendor' / 'ByteTrack'
    git = ['git', '-C', str(repo)]
    if subprocess.check_output([*git, 'rev-parse', 'HEAD'], text=True).strip() != BYTETRACK_COMMIT:
        raise ValueError('Unexpected ByteTrack commit')

    # Stage the association module with only CPU import and NumPy compatibility changes.
    runtime = run / 'runtime'
    package = runtime / 'yolox'
    tracker_dir = package / 'tracker'
    tracker_dir.mkdir(parents=True, exist_ok=True)
    (package / '__init__.py').touch()
    for name in ('basetrack.py', 'byte_tracker.py', 'kalman_filter.py', 'matching.py'):
        source = subprocess.check_output([*git, 'show', f'HEAD:yolox/tracker/{name}'], text=True)
        source = source.replace('dtype=np.float)', 'dtype=float)')
        if name == 'byte_tracker.py':
            source = source.replace('import torch\nimport torch.nn.functional as F\n', '')
        (tracker_dir / name).write_text(source, encoding='utf-8')
    sys.path.insert(0, str(runtime))
    if 'yolox' in sys.modules:
        package_paths = sys.modules['yolox'].__path__
        if str(package) not in package_paths:
            package_paths.insert(0, str(package))
    from yolox.tracker.byte_tracker import BYTETracker

    return BYTETracker


def read_detections(source, length):
    detections = defaultdict(list)
    with source.open(newline='') as file:
        for row in csv.reader(file):
            frame = int(row[0])
            x, y, width, height, score = map(float, row[2:7])
            if not 1 <= frame <= length or width <= 0 or height <= 0:
                raise ValueError(f'Invalid detection {row}')
            detections[frame].append((x, y, x + width, y + height, score))
    return detections


def predict(run, info, source=None, tracker_class=None):
    sequence = info['name']
    if source is None:
        source = DATA / 'gt' / sequence / 'det' / 'det.txt'
    length = info.getint('seqLength')
    detections = read_detections(source, length)
    options = SimpleNamespace(track_thresh=0.5, track_buffer=30, match_thresh=0.8, mot20=False)
    if tracker_class is None:
        tracker_class = load_tracker(run)
    tracker = tracker_class(options, frame_rate=info.getint('frameRate'))
    size = (info.getint('imHeight'), info.getint('imWidth'))
    path = run / 'trackers' / 'ByteTrack' / 'data' / f'{sequence}.txt'
    path.parent.mkdir(parents=True, exist_ok=True)
    tracks = defaultdict(list)
    with path.open('w', newline='') as file:
        writer = csv.writer(file, lineterminator='\n')
        for frame in range(1, length + 1):
            boxes = np.array(detections[frame], dtype=float).reshape(-1, 5)
            for track in tracker.update(boxes, size, size):
                x, y, width, height = map(float, track.tlwh)
                if width * height <= 10 or width / height > 1.6:
                    continue
                writer.writerow((frame, track.track_id, f'{x:.4f}', f'{y:.4f}',
                                 f'{width:.4f}', f'{height:.4f}', f'{track.score:.6f}', -1, -1, -1))
                tracks[frame].append((track.track_id, x, y, width, height))
    print(f'Processed {length} frames, {sum(map(len, detections.values()))} detections, '
          f'{sum(map(len, tracks.values()))} predicted boxes', flush=True)
    return tracks


def evaluate(run, length, sequence=SEQUENCE, gt_folder=DATA / 'gt', tracker='ByteTrack'):
    evaluator = trackeval.Evaluator({'USE_PARALLEL': False, 'PLOT_CURVES': False})
    dataset = trackeval.datasets.MotChallenge2DBox({
        'GT_FOLDER': str(gt_folder), 'TRACKERS_FOLDER': str(run / 'trackers'),
        'OUTPUT_FOLDER': str(run / 'trackeval'), 'TRACKERS_TO_EVAL': [tracker],
        'BENCHMARK': 'MOT17', 'SPLIT_TO_EVAL': 'train', 'SKIP_SPLIT_FOL': True,
        'SEQ_INFO': {sequence: length}, 'DO_PREPROC': True,
    })
    metrics = [trackeval.metrics.HOTA(), trackeval.metrics.CLEAR(), trackeval.metrics.Identity()]
    evaluator.evaluate([dataset], metrics)


def render(run, info, tracks):
    capture = cv2.VideoCapture(str(DATA / f'{SEQUENCE}-raw.mp4'))
    size = (info.getint('imWidth'), info.getint('imHeight'))
    fps = info.getint('frameRate')
    length = info.getint('seqLength')
    if not capture.isOpened() or abs(capture.get(cv2.CAP_PROP_FPS) - fps) > 0.01:
        raise RuntimeError('Official raw video missing or frame rate mismatches seqinfo.ini')
    writer = cv2.VideoWriter(str(run / 'tracking.mp4'), cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
    if not writer.isOpened():
        raise RuntimeError('Unable to encode the preview video')
    try:
        for frame in range(1, length + 1):
            success, image = capture.read()
            if not success or image.shape[:2] != size[::-1]:
                raise RuntimeError(f'Official raw video mismatch at frame {frame}')
            for identity, x, y, width, height in tracks[frame]:
                color = tuple(70 + (identity * factor) % 186 for factor in (37, 67, 97))
                left, top = round(x - 1), round(y - 1)
                cv2.rectangle(image, (left, top), (round(x + width - 1), round(y + height - 1)), color, 2)
                label = (max(2, left), max(46, top - 4))
                cv2.putText(image, str(identity), label,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 4, cv2.LINE_AA)
                cv2.putText(image, str(identity), label,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
            cv2.rectangle(image, (0, 0), (398, 28), (0, 0, 0), -1)
            cv2.putText(image, f'{SEQUENCE} | ByteTrack | {frame}/{length}', (7, 19),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            writer.write(image)
            if frame == 200:
                encoded, png = cv2.imencode('.png', image)
                if not encoded:
                    raise RuntimeError('Unable to encode preview frame')
                (run / 'frame-0200.png').write_bytes(png.tobytes())
        if capture.read()[0]:
            raise RuntimeError('Official raw video contains more frames than seqinfo.ini')
    finally:
        capture.release()
        writer.release()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path, help='new output directory')
    args = parser.parse_args()
    run = args.run.resolve()
    run.mkdir(parents=True, exist_ok=False)
    config = ConfigParser()
    with (DATA / 'gt' / SEQUENCE / 'seqinfo.ini').open() as file:
        config.read_file(file)
    info = config['Sequence']
    tracks = predict(run, info)
    evaluate(run, info.getint('seqLength'))
    render(run, info, tracks)


if __name__ == '__main__':
    main()
