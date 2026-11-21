"""Run ByteTrack and TrackEval on the shared MOT17-10 detection cache."""

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

from detect_mot17_dev import DATA, SEQUENCE, file_sha256, load_image, sequence_info
from run_p1 import evaluate, predict


def preview(run, info, cache, tracks, tracker='ByteTrack'):
    frame = 200
    image_path = DATA / 'gt' / SEQUENCE / info['imDir'] / f'{frame:06d}{info["imExt"]}'
    image = load_image(image_path)
    detection_view, tracking_view = image.copy(), image.copy()

    def draw(panel, box, label, color):
        x, y, width, height = box
        left, top = round(x - 1), round(y - 1)
        cv2.rectangle(panel, (left, top), (round(x + width - 1), round(y + height - 1)), color, 3)
        origin = (max(3, left), max(65, top - 6))
        cv2.putText(panel, label, origin, cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 5, cv2.LINE_AA)
        cv2.putText(panel, label, origin, cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

    with cache.open(newline='') as file:
        for row in csv.reader(file):
            if int(row[0]) == frame and float(row[6]) >= 0.10:
                draw(detection_view, tuple(map(float, row[2:6])), f'{float(row[6]):.2f}', (72, 224, 255))
    for identity, *box in tracks[frame]:
        color = tuple(85 + (identity * factor) % 171 for factor in (37, 67, 97))
        draw(tracking_view, box, f'ID {identity}', color)

    panels = []
    for view, title in ((detection_view, 'YOLOX-X  |  detection score >= 0.10'),
                        (tracking_view, f'{tracker}  |  assigned trajectory IDs')):
        image = cv2.resize(view, (960, 540), interpolation=cv2.INTER_AREA)
        cv2.rectangle(image, (0, 0), (959, 34), (18, 23, 31), -1)
        cv2.putText(image, title, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, .65,
                    (255, 255, 255), 1, cv2.LINE_AA)
        panels.append(image)
    encoded, png = cv2.imencode('.png', np.hstack(panels))
    if not encoded:
        raise RuntimeError('Could not encode the detection and tracking preview')
    (run / 'frame-0200-detection-vs-tracking.png').write_bytes(png.tobytes())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path, help='existing detection-cache run directory')
    args = parser.parse_args()
    run = args.run.resolve()
    info = sequence_info()
    cache = run / 'detections' / f'{SEQUENCE}.txt'
    manifest = json.loads(cache.with_suffix('.json').read_text(encoding='utf-8'))
    if (manifest['sequence'] != SEQUENCE or manifest['frames'] != info.getint('seqLength')
            or manifest['detections_sha256'] != file_sha256(cache)):
        raise ValueError('Shared cache metadata or content does not match')
    print(f'Using shared cache SHA-256 {manifest["detections_sha256"]}', flush=True)
    tracks = predict(run, info, cache)
    evaluate(run, info.getint('seqLength'), SEQUENCE, DATA / 'gt')
    preview(run, info, cache, tracks)


if __name__ == '__main__':
    main()
