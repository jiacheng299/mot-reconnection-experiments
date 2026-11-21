"""Run the E2 MOT17-10 control using only the frozen D-MOT cache."""

import argparse
import json
from pathlib import Path

from scripts.detect_mot17_dev import DATA, SEQUENCE, file_sha256, sequence_info
from scripts.p5_dance_eval import cached_detections, set_gate
from scripts.p5_mot_detector import verify_cache
from scripts.p5_train import load_ranker
from scripts.run_p1 import evaluate, predict
from scripts.score_p3_c1 import track_sequence


def run(output, cache, method, checkpoint):
    set_gate('reference')
    info = sequence_info()
    length = info.getint('seqLength')
    path = cache / 'val' / f'{SEQUENCE}.txt'
    manifest, detections = cached_detections(path, 'val', SEQUENCE, length, verify_cache)
    output.mkdir(parents=True, exist_ok=False)
    timings = []
    if checkpoint:
        record = json.loads(checkpoint.read_bytes())
        if record.get('dataset') != 'MOT17':
            raise ValueError('E2 MOT17 confirmation requires a MOT17-trained method')
        ranker, method = load_ranker(checkpoint)
        track_sequence(output, detections, length, SEQUENCE, method, ranker, timings=timings)
    elif method == 'ByteTrack':
        predict(output, info, path)
    else:
        track_sequence(output, detections, length, SEQUENCE, method)
    evaluate(output, length, SEQUENCE, DATA / 'gt', tracker=method)
    (output / 'input-manifest.json').write_text(json.dumps({
        'sequence': SEQUENCE, 'frames': length, 'detection_cache_sha256': manifest['detections_sha256'],
        'weights_sha256': manifest['weights_sha256'],
        'model_sha256': file_sha256(checkpoint) if checkpoint else None, 'method': method,
        'association_ms': sum(row[0] for row in timings) / (len(timings) * 1e6) if timings else None,
    }, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('cache', type=Path)
    parser.add_argument('--method', choices=('ByteTrack', 'C1'), default='C1')
    parser.add_argument('--checkpoint', type=Path)
    args = parser.parse_args()
    run(args.output, args.cache, args.method, args.checkpoint)
