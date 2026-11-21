"""Fetch final-evaluation inputs and run the frozen P5 detectors."""

import argparse
from configparser import ConfigParser
import csv
import hashlib
import io
from itertools import batched
import json
from pathlib import Path, PurePosixPath
import re
import time
from urllib.request import Request, urlopen
from zipfile import ZipFile
from zlib import crc32

import numpy as np

from scripts.detect_mot17_dev import ROOT, file_sha256, load_image, person_boxes, write_boxes
from scripts import p5_dance_data as dance, p5_mot_detector as mot


FINAL_ROOT = ROOT / 'data' / 'p6'
FROZEN = ROOT / 'runs' / 'p5-acceptance' / 'P6-frozen-config.json'
FINAL_SEQUENCES = {'MOT17': ('MOT17-11-FRCNN', 'MOT17-13-FRCNN'),
                   'MOT20': ('MOT20-01', 'MOT20-02', 'MOT20-03', 'MOT20-05')}
MOT_MIRRORS = {
    'MOT17': (2624658434, '6f1856734a3a1d70666d27834857f75897b46226a33e493bc35a651a685a60d3'),
    'MOT20': (3260257521, '17ededdf46df86f19f5a0300f80cf09486a185783888dd6773305f20f76ac9c1'),
}
MOT_MIRROR_ROOT = 'https://huggingface.co/datasets/AbdoTW/MOT16-17-20/resolve/bbe98108d58104b30c915dd4db34573008d6f352'
DANCE_ARCHIVES = {'test1.zip': '10e71d2b1c81fb954d7076863d299944b3b15951c4a4b033a454a94569a34a71',
                  'test2.zip': 'f1d5eb5ffac33ee0c2f184264cb625cd05efe246afcd898cb28db1eb4e3c9167'}


def frozen_config():
    config = json.loads(FROZEN.read_bytes())
    if config['architecture']['deployment'] != 'C3' or config['gamma'] != 0.9 or config['candidate_limit'] != 16:
        raise ValueError('Unexpected frozen P5 deployment')
    return config


def final_names(domain):
    if domain in FINAL_SEQUENCES:
        return FINAL_SEQUENCES[domain]
    file = dance.DANCE / 'official-test-seqmap.txt'
    header, *names = file.read_text(encoding='utf-8-sig').split()
    if header != 'name' or len(names) != 35 or len(set(names)) != 35 or not all(dance.NAME.fullmatch(n) for n in names):
        raise ValueError('Invalid DanceTrack official test seqmap')
    return tuple(sorted(names))


def archive_member(name, dataset, allowed):
    path = PurePosixPath(name)
    if path.is_absolute() or '..' in path.parts or '\\' in name or ':' in name:
        return None
    for offset, component in enumerate(path.parts):
        if component not in allowed:
            continue
        tail = path.parts[offset + 1:]
        if tail == ('seqinfo.ini',) or (dataset != 'DanceTrack' and tail == ('gt', 'gt.txt')):
            return path.parts[offset:]
        if len(tail) == 2 and tail[0] == 'img1' and re.fullmatch(r'\d{6,8}\.jpg', tail[1]):
            return path.parts[offset:]
    return None


class RemoteZip(io.RawIOBase):
    def __init__(self, url, size=None):
        self.url = url
        self.size = size
        self.position = self.start = 0
        self.buffer = b''
        if size is None:
            self._read_range(0, 0)

    def _read_range(self, start, end):
        headers = {'Range': f'bytes={start}-{end}', 'User-Agent': 'CGHTrack-reproducibility/1.0',
                   'Accept-Encoding': 'identity'}
        url = self.url + (('&' if '?' in self.url else '?') + f'range={start}')
        for attempt in range(8):
            try:
                with urlopen(Request(url, headers=headers), timeout=40) as response:
                    content_range = response.headers.get('Content-Range', '')
                    match = re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)', content_range)
                    if response.status != 206 or not match or tuple(map(int, match.groups()[:2])) != (start, end):
                        raise OSError(f'Unexpected archive range {response.status}: {content_range}')
                    total = int(match[3])
                    if self.size is not None and total != self.size:
                        raise OSError('Remote archive size changed')
                    data = response.read()
                    if len(data) != end - start + 1:
                        raise OSError('Incomplete archive byte range')
                    self.size = total
                    return data
            except (OSError, ValueError) as exc:
                if attempt == 7:
                    raise OSError(f'Range {start}-{end} retry limit for {self.url}') from exc
                time.sleep(min(attempt + 1, 5))

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=io.SEEK_SET):
        self.position = (0, self.position, self.size)[whence] + offset
        if self.position < 0:
            raise OSError('Negative archive offset')
        return self.position

    def read(self, size=-1):
        remaining = max(0, self.size - self.position)
        size = remaining if size < 0 else min(size, remaining)
        if not size:
            return b''
        end = self.position + size
        if self.position < self.start or end > self.start + len(self.buffer):
            limit = min(self.size, max(end, self.position + 16 * 1024**2))
            self.buffer = self._read_range(self.position, limit - 1)
            self.start = self.position
        result = self.buffer[self.position - self.start:end - self.start]
        self.position = end
        return result


def extract(domain, archives):
    allowed = set(final_names(domain))
    destination = FINAL_ROOT / domain
    decoded = set()
    for source, provenance in archives:
        with ZipFile(source) as archive:
            members = [(entry, parts) for entry in archive.infolist()
                       if not entry.is_dir() and (parts := archive_member(entry.filename, domain, allowed))]
            counts = {}
            for entry, path in sorted(members, key=lambda item: item[0].header_offset):
                if path in decoded:
                    raise ValueError(f'Duplicate official archive path {path}')
                decoded.add(path)
                out = destination.joinpath(*path)
                out.parent.mkdir(parents=True, exist_ok=True)
                if not out.is_file() or out.stat().st_size != entry.file_size or crc32(out.read_bytes()) != entry.CRC:
                    staged = out.with_suffix('.tmp')
                    staged.write_bytes(archive.read(entry))
                    staged.replace(out)
                counts[path[0]] = counts.get(path[0], 0) + 1
                if counts[path[0]] % 500 == 0:
                    print(f'{domain}/{path[0]}: {counts[path[0]]} files verified', flush=True)
            print(json.dumps({'source': provenance, 'verified_entries': counts}), flush=True)
    records = {}
    for folder, info, images in sequences(domain):
        if (folder.name, 'seqinfo.ini') not in decoded:
            raise ValueError(f'Official archive omits {folder.name}')
        expected = {(folder.name, info['imDir'], image.name) for image in images}
        if not expected <= decoded:
            raise ValueError(f'Official archive is missing imagery for {folder.name}')
        records[folder.name] = {'frames': len(images), 'seqinfo_sha256': file_sha256(folder / 'seqinfo.ini')}
        if domain != 'DanceTrack':
            if (folder.name, 'gt', 'gt.txt') not in decoded:
                raise ValueError(f'Archive is missing labels for {folder.name}')
            records[folder.name]['gt_sha256'] = file_sha256(folder / 'gt' / 'gt.txt')
    (destination / 'source-manifest.json').write_text(json.dumps(records, indent=2) + '\n', encoding='utf-8')
    print(f'{domain} prepared {len(records)} sequences, {sum(r["frames"] for r in records.values())} frames', flush=True)


def sequences(domain, development=False):
    if development:
        if domain == 'DanceTrack':
            for folder, info in dance.sequences('val'):
                yield folder, info, dance.frame_paths(folder, info)
            return
        if domain != 'MOT17':
            raise ValueError('MOT20 has no development split in this protocol')
        root, expected = mot.DATA / 'gt', mot.VAL
    else:
        root, expected = FINAL_ROOT / domain, final_names(domain)
    for name in expected:
        folder = root / name
        parser = ConfigParser()
        with (folder / 'seqinfo.ini').open() as file:
            parser.read_file(file)
        info = parser['Sequence']
        if info['name'] != name:
            raise ValueError(f'Wrong sequence metadata for {name}')
        images = sorted((folder / info['imDir']).glob(f'*{info["imExt"]}'))
        if [int(image.stem) for image in images] != list(range(1, info.getint('seqLength') + 1)):
            raise ValueError(f'Incomplete image numbering for {folder}')
        yield folder, info, images


def detector_settings(domain):
    cfg = frozen_config()
    if domain == 'DanceTrack':
        return dance.cache_config()
    return {'detector': 'D-MOT YOLOX-X', 'weights_sha256': cfg['detector_sha256']['MOT17'],
            'min_score': mot.MIN_SCORE, 'nms_iou': mot.NMS_IOU, 'input_size_hw': list(mot.SIZE),
            'batch_size': mot.BATCH, 'precision': 'float16'}


def verify_cache(path, domain, name, frame_count, development=False):
    record = json.loads(path.with_suffix('.json').read_bytes())
    settings = detector_settings(domain)
    if any(record.get(key) != value for key, value in settings.items()) or (
        record['sequence'] != name or record['frames'] != frame_count or
        record['weights_sha256'] != frozen_config()['detector_sha256']['DanceTrack' if domain == 'DanceTrack' else 'MOT17'] or
        file_sha256(path) != record['detections_sha256']
    ):
        raise ValueError(f'Cache violates frozen detector protocol: {path}')
    if not development and record.get('evaluation_domain') != domain:
        raise ValueError(f'Unexpected evaluation domain in {path}')
    return record


def detect(domain, output):
    import cv2
    import torch

    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    settings = detector_settings(domain)
    input_size = tuple(settings['input_size_hw'])
    if domain == 'DanceTrack':
        model, preproc = dance.load_detector()
        transform = lambda image: dance.normalize(image, preproc)
    else:
        path = ROOT / 'runs' / 'p5-mot-detector' / 'd-mot.pth'
        if mot.cache_weights(path)['weights_sha256'] != settings['weights_sha256']:
            raise ValueError('D-MOT checkpoint differs from frozen config')
        spec, preproc = mot.spec_and_preproc()
        model = spec.get_model()
        model.load_state_dict(torch.load(path, map_location='cpu', weights_only=True)['model'])
        model.cuda().eval().half()
        transform = lambda image: preproc(image, input_size)
    output = output / domain
    output.mkdir(parents=True, exist_ok=True)
    for folder, info, images in sequences(domain):
        name = folder.name
        path = output / f'{name}.txt'
        manifest = path.with_suffix('.json')
        if path.is_file() and manifest.is_file():
            verify_cache(path, domain, name, len(images))
            continue
        width, height = info.getint('imWidth'), info.getint('imHeight')
        digest, count = hashlib.sha256(), 0
        staged = path.with_suffix('.tmp')
        with staged.open('w', newline='') as file, torch.inference_mode():
            writer = csv.writer(file, lineterminator='\n')
            for batch in batched(images, settings['batch_size']):
                frames, ratios = [], []
                for image in batch:
                    digest.update(bytes.fromhex(file_sha256(image)))
                    pixels = load_image(image)
                    if pixels.shape[:2] != (height, width):
                        raise ValueError(f'Unexpected image dimensions: {image}')
                    transformed, ratio = transform(pixels)
                    frames.append(transformed)
                    ratios.append(ratio)
                predictions = model(torch.from_numpy(np.stack(frames)).to('cuda', torch.float16)).float()
                for image, prediction, ratio in zip(batch, predictions, ratios, strict=True):
                    count += write_boxes(writer, int(image.stem),
                                         person_boxes(prediction, ratio, width, height, settings['nms_iou']))
        staged.replace(path)
        record = {'evaluation_domain': domain, 'sequence': name, 'frames': len(images), 'count': count,
                  'frame_content_sha256': digest.hexdigest(), 'detections_sha256': file_sha256(path), **settings}
        staged = manifest.with_suffix('.tmp')
        staged.write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
        staged.replace(manifest)
        print(f'{domain}/{name}: {len(images)} frames, {count} frozen detector candidates', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=('fetch', 'detect'))
    parser.add_argument('domain', choices=('MOT17', 'MOT20', 'DanceTrack'))
    parser.add_argument('--cache', type=Path, default=ROOT / 'runs' / 'p6-detections')
    parser.add_argument('--mirror', action='store_true')
    parser.add_argument('--local-archive', type=Path)
    args = parser.parse_args()
    if args.stage == 'detect':
        detect(args.domain, args.cache)
    elif args.domain == 'DanceTrack':
        archives = []
        for name, digest in DANCE_ARCHIVES.items():
            path = dance.DANCE / name
            if file_sha256(path) != digest:
                raise ValueError(f'Mismatched publisher archive: {path}')
            archives.append((path, f'https://huggingface.co/datasets/noahcao/dancetrack/resolve/a0ba42a/{name}'))
        extract(args.domain, archives)
    else:
        url = f'https://motchallenge.net/data/{args.domain}.zip'
        size = None
        if args.mirror or args.local_archive:
            url = f'{MOT_MIRROR_ROOT}/{args.domain}-train.zip'
            size, digest = MOT_MIRRORS[args.domain]
        if args.local_archive:
            if file_sha256(args.local_archive) != digest:
                raise ValueError('Local mirror archive sha256 does not match the pinned revision')
            extract(args.domain, [(args.local_archive, url)])
        else:
            with RemoteZip(url, size) as source:
                extract(args.domain, [(source, url)])


if __name__ == '__main__':
    main()
