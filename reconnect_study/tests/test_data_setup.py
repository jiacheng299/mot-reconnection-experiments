import hashlib
import json
from pathlib import Path
import tempfile
import unittest


class DataSetupTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.sequence = self.root / 'val' / 'dancetrack0001'
        (self.sequence / 'gt').mkdir(parents=True)
        (self.sequence / 'gt' / 'gt.txt').write_text('1,1,5,6,7,8,1,1,1\n')
        (self.sequence / 'seqinfo.ini').write_text(
            '[Sequence]\nname=dancetrack0001\nframeRate=20\nseqLength=2\n'
        )
        self.cache = self.root / 'cache'
        self.cache.mkdir()
        self.detection = self.cache / 'dancetrack0001.txt'
        self.detection.write_text('1,-1,5,6,7,8,0.9\n')
        self.detection.with_suffix('.json').write_text(json.dumps({
            'sequence': 'dancetrack0001', 'frames': 2, 'count': 1,
            'detector': 'YOLOX-X COCO', 'weights_sha256': '5652330b6ae860043f091b8f550a60c10e1129f416edfdb65c259be6caf355cf',
            'detections_sha256': hashlib.sha256(self.detection.read_bytes()).hexdigest(),
            'commit': '6ddff4824372906469a7fae2dc3206c7aa4bbaee', 'min_score': 0.01,
            'nms_iou': 0.65, 'input_size_hw': [640, 640], 'precision': 'float16', 'batch_size': 8,
        }))

    def test_manifest_verifies_cache_and_records_split(self):
        from reconnect_study.data_setup import dance_manifest

        manifest = dance_manifest(self.root, [self.cache], splits=['val'])
        row, = manifest['sequences']
        self.assertEqual(row['name'], 'dancetrack0001')
        self.assertEqual(row['source_id'], 'dancetrack0001')
        self.assertEqual(row['frames'], 2)
        self.assertEqual(row['fps'], 20)
        self.assertEqual(row['split'], 'val')
        self.assertEqual(row['detections'], str(self.detection.resolve()))
        self.assertFalse(manifest['source_groups_verified'])

    def test_manifest_rejects_checksum_mismatch(self):
        from reconnect_study.data_setup import dance_manifest

        self.detection.write_text('1,-1,5,6,7,8,0.8\n')
        with self.assertRaisesRegex(ValueError, 'checksum'):
            dance_manifest(self.root, [self.cache], splits=['val'])

    def test_manifest_rejects_incompatible_detector_precision(self):
        from reconnect_study.data_setup import dance_manifest

        sidecar = self.detection.with_suffix('.json')
        metadata = json.loads(sidecar.read_bytes())
        metadata['precision'] = 'float32'
        sidecar.write_text(json.dumps(metadata))
        with self.assertRaisesRegex(ValueError, 'Inconsistent detector metadata'):
            dance_manifest(self.root, [self.cache], splits=['val'])

    def test_partial_manifest_reports_unavailable_cache(self):
        from reconnect_study.data_setup import dance_manifest

        self.detection.with_suffix('.json').unlink()
        result = dance_manifest(self.root, [self.cache], splits=['val'], partial=True)
        self.assertEqual(result['missing_sequences'], ['dancetrack0001'])
        self.assertEqual(result['sequences'], [])

    def test_personpath_uses_original_keyframes_and_preserves_ignore_classes(self):
        from reconnect_study.data_setup import personpath_rows

        entities = [
            {'id': 3, 'bb': [1, 2, 3, 4], 'labels': {'person': 1}, 'blob': {'frame_idx': 0}},
            {'id': 3, 'bb': [2, 3, 3, 4], 'labels': {'person': 1, 'fully_occluded': 1},
             'blob': {'frame_idx': 5}},
            {'id': 7, 'bb': [5, 6, 3, 4], 'labels': {'crowd': 1}, 'blob': {'frame_idx': 5}},
        ]
        detections = ['1,-1,1,2,3,4,.8,-1,-1,-1', '2,-1,1,2,3,4,.7,-1,-1,-1',
                      '6,-1,2,3,3,4,.6,-1,-1,-1']
        frames, truth, boxes = personpath_rows(entities, detections, 6)
        self.assertEqual(frames, [1, 6])
        self.assertEqual([(row[0], row[7]) for row in truth], [(1, 1), (2, 8), (2, 13)])
        self.assertEqual([(row[0], row[6]) for row in boxes], [(1, '.8'), (2, '.6')])

    def test_bdd_labels_share_mot_origin_with_detector_export(self):
        from reconnect_study.data_setup import bdd_rows

        box = {'x1': 0, 'y1': 2, 'x2': 10, 'y2': 22}
        rows = bdd_rows([{'index': 0, 'name': 'sample-0000001.jpg', 'labels': [
            {'id': '43', 'category': 'pedestrian', 'box2d': box},
            {'id': '44', 'category': 'rider', 'box2d': box},
            {'id': '40', 'category': 'pedestrian', 'box2d': box, 'attributes': {'Crowd': True}},
            {'id': '42', 'category': 'car', 'box2d': box},
        ]}], 'sample')
        self.assertEqual([(row[0], row[2:6], row[7]) for row in rows], [
            (1, (1, 3, 10, 20), 1), (1, (1, 3, 10, 20), 8), (1, (1, 3, 10, 20), 13),
        ])
