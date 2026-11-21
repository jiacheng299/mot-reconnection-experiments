import unittest

import torch

from scripts.p5_dance_data import train_member
from scripts.p5_train import LOCAL_NEIGHBOR_COLUMNS, make_model, next_values
from scripts.p4_joint import encode_group, pack_candidates


class DanceTrainingArchiveTest(unittest.TestCase):
    def test_members_from_train_part_archives_use_canonical_path(self):
        self.assertEqual(train_member('train1/dancetrack0001/img1/00000001.jpg'),
                         ('dancetrack0001', 'img1', '00000001.jpg'))
        self.assertEqual(train_member('train/dancetrack0001/gt/gt.txt'),
                         ('dancetrack0001', 'gt', 'gt.txt'))
        self.assertIsNone(train_member('val/dancetrack0004/gt/gt.txt'))

    def test_unsafe_or_unexpected_members_are_rejected(self):
        for name in ('train/dancetrack0001/../gt/gt.txt', '/train/dancetrack0001/seqinfo.ini',
                     'train/dancetrack0001/notes.txt', 'train/dancetrack0001/img1/bad.jpg'):
            with self.subTest(name=name):
                self.assertIsNone(train_member(name))

    def test_wide_additive_control_has_similar_capacity_and_no_pair_or_mix_term(self):
        count = lambda model: sum(parameter.numel() for parameter in model.parameters())
        additive = make_model('C2-wide')
        self.assertLessEqual(abs(count(additive) - count(make_model('C3'))), 10)
        self.assertIsNone(additive.pair)
        self.assertIsNone(additive.mix)

    def test_no_neighborhood_masks_exactly_neighborhood_inputs(self):
        features = encode_group((1, 2), (1, None), {1: {1: 0.8}, 2: {1: 0.7}}, {1: 0.95},
                                {1: (1, 0, 1), 2: (1, 0, 1)})
        modified = [[values.copy() for values in part] for part in features]
        for row in modified[0]:
            for column in LOCAL_NEIGHBOR_COLUMNS:
                row[column] = -5
        model = make_model('no-neighborhood')
        result = model(pack_candidates([features, modified]))
        self.assertEqual(float(result[0].detach()), float(result[1].detach()))

    def test_next_frame_uses_online_actions_and_target_values(self):
        batch = {'candidate_groups': torch.tensor([0, 0, 1, 1]),
                 'next_group_owners': torch.tensor([0, 0]),
                 'rewards': torch.tensor([1.0, -0.5]), 'successors': None}
        online = lambda _: torch.tensor([1.0, 2.0, 1.0, 1.0])
        target = lambda _: torch.tensor([20.0, 10.0, 3.0, 4.0])
        self.assertEqual(next_values(online, target, batch).tolist(), [13.0, 0.0])
