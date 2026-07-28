import unittest

import numpy as np
import torch

from tfscope.models.alignment import align_batch_torch, align_pwm


class AlignmentTest(unittest.TestCase):
    def test_batch_alignment_reverse_complements_only_valid_span(self):
        motif = torch.tensor(
            [
                [0.85, 0.70, 0.05, 0.10],
                [0.05, 0.10, 0.80, 0.10],
                [0.05, 0.10, 0.10, 0.70],
                [0.05, 0.10, 0.05, 0.10],
            ]
        )
        reference = torch.full((1, 4, 8), 0.25)
        reference[0, :, 2:6] = motif
        ref_mask = torch.zeros(1, 8)
        ref_mask[0, 2:6] = 1.0

        neighbor = torch.full((1, 3, 4, 8), 0.25)
        neighbor[0, :, :, :4] = motif[[3, 2, 1, 0]].flip(-1)
        neighbor_mask = torch.zeros(1, 3, 8)
        neighbor_mask[0, :, :4] = 1.0

        aligned, aligned_mask = align_batch_torch(
            neighbor,
            neighbor_mask,
            reference,
            ref_mask,
            max_shift=4,
            min_overlap=4,
            return_masks=True,
        )

        self.assertTrue(torch.allclose(aligned[0, :, :, 2:6], motif.expand(3, -1, -1)))
        self.assertTrue(torch.equal(aligned_mask[0, 0], ref_mask[0]))

    def test_min_overlap_rejects_short_alignment(self):
        reference = np.array(
            [
                [0.9, 0.1, 0.1, 0.9],
                [0.05, 0.8, 0.05, 0.05],
                [0.03, 0.05, 0.8, 0.03],
                [0.02, 0.05, 0.05, 0.02],
            ],
            dtype=np.float32,
        )
        neighbor = reference[:, :2]

        _, _, _, score = align_pwm(
            neighbor,
            reference,
            max_shift=2,
            consider_revcomp=False,
            min_overlap=4,
        )

        self.assertEqual(score, -2.0)

    def test_min_overlap_accepts_full_alignment(self):
        reference = np.array(
            [
                [0.9, 0.1, 0.1, 0.9],
                [0.05, 0.8, 0.05, 0.05],
                [0.03, 0.05, 0.8, 0.03],
                [0.02, 0.05, 0.05, 0.02],
            ],
            dtype=np.float32,
        )

        _, shift, orientation, score = align_pwm(
            reference,
            reference,
            max_shift=2,
            consider_revcomp=True,
            min_overlap=4,
        )

        self.assertEqual(shift, 0)
        self.assertEqual(orientation, "fwd")
        self.assertAlmostEqual(score, 1.0)


if __name__ == "__main__":
    unittest.main()
