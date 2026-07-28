import unittest

import torch

from tfscope.config import TFScopeConfig
from tfscope.models.heads import PWMRegressionHead
from tfscope.models.retrieval import (
    compute_aligned_true_trust,
    pairwise_trust_rank_loss,
)


class RetrievalTargetTest(unittest.TestCase):
    def test_positionwise_gate_tracks_local_donor_support(self):
        config = TFScopeConfig(
            use_retrieval=True,
            retrieval_k=2,
            positionwise_retrieval_gate=True,
            pwm_cross_attn=False,
        )
        head = PWMRegressionHead(config)
        head.eval()

        x = torch.zeros(1, config.proj_hidden_dim)
        donors = torch.full((1, 2, 4, config.max_motif_length), 0.25)
        masks = torch.zeros(1, 2, config.max_motif_length)
        masks[0, 0, 0] = 1.0
        masks[0, 1, 1] = 1.0
        trust = torch.tensor([[0.9, 0.1]])

        with torch.no_grad():
            head(
                x,
                retrieved_pwms=donors,
                retrieved_masks=masks,
                retrieved_sims=trust,
                trust_scores=trust,
            )

        beta = head._last_beta_gated
        self.assertEqual(tuple(beta.shape), (1, config.max_motif_length))
        self.assertGreater(float(beta[0, 0]), float(beta[0, 1]))
        self.assertEqual(float(beta[0, 2]), 0.0)

    def test_aligned_trust_recovers_shift_and_reverse_complement(self):
        motif = torch.tensor(
            [
                [0.85, 0.05, 0.05, 0.10],
                [0.05, 0.80, 0.10, 0.10],
                [0.05, 0.10, 0.80, 0.10],
                [0.05, 0.05, 0.05, 0.70],
            ],
            dtype=torch.float32,
        )
        target = torch.full((1, 4, 8), 0.25)
        target[0, :, :4] = motif
        target_mask = torch.zeros(1, 8)
        target_mask[0, :4] = 1.0

        donors = torch.full((1, 2, 4, 8), 0.25)
        donors[0, 0, :, 2:6] = motif[[3, 2, 1, 0]].flip(-1)
        donor_masks = torch.zeros(1, 2, 8)
        donor_masks[0, 0, 2:6] = 1.0
        donor_masks[0, 1, :4] = 1.0

        trust = compute_aligned_true_trust(
            donors,
            donor_masks,
            target,
            target_mask,
            max_shift=4,
            min_overlap=4,
        )

        self.assertGreater(float(trust[0, 0]), 0.99)
        self.assertGreater(float(trust[0, 0]), float(trust[0, 1]))

    def test_pairwise_rank_loss_rewards_correct_order(self):
        target = torch.tensor([[0.9, 0.2, 0.5]])
        valid = torch.ones_like(target)

        correct = pairwise_trust_rank_loss(
            torch.tensor([[3.0, -2.0, 0.0]]),
            target,
            valid,
        )
        reversed_order = pairwise_trust_rank_loss(
            torch.tensor([[-2.0, 3.0, 0.0]]),
            target,
            valid,
        )

        self.assertLess(float(correct), float(reversed_order))


if __name__ == "__main__":
    unittest.main()
