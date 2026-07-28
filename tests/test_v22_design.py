import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from eval_full_metrics import panel_full
from tfscope.config import TFScopeConfig
from tfscope.data.dataset import TFDataset
from tfscope.losses.tfscope_loss import TFScopeLoss
from tfscope.models.heads import PositionGateHead


class SpanGateTest(unittest.TestCase):
    def test_span_gate_is_contiguous_and_differentiable(self):
        cfg = TFScopeConfig(
            proj_hidden_dim=16,
            max_motif_length=30,
            min_motif_length=4,
            gate_mode="span",
        )
        head = PositionGateHead(cfg)
        x = torch.randn(8, 16, requires_grad=True)
        logits = head(x)
        hard = logits.sigmoid() > 0.5
        transitions = (hard[:, 1:] != hard[:, :-1]).sum(dim=1)
        self.assertTrue(torch.all(transitions <= 2))
        self.assertEqual(logits.shape, (8, 30))
        logits.sum().backward()
        self.assertIsNotNone(x.grad)
        self.assertTrue(torch.isfinite(x.grad).all())


class CoverageAlignedLossTest(unittest.TestCase):
    def test_missing_columns_are_penalized(self):
        target = torch.tensor(
            [[[0.8] * 8, [0.1] * 8, [0.05] * 8, [0.05] * 8]],
            dtype=torch.float32,
        )
        pred_logits = target.clamp_min(1e-6).log()
        mask = torch.ones(1, 8)
        full_gate = torch.full((1, 8), 8.0)
        half_gate = torch.tensor([[8.0] * 4 + [-8.0] * 4])
        full = TFScopeLoss._pwm_coverage_r(
            F.log_softmax(pred_logits, dim=1), full_gate, target, mask, 0.25
        )
        half = TFScopeLoss._pwm_coverage_r(
            F.log_softmax(pred_logits, dim=1), half_gate, target, mask, 0.25
        )
        self.assertLess(full.item(), half.item())
        self.assertAlmostEqual(half.item(), 0.5, places=2)

    def test_reported_r_cov_pays_for_missing_columns(self):
        core = np.array(
            [[0.8] * 8, [0.1] * 8, [0.05] * 8, [0.05] * 8],
            dtype=np.float32,
        )
        aligned = np.full_like(core, 0.25)
        aligned[:, :4] = core[:, :4]
        result = panel_full(core, aligned, np.arange(4), pred_ncols=4)
        self.assertAlmostEqual(result["r_overlap"], 1.0, places=5)
        self.assertAlmostEqual(result["coverage"], 0.5, places=5)
        self.assertAlmostEqual(result["r_cov"], 0.5, places=5)


class DatasetPolicyTest(unittest.TestCase):
    def _table(self, path: Path):
        pwm = np.full((4, 7), 0.25, dtype=np.float32)
        rows = []
        for i, eligible in enumerate((False, True)):
            rows.append(
                {
                    "filename": f"x{i}",
                    "gene_symbol": f"G{i}",
                    "sequence": "ACDEFG",
                    "partner_sequence": "HIK",
                    "multichain_eligible": eligible,
                    "pwm": pwm.tobytes(),
                    "family_id": 9,
                    "dbd_start": 0,
                    "dbd_end": 6,
                    "seq_length": 6,
                    "group_id": f"group{i}",
                }
            )
        pd.DataFrame(rows).to_parquet(path)

    def test_overflow_error_and_controlled_multichain(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "data.parquet"
            self._table(path)
            strict = TFScopeConfig(
                max_motif_length=6, motif_overflow_policy="error"
            )
            with self.assertRaisesRegex(ValueError, "exceed"):
                TFDataset(strict, str(path))

            cfg = TFScopeConfig(
                max_motif_length=7,
                two_chain_input=True,
                require_multichain_eligible=True,
            )
            ds = TFDataset(cfg, str(path))
            self.assertNotIn(2, ds[0]["sequence_tokens"].tolist())
            self.assertIn(2, ds[1]["sequence_tokens"].tolist())


if __name__ == "__main__":
    unittest.main()
