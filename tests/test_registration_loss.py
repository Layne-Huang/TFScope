import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from tfscope.config import TFScopeConfig
from tfscope.data.dataset import TFDataset
from tfscope.losses.registration import (
    enumerate_registered_targets,
    export_registered_predictions,
    latent_registration_loss,
    registration_states,
    transform_target,
)


def asymmetric_pwm():
    return torch.tensor(
        [
            [0.85, 0.05, 0.05, 0.10],
            [0.05, 0.80, 0.10, 0.10],
            [0.05, 0.10, 0.80, 0.10],
            [0.05, 0.05, 0.05, 0.70],
        ],
        dtype=torch.float32,
    )


def padded_target(length=8):
    target = torch.full((4, length), 0.25)
    target[:, :4] = asymmetric_pwm()
    mask = torch.zeros(length)
    mask[:4] = 1.0
    return target, mask


class RegistrationLossTest(unittest.TestCase):
    def test_transform_target_applies_reverse_complement_and_offset(self):
        target, mask = padded_target()

        transformed, transformed_mask, coverage = transform_target(
            target, mask, orientation=1, offset=2
        )

        expected = asymmetric_pwm()[[3, 2, 1, 0]].flip(-1)
        self.assertTrue(torch.allclose(transformed[:, 2:6], expected))
        self.assertEqual(transformed_mask.nonzero().flatten().tolist(), [2, 3, 4, 5])
        self.assertAlmostEqual(float(coverage), 1.0)

    def test_enumeration_has_all_offset_orientation_states(self):
        target, mask = padded_target()
        targets, masks, coverage, valid = enumerate_registered_targets(
            target.unsqueeze(0),
            mask.unsqueeze(0),
            max_shift=2,
            min_overlap=4,
        )

        self.assertEqual(len(registration_states(2)), 10)
        self.assertEqual(targets.shape, (1, 10, 4, 8))
        self.assertEqual(masks.shape, (1, 10, 8))
        self.assertEqual(coverage.shape, (1, 10))
        self.assertEqual(int(valid.sum()), 6)

    def test_short_motif_uses_its_full_length_as_minimum_overlap(self):
        target, mask = padded_target()
        mask[2:] = 0.0

        _, _, _, valid = enumerate_registered_targets(
            target.unsqueeze(0),
            mask.unsqueeze(0),
            max_shift=2,
            min_overlap=4,
        )

        self.assertTrue(valid.any())

    def test_anchor_selects_labeled_state_and_backpropagates(self):
        target, mask = padded_target()
        registered, registered_mask, _ = transform_target(
            target, mask, orientation=1, offset=2
        )
        pred_pwm = registered.clamp_min(1e-5).log().unsqueeze(0).requires_grad_()
        pred_gate = torch.where(
            registered_mask > 0,
            torch.tensor(8.0),
            torch.tensor(-8.0),
        ).unsqueeze(0).requires_grad_()
        register_logits = torch.zeros(1, 10, requires_grad=True)

        loss, metrics, expected_target, expected_mask = latent_registration_loss(
            pred_gate,
            pred_pwm,
            target.unsqueeze(0),
            mask.unsqueeze(0),
            max_shift=2,
            min_overlap=4,
            temperature=0.1,
            coverage_penalty=0.5,
            gate_weight=1.0,
            l1_weight=1.0,
            ic_weight=0.5,
            ic_pcc_weight=0.5,
            topbase_weight=0.1,
            topbase_margin=2.0,
            topbase_ic_thresh_bits=0.5,
            anchor_mask=torch.tensor([1.0]),
            anchor_orientation=torch.tensor([1]),
            anchor_offset=torch.tensor([2]),
            register_logits=register_logits,
            register_loss_weight=0.5,
        )
        loss.backward()

        self.assertLess(float(metrics["registration_loss"]), 0.01)
        self.assertGreater(float(metrics["register_supervision"]), 0.0)
        self.assertTrue(torch.allclose(expected_target[0], registered))
        self.assertTrue(torch.equal(expected_mask[0], registered_mask))
        self.assertTrue(torch.isfinite(pred_pwm.grad).all())
        self.assertTrue(torch.isfinite(pred_gate.grad).all())
        self.assertTrue(torch.isfinite(register_logits.grad).all())
        self.assertAlmostEqual(float(metrics["registration_anchor_fraction"]), 1.0)

    def test_orientation_anchor_allows_every_offset_in_one_orientation(self):
        target, mask = padded_target()
        pred_pwm = target.clamp_min(1e-5).log().unsqueeze(0).requires_grad_()
        pred_gate = torch.where(
            mask > 0,
            torch.tensor(8.0),
            torch.tensor(-8.0),
        ).unsqueeze(0).requires_grad_()
        register_logits = torch.zeros(1, 10, requires_grad=True)

        loss, metrics, _, _ = latent_registration_loss(
            pred_gate,
            pred_pwm,
            target.unsqueeze(0),
            mask.unsqueeze(0),
            max_shift=2,
            min_overlap=4,
            temperature=0.1,
            coverage_penalty=0.5,
            gate_weight=1.0,
            l1_weight=1.0,
            ic_weight=0.5,
            ic_pcc_weight=0.5,
            topbase_weight=0.1,
            topbase_margin=2.0,
            topbase_ic_thresh_bits=0.5,
            anchor_mask=torch.tensor([1.0]),
            anchor_mode=torch.tensor([1]),
            anchor_orientation=torch.tensor([0]),
            anchor_offset=torch.tensor([0]),
            register_logits=register_logits,
            register_loss_weight=0.5,
        )
        loss.backward()

        self.assertTrue(torch.isfinite(loss))
        self.assertAlmostEqual(
            float(metrics["register_supervision"]),
            float(torch.log(torch.tensor(2.0))),
            places=5,
        )
        self.assertTrue(torch.isfinite(register_logits.grad).all())

    def test_register_export_inverts_structural_frame(self):
        target, mask = padded_target()
        registered, registered_mask, _ = transform_target(
            target, mask, orientation=1, offset=2
        )
        internal_logits = registered.clamp_min(1e-8).log().unsqueeze(0)
        internal_gate = torch.where(
            registered_mask > 0,
            torch.tensor(8.0),
            torch.tensor(-8.0),
        ).unsqueeze(0)
        register_logits = torch.full((1, 10), -10.0)
        # max_shift=2: RC offset +2 is state 9.
        register_logits[0, 9] = 10.0

        export_gate, export_pwm = export_registered_predictions(
            internal_gate, internal_logits, register_logits, max_shift=2
        )

        self.assertTrue(
            torch.allclose(
                export_pwm.softmax(dim=1)[0, :, :4],
                asymmetric_pwm(),
                atol=1e-6,
            )
        )
        self.assertEqual(
            (export_gate[0].sigmoid() > 0.5).nonzero().flatten().tolist(),
            [0, 1, 2, 3],
        )

    def test_dataset_applies_anchors_to_train_only(self):
        target = asymmetric_pwm().numpy()
        rows = []
        for filename, gene in (("train.txt", "TRAIN"), ("val.txt", "VAL")):
            rows.append(
                {
                    "tf_name": gene,
                    "uniprot_id": f"U_{gene}",
                    "gene_symbol": gene,
                    "organism": "test",
                    "sequence": "ACDEFGHIK",
                    "seq_length": 9,
                    "dbd_start": 0,
                    "dbd_end": 9,
                    "dbd_count": 1,
                    "family_id": 0,
                    "family_name": "Test",
                    "motif_length": 4,
                    "pwm": target.astype(np.float32).tobytes(),
                    "source": "test",
                    "source_id": f"S_{gene}",
                    "assay_type": "",
                    "quality_grade": "",
                    "filename": filename,
                    "origin": "test",
                }
            )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data_path = root / "data.parquet"
            split_path = root / "split.json"
            anchor_path = root / "anchors.tsv"
            pd.DataFrame(rows).to_parquet(data_path)
            split_path.write_text(
                json.dumps(
                    {"train": ["train.txt"], "val": ["val.txt"], "test": []}
                )
            )
            pd.DataFrame(
                [
                    {
                        "filename": "train.txt",
                        "split": "train",
                        "orientation_to_reference": "rc",
                        "offset_to_reference": 1,
                    }
                ]
            ).to_csv(anchor_path, sep="\t", index=False)
            config = TFScopeConfig(
                latent_registration=True,
                registration_anchor_path=str(anchor_path),
            )

            train = TFDataset(
                config, str(data_path), str(split_path), split="train"
            )[0]
            validation = TFDataset(
                config, str(data_path), str(split_path), split="val"
            )[0]

        self.assertEqual(float(train["registration_anchor_mask"]), 1.0)
        self.assertEqual(int(train["registration_anchor_mode"]), 2)
        self.assertEqual(int(train["registration_orientation"]), 1)
        self.assertEqual(int(train["registration_offset"]), 1)
        self.assertEqual(float(validation["registration_anchor_mask"]), 0.0)
        self.assertEqual(int(validation["registration_anchor_mode"]), 0)


if __name__ == "__main__":
    unittest.main()
