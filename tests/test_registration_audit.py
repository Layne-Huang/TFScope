import unittest

import numpy as np
import pandas as pd

from tfscope.data.registration_audit import (
    AuditThresholds,
    align_pair,
    build_pair_rows,
    deduplicate_motifs,
    revcomp_pwm_np,
    summarize_gene,
)


def motif():
    return np.array(
        [
            [0.85, 0.05, 0.05, 0.10, 0.70, 0.10],
            [0.05, 0.80, 0.10, 0.10, 0.10, 0.70],
            [0.05, 0.10, 0.80, 0.10, 0.10, 0.10],
            [0.05, 0.05, 0.05, 0.70, 0.10, 0.10],
        ],
        dtype=np.float32,
    )


def row(filename, pwm, source_id="S1"):
    return {
        "filename": filename,
        "gene_symbol": "GENE",
        "family_name": "Family",
        "source": "Source",
        "source_id": source_id,
        "split": "train",
        "pwm": pwm.astype(np.float32).tobytes(),
    }


class RegistrationAuditTest(unittest.TestCase):
    def test_align_pair_recovers_offset(self):
        reference = motif()
        query = reference[:, 1:5]

        alignment = align_pair(reference, query, max_shift=3, min_overlap=4)

        self.assertEqual(alignment["orientation"], "fwd")
        self.assertEqual(alignment["shift"], 1)
        self.assertEqual(alignment["overlap"], 4)
        self.assertAlmostEqual(alignment["aligned_r"], 1.0, places=6)

    def test_align_pair_recovers_reverse_complement(self):
        reference = motif()
        query = revcomp_pwm_np(reference)

        alignment = align_pair(reference, query, max_shift=2, min_overlap=4)

        self.assertEqual(alignment["orientation"], "rc")
        self.assertEqual(alignment["shift"], 0)
        self.assertAlmostEqual(alignment["aligned_r"], 1.0, places=6)

    def test_deduplicate_motifs_collapses_repeated_structure_rows(self):
        pwm = motif()
        df = pd.DataFrame(
            [
                row("a.txt", pwm),
                row("b.txt", pwm),
                row("c.txt", pwm, source_id="S2"),
            ]
        )

        unique, report = deduplicate_motifs(df)

        self.assertEqual(len(unique), 2)
        self.assertEqual(report["collapsed_duplicate_rows"], 1)
        repeated = unique.loc[unique["source_id"] == "S1"].iloc[0]
        self.assertEqual(repeated["duplicate_row_count"], 2)

    def test_registration_disagreement_produces_relative_anchors(self):
        pwm = motif()
        df = pd.DataFrame(
            [
                row("forward.txt", pwm, source_id="S1"),
                row("reverse.txt", revcomp_pwm_np(pwm), source_id="S2"),
            ]
        )
        unique, _ = deduplicate_motifs(df)
        thresholds = AuditThresholds()
        pairs = build_pair_rows(unique, thresholds)

        summary, anchors = summarize_gene(unique, pairs, thresholds)

        self.assertEqual(summary["classification"], "registration_discordant")
        self.assertEqual(summary["n_relative_anchors"], 2)
        self.assertEqual(len(anchors), 2)
        self.assertTrue(
            any(anchor["orientation_to_reference"] == "rc" for anchor in anchors)
        )
        self.assertTrue(
            all(not anchor["absolute_orientation_resolved"] for anchor in anchors)
        )
        self.assertTrue(all(anchor["split"] == "train" for anchor in anchors))


if __name__ == "__main__":
    unittest.main()
