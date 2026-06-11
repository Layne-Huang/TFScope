import unittest

import numpy as np
import pandas as pd

from build_nn_index import build_index
from tfscope.data.dataset import GeneBalancedSampler
from tfscope.data.split_hygiene import (
    assign_groups,
    audit_split,
    build_group_ids,
    donor_exclusion_reasons,
    sequence_hash,
    specific_source_ids,
)


class SplitHygieneTest(unittest.TestCase):
    def setUp(self) -> None:
        self.df = pd.DataFrame(
            [
                {
                    "filename": "a1",
                    "gene_symbol": "A",
                    "uniprot_id": "U1",
                    "source_id": "S1",
                    "sequence": "AAAA",
                    "family_id": 1,
                    "family_name": "F1",
                },
                {
                    "filename": "a2",
                    "gene_symbol": "A",
                    "uniprot_id": "U2",
                    "source_id": "S2",
                    "sequence": "AAAT",
                    "family_id": 1,
                    "family_name": "F1",
                },
                {
                    "filename": "b1",
                    "gene_symbol": "B",
                    "uniprot_id": "U3",
                    "source_id": "S2",
                    "sequence": "CCCC",
                    "family_id": 1,
                    "family_name": "F1",
                },
                {
                    "filename": "c1",
                    "gene_symbol": "C",
                    "uniprot_id": "U4",
                    "source_id": "S4",
                    "sequence": "CCCC",
                    "family_id": 2,
                    "family_name": "F2",
                },
                {
                    "filename": "d1",
                    "gene_symbol": "D",
                    "uniprot_id": "U5",
                    "source_id": "S5",
                    "sequence": "GGGG",
                    "family_id": 2,
                    "family_name": "F2",
                },
            ]
        )

    def test_group_ids_are_transitive_across_identity_fields(self) -> None:
        groups = build_group_ids(
            self.df,
            cluster_by_uniprot={"U1": 1, "U5": 1},
        )

        self.assertEqual(len(set(groups)), 1)

    def test_missing_sequences_do_not_form_an_identity(self) -> None:
        self.assertEqual(sequence_hash(None), "")
        missing = self.df.iloc[[0, 4]].copy()
        missing["sequence"] = None
        groups = build_group_ids(missing, cluster_by_uniprot={})

        self.assertEqual(len(set(groups)), 2)

    def test_assigned_groups_have_no_identity_leakage(self) -> None:
        groups = build_group_ids(self.df, cluster_by_uniprot={})
        assignments = assign_groups(
            self.df, groups, val_frac=0.2, test_frac=0.2, seed=7
        )
        split = {"train": [], "val": [], "test": []}
        for row_index, filename in enumerate(self.df["filename"]):
            split[assignments[int(groups.iloc[row_index])]].append(filename)
        report = audit_split(self.df, split)

        self.assertTrue(report.clean)

    def test_audit_detects_gene_leakage(self) -> None:
        split = {
            "train": ["a1", "b1", "c1"],
            "val": ["d1"],
            "test": ["a2"],
        }
        report = audit_split(self.df, split)

        self.assertFalse(report.clean)
        self.assertEqual(report.overlaps["gene_symbol"]["train-test"], ["A"])

    def test_retrieval_uses_train_donors_and_excludes_related_records(self) -> None:
        split = {
            "train": ["a1", "b1", "c1"],
            "val": ["d1"],
            "test": ["a2"],
        }
        embeddings = {
            name: np.array([float(i + 1), 1.0], dtype=np.float32)
            for i, name in enumerate(["a1", "a2", "b1", "c1", "d1"])
        }

        index, manifest = build_index(
            embeddings,
            self.df,
            split,
            donor_splits=["train"],
            k=10,
            family_restricted=False,
        )

        donors = [entry["nn_filename"] for entry in index["a2"]]
        self.assertNotIn("a1", donors)  # Same gene.
        self.assertNotIn("b1", donors)  # Same source.
        self.assertIn("c1", donors)
        self.assertNotIn("d1", donors)  # Validation is not a donor split.
        self.assertEqual(manifest["n_donor_records"], 3)
        for donor in donors:
            donor_row = self.df.set_index("filename").loc[donor]
            query_row = self.df.set_index("filename").loc["a2"]
            self.assertEqual(
                donor_exclusion_reasons(
                    query_row,
                    donor_row,
                    identity_source_ids=specific_source_ids(self.df),
                ),
                [],
            )

    def test_gene_balanced_sampler_equalizes_gene_counts(self) -> None:
        genes = ["A", "A", "A", "B", "C"]
        sampler = GeneBalancedSampler(genes, num_samples=8, seed=11)
        indices = list(sampler)
        sampled_genes = [genes[index] for index in indices]
        counts = {gene: sampled_genes.count(gene) for gene in set(genes)}

        self.assertEqual(len(indices), 8)
        self.assertLessEqual(max(counts.values()) - min(counts.values()), 1)
        self.assertEqual(list(GeneBalancedSampler(genes, 8, 11)), indices)


if __name__ == "__main__":
    unittest.main()
