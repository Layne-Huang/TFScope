import unittest

from build_v19_e5_structural_anchors import npz_key


class StructuralAnchorTest(unittest.TestCase):
    def test_npz_key_parses_jaspar(self):
        self.assertEqual(
            npz_key("1cf7_A_MA0470.2.jaspar.npz"),
            ("1cf7", "A", "MA0470.2"),
        )

    def test_npz_key_parses_hocomoco(self):
        self.assertEqual(
            npz_key("6zmn_A_SMAD3_HUMAN.H11MO.0.B.npz"),
            ("6zmn", "A", "SMAD3_HUMAN.H11MO.0.B"),
        )

    def test_npz_key_rejects_unmapped_names(self):
        self.assertIsNone(npz_key("unknown_entry.npz"))


if __name__ == "__main__":
    unittest.main()
