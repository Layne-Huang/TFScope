"""Numerically verify Candidate A (ChainSetEncoder) is permutation-equivariant.

Plan §4.7 / §7.7: chain permutations must change predictions by a negligible,
documented tolerance. Here we test the architectural guarantee directly on the
pooled (permutation-invariant) read-out and on the per-residue states matched by
membership.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
from tfscope.config import TFScopeConfig
from tfscope.models.chain_set_encoder import ChainSetEncoder

TOL = 1e-5


def _make_complex(seed=0):
    """Two chains: chain 0 has 6 residues, chain 1 has 4 residues, padded to L=16."""
    torch.manual_seed(seed)
    d_in = 32
    cfg = TFScopeConfig(esm_embed_dim=d_in, max_chains=4)
    cfg.chain_set_dim = 48
    cfg.chain_set_heads = 4
    cfg.chain_set_layers = 2
    enc = ChainSetEncoder(cfg, in_dim=d_in).eval()

    L = 16
    feats = torch.randn(1, L, d_in)
    chain = torch.zeros(1, L, dtype=torch.long)
    valid = torch.zeros(1, L, dtype=torch.bool)
    # chain 0: residues 0..5 ; chain 1: residues 6..9 ; rest padding
    chain[0, 0:6] = 0
    chain[0, 6:10] = 1
    valid[0, 0:10] = True
    return enc, feats, chain, valid


def _permute_chains(feats, chain, valid):
    """Reorder residues so chain 1's residues come first, relabel chain ids.

    Biologically identical complex, different chain ordering + residue layout.
    """
    f = feats.clone(); c = chain.clone(); v = valid.clone()
    # new layout: chain1(4) then chain0(6), padding after
    new_f = torch.zeros_like(f)
    new_c = torch.zeros_like(c)
    new_v = torch.zeros_like(v)
    # chain1 -> positions 0..3, relabelled to id 0
    new_f[0, 0:4] = feats[0, 6:10]; new_c[0, 0:4] = 0; new_v[0, 0:4] = True
    # chain0 -> positions 4..9, relabelled to id 1
    new_f[0, 4:10] = feats[0, 0:6]; new_c[0, 4:10] = 1; new_v[0, 4:10] = True
    return new_f, new_c, new_v


def test_pooled_summary_is_permutation_invariant():
    enc, feats, chain, valid = _make_complex()
    with torch.no_grad():
        _, pooled_a = enc(feats, chain, valid)
        f2, c2, v2 = _permute_chains(feats, chain, valid)
        _, pooled_b = enc(f2, c2, v2)
    max_dev = (pooled_a - pooled_b).abs().max().item()
    assert max_dev < TOL, f"pooled summary not permutation-invariant: dev={max_dev:.2e}"


def test_residue_states_equivariant_by_membership():
    """A residue's refined state must be independent of chain ordering."""
    enc, feats, chain, valid = _make_complex()
    with torch.no_grad():
        h_a, _ = enc(feats, chain, valid)
        f2, c2, v2 = _permute_chains(feats, chain, valid)
        h_b, _ = enc(f2, c2, v2)
    # chain1 residue #0 lives at index 6 (orig) and index 0 (permuted)
    dev1 = (h_a[0, 6] - h_b[0, 0]).abs().max().item()
    # chain0 residue #0 lives at index 0 (orig) and index 4 (permuted)
    dev0 = (h_a[0, 0] - h_b[0, 4]).abs().max().item()
    assert dev1 < TOL and dev0 < TOL, f"residue states not equivariant: {dev0:.2e}, {dev1:.2e}"


def test_homomer_chains_are_symmetric():
    """Two identical chains → identical refined residue states (no chain-id bias)."""
    enc, feats, chain, valid = _make_complex()
    # make chain 1 an exact copy of chain 0's first 4 residues
    feats = feats.clone()
    feats[0, 6:10] = feats[0, 0:4]
    with torch.no_grad():
        h, _ = enc(feats, chain, valid)
    # residue i of chain0 (0..3) vs residue i of chain1 (6..9) should match,
    # because chains are identical multisets and no positional/id feature breaks it.
    # (chain0 has 6 residues vs chain1's 4, so intra-chain context differs; instead
    #  test the degenerate true-homomer case: restrict both chains to 4 residues.)
    valid2 = valid.clone(); valid2[0, 4:6] = False  # chain0 now residues 0..3 only
    with torch.no_grad():
        h2, _ = enc(feats, chain, valid2)
    dev = (h2[0, 0:4] - h2[0, 6:10]).abs().max().item()
    assert dev < TOL, f"homomer chains not symmetric: dev={dev:.2e}"


if __name__ == "__main__":
    test_pooled_summary_is_permutation_invariant()
    test_residue_states_equivariant_by_membership()
    test_homomer_chains_are_symmetric()
    print("All ChainSetEncoder equivariance tests passed.")
