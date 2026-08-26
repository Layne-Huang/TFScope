#!/usr/bin/env python
"""Phase-4 acceptance tests. Run on CPU with a tiny stub encoder (no ESM download).

Enforces, from the brief's acceptance criteria:
  * no model tensor contains family ID or database/source ID
  * every protein chain is encoded separately by ESM
  * flank and partner context enter through GATED residuals
  * DBD-only remains a stable path when flank/partner context is enabled
  * partner aggregation is permutation-invariant
"""
from __future__ import annotations
import sys, torch, torch.nn as nn
sys.path.insert(0, "src")

from tfscope.v26.config import V26Config
from tfscope.v26.model import TFScopeV26, assert_no_metadata_inputs


def tiny_cfg(**kw):
    c = V26Config(d_model=64, refiner_layers=1, refiner_heads=4, expert_hidden=32,
                  n_routed_experts=4, n_shared_experts=1, pwm_attn_heads=2,
                  max_motif_length=12, min_motif_length=4, dropout=0.0)
    for k, v in kw.items():
        setattr(c, k, v)
    return c


class StubESM(nn.Module):
    """Deterministic stand-in for ESM-2 so tests need no 2.5 GB download or GPU.

    Crucially it is applied ROW-WISE with no cross-row mixing, which is exactly the property
    the real per-chain encoder must have; a concatenating encoder would leak across rows and
    test_chains_encoded_independently would fail.
    """
    num_layers = 4
    def __init__(self, d): super().__init__(); self.emb = nn.Embedding(33, d); self.d = d
    def forward(self, tokens, repr_layers=None, return_contacts=False):
        h = self.emb(tokens)
        return {"representations": {l: h for l in (repr_layers or [4])}}


def build(cfg):
    m = TFScopeV26(cfg)
    m.encoder._esm = StubESM(cfg.esm_embed_dim)
    m.eval()
    return m


def make_batch(n_examples=2, n_partners=2, L=20, dbd=(4, 14)):
    rows, cidx, prim = [], [], []
    for b in range(n_examples):
        for k in range(1 + n_partners):
            rows.append(torch.randint(4, 24, (L,)))
            cidx.append(b); prim.append(k == 0)
    tok = torch.stack(rows)
    dm = torch.zeros_like(tok, dtype=torch.bool); dm[:, dbd[0]:dbd[1]] = True
    return tok, dm, torch.tensor(cidx), torch.tensor(prim)


def test_no_metadata_inputs_guard():
    for bad in ["family_id", "motif_source", "pdb_id", "gene_symbol", "has_structure"]:
        try:
            assert_no_metadata_inputs({"sequence_tokens": 1, bad: 2})
        except ValueError:
            continue
        raise AssertionError(f"guard failed to reject {bad}")
    assert_no_metadata_inputs({"sequence_tokens": 1, "dbd_mask": 2})   # must not raise


def test_forward_signature_has_no_metadata_parameter():
    import inspect
    params = set(inspect.signature(TFScopeV26.forward).parameters)
    from tfscope.v26.model import BANNED_KEYS
    assert not (params & BANNED_KEYS), f"forward accepts metadata: {params & BANNED_KEYS}"


def test_chains_encoded_independently():
    """Perturbing a PARTNER chain must not change the primary chain's residue states."""
    m = build(tiny_cfg())
    tok, dm, ci, pr = make_batch(n_examples=1, n_partners=1)
    with torch.no_grad():
        h1, _ = m.encoder(tok, dm)
        tok2 = tok.clone(); tok2[1] = torch.randint(4, 24, (tok.shape[1],))
        h2, _ = m.encoder(tok2, dm)
    assert torch.allclose(h1[0], h2[0], atol=1e-6), \
        "primary chain state changed when a partner changed -> chains are NOT independent"


def test_partner_permutation_invariance():
    m = build(tiny_cfg(use_partners=True))
    tok, dm, ci, pr = make_batch(n_examples=1, n_partners=2)
    with torch.no_grad():
        p1, g1, _ = m(tok, dm, ci, pr)
        # swap the two partner rows (indices 1 and 2)
        perm = torch.tensor([0, 2, 1])
        p2, g2, _ = m(tok[perm], dm[perm], ci[perm], pr[perm])
    assert torch.allclose(p1, p2, atol=1e-5), "partner order changed the prediction"
    assert torch.allclose(g1, g2, atol=1e-5), "partner order changed the gate"


def test_flank_gate_starts_closed():
    """At init, enabling flanks must reproduce the DBD-only prediction."""
    m = build(tiny_cfg(use_flank=True))
    tok, dm, ci, pr = make_batch(n_examples=2, n_partners=0)
    with torch.no_grad():
        p_off, _, a_off = m(tok, dm, ci, pr, use_flank=False)
        p_on, _, a_on = m(tok, dm, ci, pr, use_flank=True)
    alpha = float(a_on["alpha_flank"].max())
    assert alpha < 0.12, f"flank gate opens too far at init (alpha={alpha:.3f})"
    d = (p_on - p_off).abs().max().item()
    assert d < 0.05, f"flanked prediction deviates from DBD-only at init (max diff {d:.4f})"


def test_partner_gate_starts_closed():
    m = build(tiny_cfg(use_partners=True))
    tok, dm, ci, pr = make_batch(n_examples=2, n_partners=2)
    with torch.no_grad():
        p_off, _, _ = m(tok, dm, ci, pr, use_partners=False)
        p_on, _, a = m(tok, dm, ci, pr, use_partners=True)
    assert float(a["beta_partner"].max()) < 0.12
    assert (p_on - p_off).abs().max().item() < 0.05


def test_pwm_is_normalised_and_gate_is_monotone():
    m = build(tiny_cfg())
    tok, dm, ci, pr = make_batch()
    with torch.no_grad():
        pwm, gate, aux = m(tok, dm, ci, pr)
    assert torch.allclose(pwm.sum(1), torch.ones_like(pwm.sum(1)), atol=1e-5), \
        "PWM columns must sum to 1 over ACGT"
    diffs = gate[:, 1:] - gate[:, :-1]
    assert (diffs <= 1e-6).all(), "prefix gate must be monotone non-increasing (contiguous motif)"
    assert (gate >= -1e-6).all() and (gate <= 1 + 1e-6).all()


def test_lambda_contact_starts_small():
    m = build(tiny_cfg())
    assert float(m.pwm_head.log_lambda.exp()) < 0.2


def test_backward_pass_produces_gradients():
    m = build(tiny_cfg(use_flank=True, use_partners=True)); m.train()
    tok, dm, ci, pr = make_batch()
    pwm, gate, aux = m(tok, dm, ci, pr)
    loss = pwm.mean() + gate.mean() + aux["balance_loss"]
    loss.backward()
    n = sum(1 for p in m.parameters() if p.requires_grad and p.grad is not None
            and p.grad.abs().sum() > 0)
    assert n > 10, f"only {n} parameters received gradient"


if __name__ == "__main__":
    torch.manual_seed(0)
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn(); print(f"PASS {fn.__name__}")
    from tfscope.v26.model import TFScopeV26 as M
    m = build(tiny_cfg()); print("\nparam counts (tiny cfg):", m.param_counts())
