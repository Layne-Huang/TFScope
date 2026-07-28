"""Shape / masking / supervision tests for Candidate B (InterfacePairHead)."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import torch
from tfscope.config import TFScopeConfig
from tfscope.models.interface_pair import InterfacePairHead


def _head(residue_dim=48, ctx_dim=16):
    cfg = TFScopeConfig(max_motif_length=12)
    cfg.interface_pair_dim = 64
    return cfg, InterfacePairHead(cfg, residue_dim=residue_dim, chain_ctx_dim=ctx_dim)


def test_shapes():
    cfg, head = _head()
    B, L = 3, 20
    h = torch.randn(B, L, 48)
    ctx = torch.randn(B, L, 16)
    valid = torch.zeros(B, L, dtype=torch.bool); valid[:, :14] = True
    out = head(h, valid, chain_ctx=ctx)
    assert out["pwm_logits"].shape == (B, 4, cfg.max_motif_length)
    assert out["C"].shape == (B, L, cfg.max_motif_length)
    assert out["occ_res"].shape == (B, L)
    assert out["occ_pos"].shape == (B, cfg.max_motif_length)


def test_padding_residues_do_not_affect_output():
    """Changing features of masked-out residues must not change the PWM logits."""
    _, head = _head()
    head.eval()
    B, L = 2, 18
    h = torch.randn(B, L, 48)
    valid = torch.zeros(B, L, dtype=torch.bool); valid[:, :10] = True
    with torch.no_grad():
        out1 = head(h, valid)
        h2 = h.clone()
        h2[:, 10:] = torch.randn_like(h2[:, 10:]) * 100.0   # perturb only padding
        out2 = head(h2, valid)
    dev = (out1["pwm_logits"] - out2["pwm_logits"]).abs().max().item()
    assert dev < 1e-5, f"padding leaked into output: dev={dev:.2e}"


def test_distill_loss_masks_missing_labels():
    _, head = _head()
    B, L, J = 2, 12, 12
    logit = torch.zeros(B, L, J, requires_grad=True)
    target = torch.rand(B, L, J)
    mask = torch.zeros(B, L, J, dtype=torch.bool)
    # no labels → zero loss, no NaN
    loss0 = head.distill_loss_2d(logit, target, mask)
    assert torch.isfinite(loss0) and loss0.item() == 0.0
    # some labels → positive finite loss with gradient
    mask[:, :4, :4] = True
    loss1 = head.distill_loss_2d(logit, target, mask)
    loss1.backward()
    assert torch.isfinite(loss1) and loss1.item() > 0
    assert logit.grad is not None and logit.grad.abs().sum() > 0


def test_distillation_pulls_occupancy_toward_target():
    """A few SGD steps on the distill loss should raise occupancy where target=1."""
    _, head = _head()
    B, L, J = 1, 10, 12
    h = torch.randn(B, L, 48)
    valid = torch.ones(B, L, dtype=torch.bool)
    target = torch.zeros(B, L, J); target[0, 2, 3] = 1.0; target[0, 5, 7] = 1.0
    mask = torch.ones(B, L, J, dtype=torch.bool)
    opt = torch.optim.Adam(head.parameters(), lr=1e-2)
    first = None
    for _ in range(50):
        out = head(h, valid)
        loss = head.distill_loss_2d(out["contact_logit"], target, mask)
        if first is None: first = loss.item()
        opt.zero_grad(); loss.backward(); opt.step()
    assert loss.item() < first, f"distill loss did not decrease ({first:.3f} -> {loss.item():.3f})"
    with torch.no_grad():
        C = head(h, valid)["C"]
    assert C[0, 2, 3] > 0.5 and C[0, 5, 7] > 0.5, "occupancy did not learn the true contacts"


def test_shuffle_control_preserves_marginals():
    _, head = _head()
    B, L, J = 4, 10, 12
    target = (torch.rand(B, L, J) > 0.7).float()
    mask = torch.ones(B, L, J, dtype=torch.bool)
    g = torch.Generator().manual_seed(0)
    sh_t, sh_m = head.shuffle_contacts(target, mask, generator=g)
    # per-sample total number of contacts preserved, but arrangement changed
    assert torch.allclose(target.sum(dim=(1, 2)), sh_t.sum(dim=(1, 2)))
    assert not torch.allclose(target, sh_t)


if __name__ == "__main__":
    test_shapes()
    test_padding_residues_do_not_affect_output()
    test_distill_loss_masks_missing_labels()
    test_distillation_pulls_occupancy_toward_target()
    test_shuffle_control_preserves_marginals()
    print("All InterfacePairHead tests passed.")
