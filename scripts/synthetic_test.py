"""Synthetic test: verify TFScope model forward pass, loss, and backward."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch

from tfscope.config import TFScopeConfig
from tfscope.models.tfscope import TFScopeModel
from tfscope.losses.tfscope_loss import TFScopeLoss
from tfscope.data.dataset import SyntheticTFDataset, collate_variable_length
from tfscope.train.trainer import train


def test_forward_pass():
    print("=" * 60)
    print("Test 1: Forward pass shape check")
    print("=" * 60)

    config = TFScopeConfig()
    model = TFScopeModel(config, use_dummy_backbone=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    B, L = 4, 150
    sequence_tokens = torch.randint(4, 24, (B, L)).to(device)
    dbd_mask = torch.zeros(B, L, dtype=torch.bool, device=device)
    dbd_mask[:, 30:100] = True
    family_id = torch.tensor([0, 1, 2, 3]).to(device)

    length_logits, pwm_logits, pwm_mask, aux = model(sequence_tokens, dbd_mask, family_id)

    print(f"  length_logits shape: {length_logits.shape}  (expected: [{B}, {config.num_length_classes}])")
    print(f"  pwm_logits shape:    {pwm_logits.shape}  (expected: [{B}, 4, {config.max_motif_length}])")
    print(f"  pwm_mask shape:      {pwm_mask.shape}  (expected: [{B}, {config.max_motif_length}])")

    assert length_logits.shape == (B, config.num_length_classes), "Length shape mismatch!"
    assert pwm_logits.shape == (B, 4, config.max_motif_length), "PWM shape mismatch!"
    assert pwm_mask.shape == (B, config.max_motif_length), "Mask shape mismatch!"
    print("  PASSED\n")


def test_loss_backward():
    print("=" * 60)
    print("Test 2: Loss computation and backward pass")
    print("=" * 60)

    config = TFScopeConfig()
    model = TFScopeModel(config, use_dummy_backbone=True)
    loss_fn = TFScopeLoss(config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    loss_fn = loss_fn.to(device)

    B, L = 8, 150
    sequence_tokens = torch.randint(4, 24, (B, L)).to(device)
    dbd_mask = torch.zeros(B, L, dtype=torch.bool, device=device)
    dbd_mask[:, 30:100] = True
    family_id = torch.randint(0, config.num_families, (B,)).to(device)
    target_length = torch.randint(0, config.num_length_classes, (B,)).to(device)
    target_pwm = torch.randn(B, 4, config.max_motif_length).to(device)
    target_pwm = torch.softmax(target_pwm, dim=1)
    pwm_mask = torch.ones(B, config.max_motif_length).to(device)

    length_logits, pwm_logits, pwm_mask_out, aux = model(sequence_tokens, dbd_mask, family_id)

    loss, metrics = loss_fn(
        length_logits, pwm_logits, target_length, target_pwm, pwm_mask,
        aux.get('gate_logits'), aux.get('top_indices'), aux.get('family_id'),
    )

    print(f"  Total loss: {loss.item():.4f}")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    loss.backward()

    # Check gradients exist on key parameters
    has_grad = sum(1 for p in model.parameters() if p.grad is not None)
    total_params = sum(1 for p in model.parameters())
    print(f"  Parameters with gradients: {has_grad}/{total_params}")
    assert has_grad > 0, "No gradients computed!"
    print("  PASSED\n")


def test_overfit():
    print("=" * 60)
    print("Test 3: Overfit on 10 samples (200 steps)")
    print("=" * 60)

    config = TFScopeConfig()
    config.total_steps = 200
    config.warmup_steps = 20

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")

    dataset = SyntheticTFDataset(config, n_samples=10, max_seq_len=150)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=4, shuffle=True, collate_fn=collate_variable_length,
    )

    model = train(config, loader, device=device)
    print("  PASSED\n")


if __name__ == "__main__":
    test_forward_pass()
    test_loss_backward()
    test_overfit()
    print("=" * 60)
    print("All tests passed!")
    print("=" * 60)
