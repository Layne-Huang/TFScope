"""Unit tests for the unified two-panel evaluator scoring core (no GPU)."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from iclr.unified_eval import (panel_A, panel_B, trimmed_core, _apply_length,
                               train_length_policy, baseline_pred_len)
from tfscope.models.alignment import revcomp_pwm_np


def _peaky(L, seed=0):
    rng = np.random.default_rng(seed)
    pwm = np.full((4, L), 0.02, np.float32)
    for j in range(L):
        pwm[rng.integers(0, 4), j] = 0.94
    return pwm / pwm.sum(0, keepdims=True)


def test_perfect_prediction():
    core = _peaky(10)
    A = panel_A(core.copy(), core)
    B = panel_B(core.copy(), core, pred_len=10)
    assert A["content_r"] > 0.99, A
    assert A["overlap_frac"] == 1.0
    assert B["covR"] > 0.99 and B["coverage"] == 1.0 and B["len_mae"] == 0


def test_reverse_complement_recovered():
    core = _peaky(12, 1)
    rc = revcomp_pwm_np(core)
    A = panel_A(rc, core)             # oracle RC alignment should recover it
    assert A["content_r"] > 0.99, A


def test_uniform_prediction_scores_zero():
    core = _peaky(10, 2)
    uni = np.full((4, 10), 0.25, np.float32)
    A = panel_A(uni, core)
    B = panel_B(uni, core, pred_len=10)
    assert abs(A["content_r"]) < 0.15
    assert abs(B["covR"]) < 0.15


def test_half_length_prediction_penalised_in_B_not_A():
    core = _peaky(12, 3)
    half = core[:, :6].copy()                       # correct content, half the length
    A = panel_A(half, core)
    B = panel_B(half, core, pred_len=6)
    # Panel A (content over overlap) stays high; Panel B coverage ~0.5 penalises.
    assert A["content_r"] > 0.9, A
    assert 0.4 <= B["coverage"] <= 0.6, B
    assert B["covR"] < 0.6 and B["len_mae"] == 6 and B["len_bias"] == -6


def test_over_length_prediction_bias_positive():
    core = _peaky(8, 4)
    long = np.concatenate([core, _peaky(6, 99)], axis=1)   # correct core + junk tail
    B = panel_B(long, core, pred_len=14)
    assert B["len_bias"] == 6 and B["len_mae"] == 6


def test_apply_length_truncate_and_pad():
    p = _peaky(10)
    assert _apply_length(p, 5).shape == (4, 5)
    padded = _apply_length(p, 15)
    assert padded.shape == (4, 15)
    assert np.allclose(padded[:, 10:], 0.25)   # pad region uniform
    assert _apply_length(p, None).shape == (4, 10)


def test_length_policy_never_reads_test():
    import pandas as pd
    train = pd.DataFrame({"family_id": [1, 1, 1, 2, 2], "motif_length": [8, 10, 12, 6, 6]})
    policy = train_length_policy(train)
    assert baseline_pred_len(1, policy) == 10   # median of 8,10,12
    assert baseline_pred_len(2, policy) == 6
    assert baseline_pred_len(999, policy) == policy[1]  # global fallback for unseen family


def test_trimmed_core_drops_flanks():
    core = _peaky(6, 5)
    padded = np.concatenate([np.full((4, 3), 0.25, np.float32), core,
                             np.full((4, 3), 0.25, np.float32)], axis=1)
    tc = trimmed_core(padded)
    assert tc is not None and tc.shape[1] == 6   # uniform flanks trimmed


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn(); print("PASS", name)
    print("All unified_eval scoring tests passed.")
