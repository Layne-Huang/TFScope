#!/usr/bin/env python
"""Plot TFScope seed model architecture diagram."""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

os.makedirs("figures", exist_ok=True)

# ── colour palette (colorblind-friendly) ──────────────────────────────────────
C = {
    "input":    "#4477AA",
    "encoder":  "#228833",
    "pool":     "#66CCEE",
    "proj":     "#CCBB44",
    "moe":      "#EE6677",
    "head":     "#AA3377",
    "output":   "#BBBBBB",
    "arrow":    "#333333",
    "bg":       "#FAFAFA",
    "border":   "#DDDDDD",
}

fig, ax = plt.subplots(figsize=(16, 11))
ax.set_xlim(0, 16)
ax.set_ylim(0, 11)
ax.axis("off")
fig.patch.set_facecolor(C["bg"])
ax.set_facecolor(C["bg"])

# ── helpers ───────────────────────────────────────────────────────────────────
def box(ax, x, y, w, h, color, label, sublabel=None, fontsize=9, alpha=0.88,
        radius=0.18, labelcolor="white"):
    rect = FancyBboxPatch((x, y), w, h,
                          boxstyle=f"round,pad=0,rounding_size={radius}",
                          linewidth=1.2, edgecolor="white",
                          facecolor=color, alpha=alpha, zorder=3)
    ax.add_patch(rect)
    cy = y + h / 2
    if sublabel:
        ax.text(x + w/2, cy + 0.13, label, ha="center", va="center",
                fontsize=fontsize, fontweight="bold", color=labelcolor, zorder=4)
        ax.text(x + w/2, cy - 0.19, sublabel, ha="center", va="center",
                fontsize=fontsize - 1.5, color=labelcolor, alpha=0.9, zorder=4)
    else:
        ax.text(x + w/2, cy, label, ha="center", va="center",
                fontsize=fontsize, fontweight="bold", color=labelcolor, zorder=4)

def arrow(ax, x0, y0, x1, y1, label=None, color="#444444", lw=1.4):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle="-|>", color=color,
                                lw=lw, mutation_scale=14), zorder=5)
    if label:
        mx, my = (x0+x1)/2, (y0+y1)/2
        ax.text(mx + 0.08, my, label, fontsize=7.5, color="#555555",
                ha="left", va="center", zorder=6)

def bracket(ax, x, y, h, color, label, side="left"):
    dx = -0.18 if side == "left" else 0.18
    ax.annotate("", xy=(x, y), xytext=(x, y+h),
                arrowprops=dict(arrowstyle="-", color=color, lw=2), zorder=2)
    ax.annotate("", xy=(x+dx, y), xytext=(x, y),
                arrowprops=dict(arrowstyle="-", color=color, lw=2), zorder=2)
    ax.annotate("", xy=(x+dx, y+h), xytext=(x, y+h),
                arrowprops=dict(arrowstyle="-", color=color, lw=2), zorder=2)
    ax.text(x + dx*3.5, y + h/2, label, fontsize=8, color=color,
            ha="center", va="center", rotation=90, fontweight="bold", zorder=2)

# ══════════════════════════════════════════════════════════════════════════════
# COLUMN POSITIONS
# ══════════════════════════════════════════════════════════════════════════════
#   INPUT  |  ENCODER  |  POOLING  |  PROJ  |  MOE  |  HEADS  |  OUTPUT

# ── 0. Title ──────────────────────────────────────────────────────────────────
ax.text(8, 10.6, "TFScope Seed Model Architecture", ha="center", va="center",
        fontsize=14, fontweight="bold", color="#222222")

# ── 1. Inputs (x=0.3) ─────────────────────────────────────────────────────────
box(ax, 0.3, 7.5, 1.9, 0.7, C["input"], "TF Protein", "Sequence  (B, L)",
    fontsize=8.5)
box(ax, 0.3, 6.1, 1.9, 0.7, C["input"], "DBD Mask", "(B, L)  bool",
    fontsize=8.5)
box(ax, 0.3, 4.7, 1.9, 0.7, C["input"], "Family ID", "(B,)  int",
    fontsize=8.5)

ax.text(1.25, 9.5, "INPUT", ha="center", fontsize=8, color=C["input"],
        fontweight="bold")

# ── 2. ESM-2 Backbone (x=2.6) ─────────────────────────────────────────────────
box(ax, 2.6, 6.5, 2.1, 1.8, C["encoder"],
    "ESM-2 650M",
    "frozen  |  33 layers\n(B, L, 1280)",
    fontsize=8.5)

# layer-weight averaging note
ax.text(3.65, 6.3, "weighted avg\nlast 4 layers", ha="center",
        fontsize=7, color=C["encoder"], style="italic")

arrow(ax, 2.25, 7.85, 2.6, 7.7)    # sequence → ESM
arrow(ax, 2.25, 6.45, 2.6, 7.0,    # dbd mask → ESM (dashed — not consumed here)
      color="#888888")

ax.text(2.7, 9.5, "ENCODER", ha="center", fontsize=8, color=C["encoder"],
        fontweight="bold")

# ── 3. Dual-stream Attention Pooling (x=5.1) ──────────────────────────────────
box(ax, 5.1, 7.9, 2.0, 0.8, C["pool"],
    "Global AttnPool", "(B, 1280)", fontsize=8.5)
box(ax, 5.1, 6.5, 2.0, 0.8, C["pool"],
    "DBD AttnPool", "(B, 1280) masked",
    fontsize=8.5)

arrow(ax, 4.7, 7.85, 5.1, 8.1)     # ESM → global pool
arrow(ax, 4.7, 7.2, 5.1, 6.9)      # ESM → dbd pool
# dbd_mask → dbd pool
ax.annotate("", xy=(5.1, 6.75), xytext=(2.25, 6.45),
            arrowprops=dict(arrowstyle="-|>", color="#888888", lw=1.2,
                            connectionstyle="arc3,rad=-0.25"), zorder=5)
ax.text(4.0, 5.9, "dbd_mask", fontsize=7, color="#888888", ha="center")

ax.text(6.1, 9.5, "DUAL POOLING", ha="center", fontsize=8,
        color=C["pool"], fontweight="bold")

# concat bracket
ax.plot([7.1, 7.35, 7.35, 7.1], [8.3, 8.3, 6.5, 6.5],
        color="#888888", lw=1.5, zorder=4)
ax.text(7.55, 7.4, "cat", ha="center", va="center", fontsize=8,
        color="#888888", style="italic")

# ── 4. Projection Head (x=7.7) ────────────────────────────────────────────────
box(ax, 7.7, 6.9, 1.9, 1.0, C["proj"],
    "Projection Head",
    "Linear→GELU\n→LayerNorm→Drop\n(B, 512)",
    fontsize=7.8)

arrow(ax, 7.35, 7.4, 7.7, 7.4)

ax.text(8.65, 9.5, "PROJECTION", ha="center", fontsize=8,
        color=C["proj"], fontweight="bold")

# ── 5. MOE Block (x=9.9) ──────────────────────────────────────────────────────
# outer box
rect_moe = FancyBboxPatch((9.9, 4.4), 2.5, 4.2,
                           boxstyle="round,pad=0,rounding_size=0.18",
                           linewidth=1.5, edgecolor=C["moe"],
                           facecolor=C["moe"], alpha=0.12, zorder=2)
ax.add_patch(rect_moe)
ax.text(11.15, 8.75, "MOE BLOCK", ha="center", fontsize=8,
        color=C["moe"], fontweight="bold")

# Family embedding
box(ax, 10.05, 7.7, 2.2, 0.65, C["moe"],
    "Family Embedding", "(B, 64)", fontsize=8)
# FiLM
box(ax, 10.05, 6.85, 2.2, 0.65, C["moe"],
    "FiLM Conditioning", "γ·x + β", fontsize=8)
# Gating
box(ax, 10.05, 5.95, 2.2, 0.65, C["moe"],
    "Family-Aware Gate", "top-2 of 12 experts", fontsize=8)
# Experts
box(ax, 10.05, 4.55, 2.2, 1.1, C["moe"],
    "Expert MLPs (×12)",
    "512→2048→512\nGELU  |  top-2 active",
    fontsize=8)

# MOE internal arrows
arrow(ax, 11.15, 8.3, 11.15, 8.05, color=C["moe"])    # emb→film
arrow(ax, 11.15, 7.5, 11.15, 7.15, color=C["moe"])    # film→gate
arrow(ax, 11.15, 6.6, 11.15, 5.9, color=C["moe"])     # gate→experts

# family_id → MOE
ax.annotate("", xy=(10.05, 8.0), xytext=(2.25, 4.95),
            arrowprops=dict(arrowstyle="-|>", color=C["input"], lw=1.2,
                            connectionstyle="arc3,rad=0.3"), zorder=5)

# projection → MOE
arrow(ax, 9.6, 7.4, 10.05, 7.4, color=C["proj"])

# residual arrow (bypass)
ax.annotate("", xy=(12.65, 5.1), xytext=(9.65, 7.4),
            arrowprops=dict(arrowstyle="-|>", color="#AAAAAA", lw=1.2,
                            connectionstyle="arc3,rad=0.5"), zorder=5)
ax.text(12.1, 6.5, "+ residual", fontsize=7, color="#AAAAAA",
        ha="center", style="italic")

ax.text(11.15, 9.5, "MOE", ha="center", fontsize=8,
        color=C["moe"], fontweight="bold")

# ── 6. Output Heads (x=12.8) ──────────────────────────────────────────────────
box(ax, 12.8, 7.6, 2.1, 1.0, C["head"],
    "Position Gate Head",
    "MLP → (B, 20)\npre-sigmoid logits",
    fontsize=8)
box(ax, 12.8, 5.9, 2.1, 1.4, C["head"],
    "PWM Regression Head",
    "pos-embed + proj\n+ self-attn\n→ (B, 4, 20) logits",
    fontsize=8)

arrow(ax, 12.55, 5.1, 12.8, 8.0, color=C["head"])   # moe→gate head
arrow(ax, 12.55, 5.1, 12.8, 6.5, color=C["head"])   # moe→pwm head

ax.text(13.85, 9.5, "HEADS", ha="center", fontsize=8,
        color=C["head"], fontweight="bold")

# ── 7. Outputs (x=15.1) ───────────────────────────────────────────────────────
box(ax, 15.1, 7.7, 0.8, 0.75, C["output"],
    "Gate\nLogits", fontsize=7.5, labelcolor="#333333")
box(ax, 15.1, 6.0, 0.8, 1.25, C["output"],
    "PWM\nLogits\n(4×20)", fontsize=7.5, labelcolor="#333333")

arrow(ax, 14.9, 8.1, 15.1, 8.08)
arrow(ax, 14.9, 6.6, 15.1, 6.62)

ax.text(15.5, 9.5, "OUTPUT", ha="center", fontsize=8,
        color="#888888", fontweight="bold")

# ── 8. Loss annotation (bottom) ───────────────────────────────────────────────
loss_y = 3.3
rect_loss = FancyBboxPatch((1.0, 1.5), 14.0, 1.65,
                            boxstyle="round,pad=0,rounding_size=0.15",
                            linewidth=1.2, edgecolor="#CCCCCC",
                            facecolor="#F0F0F0", alpha=0.9, zorder=2)
ax.add_patch(rect_loss)
ax.text(8.0, 3.0, "LOSS  =  σ_gate⁻¹ · L_gate  +  log σ_gate  +  σ_pwm⁻¹ · L_pwm  +  log σ_pwm  +  L_balance  +  L_diversity",
        ha="center", va="center", fontsize=9, color="#333333", zorder=3)
ax.text(1.6, 2.4, "L_gate = BCE( gate_logits, pwm_mask )\n         + λ · ordinal_penalty",
        ha="left", va="center", fontsize=7.8, color="#555555", zorder=3)
ax.text(8.5, 2.4,  "L_pwm = masked KL-div( pred_pwm || target_pwm )",
        ha="left", va="center", fontsize=7.8, color="#555555", zorder=3)
ax.text(1.6, 1.75, "L_balance = load-balance aux loss (MOE)",
        ha="left", va="center", fontsize=7.8, color="#555555", zorder=3)
ax.text(8.5, 1.75, "L_diversity = family-diversity aux loss (MOE)",
        ha="left", va="center", fontsize=7.8, color="#555555", zorder=3)

# arrows from outputs to loss box
ax.annotate("", xy=(15.0, 3.15), xytext=(15.45, 6.0),
            arrowprops=dict(arrowstyle="-|>", color="#AAAAAA", lw=1.1,
                            connectionstyle="arc3,rad=0.2"), zorder=5)
ax.annotate("", xy=(15.0, 3.15), xytext=(15.45, 8.07),
            arrowprops=dict(arrowstyle="-|>", color="#AAAAAA", lw=1.1,
                            connectionstyle="arc3,rad=0.25"), zorder=5)

# ── 9. Dimension annotations on arrows ────────────────────────────────────────
ax.text(5.05, 9.1, "(B, L, 1280)", fontsize=7, color="#555555", ha="center")
ax.text(7.55, 9.1, "(B, 1280) ×2", fontsize=7, color="#555555", ha="center")
ax.text(9.75, 9.1, "(B, 512)",      fontsize=7, color="#555555", ha="center")
ax.text(12.5, 9.1, "(B, 512)",      fontsize=7, color="#555555", ha="center")

plt.tight_layout(pad=0.5)
out_path = "figures/tfscope_architecture.pdf"
plt.savefig(out_path, dpi=200, bbox_inches="tight", facecolor=C["bg"])
out_png = "figures/tfscope_architecture.png"
plt.savefig(out_png, dpi=200, bbox_inches="tight", facecolor=C["bg"])
print(f"Saved: {out_path}")
print(f"Saved: {out_png}")
