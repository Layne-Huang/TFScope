#!/usr/bin/env python
"""TFScope v10/v14 architecture diagram (current model)."""
import os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

os.makedirs("figures", exist_ok=True)
C = {"input":"#4477AA","enc":"#228833","pool":"#66CCEE","proj":"#CCBB44",
     "moe":"#EE6677","ret":"#AA3377","head":"#EE8866","loss":"#999933",
     "out":"#BBBBBB","arrow":"#333333"}

fig, ax = plt.subplots(figsize=(15, 11)); ax.set_xlim(0,15); ax.set_ylim(0,11); ax.axis("off")

def box(x,y,w,h,c,title,sub="",fs=10,tc="white"):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.04,rounding_size=0.12",
                 fc=c,ec="#222",lw=1.3,alpha=0.95))
    ax.text(x+w/2,y+h/2+(0.16 if sub else 0),title,ha="center",va="center",
            fontsize=fs,fontweight="bold",color=tc)
    if sub: ax.text(x+w/2,y+h/2-0.22,sub,ha="center",va="center",fontsize=7.5,color=tc)

def arrow(x1,y1,x2,y2,style="-|>",c=None,lw=1.8,ls="-"):
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle=style,mutation_scale=16,
                 lw=lw,color=c or C["arrow"],ls=ls,shrinkA=2,shrinkB=2))

ax.text(7.5,10.6,"TFScope — Seed Model Architecture (v10 / v14)",
        ha="center",fontsize=15,fontweight="bold")

# ── Inputs ──
box(0.3,8.8,3.0,1.0,C["input"],"TF protein seq","(up to 1024 aa)",10)
box(0.3,7.5,3.0,1.0,C["input"],"DBD mask","binary, DNA-binding domain",9)
box(0.3,6.2,3.0,1.0,C["input"],"Family ID","8 core + other + multi",9)

# ── Encoder ──
box(4.0,8.0,2.8,1.8,C["enc"],"ESM-2 (650M)","frozen + LoRA\n(last 6 layers, r=16)",10)
arrow(3.3,9.3,4.0,9.0); arrow(3.3,8.0,4.0,8.6)

# ── Pooling ──
box(7.4,9.2,2.7,0.85,C["pool"],"Global gated pool","",9)
box(7.4,8.1,2.7,0.85,C["pool"],"DBD gated pool","masked attention",8.5)
arrow(6.8,9.0,7.4,9.55); arrow(6.8,8.7,7.4,8.5)
arrow(3.3,7.9,7.4,8.35,ls="--",c="#888")  # dbd mask -> dbd pool

# ── Projection ──
box(10.6,8.4,2.4,1.1,C["proj"],"Projection","[global‖DBD]→512",9)
arrow(10.1,9.6,10.6,9.2); arrow(10.1,8.5,10.6,8.7)

# ── MoE ──
box(10.4,6.2,2.8,1.5,C["moe"],"MoE block","12 experts, top-2\n+1 shared (DeepSeek)\nfamily-aware routing",9)
arrow(11.8,8.4,11.8,7.7)
arrow(3.3,6.7,10.4,6.9,ls="--",c="#888")  # family id -> moe

# ── Retrieval branch ──
box(0.3,3.2,3.4,2.2,C["ret"],"Retrieval (RAG)",
    "ESM-DBD cosine NN\ntop-K=3 PWMs\nTrustPredictor\n(learned per-NN trust)",8.5)
box(0.3,1.6,3.4,1.0,C["ret"],"NN index","tf_nn_index.json\n(donor pool)",8,"white")
arrow(2.0,3.2,2.0,2.6)

# ── PWM head ──
box(7.2,4.0,3.2,2.0,C["head"],"PWM Regression Head",
    "pos self-attn\n+ cross-attn to ESM-DBD\n+ retrieval log-prior\nβ-gated by trust",8.5,"white")
arrow(11.8,6.2,9.6,6.0)              # moe -> pwm head
arrow(3.7,4.3,7.2,4.8)               # retrieval -> pwm head
arrow(6.8,8.4,8.0,6.0,ls="--",c="#888")  # esm emb -> cross attn

# ── Gate head ──
box(10.8,4.2,2.6,1.3,C["head"],"Position Gate","motif length\n(per-pos sigmoid)",8.5,"white")
arrow(11.9,6.2,12.0,5.5)

# ── Outputs ──
box(7.6,2.0,2.6,1.0,C["out"],"PWM (4×L)","softmax",9,"#222")
box(10.8,2.4,2.6,1.0,C["out"],"Motif mask","",9,"#222")
arrow(8.8,4.0,8.8,3.0); arrow(12.1,4.2,12.1,3.4)

# ── Loss ──
box(6.2,0.3,7.2,1.1,C["loss"],"Loss",
    "L1 + IC-match + entropy  |  v14:+ IC-weighted Pearson + top-base margin  |  gate BCE  |  trust BCE  |  MoE balance/diversity",
    7.2,"white")
arrow(8.8,2.0,8.8,1.4); arrow(12.1,2.4,12.1,1.4)

# legend
ax.text(0.3,0.5,"Solid = data flow   Dashed = conditioning/auxiliary input",
        fontsize=8,color="#555",style="italic")

plt.tight_layout()
out="figures/tfscope_architecture_v14.pdf"
fig.savefig(out,bbox_inches="tight"); fig.savefig(out.replace(".pdf",".png"),dpi=150,bbox_inches="tight")
plt.close(fig)
print(f"Saved {out} and .png")
