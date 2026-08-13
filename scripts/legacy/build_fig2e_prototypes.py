"""Fig 2e — the learned prototype dictionary exposes interpretable binding concepts.

The MoE carries a dictionary of 32 prototype vectors (moe.proto.prototypes). For every
test TF we record which prototypes its representation activates (softmax attention weights),
then (a) show that prototypes specialise by structural family (family x prototype usage map)
and (b) read out the binding concept each dominant prototype encodes via the consensus motif
of the TFs that most activate it.

GPU 0 (training runs are on other GPUs). Outputs:
  results/per_family/fig2e_prototypes.json
  figures/figure2e_prototypes/figure2e_prototypes.{png,pdf}
"""
import os, sys, json
os.environ["TORCH_HOME"] = "/data1/leihuang/.cache/torch"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
sys.path.insert(0, "src"); sys.path.insert(0, "scripts")
import numpy as np, torch, pandas as pd
from tfscope.config import TFScopeConfig
from tfscope.models.tfscope import TFScopeModel
from tfscope.data.dataset import AA_TO_TOKEN

CKPT = "/data1/leihuang/project/TFScope/checkpoints/v19_combined_fm_deeppbs_contact/rag_seed42/ckpt_best.pt"
PARQ = "data/processed/tf_pwm_deeppbs_only_canon_trim.parquet"
SPLIT = "data/processed/splits/deeppbs_cluster40/split.json"
OUTD = "figures/figure2e_prototypes"; os.makedirs(OUTD, exist_ok=True)
dev = "cuda:0" if torch.cuda.is_available() else "cpu"

cfg = TFScopeConfig()
for k, v in json.load(open(os.path.join(os.path.dirname(CKPT), "config.json"))).items():
    if hasattr(cfg, k):
        try: setattr(cfg, k, type(getattr(cfg, k))(v))
        except Exception: pass
cfg.use_retrieval = False
m = TFScopeModel(cfg).to(dev).eval()
m.load_state_dict(torch.load(CKPT, map_location=dev, weights_only=False)["model"], strict=False)

cap = {}
m.moe.proto.register_forward_hook(lambda mod, i, o: cap.__setitem__("w", o[1].detach()))

@torch.no_grad()
def proto_weights(seq, fid):
    t = torch.tensor([[AA_TO_TOKEN.get(a, 4) for a in seq]], dtype=torch.long, device=dev)
    dm = torch.ones(1, len(seq), dtype=torch.bool, device=dev)
    fi = torch.tensor([fid], dtype=torch.long, device=dev)
    m(t, dm, fi, retrieved_pwms=None, retrieved_masks=None, retrieved_sims=None, recog_prior=None)
    return cap["w"][0].cpu().numpy()   # (32,)

sp = set(json.load(open(SPLIT))["test"])
df = pd.read_parquet(PARQ)
df = df[df.filename.astype(str).isin(sp)].reset_index(drop=True)

NP = cfg.n_prototypes
recs = []
for r in df.itertuples():
    s, e = int(r.dbd_start), int(r.dbd_end); dbd = str(r.sequence)[s:e]
    if len(dbd) < 8: continue
    w = proto_weights(dbd, int(r.family_id))
    gt = np.frombuffer(r.pwm, dtype=np.float32).reshape(4, -1).astype(float)
    recs.append(dict(gene=r.gene_symbol, family=r.family_name, w=w, gt=gt))
print(f"scored {len(recs)} test TFs over {NP} prototypes")

fams = sorted(set(r["family"] for r in recs))
W = np.stack([r["w"] for r in recs])                     # (N, 32)
# family x prototype mean usage
M = np.zeros((len(fams), NP))
for fi, f in enumerate(fams):
    idx = [i for i, r in enumerate(recs) if r["family"] == f]
    M[fi] = W[idx].mean(0)

# prototype specialisation: dominant family + concentration (max share)
proto_dom, proto_share = [], []
Mn = M / (M.sum(0, keepdims=True) + 1e-9)                # normalise each prototype across families
for p in range(NP):
    proto_dom.append(fams[int(Mn[:, p].argmax())]); proto_share.append(float(Mn[:, p].max()))
# pick the most-used, most-specialised prototypes for concept logos
usage = W.mean(0)
cand = sorted(range(NP), key=lambda p: -(usage[p] * proto_share[p]))
chosen, seen_fam = [], set()
for p in cand:                                            # diverse families
    if proto_dom[p] in seen_fam and len(chosen) >= 4: continue
    chosen.append(p); seen_fam.add(proto_dom[p])
    if len(chosen) == 6: break

# exemplar = top-activating TF per chosen prototype
concept = []
for p in chosen:
    top = max(recs, key=lambda r: r["w"][p])
    concept.append(dict(proto=p, gene=top["gene"], family=top["family"],
                        dom_family=proto_dom[p], share=round(proto_share[p], 2), gt=top["gt"]))

json.dump(dict(families=fams, usage=usage.round(4).tolist(), proto_dom=proto_dom,
               proto_share=[round(x, 3) for x in proto_share],
               concepts=[{k: v for k, v in c.items() if k != "gt"} for c in concept],
               n_tf=len(recs)),
          open("results/per_family/fig2e_prototypes.json", "w"), indent=1)

# ── figure ──
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
def consensus_logo(ax, pwm, title):
    pwm = np.clip(pwm, 1e-9, 1); pwm = pwm / pwm.sum(0, keepdims=True)
    ic = (2 + (pwm * np.log2(pwm)).sum(0))                 # per-col info (bits)
    heights = pwm * ic                                     # (4, L)
    colors = {"A": "#2ca02c", "C": "#1f77b4", "G": "#ff7f0e", "T": "#d62728"}
    order = "ACGT"
    L = pwm.shape[1]
    for j in range(L):
        ys = 0.0
        for b in np.argsort(heights[:, j]):
            h = heights[b, j]
            if h < 0.01: continue
            ax.text(j + 0.5, ys + h / 2, order[b], ha="center", va="center",
                    fontsize=min(26, 7 + 60 * h), fontweight="bold", color=colors[order[b]],
                    family="monospace")
            ys += h
    ax.set_xlim(0, L); ax.set_ylim(0, 2.1); ax.set_xticks([]); ax.set_yticks([0, 1, 2])
    ax.set_ylabel("bits", fontsize=7); ax.set_title(title, fontsize=8.5)
    for sp_ in ["top", "right"]: ax.spines[sp_].set_visible(False)

fig = plt.figure(figsize=(12, 5.2))
gs = fig.add_gridspec(3, 3, width_ratios=[2.0, 1, 1], hspace=0.75, wspace=0.3)

# (a) family x prototype usage heatmap
axh = fig.add_subplot(gs[:, 0])
im = axh.imshow(M, aspect="auto", cmap="magma")
axh.set_yticks(range(len(fams))); axh.set_yticklabels(fams, fontsize=8)
axh.set_xticks(range(0, NP, 4)); axh.set_xticklabels(range(0, NP, 4), fontsize=7)
axh.set_xlabel("prototype index (of 32)", fontsize=9)
axh.set_title("a  Prototypes specialise by family\n(mean activation weight)", fontsize=10, fontweight="bold", loc="left")
for p in chosen:
    axh.add_patch(plt.Rectangle((p - 0.5, -0.5), 1, len(fams), fill=False, edgecolor="cyan", lw=1.5))
fig.colorbar(im, ax=axh, shrink=0.5, pad=0.02, label="activation")

# (b) concept logos for chosen prototypes
for i, c in enumerate(concept):
    ax = fig.add_subplot(gs[i % 3, 1 + i // 3])
    consensus_logo(ax, c["gt"], f"proto {c['proto']} → {c['dom_family']}\n{c['gene']} (share {c['share']})")
fig.suptitle("Learned prototypes expose interpretable binding concepts",
             fontsize=12, fontweight="bold", y=0.99)
out = f"{OUTD}/figure2e_prototypes"
fig.savefig(out + ".png", dpi=300, bbox_inches="tight"); fig.savefig(out + ".pdf", bbox_inches="tight")
print("chosen prototypes:", [(c["proto"], c["dom_family"], c["gene"]) for c in concept])
print(f"saved {out}.png/.pdf")
