"""Fig 2d input prep: pick high-confidence, STRUCTURE-LESS, in-distribution TFs and
emit AF3-ready sequences = TF protein + TFScope-predicted consensus dsDNA.

Structure-less = not a crystal-pattern filename (augmented from HOCOMOCO/JASPAR/CISBP).
For each candidate we run TFScope (no retrieval), take the gated motif core, and write
the consensus (argmax) + 3 bp predicted flanks as double-stranded DNA. Dimeric families
(bHLH/bZIP/Nuclear_Receptor) get two protein copies; others a monomer.
"""
import os, sys, json, re
os.environ["TORCH_HOME"] = "/data1/leihuang/.cache/torch"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
sys.path.insert(0, "src"); sys.path.insert(0, "scripts")
import numpy as np, torch, torch.nn.functional as F, pandas as pd
from tfscope.config import TFScopeConfig
from tfscope.models.tfscope import TFScopeModel
from tfscope.data.dataset import AA_TO_TOKEN

CKPT = "/data1/leihuang/project/TFScope/checkpoints/v19_combined_fm_deeppbs_contact/rag_seed42/ckpt_best.pt"
PARQ = "data/processed/tf_pwm_aug_dbd.parquet"
OUT = "results/fig2d_af3_inputs"
os.makedirs(OUT, exist_ok=True)
dev = "cuda:0" if torch.cuda.is_available() else "cpu"
DIMERIC = {"bHLH", "bZIP", "Nuclear_Receptor"}
COMP = {"A": "T", "C": "G", "G": "C", "T": "A"}
# families TFScope handles well; pick recognisable structure-less TFs per family
WANT = {"Homeodomain": 2, "bHLH": 2, "bZIP": 2, "ETS": 1, "Forkhead": 1,
        "C2H2_medium": 1, "C2H2_short": 1, "Nuclear_Receptor": 1}

cfg = TFScopeConfig()
for k, v in json.load(open(os.path.join(os.path.dirname(CKPT), "config.json"))).items():
    if hasattr(cfg, k):
        try: setattr(cfg, k, type(getattr(cfg, k))(v))
        except Exception: pass
cfg.use_retrieval = False
m = TFScopeModel(cfg).to(dev).eval()
m.load_state_dict(torch.load(CKPT, map_location=dev, weights_only=False)["model"], strict=False)

@torch.no_grad()
def predict(seq, fid):
    t = torch.tensor([[AA_TO_TOKEN.get(a, 4) for a in seq]], dtype=torch.long, device=dev)
    dm = torch.ones(1, len(seq), dtype=torch.bool, device=dev)
    fi = torch.tensor([fid], dtype=torch.long, device=dev)
    gl, pl, _ = m(t, dm, fi, retrieved_pwms=None, retrieved_masks=None, retrieved_sims=None, recog_prior=None)
    gate = gl.sigmoid()[0].cpu().numpy(); pwm = F.softmax(pl, 1)[0].cpu().numpy()  # (4, W)
    return pwm, gate

def consensus_from(pwm, gate, flank=3):
    cols = np.where(gate > 0.5)[0]
    if len(cols) < 4:  # fall back to top-IC contiguous window
        ic = (pwm * np.log2(pwm + 1e-9)).sum(0) + 2
        cols = np.arange(max(0, ic.argmax() - 4), min(pwm.shape[1], ic.argmax() + 5))
    lo, hi = cols.min(), cols.max()
    lo, hi = max(0, lo - flank), min(pwm.shape[1] - 1, hi + flank)
    bases = "ACGT"
    seq = "".join(bases[pwm[:, j].argmax()] for j in range(lo, hi + 1))
    core_lo = cols.min() - lo
    ic_core = float(((pwm[:, cols] * np.log2(pwm[:, cols] + 1e-9)).sum(0) + 2).mean())
    return seq, core_lo, core_lo + len(cols), round(ic_core, 3)

d = pd.read_parquet(PARQ)
iscrys = d.filename.astype(str).str.match(r"^[0-9][a-z0-9]{3}_[A-Za-z]_")
pool = d[(~iscrys) & (d.quality_grade == "A")].copy()
pool = pool.sort_values("gene_symbol").drop_duplicates("gene_symbol")

picked = []
for fam, n in WANT.items():
    cand = pool[pool.family_name == fam]
    rows = []
    for r in cand.itertuples():
        seq = str(r.sequence); s, e = int(r.dbd_start), int(r.dbd_end)
        dbd = seq[s:e]
        if not (15 <= len(dbd) <= 120): continue
        pwm, gate = predict(dbd, int(r.family_id))
        cons, clo, chi, ic = consensus_from(pwm, gate)
        rows.append((ic, r.gene_symbol, r.uniprot_id, fam, dbd, seq, cons, clo, chi))
    rows.sort(reverse=True)   # by IC (confidence)
    picked += rows[:n]

# ── write outputs ──
lines = ["# Fig 2d — AF3 input sequences for structure-less TFs (TFScope-predicted consensus)\n"]
lines.append("Fold each as protein + double-stranded DNA. Dimeric families use TWO protein copies.\n")
recs = []
for ic, gene, uni, fam, dbd, full, cons, clo, chi in picked:
    rc = "".join(COMP[b] for b in cons[::-1])
    olig = "homodimer (2 copies)" if fam in DIMERIC else "monomer"
    core = cons[clo:chi]
    lines += [f"\n## {gene}  ({fam}, {olig})  uniprot={uni}  motif_IC={ic}",
              f"- DNA top strand (5'->3'): {cons}     [motif core: {core}]",
              f"- DNA bottom strand (5'->3'): {rc}",
              f"- protein DBD ({len(dbd)} aa): {dbd}"]
    recs.append(dict(gene=gene, uniprot=uni, family=fam, oligomer=olig, motif_ic=ic,
                     dna_top=cons, dna_bottom=rc, motif_core=core,
                     protein_dbd=dbd, protein_full=full))
open(f"{OUT}/af3_sequences.md", "w").write("\n".join(lines) + "\n")
json.dump(recs, open(f"{OUT}/af3_sequences.json", "w"), indent=1)
# fasta of proteins (DBD)
with open(f"{OUT}/proteins_dbd.fasta", "w") as f:
    for r in recs: f.write(f">{r['gene']}_{r['family']}\n{r['protein_dbd']}\n")

print(f"selected {len(recs)} structure-less TFs -> {OUT}/af3_sequences.md")
print(f"{'gene':<10} {'family':<16} {'oligomer':<20} {'IC':>5}  consensus")
for r in recs:
    print(f"{r['gene']:<10} {r['family']:<16} {r['oligomer']:<20} {r['motif_ic']:>5}  {r['dna_top']}")
