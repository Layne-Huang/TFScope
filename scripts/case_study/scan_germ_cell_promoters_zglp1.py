#!/usr/bin/env python
"""Germ-cell / oogenesis promoter enrichment for the ZGLP1 predicted GATA motif.

Scans promoters (TSS +/-1 kb, hg38) of curated oogenesis / meiosis-entry genes (the
ZGLP1-driven oogenic program; Nagaoka et al. 2020) with the ZGLP1 RAG PWM and GATA
references, against a DINUCLEOTIDE-PRESERVING shuffled background (holds GC *and* CpG
fixed). GATA elements are AT-rich (not CpG-centered), so this is a fairer test than the
SOHLH1 E-box; the conclusion is data-driven (main panel if it beats the shuffled
control, Extended Data otherwise).

This nominates candidate regulatory elements; it does NOT establish in vivo ZGLP1
occupancy. hg38 = /n/holylabs/.../WholeGenomeFasta/genome.fa (+ .fai).
"""
import os, sys, json, urllib.request, urllib.parse, random
sys.path.insert(0, "scripts/case_study"); sys.path.insert(0, "pwm_rosetta")
import numpy as np, pandas as pd, yaml
from scipy.stats import mannwhitneyu
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

cfg = yaml.safe_load(open("configs/case_study_zglp1.yaml"))
OUT = f"{cfg['output_dir']}/targets"; FIG = cfg["figure_dir"]; EXT = cfg["extended_figure_dir"]
os.makedirs(OUT, exist_ok=True); os.makedirs(EXT, exist_ok=True)
GENOME = "/n/holylabs/lpinello_lab/Lab/leihuang/WholeGenomeFasta/genome.fa"
random.seed(0); np.random.seed(0)
WIN = 1000
plt.rcParams.update({"font.size": 8, "pdf.fonttype": 42})

# curated ZGLP1-driven oogenic program: meiosis-entry + oocyte/folliculogenesis genes
GERM = ["ZGLP1","STRA8","MEIOSIN","REC8","SYCP1","SYCP2","SYCP3","SPO11","DMC1","HORMAD1",
        "HORMAD2","STAG3","SMC1B","MEIOC","RAD21L1","TEX12","TEX14","SYCE1","SYCE2","MAJIN",
        "NOBOX","FIGLA","LHX8","GDF9","BMP15","ZP1","ZP2","ZP3","OOSP2","DAZL","DDX4","MAEL",
        "SOHLH1","SOHLH2","KIT","ZBTB16","DPPA3","NANOS3","H1-6","RNF17"]

# ── faidx fetcher (no deps) ──────────────────────────────────────────────────
FAI = {}
for ln in open(GENOME + ".fai"):
    n, L, off, lb, lw = ln.split("\t"); FAI[n] = (int(L), int(off), int(lb), int(lw))
_gf = open(GENOME, "rb")
def fetch_fast(chrom, start, end):
    if chrom not in FAI: return None
    L, off, lb, lw = FAI[chrom]; start = max(0, start); end = min(L, end)
    if end <= start: return None
    b0 = off + (start // lb) * lw + (start % lb)
    b1 = off + (end // lb) * lw + (end % lb)
    _gf.seek(b0); raw = _gf.read(b1 - b0).decode()
    return raw.replace("\n", "").upper()

def get_tss(symbols):
    data = urllib.parse.urlencode({"q": ",".join(symbols), "scopes": "symbol",
                                   "fields": "genomic_pos", "species": "human"}).encode()
    req = urllib.request.Request("https://mygene.info/v3/query", data=data,
                                 headers={"Content-Type": "application/x-www-form-urlencoded"})
    res = json.loads(urllib.request.urlopen(req, timeout=60).read().decode())
    out = {}
    for r in res:
        gp = r.get("genomic_pos")
        if not gp: continue
        gp = gp[0] if isinstance(gp, list) else gp
        chrom = "chr" + str(gp["chr"])
        if chrom not in FAI: continue
        tss = gp["end"] if gp["strand"] == -1 else gp["start"]
        out[r.get("query")] = (chrom, int(tss), int(gp["strand"]))
    return out

tss = get_tss(GERM)
print(f"resolved TSS for {len(tss)}/{len(GERM)} oogenesis genes")

def gc(s): return (s.count("G") + s.count("C")) / max(1, len(s))
def cpg(s): return s.count("CG") / max(1, len(s) - 1)
def revcomp(s): return s.translate(str.maketrans("ACGT", "TGCA"))[::-1]

targets = {}
for g, (c, t, st) in tss.items():
    s = fetch_fast(c, t - WIN, t + WIN)
    if s and s.count("N") < 0.1 * len(s):
        targets[g] = s
print(f"target promoters: {len(targets)}")
tgt_seqs = list(targets.values())

# ── Altschul-Erikson dinucleotide-preserving shuffle (holds GC + CpG fixed) ───
def dinuc_shuffle(seq, rng):
    from collections import defaultdict
    s = [c if c in "ACGT" else "A" for c in seq]; n = len(s)
    if n < 3: return "".join(s)
    last = s[-1]; graph = defaultdict(list)
    for i in range(n - 1): graph[s[i]].append(s[i + 1])
    while True:
        last_edge = {}
        for v in graph:
            if v != last: last_edge[v] = rng.choice(graph[v])
        good = True
        for v in graph:
            if v == last: continue
            seen, x = set(), v
            while x != last:
                if x in seen or x not in last_edge: good = False; break
                seen.add(x); x = last_edge[x]
            if not good: break
        if good: break
    edge_lists, ptr = {}, defaultdict(int)
    for v in graph:
        e = list(graph[v])
        if v != last:
            e.remove(last_edge[v]); rng.shuffle(e); e.append(last_edge[v])
        else:
            rng.shuffle(e)
        edge_lists[v] = e
    out = [s[0]]; cur = s[0]
    for _ in range(n - 1):
        nxt = edge_lists[cur][ptr[cur]]; ptr[cur] += 1; out.append(nxt); cur = nxt
    return "".join(out)

_rng = random.Random(0); K_SHUF = 5
bg = [dinuc_shuffle(s, _rng) for s in tgt_seqs for _ in range(K_SHUF)]
print(f"dinuc-shuffled bg: {len(bg)} | target GC {np.mean([gc(s) for s in tgt_seqs]):.2f} "
      f"CpG {np.mean([cpg(s) for s in tgt_seqs]):.3f} | bg GC {np.mean([gc(s) for s in bg]):.2f} "
      f"CpG {np.mean([cpg(s) for s in bg]):.3f}")

# ── motifs ────────────────────────────────────────────────────────────────────
def read_pwm(path): return pd.read_csv(path, sep="\t", index_col=0)[list("ACGT")].values.T
def gata_pwm(consensus="AGATAA"):
    iupac = {"A":"A","C":"C","G":"G","T":"T","W":"AT","R":"AG","N":"ACGT"}
    m = np.full((4, len(consensus)), 1e-3)
    for j, ch in enumerate(consensus):
        for b in iupac[ch]: m["ACGT".index(b), j] = 1.0
    return m / m.sum(0, keepdims=True)
_df = pd.read_parquet(cfg["donor_parquet"])
def pwm_of(fn):
    p = np.frombuffer(_df[_df.filename == fn].iloc[0]["pwm"], np.float32).reshape(4, -1)
    L = int((p.sum(0) > 1e-6).sum()); return p[:, :L] if L >= 2 else p
rag = read_pwm(f"{cfg['output_dir']}/predictions/ZGLP1_prod_RAG_LGO.pwm.tsv")
gt = pwm_of(cfg["ground_truth_filename"])                 # ZGLP1 experimental H13CORE
gata_ex = pwm_of(cfg["gata_exemplar_filename"])           # GATA3 MA0037.3
shuf = rag[:, np.random.permutation(rag.shape[1])]
MOTIFS = {"ZGLP1 RAG": rag, "ZGLP1 H13CORE (exp)": gt, "GATA3 exemplar": gata_ex,
          "canonical GATA": gata_pwm(cfg["canonical_gata"]), "shuffled ZGLP1 RAG": shuf}

_TAB = np.full(256, 4, np.uint8)
for b, i in {65: 0, 67: 1, 71: 2, 84: 3}.items(): _TAB[b] = i
def encode(seq): return _TAB[np.frombuffer(seq.encode("latin1"), np.uint8)]
def logodds(pwm): return np.log2(np.clip(pwm, 1e-3, 1) / 0.25)
def _scan(idx, lo):
    L = lo.shape[1]
    if len(idx) < L: return -1e9
    win = np.lib.stride_tricks.sliding_window_view(idx, L)
    valid = (win < 4).all(1)
    if not valid.any(): return -1e9
    w = win[valid].astype(np.intp)
    return float(lo[w, np.arange(L)].sum(1).max())
def best_score(seq, lo): return max(_scan(encode(seq), lo), _scan(encode(revcomp(seq)), lo))

rows = []; per_seq = {}
for name, pwm in MOTIFS.items():
    lo = logodds(pwm)
    ts = np.array([best_score(s, lo) for s in tgt_seqs])
    bs = np.array([best_score(s, lo) for s in bg])
    per_seq[name] = (ts, bs)
    U, p = mannwhitneyu(ts, bs, alternative="greater")
    rows.append(dict(motif=name, auroc=round(U / (len(ts) * len(bs)), 3), mwu_p=f"{p:.2e}",
                     target_median=round(float(np.median(ts)), 2),
                     bg_median=round(float(np.median(bs)), 2)))
enr = pd.DataFrame(rows); enr.to_csv(f"{OUT}/promoter_scan_scores.tsv", sep="\t", index=False)
print("\nPromoter enrichment (oogenesis vs dinucleotide-shuffled background):")
print(enr.to_string(index=False))

# candidate target table (ranked by ZGLP1 RAG best score)
lo_rag = logodds(rag)
cand = sorted(((g, best_score(s, lo_rag)) for g, s in targets.items()), key=lambda x: -x[1])
pd.DataFrame([dict(rank=i+1, gene=g, zglp1_rag_best_logodds=round(sc, 2)) for i, (g, sc) in enumerate(cand)]
             ).to_csv(f"{OUT}/top_candidate_targets.tsv", sep="\t", index=False)
pd.DataFrame([dict(gene=g, chrom=tss[g][0], tss=tss[g][1], strand=tss[g][2]) for g in targets]
             ).to_csv(f"{OUT}/germ_cell_gene_set.tsv", sep="\t", index=False)

# ── verdict: is ZGLP1 RAG enriched beyond the composition (shuffled) control? ──
ei = enr.set_index("motif")
au_rag = ei.loc["ZGLP1 RAG", "auroc"]; au_shuf = ei.loc["shuffled ZGLP1 RAG", "auroc"]
p_rag = float(ei.loc["ZGLP1 RAG", "mwu_p"])
enriched = (p_rag < 0.05) and (au_rag - au_shuf > 0.03)
verdict = ("composition-INDEPENDENT enrichment" if enriched
           else "no enrichment beyond composition")
print(f"\nVERDICT: ZGLP1 RAG AUROC={au_rag} vs shuffled {au_shuf} (p={p_rag:.1e}) -> {verdict}")

# ── panel ─────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.9), gridspec_kw=dict(width_ratios=[1, 1.15], wspace=0.5))
ts, bs = per_seq["ZGLP1 RAG"]
ax[0].boxplot([bs, ts], tick_labels=["dinucleotide-\nshuffled", "oogenesis\npromoters"],
              widths=0.6, showfliers=False, patch_artist=True,
              boxprops=dict(facecolor="#cfe3d6"), medianprops=dict(color="k"))
ax[0].set_ylabel("best ZGLP1-RAG log-odds", fontsize=7)
ax[0].set_title(f"ZGLP1-RAG motif scan\nAUROC={au_rag} ({'sig.' if enriched else 'n.s.'})", fontsize=7.5)
ax[0].tick_params(labelsize=6.5)
order = ["ZGLP1 RAG", "ZGLP1 H13CORE (exp)", "GATA3 exemplar", "canonical GATA", "shuffled ZGLP1 RAG"]
au_v = [ei.loc[m, "auroc"] for m in order]
cols = ["#c0392b", "#7b3fa0", "#3b5b92", "#4c9f70", "#dcdcdc"]
ax[1].barh(range(len(order))[::-1], au_v, color=cols, edgecolor="k", lw=0.4)
ax[1].set_yticks(range(len(order))[::-1]); ax[1].set_yticklabels(order, fontsize=6.5)
ax[1].axvline(0.5, color="k", ls=":", lw=0.8); ax[1].set_xlim(0.3, 1.0)
ax[1].set_xlabel("AUROC (oogenesis vs dinuc-shuffled)", fontsize=7); ax[1].tick_params(labelsize=6.5)
ax[1].set_title("motif vs shuffled control", fontsize=7.5)
fig.suptitle(f"Germ-cell promoter scan — {verdict}", fontsize=8.5, y=1.06)
dest = FIG if enriched else EXT
fig.savefig(f"{dest}/ZGLP1_panel_promoter_enrichment.pdf", bbox_inches="tight")
fig.savefig(f"{dest}/ZGLP1_panel_promoter_enrichment.png", dpi=300, bbox_inches="tight")
print(f"wrote {dest}/ZGLP1_panel_promoter_enrichment.pdf "
      f"({'main figure' if enriched else 'Extended Data'})")
print("top candidate oogenesis promoters:", ", ".join(g for g, _ in cand[:8]))
