#!/usr/bin/env python
"""Figure 5e — germ-cell promoter enrichment for the SOHLH1 predicted motif.

Scans promoters (TSS +/-1 kb) of curated germ-cell / SOHLH1-pathway genes with the
SOHLH1 RAG PWM and references, vs a GC-matched random-genomic background, and reports
AUROC / Mann-Whitney enrichment + a ranked candidate-target table.

NOTE: this nominates candidate regulatory elements; it does NOT establish in vivo
SOHLH1 occupancy. hg38 = /n/holylabs/.../WholeGenomeFasta/genome.fa (+ .fai).
"""
import os, sys, json, urllib.request, urllib.parse, random
sys.path.insert(0, "scripts/case_study"); sys.path.insert(0, "pwm_rosetta")
import numpy as np, pandas as pd, yaml
from scipy.stats import mannwhitneyu
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

cfg = yaml.safe_load(open("configs/case_study_sohlh1.yaml"))
OUT = f"{cfg['output_dir']}/targets"; FIG = cfg["figure_dir"]; os.makedirs(OUT, exist_ok=True)
GENOME = "/n/holylabs/lpinello_lab/Lab/leihuang/WholeGenomeFasta/genome.fa"
random.seed(0); np.random.seed(0)
WIN = 1000                          # TSS +/- WIN
plt.rcParams.update({"font.size": 8, "pdf.fonttype": 42})

# curated germ-cell / SOHLH1-pathway genes (oocyte/spermatogonia regulators + SOHLH1 targets)
GERM = ["SOHLH1","SOHLH2","LHX8","NOBOX","FIGLA","KIT","KITLG","ZBTB16","GDF9","BMP15",
        "ZP1","ZP2","ZP3","DAZL","DDX4","SYCP3","STRA8","NANOS2","NANOS3","DPPA3",
        "MAEL","PIWIL1","PIWIL2","TDRD1","GFRA1","DMRT1","SYCP1","TAF4B","FOXL2","SOX3",
        "INHA","NR5A1","H1-6","STK31","RFX2"]

# ── faidx fetcher (no deps) ──────────────────────────────────────────────────
FAI = {}
for ln in open(GENOME + ".fai"):
    n, L, off, lb, lw = ln.split("\t"); FAI[n] = (int(L), int(off), int(lb), int(lw))
_gf = open(GENOME, "rb")
def fetch(chrom, start, end):       # 0-based, end-exclusive
    if chrom not in FAI: return None
    L, off, lb, lw = FAI[chrom]
    start = max(0, start); end = min(L, end)
    if end <= start: return None
    seq = []
    for p in range(start, end):
        _gf.seek(off + (p // lb) * lw + (p % lb)); seq.append(_gf.read(1).decode())
    return "".join(seq).upper()
def fetch_fast(chrom, start, end):  # contiguous read (faster)
    if chrom not in FAI: return None
    L, off, lb, lw = FAI[chrom]; start = max(0, start); end = min(L, end)
    if end <= start: return None
    b0 = off + (start // lb) * lw + (start % lb)
    b1 = off + (end // lb) * lw + (end % lb)
    _gf.seek(b0); raw = _gf.read(b1 - b0).decode()
    return raw.replace("\n", "").upper()

# ── TSS via mygene ───────────────────────────────────────────────────────────
def get_tss(symbols):
    data = urllib.parse.urlencode({"q": ",".join(symbols), "scopes": "symbol",
                                   "fields": "genomic_pos", "species": "human"}).encode()
    req = urllib.request.Request("https://mygene.info/v3/query", data=data,
                                 headers={"Content-Type": "application/x-www-form-urlencoded"})
    res = json.loads(urllib.request.urlopen(req, timeout=60).read().decode())
    out = {}
    for r in res:
        gp = r.get("genomic_pos");
        if not gp: continue
        gp = gp[0] if isinstance(gp, list) else gp
        chrom = "chr" + str(gp["chr"])
        if chrom not in FAI: continue
        tss = gp["end"] if gp["strand"] == -1 else gp["start"]
        out[r.get("query")] = (chrom, int(tss), int(gp["strand"]))
    return out

tss = get_tss(GERM)
print(f"resolved TSS for {len(tss)}/{len(GERM)} germ-cell genes")

def gc(s): return (s.count("G") + s.count("C")) / max(1, len(s))
def revcomp(s): return s.translate(str.maketrans("ACGT", "TGCA"))[::-1]

# target promoters
targets = {}
for g, (c, t, st) in tss.items():
    s = fetch_fast(c, t - WIN, t + WIN)
    if s and s.count("N") < 0.1 * len(s):
        targets[g] = s
print(f"target promoters: {len(targets)}")
tgt_seqs = list(targets.values()); tgt_gc = np.array([gc(s) for s in tgt_seqs])

# Background = dinucleotide-preserving shuffles of the target promoters.
# This holds GC *and* CpG content fixed per-sequence (the E-box CACGTG contains a
# CpG, and germ-cell/CpG-island promoters are CpG-rich), so any residual signal is
# motif-specific rather than composition-driven.
def dinuc_shuffle(seq, rng):
    """Altschul-Erikson Eulerian dinucleotide-preserving shuffle (correct form)."""
    from collections import defaultdict
    s = [c if c in "ACGT" else "A" for c in seq]
    n = len(s)
    if n < 3: return "".join(s)
    last = s[-1]
    graph = defaultdict(list)
    for i in range(n - 1): graph[s[i]].append(s[i + 1])
    # choose a random arborescence to `last` via per-vertex last-edges
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
    # shuffle remaining edges, place the chosen last-edge at the tail of each list
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

_rng = random.Random(0)
K_SHUF = 5
bg = []
for s in tgt_seqs:
    for _ in range(K_SHUF):
        bg.append(dinuc_shuffle(s, _rng))
print(f"dinucleotide-shuffled background: {len(bg)} (target GC {tgt_gc.mean():.2f}, "
      f"bg GC {np.mean([gc(s) for s in bg]):.2f})")

# ── motifs -> log-odds; best score per sequence (both strands) ────────────────
def read_pwm(path): return pd.read_csv(path, sep="\t", index_col=0)[list("ACGT")].values.T
def ebox():
    m = np.full((4, 6), 1e-3)
    for j, ch in enumerate("CACGTG"): m["ACGT".index(ch), j] = 1.0
    return m / m.sum(0, keepdims=True)
rag = read_pwm(f"{cfg['output_dir']}/predictions/SOHLH1_RAG_LGO.pwm.tsv")
norag = read_pwm(f"{cfg['output_dir']}/predictions/SOHLH1_noRAG.pwm.tsv")
_df = pd.read_parquet(cfg["donor_parquet"])
s2 = np.frombuffer(_df[_df.filename == cfg["paralog_reference_filename"]].iloc[0]["pwm"], np.float32).reshape(4, -1)
shuf = rag[:, np.random.permutation(rag.shape[1])]
MOTIFS = {"SOHLH1 RAG": rag, "SOHLH1 noRAG": norag, "SOHLH2 JASPAR": s2,
          "canonical E-box": ebox(), "shuffled SOHLH1 RAG": shuf}

_TAB = np.full(256, 4, np.uint8)
for b, i in {65: 0, 67: 1, 71: 2, 84: 3}.items(): _TAB[b] = i   # A C G T
def encode(seq): return _TAB[np.frombuffer(seq.encode("latin1"), np.uint8)]
def logodds(pwm): return np.log2(np.clip(pwm, 1e-3, 1) / 0.25)
def _scan_strand(idx, lo):
    L = lo.shape[1]
    if len(idx) < L: return -1e9
    win = np.lib.stride_tricks.sliding_window_view(idx, L)    # (n, L)
    valid = (win < 4).all(1)
    if not valid.any(): return -1e9
    w = win[valid].astype(np.intp)
    sc = lo[w, np.arange(L)].sum(1)                           # (n_valid,)
    return float(sc.max())
def best_score(seq, lo):
    return max(_scan_strand(encode(seq), lo), _scan_strand(encode(revcomp(seq)), lo))

rows = []; per_seq = {}
for name, pwm in MOTIFS.items():
    lo = logodds(pwm)
    ts = np.array([best_score(s, lo) for s in tgt_seqs])
    bs = np.array([best_score(s, lo) for s in bg])
    per_seq[name] = (ts, bs)
    U, p = mannwhitneyu(ts, bs, alternative="greater")
    auroc = U / (len(ts) * len(bs))
    rows.append(dict(motif=name, auroc=round(auroc, 3), mwu_p=f"{p:.2e}",
                     target_median=round(float(np.median(ts)), 2),
                     bg_median=round(float(np.median(bs)), 2)))
enr = pd.DataFrame(rows)
enr.to_csv(f"{OUT}/promoter_scan_scores.tsv", sep="\t", index=False)
print("\nPromoter enrichment (target germ-cell vs GC-matched background):")
print(enr.to_string(index=False))

# candidate target table (germ-cell genes ranked by SOHLH1 RAG best score)
lo_rag = logodds(rag)
cand = sorted(((g, best_score(s, lo_rag)) for g, s in targets.items()), key=lambda x: -x[1])
pd.DataFrame([dict(rank=i+1, gene=g, sohlh1_rag_best_logodds=round(sc, 2)) for i, (g, sc) in enumerate(cand)]
             ).to_csv(f"{OUT}/top_candidate_targets.tsv", sep="\t", index=False)
pd.DataFrame([dict(gene=g, chrom=tss[g][0], tss=tss[g][1], strand=tss[g][2]) for g in targets]
             ).to_csv(f"{OUT}/germ_cell_gene_set.tsv", sep="\t", index=False)

# ── panel (Extended Data: composition-controlled, honest negative) ───────────
EXT = cfg["extended_figure_dir"]; os.makedirs(EXT, exist_ok=True)
fig, ax = plt.subplots(1, 2, figsize=(6.6, 2.9), gridspec_kw=dict(width_ratios=[1, 1.15], wspace=0.5))
ts, bs = per_seq["SOHLH1 RAG"]
ax[0].boxplot([bs, ts], tick_labels=["dinucleotide-\nshuffled", "germ-cell\npromoters"],
              widths=0.6, showfliers=False, patch_artist=True,
              boxprops=dict(facecolor="#cfd8e3"), medianprops=dict(color="k"))
ax[0].set_ylabel("best SOHLH1-RAG log-odds", fontsize=7)
au = enr.set_index("motif").loc["SOHLH1 RAG", "auroc"]
ax[0].set_title(f"SOHLH1-RAG motif scan\nAUROC={au} (n.s.)", fontsize=7.5)
ax[0].tick_params(labelsize=6.5)
order = ["SOHLH1 RAG", "SOHLH2 JASPAR", "canonical E-box", "SOHLH1 noRAG", "shuffled SOHLH1 RAG"]
au_v = [enr.set_index("motif").loc[m, "auroc"] for m in order]
cols = ["#c0392b", "#3b5b92", "#4c9f70", "#b0b0b0", "#dcdcdc"]
ax[1].barh(range(len(order))[::-1], au_v, color=cols, edgecolor="k", lw=0.4)
ax[1].set_yticks(range(len(order))[::-1]); ax[1].set_yticklabels(order, fontsize=6.5)
ax[1].axvline(0.5, color="k", ls=":", lw=0.8); ax[1].set_xlim(0.3, 1.0)
ax[1].set_xlabel("AUROC (germ-cell vs dinuc-shuffled)", fontsize=7); ax[1].tick_params(labelsize=6.5)
ax[1].set_title("motif vs shuffled control", fontsize=7.5)
fig.suptitle("Extended Data: germ-cell promoter scan — no enrichment beyond composition",
             fontsize=8.5, y=1.06)
for p, d in [("pdf", FIG), ("pdf", EXT)]:
    fig.savefig(f"{d}/panel_e_target_enrichment.{p}", bbox_inches="tight")
fig.savefig(f"{FIG}/panel_e_target_enrichment.png", dpi=300, bbox_inches="tight")
print(f"\nwrote {EXT}/panel_e_target_enrichment.pdf (Extended Data; honest negative)")
print("candidate germ-cell promoters by motif score (composition-confounded):",
      ", ".join(g for g, _ in cand[:8]))
