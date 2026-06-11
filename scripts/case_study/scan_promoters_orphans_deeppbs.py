#!/usr/bin/env python
"""Promoter enrichment for each orphan's predicted motif (ADNP2, ZHX2, ZHX3).

Per orphan: scan TSS+/-1 kb (hg38) promoters of a biologically matched gene set
(ADNP2 = neurodevelopmental; ZHX2/ZHX3 = their hepatic/tumour-suppressor target
networks) with the RAG motif + canonical TAAT + a column-shuffled control, against a
dinucleotide-preserving shuffled background (GC + CpG fixed). Verdict per orphan is
data-driven. Candidate elements only; not in-vivo occupancy.
"""
import os, sys, json, urllib.request, urllib.parse, random
sys.path.insert(0, "scripts/case_study")
import numpy as np, pandas as pd, yaml
from scipy.stats import mannwhitneyu
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

cfg = yaml.safe_load(open("configs/case_study_orphans_deeppbs.yaml"))
OUT = f"{cfg['output_dir']}/targets"; FIG = cfg["figure_dir"]; EXT = cfg["extended_figure_dir"]
os.makedirs(OUT, exist_ok=True); os.makedirs(EXT, exist_ok=True)
GENOME = "/n/holylabs/lpinello_lab/Lab/leihuang/WholeGenomeFasta/genome.fa"
random.seed(0); np.random.seed(0); WIN = 1000
plt.rcParams.update({"font.size": 8, "pdf.fonttype": 42})

FAI = {}
for ln in open(GENOME + ".fai"):
    n, L, off, lb, lw = ln.split("\t"); FAI[n] = (int(L), int(off), int(lb), int(lw))
_gf = open(GENOME, "rb")
def fetch_fast(chrom, start, end):
    if chrom not in FAI: return None
    L, off, lb, lw = FAI[chrom]; start = max(0, start); end = min(L, end)
    if end <= start: return None
    b0 = off + (start // lb) * lw + (start % lb); b1 = off + (end // lb) * lw + (end % lb)
    _gf.seek(b0); return _gf.read(b1 - b0).decode().replace("\n", "").upper()
def get_tss(symbols):
    data = urllib.parse.urlencode({"q": ",".join(sorted(set(symbols))), "scopes": "symbol",
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
        out[r.get("query")] = (chrom, int(gp["end"] if gp["strand"] == -1 else gp["start"]), int(gp["strand"]))
    return out
def revcomp(s): return s.translate(str.maketrans("ACGT", "TGCA"))[::-1]
def dinuc_shuffle(seq, rng):
    from collections import defaultdict
    s = [c if c in "ACGT" else "A" for c in seq]; n = len(s)
    if n < 3: return "".join(s)
    last = s[-1]; graph = defaultdict(list)
    for i in range(n - 1): graph[s[i]].append(s[i + 1])
    while True:
        last_edge = {v: rng.choice(graph[v]) for v in graph if v != last}; good = True
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
        if v != last: e.remove(last_edge[v]); rng.shuffle(e); e.append(last_edge[v])
        else: rng.shuffle(e)
        edge_lists[v] = e
    out = [s[0]]; cur = s[0]
    for _ in range(n - 1):
        nxt = edge_lists[cur][ptr[cur]]; ptr[cur] += 1; out.append(nxt); cur = nxt
    return "".join(out)
def read_pwm(path): return pd.read_csv(path, sep="\t", index_col=0)[list("ACGT")].values.T
def hd_pwm(c="TAATTA"):
    iu = {"A":"A","C":"C","G":"G","T":"T","W":"AT","R":"AG","Y":"CT"}
    m = np.full((4, len(c)), 1e-3)
    for j, ch in enumerate(c):
        for b in iu[ch]: m["ACGT".index(b), j] = 1.0
    return m / m.sum(0, keepdims=True)
_TAB = np.full(256, 4, np.uint8)
for b, i in {65: 0, 67: 1, 71: 2, 84: 3}.items(): _TAB[b] = i
def encode(seq): return _TAB[np.frombuffer(seq.encode("latin1"), np.uint8)]
def logodds(p): return np.log2(np.clip(p, 1e-3, 1) / 0.25)
def _scan(idx, lo):
    L = lo.shape[1]
    if len(idx) < L: return -1e9
    win = np.lib.stride_tricks.sliding_window_view(idx, L); v = (win < 4).all(1)
    if not v.any(): return -1e9
    w = win[v].astype(np.intp); return float(lo[w, np.arange(L)].sum(1).max())
def best_score(seq, lo): return max(_scan(encode(seq), lo), _scan(encode(revcomp(seq)), lo))

taat = hd_pwm(cfg["canonical_homeodomain"])
allrows = []
for o in cfg["orphans"]:
    gene = o["gene"]; genes = cfg["promoter_sets"][gene]
    tss = get_tss(genes)
    targets = {g: s for g, (c, t, st) in tss.items()
               if (s := fetch_fast(c, t - WIN, t + WIN)) and s.count("N") < 0.1 * len(s)}
    tgt = list(targets.values())
    rng = random.Random(0); bg = [dinuc_shuffle(s, rng) for s in tgt for _ in range(5)]
    rag = read_pwm(f"{cfg['output_dir']}/predictions/{gene}_RAG_LGO.pwm.tsv")
    shuf = rag[:, np.random.permutation(rag.shape[1])]
    MOT = {f"{gene} RAG": rag, "canonical TAAT": taat, f"shuffled {gene} RAG": shuf}
    res = {}
    for name, pwm in MOT.items():
        lo = logodds(pwm)
        ts = np.array([best_score(s, lo) for s in tgt]); bs = np.array([best_score(s, lo) for s in bg])
        U, p = mannwhitneyu(ts, bs, alternative="greater")
        res[name] = (round(U / (len(ts) * len(bs)), 3), p)
    au_rag, p_rag = res[f"{gene} RAG"]; au_shuf = res[f"shuffled {gene} RAG"][0]
    enriched = (p_rag < 0.05) and (au_rag - au_shuf > 0.03)
    lo_rag = logodds(rag)
    cand = sorted(((g, best_score(s, lo_rag)) for g, s in targets.items()), key=lambda x: -x[1])
    allrows.append(dict(gene=gene, n_promoters=len(targets), auroc_RAG=au_rag, p_RAG=f"{p_rag:.2g}",
                        auroc_canonicalTAAT=res["canonical TAAT"][0], auroc_shuffled=au_shuf,
                        verdict=("enriched" if enriched else "n.s. (composition)"),
                        top_candidates=";".join(g for g, _ in cand[:6])))
    print(f"{gene}: {len(targets)} promoters | RAG AUROC={au_rag} p={p_rag:.2g} | "
          f"shuffled {au_shuf} | TAAT {res['canonical TAAT'][0]} -> "
          f"{'ENRICHED' if enriched else 'n.s.'}")

summary = pd.DataFrame(allrows)
summary.to_csv(f"{OUT}/orphan_promoter_scan_summary.tsv", sep="\t", index=False)

# combined panel (one bar group per orphan)
fig, ax = plt.subplots(figsize=(6.4, 3.0))
x = np.arange(len(summary)); w = 0.27
ax.bar(x - w, summary.auroc_RAG, w, color="#c0392b", ec="k", lw=0.4, label="RAG motif")
ax.bar(x, summary.auroc_canonicalTAAT, w, color="#4c9f70", ec="k", lw=0.4, label="canonical TAAT")
ax.bar(x + w, summary.auroc_shuffled, w, color="#dcdcdc", ec="k", lw=0.4, label="shuffled control")
ax.axhline(0.5, color="k", ls=":", lw=0.8)
ax.set_xticks(x); ax.set_xticklabels(summary.gene, fontsize=8)
ax.set_ylabel("AUROC (targets vs dinuc-shuffled)", fontsize=8); ax.set_ylim(0.3, 0.8)
ax.legend(fontsize=6.5, loc="upper right"); ax.set_title("Orphan promoter scans (composition-controlled)", fontsize=9)
fig.savefig(f"{EXT}/orphan_promoter_scans.pdf", bbox_inches="tight")
fig.savefig(f"{EXT}/orphan_promoter_scans.png", dpi=300, bbox_inches="tight")
print(f"\nwrote {EXT}/orphan_promoter_scans.pdf")
print(summary.to_string(index=False))
