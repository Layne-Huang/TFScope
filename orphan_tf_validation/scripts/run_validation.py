"""Phase-1 MVP: validate TFScope orphan-TF motifs against public ChIP-seq occupancy.
For ADNP, ZHX2, ZHX3: summit +/-250bp peaks -> real sequences; per-peak dinucleotide-shuffle
control; MOODS scan with the TFScope PWM; composition-controlled enrichment (log2 E, z, emp p),
summit-centered density, and a 100x column-shuffled-PWM null (percentile of the real motif).

Usage: python scripts/run_validation.py [TF ...]   (default: all three)
Outputs: results/enrichment/<TF>.json  + results/orphan_chip_validation.json (combined)
"""
import os, sys, json, subprocess
import numpy as np
import MOODS.scan, MOODS.tools
sys.path.insert(0, os.path.dirname(__file__))
from lib_shuffle import dinucleotide_shuffle

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GENOME = "/data1/leihuang/WholeGenomeFasta/genome.fa"
PWM_DIR = os.path.join(ROOT, "..", "results", "genome_cre_scan", "pwms")
WIN = 250                      # +/- around summit (500 bp)
PVAL = 1e-4
N_SHUF = 20                    # dinucleotide shuffles per peak (enrichment + z)
N_NULL = 100                   # column-shuffled PWMs
NULL_PEAKS = 2000              # peaks used for the (heavier) null-PWM scan
MAIN_PEAKS = 8000              # peaks used for enrichment + centrality
BG = [0.295, 0.205, 0.205, 0.295]
CHROMS = {f"chr{c}" for c in list(range(1, 23)) + ["X", "Y"]}
rng = np.random.default_rng(42)
TFS = sys.argv[1:] or ["ADNP", "ZHX2", "ZHX3"]


def read_chrom_sizes():
    sz = {}
    for line in open(GENOME + ".fai"):
        c, n = line.split()[:2]; sz[c] = int(n)
    return sz


def peaks_to_windows(tf, sizes, cap):
    rows = []
    for line in open(f"{ROOT}/data/raw_peaks/{tf}.bed"):
        p = line.split()
        chrom = p[0]
        if chrom not in CHROMS: continue
        start, end = int(p[1]), int(p[2]); sig = float(p[6])
        off = int(p[9]); summit = start + (off if off >= 0 else (end - start) // 2)
        s, e = summit - WIN, summit + WIN
        if s < 0 or e > sizes.get(chrom, 0): continue
        rows.append((chrom, s, e, sig))
    rows.sort(key=lambda r: -r[3])
    rows = rows[:cap]
    bed = f"{ROOT}/data/processed_peaks/{tf}.{2*WIN}bp.bed"
    with open(bed, "w") as o:
        for i, (c, s, e, sig) in enumerate(rows):
            o.write(f"{c}\t{s}\t{e}\t{tf}_{i}\t{sig:.2f}\t+\n")
    return bed, len(rows)


def getfasta(bed):
    out = bed.replace(".bed", ".fa")
    subprocess.run(["bedtools", "getfasta", "-fi", GENOME, "-bed", bed, "-fo", out],
                   check=True, capture_output=True)
    seqs = []
    for line in open(out):
        if not line.startswith(">"): seqs.append(line.strip().upper())
    return seqs


def scanner_for(mats_pwm):
    mats, owner, thr = [], [], []
    for name, P in mats_pwm:
        cm = (P / P.sum(0, keepdims=True)).tolist()
        lo = MOODS.tools.log_odds(cm, BG, 0.01); t = MOODS.tools.threshold_from_p(lo, BG, PVAL)
        for m in (lo, MOODS.tools.reverse_complement(lo)):
            mats.append(m); owner.append(name); thr.append(t)
    sc = MOODS.scan.Scanner(7); sc.set_motifs(mats, BG, thr)
    return sc, owner


def count_hits(sc, owner, seqs, names):
    h = {n: 0 for n in names}
    for s in seqs:
        if len(s) < 5: continue
        for mi, mm in enumerate(sc.scan(s)):
            h[owner[mi]] += len(mm)
    return h


def col_shuffle(P):
    return P[:, rng.permutation(P.shape[1])]


sizes = read_chrom_sizes()
combined = {}
for tf in TFS:
    P = np.load(f"{PWM_DIR}/{tf}.npy"); P = P / P.sum(0, keepdims=True)
    L = P.shape[1]
    bed, npk = peaks_to_windows(tf, sizes, MAIN_PEAKS)
    real = getfasta(bed)
    print(f"[{tf}] {npk} peaks (summit +/-{WIN}), motif L={L}")

    # ---- main enrichment + summit centrality (real PWM only, all peaks, N_SHUF) ----
    sc1, own1 = scanner_for([("M", P)])
    real_total = count_hits(sc1, own1, real, ["M"])["M"]
    shuf_counts = []
    for r in range(N_SHUF):
        surr = [dinucleotide_shuffle(s, rng) for s in real]
        shuf_counts.append(count_hits(sc1, own1, surr, ["M"])["M"])
    shuf_counts = np.array(shuf_counts, float)
    bp = sum(len(s) for s in real)
    enr = real_total / (shuf_counts.mean() + 1e-9)
    z = (real_total - shuf_counts.mean()) / (shuf_counts.std() + 1e-9)
    emp_p = (1 + np.sum(shuf_counts >= real_total)) / (1 + N_SHUF)

    # summit centrality: distance of each hit center to window center
    dists = []
    for s in real:
        if len(s) < L: continue
        best = None
        for mi, mm in enumerate(sc1.scan(s)):
            for match in mm:
                pos, score = match.pos, match.score
                c = pos + L / 2 - len(s) / 2          # signed distance to window center
                if best is None or score > best[1]: best = (c, score)
        if best is not None: dists.append(best[0])
    dists = np.array(dists)
    frac50 = float(np.mean(np.abs(dists) <= 50)) if len(dists) else 0.0
    frac100 = float(np.mean(np.abs(dists) <= 100)) if len(dists) else 0.0

    # ---- null-PWM percentile (real + 100 column-shuffled PWMs, fewer peaks) ----
    real_n = real[:NULL_PEAKS]
    nmeff = min(10, N_SHUF)
    names = ["REAL"] + [f"N{i}" for i in range(N_NULL)]
    mats_pwm = [("REAL", P)] + [(f"N{i}", col_shuffle(P)) for i in range(N_NULL)]
    sc2, own2 = scanner_for(mats_pwm)
    rc = count_hits(sc2, own2, real_n, names)
    sc_tot = {n: 0 for n in names}
    for r in range(nmeff):
        surr = [dinucleotide_shuffle(s, rng) for s in real_n]
        h = count_hits(sc2, own2, surr, names)
        for n in names: sc_tot[n] += h[n]
    nullE = []
    realE_sub = rc["REAL"] / ((sc_tot["REAL"] / nmeff) + 1e-9)
    for i in range(N_NULL):
        e = rc[f"N{i}"] / ((sc_tot[f"N{i}"] / nmeff) + 1e-9)
        nullE.append(e)
    nullE = np.array(nullE)
    pct = float(np.mean(nullE < realE_sub))

    res = dict(tf=tf, n_peaks=npk, motif_len=L, peak_bp=bp,
               real_hits=int(real_total), shuf_mean=round(float(shuf_counts.mean()), 1),
               log2_enrich=round(float(np.log2(enr + 1e-9)), 3), enrich=round(float(enr), 3),
               z=round(float(z), 2), emp_p=round(float(emp_p), 4),
               frac_hits_within_50bp=round(frac50, 3), frac_hits_within_100bp=round(frac100, 3),
               n_hits_centrality=len(dists),
               null_pwm_percentile=round(pct, 3), real_enrich_sub=round(float(realE_sub), 3),
               null_enrich_p95=round(float(np.percentile(nullE, 95)), 3))
    json.dump(dict(res, dist_hist=np.histogram(dists, bins=np.arange(-WIN, WIN + 25, 25))[0].tolist(),
                   null_enrich=[round(float(x), 4) for x in nullE],
                   shuf_counts=[int(x) for x in shuf_counts]),
              open(f"{ROOT}/results/enrichment/{tf}.json", "w"), indent=1)
    combined[tf] = res
    print(f"  log2E={res['log2_enrich']:+.2f} (z={res['z']}, p={res['emp_p']}) | "
          f"central<=50bp={res['frac_hits_within_50bp']:.2f} | nullPWM pctile={res['null_pwm_percentile']:.2f}")

json.dump(combined, open(f"{ROOT}/results/orphan_chip_validation.json", "w"), indent=1)
print(f"\nsaved {ROOT}/results/orphan_chip_validation.json")
