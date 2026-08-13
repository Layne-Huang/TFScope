#!/usr/bin/env python
"""Stage 2: cluster-aware DBD cropping (local; no network).

Problem it fixes: the old rule cropped from the FIRST annotated domain to the
LAST, which swallows everything between separated clusters. HIVEP1 is the
worst case seen: 5 zinc-finger fragments in 3 clusters separated by 501 and
1101 residues -> a 1736-residue "DBD" of which only ~134 residues (8%) are
actually zinc finger.

Algorithm
  1. sort fragments; start a new cluster whenever the gap to the previous
     fragment exceeds GAP_THRESHOLD
  2. single cluster -> use it (the common, already-correct case: tandem arrays
     like CTCF's zinc fingers have ~7-residue TGEKP linkers)
  3. multiple clusters -> keep the LARGEST by residue span; ties broken by
     closeness to the family's typical structural DBD length measured from
     tf_pwm_deeppbs_v2_deduped.parquet (real DNA-contact crops)
  4. set crop_ambiguous=True when the runner-up cluster is within AMBIG_FRAC
     of the winner, so the guess is recorded in the data rather than hidden

Honest limitation: without a structure we cannot know which cluster actually
contacts DNA in the complex that produced the PWM. Real structures often show
a subset of fingers spanning parts of two clusters (deeppbs_v2's CTCF crops
cover ~4-5 of its 11 fingers). This is a principled heuristic, not truth --
hence the flag.
"""
import json
import numpy as np
import pandas as pd

FRAGS = "/tmp/domain_fragments.jsonl"
GAP_THRESHOLD = 40      # max residues between fragments of one cluster
AMBIG_FRAC = 0.20       # runner-up within 20% of winner -> flag


def load_fragments(path=FRAGS):
    out = {}
    for line in open(path):
        try: r = json.loads(line)
        except Exception: continue
        if "error" in r or not r.get("frags"):
            continue
        out[r["gene"]] = r
    return out


def cluster(frags, gap=GAP_THRESHOLD):
    """frags: sorted [[start,end],...] -> list of (start, end, n_fragments)."""
    if not frags:
        return []
    clusters = [[frags[0][0], frags[0][1], 1]]
    for a, b in frags[1:]:
        if a - clusters[-1][1] <= gap:
            clusters[-1][1] = max(clusters[-1][1], b)
            clusters[-1][2] += 1
        else:
            clusters.append([a, b, 1])
    return [tuple(c) for c in clusters]


def family_target_lengths():
    """Median real structural crop length per family, from deeppbs_v2."""
    st = pd.read_parquet("data/processed/tf_pwm_deeppbs_v2_deduped.parquet")
    main = pd.read_parquet("data/processed/tf_pwm_aug_dbd_canon_trim.parquet")
    g2f = {}
    for _, r in main.iterrows():
        g2f.setdefault(str(r["gene_symbol"]).upper(), r["family_name"])
    st["family"] = st["gene"].str.upper().map(g2f)
    return st.groupby("family")["seq_length"].median().to_dict()


def choose(clusters, target_len=None):
    """Return (start, end, ambiguous). Largest span wins; ties -> closest to target."""
    if len(clusters) == 1:
        c = clusters[0]
        return c[0], c[1], False
    sizes = [(c[1] - c[0], c) for c in clusters]
    sizes.sort(key=lambda x: -x[0])
    best_size, best = sizes[0]
    runner_size = sizes[1][0]
    ambiguous = runner_size >= best_size * (1 - AMBIG_FRAC)
    if ambiguous and target_len:
        # tie-break on closeness to the family's real structural length
        cand = [c for s, c in sizes if s >= best_size * (1 - AMBIG_FRAC)]
        best = min(cand, key=lambda c: abs((c[1] - c[0]) - target_len))
    return best[0], best[1], ambiguous


def main():
    data = load_fragments()
    print(f"genes with fragments: {len(data)}", flush=True)

    # --- gap distribution: verify the bimodality seen on the first 8 genes ---
    gaps = []
    for r in data.values():
        f = r["frags"]
        gaps += [f[i+1][0] - f[i][1] for i in range(len(f) - 1)]
    gaps = np.array([g for g in gaps if g >= 0])
    if len(gaps):
        print(f"\ngap distribution over {len(gaps)} inter-fragment gaps:")
        for q in [10, 25, 50, 75, 90, 95, 99]:
            print(f"  p{q}: {np.percentile(gaps, q):.0f}")
        for lo, hi in [(0,10),(10,20),(20,40),(40,60),(60,100),(100,200),(200,500),(500,10**9)]:
            print(f"  gap {lo}-{hi}: {((gaps>=lo)&(gaps<hi)).sum()}")

    targets = family_target_lengths()
    aug = pd.read_parquet("data/processed/tf_pwm_aug_dbd_canon_trim_v2.parquet")
    g2fam = dict(zip(aug["gene_symbol"].str.upper(), aug["family_name"]))

    rows = []
    for gene, r in data.items():
        cl = cluster(r["frags"])
        fam = g2fam.get(gene.upper())
        tgt = targets.get(fam)
        s, e, amb = choose(cl, tgt)
        old_span = r["frags"][-1][1] - r["frags"][0][0]
        rows.append({"gene": gene, "uniprot_id": r["uniprot_id"],
                      "n_clusters": len(cl), "old_span": old_span,
                      "new_len": e - s, "saved": old_span - (e - s),
                      "start": s, "end": e, "crop": r["seq"][s:e],
                      "families": "_".join(r["families"]),
                      "n_frag_in_crop": sum(1 for a,b in r["frags"] if a>=s and b<=e),
                      "crop_ambiguous": amb})
    df = pd.DataFrame(rows)
    multi = df[df["n_clusters"] > 1]
    print(f"\ngenes total={len(df)}  single-cluster={len(df)-len(multi)}  multi-cluster={len(multi)}")
    print(f"ambiguous (runner-up within {int(AMBIG_FRAC*100)}%): {df['crop_ambiguous'].sum()}")
    if len(multi):
        print(f"\nresidues saved on multi-cluster genes: total={multi['saved'].sum()} "
              f"median={multi['saved'].median():.0f} max={multi['saved'].max()}")
        print("\nbiggest reductions:")
        print(multi.sort_values("saved", ascending=False)
                   [["gene","n_clusters","old_span","new_len","saved","crop_ambiguous","families"]]
                   .head(15).to_string(index=False))
    df.to_parquet("/tmp/cluster_crops.parquet")
    print("\nsaved /tmp/cluster_crops.parquet")


if __name__ == "__main__":
    main()
