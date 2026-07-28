#!/usr/bin/env python
"""Rebuild partner_sequence for BOTH sets, consistent with our dimer decision,
then write tf_pwm_training_v2p.parquet (split-preserving).

STRUCTURE set (partner is OBSERVED in the co-crystal):
  attach the partner chain for EVERY is_dimer=True row via partner_chains --
  this is exactly find_dimer_partners' decision (shared primary DNA duplex +
  <=8A protein-protein contact). Homodimers get their self-chain (GCN4->chain
  D), heterodimers get the other gene (NFE2L2->MAFG). Monomers single-chain.

SEQUENCE set (no structure -> partner ASSIGNED by rule):
  only rows in a dimerizing family (bHLH/bZIP/Nuclear_Receptor) with a LONG
  motif (IC-core >= 13) -- the ones a single DBD cannot cover. Partner chosen
  by motif symmetry (both signals from the PWM we already have):
    palindromic (RC-symmetric, score >= PALIN_THR)  -> self-duplicate DBD
    asymmetric / direct-repeat                       -> canonical partner DBD
      NR -> RXRA ; CNC-bZIP -> MAFG ; class-II bHLH -> TCF3 ; bHLH-PAS -> ARNT
      (fallback self-duplicate when no curated partner)
  everything else single-chain.

Outputs:
  data/processed/tf_pwm_deeppbs_v2_partner.parquet   (structure, rebuilt)
  data/processed/tf_pwm_aug_partner_v2.parquet        (seq, with partner col)
  data/processed/tf_pwm_training_v2p.parquet          (merged, split-preserving)
"""
import os, sys, importlib.util, time
import numpy as np, pandas as pd

sys.path.insert(0, "src")
spec = importlib.util.spec_from_file_location("bd", "scripts/build_deeppbs_structural_v2.py")
bd = importlib.util.module_from_spec(spec); spec.loader.exec_module(bd)

STR = "data/processed/tf_pwm_deeppbs_v2_deduped.parquet"
AUG = "data/processed/tf_pwm_aug_dbd_canon_trim_v2.parquet"
TRAIN_IN = "data/processed/tf_pwm_training_v2.parquet"
CIF = "data/raw/pdb_cif_cache"
STR_OUT = "data/processed/tf_pwm_deeppbs_v2_partner.parquet"
AUG_OUT = "data/processed/tf_pwm_aug_partner_v2.parquet"
TRAIN_OUT = "data/processed/tf_pwm_training_v2p.parquet"

DIMFAM = {"bHLH", "bZIP", "Nuclear_Receptor"}
LONG_CORE = 13
PALIN_THR = 0.5
# subtype gene sets for curated heterodimer partners (seq set, asymmetric motif)
CNC_BZIP = {"NFE2", "NFE2L1", "NFE2L2", "NFE2L3", "BACH1", "BACH2"}
BHLH_CLASSII = {"MYOD1", "MYOG", "MYF5", "MYF6", "HAND1", "HAND2", "NHLH1", "NHLH2",
                "LYL1", "ASCL1", "ASCL2", "NEUROD1", "NEUROD2", "NEUROD4", "NEUROD6",
                "ATOH1", "ATOH7", "TAL1", "TAL2", "FERD3L", "MSC", "TWIST1", "TWIST2",
                "SCX", "PTF1A", "BHLHA15", "MYF6"}
BHLH_PAS = {"HIF1A", "HIF3A", "EPAS1", "ARNT", "ARNTL", "ARNTL2", "CLOCK", "NPAS1",
            "NPAS2", "NPAS3", "NPAS4", "AHR", "AHRR", "SIM1", "SIM2", "BMAL1"}


def as_pwm(x, ml):
    if isinstance(x, (bytes, bytearray)):
        a = np.frombuffer(x, dtype=np.float32).copy()
    else:
        a = np.asarray(x, dtype=np.float32).ravel()
    L = int(ml)
    if a.size == 4 * L:
        a4 = a.reshape(4, L)
        return a4 if np.allclose(a4.sum(0), 1, atol=0.1) else a.reshape(L, 4).T
    return a.reshape(4, -1)


def ic_core(p):
    pc = np.clip(p, 1e-8, 1); ic = 2 + (pc * np.log2(pc)).sum(0); inf = np.where(ic >= 0.25)[0]
    if len(inf) == 0:
        return p, 0
    core = p[:, inf[0]:inf[-1] + 1]
    return core, core.shape[1]


def palindrome(core):
    rc = core[[3, 2, 1, 0]][:, ::-1]
    return float(np.corrcoef(core.ravel(), rc.ravel())[0, 1])


# ---------- STRUCTURE SET: partner for every is_dimer row ----------
def rebuild_structure():
    st = pd.read_parquet(STR).reset_index(drop=True); st["G"] = st.gene.str.upper()
    pseq = [""] * len(st); pgene = [""] * len(st)
    # chain_id -> gene, per pdb, from the parquet rows (for labeling partners)
    chain_gene = {}
    for _, r in st.iterrows():
        chain_gene[(r["pdb_id"], r["chain_id"])] = r["G"]
    by_pdb = {}
    for i, r in st.iterrows():
        if bool(r["is_dimer"]):
            by_pdb.setdefault(r["pdb_id"], []).append((i, r["chain_id"]))
    print(f"[structure] {len(st)} rows | {sum(len(v) for v in by_pdb.values())} is_dimer rows "
          f"across {len(by_pdb)} pdbs", flush=True)
    ok = 0; t0 = time.time()
    for n, (pdb, rows) in enumerate(by_pdb.items(), 1):
        if n % 100 == 0 or n == len(by_pdb):
            print(f"  [{n}/{len(by_pdb)} pdbs] ok={ok} {time.time()-t0:.0f}s", flush=True)
        path = os.path.join(CIF, f"{str(pdb).lower()}.cif")
        if not os.path.exists(path):
            continue
        try:
            prot, dna, dna_by_chain = bd.load_chains(path)
            # find_dimer_partners returns a PROPER DBD crop per chain (own primary
            # duplex) + the authoritative partner list -- reuse both so partner
            # crops are sized exactly like the primary sequences.
            out, _, partners = bd.find_dimer_partners(prot, dna, dna_by_chain)
        except Exception:
            continue
        for i, cid in rows:
            plist = partners.get(cid, [])
            best, bg, bl = "", "", 0
            for pc in plist:
                if pc in out:
                    seq = out[pc][0]
                    if len(seq) > bl:
                        g = chain_gene.get((pdb, pc), st.at[i, "G"])
                        best, bg, bl = seq, g, len(seq)
            if best:
                pseq[i] = best; pgene[i] = bg; ok += 1
    st["partner_sequence"] = pseq; st["partner_gene_used"] = pgene
    st = st.drop(columns=["G"])
    has = st.partner_sequence.str.len() > 0
    homo = sum(1 for a, b in zip(st.gene.str.upper(), st.partner_gene_used) if b and a == b)
    print(f"[structure] partner set on {has.sum()} rows (homodimer~{homo}, hetero~{has.sum()-homo})")
    st.to_parquet(STR_OUT)
    return st


# ---------- SEQUENCE SET: motif-gated partners ----------
def build_sequence(canon):
    # reset index so label == position (seq_i filenames are positional)
    aug = pd.read_parquet(AUG).reset_index(drop=True)
    pseq = [""] * len(aug); pgene = [""] * len(aug)
    n_self = n_cur = 0
    for i, r in aug.iterrows():
        fam = str(r["family_name"]); gene = str(r["gene_symbol"]).upper()
        if fam not in DIMFAM:
            continue
        core, L = ic_core(as_pwm(r["pwm"], r["motif_length"]))
        if L < LONG_CORE:
            continue
        own = str(r["sequence"])
        if palindrome(core) >= PALIN_THR:
            pseq[i] = own; pgene[i] = gene + "(self)"; n_self += 1           # homodimer
            continue
        # asymmetric -> curated partner by subtype
        partner = None
        if fam == "Nuclear_Receptor":
            partner = ("RXRA", canon.get("RXRA"))
        elif fam == "bZIP" and gene in CNC_BZIP:
            partner = ("MAFG", canon.get("MAFG"))
        elif fam == "bHLH" and gene in BHLH_CLASSII:
            partner = ("TCF3", canon.get("TCF3"))
        elif fam == "bHLH" and gene in BHLH_PAS:
            partner = ("ARNT", canon.get("ARNT"))
        if partner and partner[1]:
            pseq[i] = partner[1]; pgene[i] = partner[0]; n_cur += 1
        else:
            pseq[i] = own; pgene[i] = gene + "(self)"; n_self += 1           # fallback
    aug["partner_sequence"] = pseq; aug["partner_gene_used"] = pgene
    has = aug.partner_sequence.str.len() > 0
    print(f"[sequence] partner set on {has.sum()} rows ({aug.loc[has,'gene_symbol'].str.upper().nunique()} genes) "
          f"| self-dup={n_self} curated-hetero={n_cur}")
    print("  curated partner usage:",
          aug.loc[has & ~aug.partner_gene_used.str.endswith("(self)"), "partner_gene_used"].value_counts().to_dict())
    aug.to_parquet(AUG_OUT)
    return aug


# ---------- MERGE into split-preserving training table ----------
def merge(st, aug):
    tr = pd.read_parquet(TRAIN_IN)
    seq_map = {f"seq_{i}": s for i, s in enumerate(aug["partner_sequence"].tolist())}
    seq_g = {f"seq_{i}": g for i, g in enumerate(aug["partner_gene_used"].tolist())}
    str_map = {f"str_{i}": s for i, s in enumerate(st["partner_sequence"].tolist())}
    str_g = {f"str_{i}": g for i, g in enumerate(st["partner_gene_used"].tolist())}
    m = {**seq_map, **str_map}; mg = {**seq_g, **str_g}
    tr["partner_sequence"] = tr["filename"].map(m).fillna("")
    tr["partner_gene_used"] = tr["filename"].map(mg).fillna("")
    has = tr.partner_sequence.str.len() > 0
    print(f"[merge] {len(tr)} rows | partner on {has.sum()} "
          f"(struct {tr[has].filename.str.startswith('str_').sum()}, seq {tr[has].filename.str.startswith('seq_').sum()})")
    tr.to_parquet(TRAIN_OUT)
    print(f"saved {TRAIN_OUT}")


def main():
    st = rebuild_structure()
    # canonical partner DBDs sourced from the structure heterodimer extractions
    has = st.partner_sequence.str.len() > 0
    canon = {}
    for pg in ["RXRA", "MAFG", "TCF3", "ARNT", "JUN"]:
        sub = st[has & (st.partner_gene_used == pg)]
        if len(sub):
            canon[pg] = sub.iloc[0]["partner_sequence"]
    print("canonical partner DBDs:", {k: len(v) for k, v in canon.items()})
    aug = build_sequence(canon)
    merge(st, aug)


if __name__ == "__main__":
    main()
