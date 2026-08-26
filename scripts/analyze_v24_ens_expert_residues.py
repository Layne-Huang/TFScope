#!/usr/bin/env python
"""What does each v24 ResidueMoE expert actually recognize? (5-seed ensemble)

v24 routes PER DBD RESIDUE (moe_granularity='residue', 8 routed + 2 shared experts,
top-2). That gives ~300k routing decisions over the v23 table, so per-expert residue
statistics are well powered. For every DBD token we record its top-1 expert and ask:

  (A) amino-acid enrichment      log2 P(aa | e) / P(aa)
  (B) family-composition control log2 P(aa | e) / sum_f P(f|e) P(aa|f)
      -> removes the trivial explanation "expert e just took the C2H2 proteins,
         and C2H2 is C/H-rich".  What survives is chemistry the router reads
         *within* families.
  (C) DNA-contact enrichment     log2 P(contact | e) / P(contact), where `contact`
      is the co-crystal base-contact residue set (contact_targets_v23.json, the
      exact supervision v24 was trained on; 309 proteins).
  (D) family-embedding ablation  re-route every protein with family_id forced to a
      single constant. If routing barely moves, it is token chemistry, not the
      family label being re-encoded through the router's family bias.
  (E) cross-seed reproducibility  experts are permutation-symmetric across seeds, so
      match seed_s -> seed42 by Hungarian on the AA-enrichment profiles and report
      the matched correlation against a random-permutation null. Reproducible
      archetypes = a real decomposition; chance-level = an arbitrary partition.

Usage:
  python scripts/analyze_v24_ens_expert_residues.py [--device cuda:0] [--max-proteins N]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

os.environ.setdefault("TORCH_HOME", "/data1/leihuang/.cache/torch")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
sys.path.insert(0, ".")
sys.path.insert(0, "src")

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from tfscope.config import TFScopeConfig
from tfscope.models.tfscope import TFScopeModel
from tfscope.data.dataset import AA_TO_TOKEN, ESM2_PAD_TOKEN

DATA = "data/processed/tf_pwm_training_v23.parquet"
CONTACTS = "data/contact_maps/contact_targets_v23.json"
CKPTS = [
    ("seed42", "/data1/leihuang/project/TFScope/checkpoints/v24_contact/contact_v24_seed42/ckpt_best.pt"),
    ("seed1",  "/data1/leihuang/TFScope_store/checkpoints/iclr_phase1/v24_ens/seed1/ckpt_best.pt"),
    ("seed7",  "/data1/leihuang/TFScope_store/checkpoints/iclr_phase1/v24_ens/seed7/ckpt_best.pt"),
    ("seed13", "/data1/leihuang/TFScope_store/checkpoints/iclr_phase1/v24_ens/seed13/ckpt_best.pt"),
    ("seed23", "/data1/leihuang/TFScope_store/checkpoints/iclr_phase1/v24_ens/seed23/ckpt_best.pt"),
]
OUTDIR = "results/moe_expert_interpretation"
AAS = list("ACDEFGHIKLMNPQRSTVWY")
AA2I = {c: i for i, c in enumerate(AAS)}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
def build_model(ckpt, device):
    cfg = TFScopeConfig()
    cfgp = os.path.join(os.path.dirname(ckpt), "config.json")
    for k, v in json.load(open(cfgp)).items():
        if hasattr(cfg, k):
            try:
                setattr(cfg, k, type(getattr(cfg, k))(v))
            except Exception:
                setattr(cfg, k, v)
    cfg.use_retrieval = False
    assert getattr(cfg, "moe_granularity", "protein") == "residue", \
        f"{ckpt}: expected residue-granularity MoE, got {cfg.moe_granularity}"
    m = TFScopeModel(cfg).to(device).eval()
    sd = torch.load(ckpt, map_location=device, weights_only=False)
    m.load_state_dict(sd.get("model", sd), strict=False)
    return m, cfg


def load_proteins(max_proteins=None, max_len=1024):
    df = pd.read_parquet(DATA)
    df = df.drop_duplicates("sequence").reset_index(drop=True)
    df["seq"] = df.sequence.astype(str).str.slice(0, max_len)
    df = df[df.seq.str.len() >= 8].reset_index(drop=True)
    if max_proteins:
        df = df.sample(min(max_proteins, len(df)), random_state=0).reset_index(drop=True)

    ct = json.load(open(CONTACTS))
    contact_sets, n_ct, n_clip = {}, 0, 0
    for i, r in enumerate(df.itertuples()):
        e = ct.get(str(r.filename))
        if not e:
            continue
        idx = set()
        for _col, rows in e["cols"].items():
            for ridx, w in rows:
                if w > 0:
                    if ridx < len(r.seq):
                        idx.add(int(ridx))
                    else:
                        n_clip += 1
        contact_sets[i] = idx
        n_ct += 1
    log(f"proteins={len(df)}  with co-crystal contacts={n_ct}  "
        f"(contact indices past the crop, dropped: {n_clip})")
    return df, contact_sets


@torch.no_grad()
def route_all(model, cfg, df, contact_sets, device, const_fid, batch=8):
    """Route every DBD token twice (true family_id, constant family_id).

    Returns a dict of flat per-token arrays, aligned across both passes.
    """
    E = cfg.num_experts
    order = np.argsort([len(s) for s in df.seq.values])   # length-sorted -> less padding
    fams = sorted(df.family_name.unique())
    fam2i = {f: i for i, f in enumerate(fams)}

    out = {k: [] for k in
           ("expert", "expert_const", "w1", "ent", "aa", "fam", "relpos", "contact", "prot")}
    # Token-budget batching: the tail of the length-sorted order holds 1000+ aa
    # sequences, and ESM attention is O(B*L^2). A fixed batch there spikes memory
    # on a shared card, so cap padded tokens per batch instead.
    TOK_BUDGET = 4096
    batches, cur = [], []
    for i in order:
        cand = cur + [i]
        Lmax = max(len(df.seq.values[j]) for j in cand)
        if cur and (len(cand) * Lmax > TOK_BUDGET or len(cand) > batch):
            batches.append(cur); cur = [i]
        else:
            cur = cand
    if cur:
        batches.append(cur)

    t0, done = time.time(), 0
    for bnum, idx in enumerate(batches):
        seqs = [df.seq.values[i] for i in idx]
        L = max(len(s) for s in seqs)
        tok = torch.full((len(idx), L), ESM2_PAD_TOKEN, dtype=torch.long)
        dm = torch.zeros(len(idx), L, dtype=torch.bool)
        for j, s in enumerate(seqs):
            tok[j, :len(s)] = torch.tensor([AA_TO_TOKEN.get(a, 3) for a in s])
            dm[j, :len(s)] = True
        tok, dm = tok.to(device), dm.to(device)
        fid = torch.tensor([int(df.family_id.values[i]) for i in idx], device=device)
        fid_c = torch.full_like(fid, const_fid)

        _, _, aux = model(tok, dm, fid)
        gl = aux["gate_logits"].float()                    # (N_dbd, E) DBD tokens, row-major
        top = aux["top_indices"][:, 0].cpu().numpy()
        p = F.softmax(gl, dim=-1)
        w1 = p.max(dim=-1).values.cpu().numpy()
        ent = (-(p * (p + 1e-9).log()).sum(-1) / np.log(E)).cpu().numpy()

        _, _, aux_c = model(tok, dm, fid_c)
        top_c = aux_c["top_indices"][:, 0].cpu().numpy()

        # rebuild (protein, position) for the flattened DBD-token order (row-major)
        pos = 0
        for j, i in enumerate(idx):
            n = len(seqs[j])
            sl = slice(pos, pos + n); pos += n
            out["expert"].append(top[sl]); out["expert_const"].append(top_c[sl])
            out["w1"].append(w1[sl]); out["ent"].append(ent[sl])
            out["aa"].append(np.array(list(seqs[j])))
            out["fam"].append(np.full(n, fam2i[df.family_name.values[i]]))
            out["relpos"].append(np.arange(n) / max(n - 1, 1))
            cs = contact_sets.get(int(i))
            out["contact"].append(
                np.array([1 if k in cs else 0 for k in range(n)]) if cs is not None
                else np.full(n, -1))
            out["prot"].append(np.full(n, int(i)))
        assert pos == len(top), f"token bookkeeping mismatch {pos} vs {len(top)}"
        done += len(idx)
        if bnum % 50 == 0 or done == len(order):
            log(f"    routed {done}/{len(order)} proteins "
                f"(batch {bnum+1}/{len(batches)}, {time.time()-t0:.0f}s)")
    return {k: np.concatenate(v) for k, v in out.items()}, fams


# ─────────────────────────────────────────────────────────────────────────────
def nmi(expert, label, E, nlab):
    J = np.zeros((nlab, E))
    for l in range(nlab):
        J[l] = np.bincount(expert[label == l], minlength=E)
    J = J / max(J.sum(), 1)
    pl = J.sum(1, keepdims=True); pe = J.sum(0, keepdims=True)
    nz = J > 0
    I = float((J[nz] * np.log(J[nz] / (pl @ pe)[nz])).sum())
    H = float(-(pl[pl > 0] * np.log(pl[pl > 0])).sum())
    return I / max(H, 1e-9)


def analyse(R, fams, E):
    aa_lab = np.array([AA2I.get(c, -1) for c in R["aa"]])
    keep = aa_lab >= 0
    et, ec, aal, fam = R["expert"][keep], R["expert_const"][keep], aa_lab[keep], R["fam"][keep]
    contact, relpos = R["contact"][keep], R["relpos"][keep]
    N = len(et)

    usage = np.bincount(et, minlength=E) / N
    bg_aa = np.array([(aal == i).mean() for i in range(20)]) + 1e-9
    # P(aa | family) for the family-composition control
    p_aa_given_f = np.zeros((len(fams), 20))
    for f in range(len(fams)):
        s = fam == f
        if s.sum():
            p_aa_given_f[f] = np.array([(aal[s] == i).mean() for i in range(20)])

    has_ct = contact >= 0
    bg_ct = float(contact[has_ct].mean()) if has_ct.any() else float("nan")

    experts = {}
    for e in range(E):
        s = et == e
        n = int(s.sum())
        if n < 50:
            experts[e] = {"n": n, "note": "too few tokens"}
            continue
        p_aa = np.array([(aal[s] == i).mean() for i in range(20)]) + 1e-9
        enr = np.log2(p_aa / bg_aa)
        fam_mix = np.array([(fam[s] == f).mean() for f in range(len(fams))])
        exp_aa = fam_mix @ p_aa_given_f + 1e-9
        enr_wf = np.log2(p_aa / exp_aa)                     # within-family enrichment
        sc = s & has_ct
        ct_rate = float(contact[sc].mean()) if sc.sum() > 20 else float("nan")
        experts[e] = {
            "n": n, "usage": round(float(usage[e]), 4),
            "mean_relpos": round(float(relpos[s].mean()), 3),
            "family_mix": {fams[f]: round(float(fam_mix[f]), 3)
                           for f in np.argsort(-fam_mix)[:4] if fam_mix[f] > 0.01},
            "aa_enrichment": {AAS[i]: round(float(enr[i]), 2) for i in range(20)},
            "aa_enrichment_within_family": {AAS[i]: round(float(enr_wf[i]), 2) for i in range(20)},
            "top_aa": [f"{AAS[i]}{enr[i]:+.2f}" for i in np.argsort(-enr)[:6]],
            "top_aa_within_family": [f"{AAS[i]}{enr_wf[i]:+.2f}" for i in np.argsort(-enr_wf)[:6]],
            "contact_n": int(sc.sum()),
            "contact_rate": None if np.isnan(ct_rate) else round(ct_rate, 4),
            "contact_log2_enrichment": None if np.isnan(ct_rate) else
                round(float(np.log2((ct_rate + 1e-9) / (bg_ct + 1e-9))), 3),
        }

    ctl = contact[has_ct].astype(int)
    return {
        "n_tokens": int(N),
        "expert_usage": [round(float(u), 4) for u in usage],
        "usage_entropy_norm": round(float(-(usage[usage > 0] * np.log(usage[usage > 0])).sum()
                                          / np.log(E)), 4),
        "mean_top1_gate_weight": round(float(R["w1"][keep].mean()), 4),
        "mean_routing_entropy_norm": round(float(R["ent"][keep].mean()), 4),
        "nmi_expert_family": round(nmi(et, fam, E, len(fams)), 4),
        "nmi_expert_aa": round(nmi(et, aal, E, 20), 4),
        "nmi_expert_contact": round(nmi(et[has_ct], ctl, E, 2), 4) if has_ct.any() else None,
        "nmi_expert_family_CONSTFAM": round(nmi(ec, fam, E, len(fams)), 4),
        "nmi_expert_aa_CONSTFAM": round(nmi(ec, aal, E, 20), 4),
        "frac_routing_unchanged_no_family": round(float((et == ec).mean()), 4),
        "contact_background_rate": None if np.isnan(bg_ct) else round(bg_ct, 4),
        "experts": experts,
    }


def cross_seed(profiles, key="aa_enrichment", n_null=2000, seed=0):
    """Hungarian-match every seed's expert profiles onto seed42's, vs a permutation null."""
    from scipy.optimize import linear_sum_assignment
    names = list(profiles)
    ref = profiles[names[0]]
    E = ref.shape[0]
    rng = np.random.RandomState(seed)

    def corr(a, b):
        a = a - a.mean(); b = b - b.mean()
        d = np.linalg.norm(a) * np.linalg.norm(b)
        return float(a @ b / d) if d > 1e-9 else 0.0

    res = {}
    for nm in names[1:]:
        M = np.array([[corr(ref[i], profiles[nm][j]) for j in range(E)] for i in range(E)])
        ri, ci = linear_sum_assignment(-M)
        matched = float(M[ri, ci].mean())
        null = []
        for _ in range(n_null):
            perm = rng.permutation(E)
            null.append(float(M[np.arange(E), perm].mean()))
        null = np.array(null)
        res[nm] = {
            "matched_mean_r": round(matched, 3),
            "assignment": {int(i): int(j) for i, j in zip(ri, ci)},
            "per_pair_r": [round(float(M[i, j]), 3) for i, j in zip(ri, ci)],
            "null_mean_r": round(float(null.mean()), 3),
            "null_p95_r": round(float(np.percentile(null, 95)), 3),
            "p_vs_null": round(float((null >= matched).mean()), 5),
        }
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--max-proteins", type=int, default=0)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--out", default=os.path.join(OUTDIR, "v24_ens_expert_residues.json"))
    a = ap.parse_args()
    os.makedirs(OUTDIR, exist_ok=True)

    df, contact_sets = load_proteins(a.max_proteins or None)
    # constant family for the ablation: the most common family_id of "Other"
    const_fid = int(df[df.family_name == "Other"].family_id.mode().iloc[0]) \
        if (df.family_name == "Other").any() else int(df.family_id.mode().iloc[0])
    log(f"family-ablation constant family_id = {const_fid}")

    results, profiles, profiles_wf = {}, {}, {}
    for tag, ck in CKPTS:
        if not os.path.exists(ck):
            log(f"!! missing {ck} — skipping {tag}")
            continue
        log(f"=== {tag}: {ck}")
        model, cfg = build_model(ck, a.device)
        E = cfg.num_experts
        R, fams = route_all(model, cfg, df, contact_sets, a.device, const_fid, a.batch)
        res = analyse(R, fams, E)
        results[tag] = res
        profiles[tag] = np.array([[res["experts"][e].get("aa_enrichment", {}).get(c, 0.0)
                                   for c in AAS] for e in range(E)])
        profiles_wf[tag] = np.array([[res["experts"][e].get("aa_enrichment_within_family", {}).get(c, 0.0)
                                      for c in AAS] for e in range(E)])
        log(f"  usage_entropy={res['usage_entropy_norm']:.3f}  "
            f"NMI(e;fam)={res['nmi_expert_family']:.3f}  NMI(e;aa)={res['nmi_expert_aa']:.3f}  "
            f"NMI(e;contact)={res['nmi_expert_contact']}  "
            f"unchanged_wo_family={res['frac_routing_unchanged_no_family']:.3f}")
        for e in range(E):
            x = res["experts"][e]
            if "top_aa" not in x:
                continue
            log(f"    e{e} n={x['n']:6d} use={x['usage']:.3f} "
                f"ctc={x['contact_log2_enrichment']} | {' '.join(x['top_aa'])} "
                f"|| wf: {' '.join(x['top_aa_within_family'])}")
        del model
        torch.cuda.empty_cache()

    out = {"data": DATA, "contacts": CONTACTS, "ckpts": dict(CKPTS),
           "n_proteins": int(len(df)), "per_seed": results}
    if len(profiles) > 1:
        out["cross_seed_aa"] = cross_seed(profiles)
        out["cross_seed_aa_within_family"] = cross_seed(profiles_wf)
        log("=== cross-seed expert-archetype reproducibility (AA profiles) ===")
        for k, v in out["cross_seed_aa"].items():
            log(f"  seed42 vs {k}: matched r={v['matched_mean_r']} "
                f"null={v['null_mean_r']} (p95 {v['null_p95_r']}) p={v['p_vs_null']}")
    json.dump(out, open(a.out, "w"), indent=1)
    log(f"saved {a.out}")


if __name__ == "__main__":
    main()
