"""Test the v24 5-seed ENSEMBLE (vs single-seed42) on the two application cases:
  (A) MyoD1 L112R specificity switch  -> directional Δ_switch (CACGTG vs CACCTG)
  (B) 4 de novo DBD designs (DBP005/006/009/035) -> CAC recovery + core-r vs the
      experimental single-substitution preference (Glasscock et al.).
Ensemble = mean of softmax(PWM) and sigmoid(gate) over {seed42,1,7,13,23}.

  PYTHONPATH=src python -m iclr.eval_v24ens_mut_design --device cuda:0
"""
from __future__ import annotations
import os, sys, json, argparse
os.environ.setdefault("TORCH_HOME", "/data1/leihuang/.cache/torch"); os.environ["TRANSFORMERS_OFFLINE"] = "1"
sys.path.insert(0, "src"); sys.path.insert(0, "scripts")
import numpy as np, torch, torch.nn.functional as F, pandas as pd
from tfscope.config import TFScopeConfig
from tfscope.models.tfscope import TFScopeModel
from tfscope.data.dataset import AA_TO_TOKEN
from eval_full_metrics import aligned_cols, panel

CK = "/data1/leihuang/project/TFScope/checkpoints"
V24 = [f"{CK}/v24_contact/contact_v24_seed42"] + \
      [f"checkpoints/iclr_phase1/v24_ens/seed{s}" for s in (1, 7, 13, 23)]
B = np.array(list("ACGT")); B2I = {"A": 0, "C": 1, "G": 2, "T": 3}
DEV = "cuda:0" if torch.cuda.is_available() else "cpu"


def load(ckpt_dir):
    cfg = TFScopeConfig()
    for k, v in json.load(open(os.path.join(ckpt_dir, "config.json"))).items():
        if hasattr(cfg, k):
            try: setattr(cfg, k, type(getattr(cfg, k))(v))
            except Exception: pass
    cfg.use_retrieval = False
    m = TFScopeModel(cfg).to(DEV).eval()
    m.load_state_dict(torch.load(os.path.join(ckpt_dir, "ckpt_best.pt"),
                                 map_location=DEV, weights_only=False)["model"], strict=False)
    return m


@torch.no_grad()
def _one(m, seq, fid):
    t = torch.tensor([[AA_TO_TOKEN.get(a, 4) for a in seq]], dtype=torch.long, device=DEV)
    dm = torch.ones(1, len(seq), dtype=torch.bool, device=DEV); fi = torch.tensor([fid], device=DEV)
    gl, pl, _ = m(t, dm, fi, retrieved_pwms=None, retrieved_masks=None, retrieved_sims=None, recog_prior=None)
    return gl.sigmoid()[0].cpu().numpy(), F.softmax(pl, 1)[0].cpu().numpy()


def make_predictor(models):
    """Returns predict(seq, fid) -> (gate_avg(42,), pwm_avg(4,42)) averaged over members."""
    def predict(seq, fid):
        gs, ps = [], []
        for m in models:
            g, p = _one(m, seq, fid); gs.append(g); ps.append(p)
        return np.mean(gs, 0), np.mean(ps, 0)
    return predict


def gated_core(gate, pwm):
    L = max(4, int((gate > 0.5).sum()))
    return pwm[:, :L]


# ── (A) MyoD1 switch ──────────────────────────────────────────────────────────
def rc(s): return s[::-1].translate(str.maketrans("ACGT", "TGCA"))
def loscore(P, seq):
    lo = np.log2(np.clip(P, 1e-6, 1) / 0.25); W = P.shape[1]; L = len(seq); best = -1e9
    for s in (seq, rc(seq)):
        idx = [B2I[c] for c in s]
        for off in range(0, W - L + 1):
            best = max(best, float(sum(lo[idx[j], off + j] for j in range(L))))
    return best

def myod1_switch(predict):
    WT = "RKAATMRERRRLSKVNEAFETLKRCTSSNPNQRLPKVEILRNAIRYIEGLQA"
    MUT = WT[:11] + "R" + WT[12:]                     # L112R
    FID = 3                                           # bHLH
    _, pw = predict(WT, FID); _, pm = predict(MUT, FID)
    S = {"WT": {"CACGTG": loscore(pw, "CACGTG"), "CACCTG": loscore(pw, "CACCTG")},
         "mut": {"CACGTG": loscore(pm, "CACGTG"), "CACCTG": loscore(pm, "CACCTG")}}
    dWT = S["WT"]["CACGTG"] - S["WT"]["CACCTG"]; dMUT = S["mut"]["CACGTG"] - S["mut"]["CACCTG"]
    return {"d_switch": dMUT - dWT, "dWT": dWT, "dMUT": dMUT,
            "reproduced": bool(dMUT - dWT > 0)}


# ── (B) 4 DBD designs ─────────────────────────────────────────────────────────
def _design_setup():
    XLS = "case_study/pdb/design_pdbs/41594_2025_1669_MOESM16_ESM.xls"
    SHEET = {"DBP005": "Extended_Data_Figure_1_C_DBP005", "DBP006": "Extended_Data_Figure_1_D_DBP006",
             "DBP009": "Extended_Data_Figure_1_E_DBP009", "DBP035": "Extended_Data_Figure_1_G_DBP035"}
    VALCOL = pd.ExcelFile(XLS).parse(SHEET["DBP005"], nrows=1).columns.tolist()[-1]
    WTOV = {"DBP006": 0.1202}
    def exp_pref(d):                                  # LOWER value = STRONGER binding -> 1/value
        da = pd.ExcelFile(XLS).parse(SHEET[d]); wt = da[da.position.astype(str) == "WT"]
        wtv = WTOV.get(d) or (float(wt[VALCOL].iloc[0]) if len(wt) else 1.0)
        df = da[da.position.astype(str).str.isdigit()].copy(); L = int(df.position.astype(int).max())
        P = np.full((4, L), 1e-6, np.float32)
        for p in range(1, L + 1):
            sub = df[df.position.astype(int) == p]; ob = str(sub.original_base.iloc[0])
            if ob in B2I: P[B2I[ob], p - 1] = 1.0 / max(wtv, 1e-3)
            for _, r in sub.iterrows():
                nb = str(r["new_base"])
                if nb in B2I: P[B2I[nb], p - 1] = 1.0 / max(float(r[VALCOL]), 1e-3)
        return P / P.sum(0, keepdims=True)
    designs = list(SHEET); prefs = {d: exp_pref(d) for d in designs}
    e2 = {e["name"]: e for e in json.load(open("results/design_case_study/design_e2_predictions.json"))}
    dfv = pd.read_parquet("data/processed/tf_pwm_training_v23.parquet")
    def fid_for(gene):
        sub = dfv[dfv["gene_symbol"].astype(str).str.upper().str.contains(gene.upper())]
        return int(sub["family_id"].mode().iloc[0]) if len(sub) else None
    return designs, prefs, e2, fid_for

def designs_eval(predict, designs, prefs, e2, fid_for):
    rec = {}
    for d in designs:
        donor = str(e2[d].get("top_donor", "POU2F1"))
        fid = fid_for(donor) or fid_for("POU2F1") or 4
        g, p = predict(e2[d]["prot_seq"], fid); pv = gated_core(g, p)
        con = "".join(B[pv.argmax(0)])
        aligned, cols, _ = aligned_cols(pv, prefs[d]); dd = panel(prefs[d], aligned, cols)
        rec[d] = {"consensus": con, "cac": ("CAC" in con or "GTG" in con),
                  "core_r": (round(float(dd["r"]), 3) if dd else None)}
    rs = [rec[d]["core_r"] for d in designs if rec[d]["core_r"] is not None]
    rec["mean_core_r"] = round(float(np.nanmean(rs)), 3)
    rec["cac_count"] = int(sum(rec[d]["cac"] for d in designs))
    return rec


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", default="results/iclr_phase1_apples_to_apples/v24ens_mut_design.json")
    a = ap.parse_args()
    print("loading 5 v24 members ...")
    members = [load(d) for d in V24]
    variants = {"v24_seed42": make_predictor(members[:1]),
                "v24_ens5": make_predictor(members)}

    designs, prefs, e2, fid_for = _design_setup()
    out = {}
    for name, predict in variants.items():
        sw = myod1_switch(predict)
        de = designs_eval(predict, designs, prefs, e2, fid_for)
        out[name] = {"myod1_switch": sw, "designs": de}
        print(f"\n===== {name} =====")
        print(f"MyoD1 L112R: Δ_switch={sw['d_switch']:+.2f}  (dWT={sw['dWT']:+.2f} dMUT={sw['dMUT']:+.2f})  "
              f"reproduced={sw['reproduced']}")
        print(f"Designs: CAC {de['cac_count']}/4   mean core-r {de['mean_core_r']}")
        for d in designs:
            print(f"   {d}: {de[d]['consensus'][:16]:<16} CAC={'Y' if de[d]['cac'] else 'n'} core-r={de[d]['core_r']}")
    json.dump(out, open(a.out, "w"), indent=2)
    print("\nwrote", a.out)
    # headline delta
    print("\n=== ensemble vs seed42 ===")
    print(f"Δ_switch: seed42 {out['v24_seed42']['myod1_switch']['d_switch']:+.2f} -> "
          f"ens5 {out['v24_ens5']['myod1_switch']['d_switch']:+.2f}")
    print(f"design mean core-r: seed42 {out['v24_seed42']['designs']['mean_core_r']} -> "
          f"ens5 {out['v24_ens5']['designs']['mean_core_r']}  | "
          f"CAC seed42 {out['v24_seed42']['designs']['cac_count']}/4 -> ens5 {out['v24_ens5']['designs']['cac_count']}/4")


if __name__ == "__main__":
    main()
