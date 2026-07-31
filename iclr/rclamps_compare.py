"""Head-to-head: v24 (PLM, sequence-only) vs rCLAMPS (recognition code) vs
family-prior, on rCLAMPS's classic-homeodomain panel (763 TFs, CIS-BP GT, 6-bp
aligned core). rCLAMPS predictions are hold-one-out; v24 is sequence-only.

Reports oracle-aligned per-column Pearson r (content), mean/median, on the FULL
set and on the CLEAN subset (rCLAMPS TFs NOT in v24's training genes) to control
for v24 leakage. CPU-parse + GPU v24 inference.
"""
import os, sys, json, collections
import numpy as np, pandas as pd
sys.path.insert(0, "src")
os.environ.setdefault("TORCH_HOME", "/data1/leihuang/.cache/torch")
import torch, torch.nn.functional as F
from tfscope.config import TFScopeConfig
from tfscope.models.tfscope import TFScopeModel
from tfscope.data.dataset import AA_TO_TOKEN
from tfscope.models.alignment import align_pwm, revcomp_pwm_np

RC = "/data1/leihuang/rCLAMPS"
V24 = "/data1/leihuang/project/TFScope/checkpoints/v24_contact/contact_v24_seed42/ckpt_best.pt"
DEV = "cuda:0"


def load_pwms(path):
    d = collections.defaultdict(lambda: collections.defaultdict(dict))
    with open(path) as f:
        next(f)
        for L in f:
            p, bpos, base, prob = L.split()
            d[p][int(bpos)][base] = float(prob)
    out = {}
    for p, cols in d.items():
        Ln = max(cols) + 1; M = np.zeros((4, Ln))
        for j in range(Ln):
            for bi, b in enumerate("ACGT"):
                M[bi, j] = cols[j].get(b, 0)
        out[p] = (M / M.sum(0, keepdims=True).clip(1e-9)).astype(np.float32)
    return out


def r_align(pred, gt):
    aligned, shift, orient, r = align_pwm(pred, gt, max_shift=10, consider_revcomp=True, min_overlap=2)
    o = revcomp_pwm_np(pred) if orient == "rc" else pred
    cols = [i + shift for i in range(o.shape[1]) if 0 <= i + shift < gt.shape[1]]
    if len(cols) < 2:
        return 0.0
    from scipy.stats import pearsonr
    t = gt[:, cols]; p = np.clip(aligned[:, cols], 1e-8, 1); p = p / p.sum(0, keepdims=True)
    rs = [0.0 if (t[:, j].std() == 0 or p[:, j].std() == 0) else pearsonr(t[:, j], p[:, j])[0]
          for j in range(t.shape[1])]
    return float(np.nanmean(rs))


def main():
    gt = load_pwms(f"{RC}/my_results/allHomeodomainProts/pwms_testAli_holdOneOut.txt")
    pred = load_pwms(f"{RC}/my_results/allHomeodomainProts/pwms_pred_holdOneOut.txt")
    dbd = {}
    with open(f"{RC}/cis_bp/prot_seq.txt") as f:
        hdr = f.readline().rstrip("\n").split("\t"); ix = {c: i for i, c in enumerate(hdr)}
        for L in f:
            r = L.rstrip("\n").split("\t")
            if len(r) <= ix["DBD_seqs"]:
                continue
            seq = r[ix["DBD_seqs"]].split(",")[0].replace(" ", "")
            if seq:
                dbd[r[ix["TF_Name"]]] = seq; dbd[r[ix["TF_Name"]].upper()] = seq
    keys = [k for k in (set(gt) & set(pred)) if dbd.get(k, dbd.get(k.upper()))]

    # v24 leakage set (v24 training genes)
    df = pd.read_parquet("data/processed/tf_pwm_training_v23.parquet"); df["filename"] = df.filename.astype(str)
    trg = set(df[df.filename.isin(set(json.load(open("data/processed/splits/train_v22/split.json"))["train"]))]
              .gene_symbol.astype(str).str.upper())

    # family prior = leave-one-out mean of the 6-bp GT cores
    G = np.stack([gt[k] for k in keys])                       # (N,4,6)
    prior_all = G.mean(0)

    # v24 inference
    cfg = TFScopeConfig()
    for k, v in json.load(open(os.path.dirname(V24) + "/config.json")).items():
        if hasattr(cfg, k):
            try: setattr(cfg, k, type(getattr(cfg, k))(v))
            except Exception: pass
    cfg.use_retrieval = False
    m = TFScopeModel(cfg).to(DEV).eval()
    m.load_state_dict(torch.load(V24, map_location=DEV, weights_only=False)["model"], strict=False)

    @torch.no_grad()
    def v24_pred(seq, fid=4):                                  # 4 = Homeodomain
        t = torch.tensor([[AA_TO_TOKEN.get(a, 4) for a in seq]], device=DEV)
        dm = torch.ones(1, len(seq), dtype=torch.bool, device=DEV); fi = torch.tensor([fid], device=DEV)
        gl, pl, _ = m(t, dm, fi)
        g = torch.sigmoid(gl)[0].cpu().numpy(); p = F.softmax(pl, 1)[0].cpu().numpy()
        c = np.where(g > 0.5)[0]
        if len(c) < 4:
            ic = (p * np.log2(p + 1e-9)).sum(0) + 2; a = ic.argmax(); c = np.arange(max(0, a - 4), min(p.shape[1], a + 5))
        return p[:, c.min():c.max() + 1]

    rows = []
    for i, k in enumerate(keys):
        seq = dbd.get(k, dbd.get(k.upper()))
        loo_prior = (G.sum(0) - gt[k]) / (len(keys) - 1)      # leave-one-out family prior
        rows.append({
            "prot": k, "leak": k.upper() in trg,
            "r_v24": r_align(v24_pred(seq), gt[k]),
            "r_rclamps": r_align(pred[k], gt[k]),
            "r_prior": r_align(loo_prior, gt[k]),
        })
        if (i + 1) % 100 == 0:
            print(f"  scored {i+1}/{len(keys)}", flush=True)

    def agg(rs):
        return {m_: {"mean": float(np.mean([r[m_] for r in rs])), "median": float(np.median([r[m_] for r in rs]))}
                for m_ in ["r_v24", "r_rclamps", "r_prior"]}
    clean = [r for r in rows if not r["leak"]]
    out = {"n_all": len(rows), "n_clean": len(clean), "n_leak_v24": len(rows) - len(clean),
           "ALL": agg(rows), "CLEAN_v24_nonleak": agg(clean), "per_prot": rows}
    os.makedirs("results/iclr_phase1_apples_to_apples", exist_ok=True)
    json.dump(out, open("results/iclr_phase1_apples_to_apples/rclamps_compare.json", "w"), indent=2)
    print(f"\n=== v24 vs rCLAMPS vs family-prior on homeodomains (mean r / median r) ===")
    for grp, lbl in [("ALL", f"ALL n={len(rows)}"), ("CLEAN_v24_nonleak", f"CLEAN (v24-nonleak) n={len(clean)}")]:
        a = out[grp]
        print(f"[{lbl}]")
        for m_, nm in [("r_rclamps", "rCLAMPS (recog code, holdout)"), ("r_v24", "v24 (PLM, seq-only)"), ("r_prior", "homeodomain prior")]:
            print(f"   {nm:<34} mean {a[m_]['mean']:.3f}  median {a[m_]['median']:.3f}")


if __name__ == "__main__":
    main()
