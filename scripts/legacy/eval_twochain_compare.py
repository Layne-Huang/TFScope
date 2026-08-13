#!/usr/bin/env python
"""Head-to-head: single-chain v20 vs two-chain v21 on the held-out test set,
coverage-aware metric, with per-gene breakdown for the heterodimer genes.

Each model is evaluated with ITS OWN two_chain_input setting (read from the
checkpoint's config.json), so v21 sees the partner chain and v20 does not --
both on the same tf_pwm_training_v2p.parquet test split.
"""
import os, sys, json, numpy as np, torch, torch.nn.functional as F, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "src"); sys.path.insert(0, "scripts")
from tfscope.config import TFScopeConfig
from tfscope.data.dataset import TFDataset, collate_variable_length
from tfscope.models.tfscope import TFScopeModel
from tfscope.models.alignment import align_pwm
from eval_full_metrics import trimmed_core
from torch.utils.data import DataLoader

DATA = "data/processed/tf_pwm_training_v2p.parquet"
SPLIT = "data/processed/splits/train_v2/split.json"
IC, MAXSHIFT, MINPOS = 0.25, 10, 4
dev = "cuda"
HETERO = ["THRB", "NFE2L2", "POU5F1::SOX2", "POU2F1", "FOXP2", "ELF1", "ETS1", "FLI1"]

MODELS = {
    "v20_single": "/data1/leihuang/project/TFScope/checkpoints/v20_residue_moe_newdata/residue_moe_v2_seed42",
    "v21_twochain": "/data1/leihuang/project/TFScope/checkpoints/v21_twochain_heterodimer/twochain_v2p_ddp6_seed42",
}


def _trimg(pwm, th=IC):
    p = np.clip(pwm, 1e-8, 1); ic = 2 + (p * np.log2(p)).sum(0); inf = np.where(ic >= th)[0]
    return pwm if len(inf) == 0 else pwm[:, inf[0]:inf[-1] + 1]


def load(ckpt_dir):
    cfg = TFScopeConfig()
    for k, v in json.load(open(os.path.join(ckpt_dir, "config.json"))).items():
        if hasattr(cfg, k):
            try: setattr(cfg, k, type(getattr(cfg, k))(v))
            except Exception: setattr(cfg, k, v)
    m = TFScopeModel(cfg).to(dev).eval()
    m.load_state_dict(torch.load(os.path.join(ckpt_dir, "ckpt_best.pt"), map_location=dev,
                                 weights_only=False)["model"], strict=False)
    return m, cfg


def evaluate(m, cfg):
    ds = TFDataset(cfg, DATA, SPLIT, split="test", max_seq_len=1024)
    genes = [str(g).upper() for g in ds.df["gene_symbol"].values]
    ld = DataLoader(ds, batch_size=12, shuffle=False, num_workers=2, collate_fn=collate_variable_length)
    rows = []; idx = 0
    with torch.no_grad():
        for b in ld:
            bb = {k: v.to(dev, dtype=torch.float32 if v.is_floating_point() else torch.long) for k, v in b.items()}
            gl, pl, _ = m(bb['sequence_tokens'], bb['dbd_mask'], bb['family_id'],
                          recog_prior=bb.get('recog_prior'))
            pp = F.softmax(pl, 1).cpu().numpy(); gp = torch.sigmoid(gl).cpu().numpy()
            tg = bb['target_pwm'].cpu().numpy(); mk = bb['pwm_mask'].cpu().numpy()
            for j in range(pp.shape[0]):
                g = genes[idx]; idx += 1
                core = trimmed_core(tg[j], mk[j], IC)
                if core is None or core.shape[1] < MINPOS: continue
                gate = gp[j]; active = gate > 0.5
                if not active.any(): active = gate > gate.max() * 0.5
                pc = pp[j][:, active]; gc = _trimg(tg[j][:, mk[j].astype(bool)])
                if pc.shape[1] == 0 or gc.shape[1] == 0: continue
                _, sh, _, r = align_pwm(pc, gc, max_shift=MAXSHIFT, consider_revcomp=True)
                nov = sum(1 for i in range(pc.shape[1]) if 0 <= i + sh < gc.shape[1])
                cov = nov / gc.shape[1]
                rows.append(dict(g=g, covr=r * cov, r=r, cov=cov, lp=pc.shape[1], lg=gc.shape[1]))
    return rows


def main():
    res = {}
    for name, d in MODELS.items():
        m, cfg = load(d)
        res[name] = evaluate(m, cfg)
        print(f"loaded {name}: two_chain_input={getattr(cfg,'two_chain_input',False)}, n={len(res[name])}")
        del m; torch.cuda.empty_cache()

    def agg(rows, sub=None):
        rr = [x for x in rows if sub is None or x["g"] in sub]
        if not rr: return None
        return (np.mean([x["covr"] for x in rr]), np.mean([x["lp"] for x in rr]),
                np.mean([x["lg"] for x in rr]), len(rr))

    print("\n" + "=" * 66)
    print(f"{'metric':<26}{'v20_single':>18}{'v21_twochain':>20}")
    print("-" * 66)
    a20, a21 = agg(res["v20_single"]), agg(res["v21_twochain"])
    print(f"{'ALL test covR':<26}{a20[0]:>18.4f}{a21[0]:>20.4f}")
    print(f"{'ALL pred/gt len':<26}{a20[1]:>8.1f}/{a20[2]:<8.1f}{a21[1]:>10.1f}/{a21[2]:<8.1f}")

    # heterodimer test genes present in both
    het_present = sorted({x["g"] for x in res["v21_twochain"]} & set(HETERO))
    h20 = agg(res["v20_single"], set(het_present)); h21 = agg(res["v21_twochain"], set(het_present))
    if h20 and h21:
        print(f"{'HETERODIMER covR':<26}{h20[0]:>18.4f}{h21[0]:>20.4f}")
        print(f"{'HETERODIMER pred/gt len':<26}{h20[1]:>8.1f}/{h20[2]:<8.1f}{h21[1]:>10.1f}/{h21[2]:<8.1f}")

    print("\n--- per heterodimer gene (best row per gene): covR | pred/gt len ---")
    print(f"{'gene':<14}{'v20 covR':>10}{'v20 len':>10}   {'v21 covR':>10}{'v21 len':>10}")
    for g in het_present:
        r20 = [x for x in res["v20_single"] if x["g"] == g]
        r21 = [x for x in res["v21_twochain"] if x["g"] == g]
        b20 = max(r20, key=lambda x: x["covr"]); b21 = max(r21, key=lambda x: x["covr"])
        print(f"{g:<14}{b20['covr']:>10.3f}{b20['lp']:>7.0f}/{b20['lg']:<3.0f}   "
              f"{b21['covr']:>10.3f}{b21['lp']:>7.0f}/{b21['lg']:<3.0f}")

    os.makedirs("results/twochain", exist_ok=True)
    json.dump({k: v for k, v in res.items()}, open("results/twochain/compare_test.json", "w"), default=float)
    print("\nsaved results/twochain/compare_test.json")


if __name__ == "__main__":
    main()
