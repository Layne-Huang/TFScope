"""Collect Boltz-2 foldability: v24-ENSEMBLE vs DeepPBS predicted-consensus DNA.

TFScope side: ensemble fold {gene}_ens if it exists, else the cached single-seed
{gene}_v24 fold (the ensemble consensus was byte-identical for those genes).
DeepPBS side: reused verbatim from the boltz_v24 campaign.
"""
import glob, json, os
import numpy as np

V24_OUT = "/data1/leihuang/TFScope_store/boltz_v24/out"
ENS_OUT = "/data1/leihuang/TFScope_store/boltz_v24_ens/out"
V24 = "results/af3_v24_foldability/v24_consensus.json"
ENS = "results/af3_v24_foldability/ens_consensus.json"
RES = "results/af3_v24_foldability"


def _conf(root, name):
    fs = glob.glob(f"{root}/boltz_results_{name}/predictions/{name}/confidence_{name}_model_0.json")
    if not fs:
        return None
    d = json.load(open(fs[0]))
    return {"iptm": d.get("iptm"), "plddt": d.get("complex_plddt")}


def tfscope_conf(gene):
    c = _conf(ENS_OUT, f"{gene}_ens")               # ensemble fold (differing genes)
    return c if c else _conf(V24_OUT, f"{gene}_v24")  # else reuse identical v24 fold


def main():
    ens = {r["gene"]: r for r in json.load(open(ENS))}
    dlen_ens = {g: len(r["ens"]) for g, r in ens.items()}
    dlen_dpp = {}
    for r in json.load(open(V24)):
        dlen_dpp[r["gene"]] = None  # deeppbs len unknown here; length control optional
    rows = []
    for gene in sorted(ens):
        v = tfscope_conf(gene); d = _conf(V24_OUT, f"{gene}_deeppbs")
        if not v or not d or v["iptm"] is None or d["iptm"] is None:
            continue
        rows.append({"gene": gene, "iptm_ens": v["iptm"], "iptm_deeppbs": d["iptm"],
                     "plddt_ens": v["plddt"], "plddt_deeppbs": d["plddt"],
                     "d_iptm": v["iptm"] - d["iptm"]})
    n = len(rows)
    if n == 0:
        print("no completed pairs yet"); return
    div = np.array([r["d_iptm"] for r in rows]); wins = int((div > 0).sum())
    v_mean = float(np.mean([r["iptm_ens"] for r in rows]))
    d_mean = float(np.mean([r["iptm_deeppbs"] for r in rows]))
    out = {"n_pairs": n, "ens_mean_iptm": round(v_mean, 3), "deeppbs_mean_iptm": round(d_mean, 3),
           "mean_delta_iptm": round(float(div.mean()), 3), "ens_wins": wins, "ens_win_frac": round(wins / n, 3),
           "ens_mean_plddt": round(float(np.mean([r["plddt_ens"] for r in rows])), 3),
           "deeppbs_mean_plddt": round(float(np.mean([r["plddt_deeppbs"] for r in rows])), 3),
           "n_ens_folded": len(glob.glob(f"{ENS_OUT}/boltz_results_*_ens")), "per_tf": rows}
    try:
        from scipy.stats import wilcoxon
        out["wilcoxon_p_iptm"] = float(wilcoxon([r["iptm_ens"] for r in rows],
                                                [r["iptm_deeppbs"] for r in rows]).pvalue)
    except Exception:
        pass
    os.makedirs(RES, exist_ok=True)
    json.dump(out, open(f"{RES}/boltz_foldability_ens.json", "w"), indent=2)
    print(f"=== Boltz-2 foldability: v24-ENSEMBLE vs DeepPBS ({n} TF pairs) ===")
    print(f"  ipTM  : ens {v_mean:.3f}  vs  DeepPBS {d_mean:.3f}   (Δ {div.mean():+.3f}, "
          f"ens wins {wins}/{n} = {100*wins/n:.0f}%)")
    if "wilcoxon_p_iptm" in out:
        print(f"  Wilcoxon p = {out['wilcoxon_p_iptm']:.2e}")
    print(f"  saved {RES}/boltz_foldability_ens.json")


if __name__ == "__main__":
    main()
