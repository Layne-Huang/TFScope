"""Collect Boltz-2 foldability results: v24 vs DeepPBS predicted-consensus DNA.

Reads confidence_<name>_model_0.json for each {gene}_{v24,deeppbs} fold, pairs by
gene, and reports the ipTM / pLDDT head-to-head (mean, wins, Wilcoxon) with a
length control (Spearman of Δ-ipTM vs Δ-DNA-length), mirroring the original Fig1f
AF3 analysis — but on v24 (not the combined model) and Boltz-2 (not AF3).
"""
import glob, json, os
import numpy as np

OUT = "/data1/leihuang/TFScope_store/boltz_v24/out"
PLAN = "/data1/leihuang/TFScope_store/boltz_v24/plan.json"
RES = "results/af3_v24_foldability"


def conf(name):
    fs = glob.glob(f"{OUT}/boltz_results_{name}/predictions/{name}/confidence_{name}_model_0.json")
    if not fs:
        return None
    d = json.load(open(fs[0]))
    return {"iptm": d.get("iptm"), "plddt": d.get("complex_plddt"), "iplddt": d.get("complex_iplddt")}


def main():
    plan = json.load(open(PLAN))
    dlen = {}
    for j in plan["jobs"]:
        dlen[(j["gene"], j["source"])] = j["dna_len"]
    genes = sorted({j["gene"] for j in plan["jobs"]})
    rows = []
    for g in genes:
        v = conf(f"{g}_v24"); d = conf(f"{g}_deeppbs")
        if not v or not d or v["iptm"] is None or d["iptm"] is None:
            continue
        rows.append({"gene": g, "iptm_v24": v["iptm"], "iptm_deeppbs": d["iptm"],
                     "plddt_v24": v["plddt"], "plddt_deeppbs": d["plddt"],
                     "d_iptm": v["iptm"] - d["iptm"],
                     "d_len": dlen.get((g, "v24"), 0) - dlen.get((g, "deeppbs"), 0)})
    n = len(rows)
    if n == 0:
        print("no completed pairs yet"); return
    div = np.array([r["d_iptm"] for r in rows])
    wins = int((div > 0).sum())
    v_mean = float(np.mean([r["iptm_v24"] for r in rows]))
    d_mean = float(np.mean([r["iptm_deeppbs"] for r in rows]))
    vp_mean = float(np.mean([r["plddt_v24"] for r in rows]))
    dp_mean = float(np.mean([r["plddt_deeppbs"] for r in rows]))
    out = {"n_pairs": n, "v24_mean_iptm": round(v_mean, 3), "deeppbs_mean_iptm": round(d_mean, 3),
           "mean_delta_iptm": round(float(div.mean()), 3), "v24_wins": wins, "v24_win_frac": round(wins / n, 3),
           "v24_mean_plddt": round(vp_mean, 3), "deeppbs_mean_plddt": round(dp_mean, 3), "per_tf": rows}
    try:
        from scipy.stats import wilcoxon, spearmanr
        out["wilcoxon_p_iptm"] = float(wilcoxon([r["iptm_v24"] for r in rows],
                                                [r["iptm_deeppbs"] for r in rows]).pvalue)
        rho, pl = spearmanr(div, [r["d_len"] for r in rows])
        out["len_control_spearman_rho"] = round(float(rho), 3); out["len_control_p"] = round(float(pl), 3)
    except Exception:
        pass
    os.makedirs(RES, exist_ok=True)
    json.dump(out, open(f"{RES}/boltz_foldability.json", "w"), indent=2)
    print(f"=== Boltz-2 foldability: v24 vs DeepPBS consensus ({n} TF pairs) ===")
    print(f"  ipTM  : v24 {v_mean:.3f}  vs  DeepPBS {d_mean:.3f}   (Δ {div.mean():+.3f}, "
          f"v24 wins {wins}/{n} = {100*wins/n:.0f}%)")
    print(f"  pLDDT : v24 {vp_mean:.3f}  vs  DeepPBS {dp_mean:.3f}")
    if "wilcoxon_p_iptm" in out:
        print(f"  Wilcoxon p = {out['wilcoxon_p_iptm']:.2e}   len-control Spearman ρ = "
              f"{out.get('len_control_spearman_rho')} (p={out.get('len_control_p')})")
    print(f"  saved {RES}/boltz_foldability.json")


if __name__ == "__main__":
    main()
