"""PBX1 AF3+Rosetta calibration case study — step 1: TFScope seed PWM.

PBX1 (Homeodomain, DBD 73 aa, monomer, motif ~7bp) is genuinely held out:
0 rows in tf_pwm_combined_fm_deeppbs.parquet (gene-level), only appears as the
cluster40 TEST crystal 1b72_B_PBX1.PBX1_HUMAN.H11MO.0.A.txt.

Uses the canonical COMBINED checkpoint (v19_combined_fm_deeppbs_contact/rag_seed42,
learned-10 family scheme, no-RAG, contact-supervision) — NOT dual-family, which
regresses on every metric (see memory dual-family-vs-combined).

Outputs:
  results/pbx1_case_study/seed_pwm.json       - predicted PWM, gate, consensus, gt comparison
  results/pbx1_case_study/fold_pbx1_wt.json   - AlphaFold Server job_request for the WT complex
"""
import os, sys, json
os.environ.setdefault("TORCH_HOME", "/data1/leihuang/.cache/torch")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
sys.path.insert(0, "src")
import numpy as np, torch, torch.nn.functional as F, pandas as pd
from tfscope.config import TFScopeConfig
from tfscope.models.tfscope import TFScopeModel
from tfscope.data.dataset import AA_TO_TOKEN

CKPT = "/data1/leihuang/project/TFScope/checkpoints/v19_combined_fm_deeppbs_contact/rag_seed42/ckpt_best.pt"
PARQ = "data/processed/tf_pwm_deeppbs_only.parquet"
OUT = "results/pbx1_case_study"
os.makedirs(OUT, exist_ok=True)
dev = "cuda:0" if torch.cuda.is_available() else "cpu"
BASES = "ACGT"
COMP = {"A": "T", "C": "G", "G": "C", "T": "A"}

d = pd.read_parquet(PARQ)
r = d[d.gene_symbol == "PBX1"].iloc[0]
dbd = str(r.sequence)[int(r.dbd_start):int(r.dbd_end)]
fid = int(r.family_id)
gt_pwm = np.frombuffer(r.pwm, dtype=np.float32).reshape(4, int(r.motif_length))
print(f"PBX1 DBD ({len(dbd)} aa, family_id={fid} [{r.family_name}]): {dbd}")
print(f"GT motif (HOCOMOCO {r.filename}), {gt_pwm.shape[1]} bp:")
print("  consensus:", "".join(BASES[i] for i in gt_pwm.argmax(0)))

cfg = TFScopeConfig()
for k, v in json.load(open(os.path.join(os.path.dirname(CKPT), "config.json"))).items():
    if hasattr(cfg, k):
        try: setattr(cfg, k, type(getattr(cfg, k))(v))
        except Exception: pass
cfg.use_retrieval = False
m = TFScopeModel(cfg).to(dev).eval()
sd = torch.load(CKPT, map_location=dev, weights_only=False)["model"]
missing, unexpected = m.load_state_dict(sd, strict=False)
print(f"loaded ckpt (missing={len(missing)}, unexpected={len(unexpected)})")

@torch.no_grad()
def predict(seq, fid):
    t = torch.tensor([[AA_TO_TOKEN.get(a, 4) for a in seq]], dtype=torch.long, device=dev)
    dm = torch.ones(1, len(seq), dtype=torch.bool, device=dev)
    fi = torch.tensor([fid], dtype=torch.long, device=dev)
    gl, pl, _ = m(t, dm, fi, retrieved_pwms=None, retrieved_masks=None, retrieved_sims=None, recog_prior=None)
    gate = gl.sigmoid()[0].cpu().numpy()
    pwm = F.softmax(pl, 1)[0].cpu().numpy()  # (4, W)
    return pwm, gate

pwm, gate = predict(dbd, fid)
cols = np.where(gate > 0.5)[0]
if len(cols) < 4:
    ic = (pwm * np.log2(pwm + 1e-9)).sum(0) + 2
    cols = np.arange(max(0, ic.argmax() - 3), min(pwm.shape[1], ic.argmax() + 4))
lo, hi = cols.min(), cols.max()
flank = 3
lo_f, hi_f = max(0, lo - flank), min(pwm.shape[1] - 1, hi + flank)
core_consensus = "".join(BASES[pwm[:, j].argmax()] for j in range(lo, hi + 1))
full_consensus = "".join(BASES[pwm[:, j].argmax()] for j in range(lo_f, hi_f + 1))
ic_core = float(((pwm[:, cols] * np.log2(pwm[:, cols] + 1e-9)).sum(0) + 2).mean())
print(f"\nTFScope predicted width={pwm.shape[1]}, gated core cols=[{lo},{hi}] ({hi-lo+1} bp), IC={ic_core:.3f}")
print("  core consensus:      ", core_consensus)
print("  full (core+3bp flank):", full_consensus)

# Compare core PWM to GT PWM at best ungapped alignment (allow RC)
def best_align_corr(pred, gt):
    best = (-2, 0, False)
    for rc in (False, True):
        p = pred[:, ::-1][::-1] if rc else pred  # reverse-complement: flip positions + base order ACGT->TGCA reversed
        if rc:
            p = pred[::-1, ::-1]
        W, G = p.shape[1], gt.shape[1]
        for shift in range(-(W - 1), G):
            cols_p, cols_g = [], []
            for j in range(G):
                k = j + shift
                if 0 <= k < W:
                    cols_p.append(k); cols_g.append(j)
            if len(cols_p) < min(4, G):
                continue
            a = p[:, cols_p].ravel(); b = gt[:, cols_g].ravel()
            rho = np.corrcoef(a, b)[0, 1]
            if rho > best[0]:
                best = (rho, shift, rc)
    return best

rho, shift, rc = best_align_corr(pwm[:, lo:hi + 1], gt_pwm)
print(f"\ncore-vs-GT best-aligned Pearson r={rho:.3f} (shift={shift}, rc={rc})")

rec = dict(gene="PBX1", uniprot=r.uniprot_id, family=r.family_name, family_id=fid,
           dbd=dbd, checkpoint=CKPT,
           pwm=pwm.tolist(), gate=gate.tolist(), core_lo=int(lo), core_hi=int(hi),
           core_consensus=core_consensus, full_consensus=full_consensus,
           ic_core=round(ic_core, 3), gt_consensus="".join(BASES[i] for i in gt_pwm.argmax(0)),
           gt_pwm=gt_pwm.tolist(), align_r=round(float(rho), 3), align_shift=int(shift), align_rc=bool(rc))
json.dump(rec, open(f"{OUT}/seed_pwm.json", "w"), indent=1)

# ── AlphaFold Server job request for the WT monomer complex ──
top = full_consensus
bot = "".join(COMP[b] for b in top[::-1])
job = [{
    "name": "pbx1_wt_homeodomain",
    "modelSeeds": [],
    "sequences": [
        {"proteinChain": {"sequence": dbd, "count": 1, "useStructureTemplate": True}},
        {"dnaSequence": {"sequence": top, "count": 1}},
        {"dnaSequence": {"sequence": bot, "count": 1}},
    ],
    "dialect": "alphafoldserver",
    "version": 3,
}]
json.dump(job, open(f"{OUT}/fold_pbx1_wt_job_request.json", "w"), indent=1)
print(f"\nWrote {OUT}/seed_pwm.json and {OUT}/fold_pbx1_wt_job_request.json")
print(f"AF3 job: protein {len(dbd)}aa (monomer) + dsDNA {len(top)}bp top/bottom")
