#!/usr/bin/env python
"""Shared utilities for the revised SOHLH1 case study."""
import os, sys, json
sys.path.insert(0, "src"); sys.path.insert(0, "pwm_rosetta")
os.environ.setdefault("TORCH_HOME", "/n/holylabs/lpinello_lab/Lab/leihuang/.cache/torch")
import numpy as np, torch, torch.nn.functional as F, yaml
from scipy.stats import pearsonr
from tfscope.config import TFScopeConfig
from tfscope.models.tfscope import TFScopeModel
from tfscope.data.dataset import AA_TO_TOKEN
from tfscope.models.alignment import align_pwm

BASES = np.array(list("ACGT"))
device = "cuda" if torch.cuda.is_available() else "cpu"


def load_cfg(path="configs/case_study_sohlh1.yaml"):
    return yaml.safe_load(open(path))


def load_model(ckpt, force_retrieval=None):
    cfg = TFScopeConfig()
    cp = os.path.join(os.path.dirname(ckpt), "config.json")
    if os.path.isfile(cp):
        for k, v in json.load(open(cp)).items():
            if hasattr(cfg, k):
                try: setattr(cfg, k, type(getattr(cfg, k))(v))
                except Exception: pass
    if force_retrieval is not None:
        cfg.use_retrieval = force_retrieval
    m = TFScopeModel(cfg).to(device)
    m.load_state_dict(torch.load(ckpt, map_location=device, weights_only=False)["model"], strict=False)
    m.eval()
    return m, cfg


def column_ic(pwm):                                    # (4,L) -> (L,)
    p = np.clip(pwm, 1e-9, 1.0); p = p / p.sum(0, keepdims=True)
    return 2.0 + (p * np.log2(p)).sum(0)


def active_cols(gate, thr=0.5):
    m = gate > thr
    if m.sum() < 2:
        m = np.zeros_like(gate, bool); m[:4] = True
    return m


def aligned_r(pred, target, max_shift=10, rc=True):
    """oracle-aligned, RC-aware mean per-column Pearson r + IC-weighted r."""
    al, _, _, _ = align_pwm(pred, target, max_shift=max_shift, consider_revcomp=rc)
    Lc = min(al.shape[1], target.shape[1])
    rs, w = [], []
    for j in range(Lc):
        a, b = target[:, j], al[:, j]
        if np.std(a) < 1e-8 or np.std(b) < 1e-8:
            r = 0.0
        else:
            r = pearsonr(a, b)[0]
        rs.append(r); w.append(2.0 + float((np.clip(a,1e-9,1)*np.log2(np.clip(a,1e-9,1))).sum()))
    rs = np.array(rs); w = np.array(w)
    mean_r = float(np.nanmean(rs))
    icw = float(np.nansum(rs * w) / (w.sum() + 1e-8))
    return mean_r, icw, al[:, :Lc]


def tokens_from_seq(seq):
    t = torch.tensor([[AA_TO_TOKEN.get(a, 4) for a in seq]], dtype=torch.long, device=device)
    mask = torch.ones(1, len(seq), dtype=torch.bool, device=device)
    return t, mask


def esm_dbd_embedding(seq, _cache={}):
    """ESM2 L33 DBD-mean-pool embedding (matches donor pool). Caches the ESM model."""
    if "esm" not in _cache:
        import esm
        em, alpha = esm.pretrained.esm2_t33_650M_UR50D()
        _cache["esm"] = (em.to(device).eval(), alpha, alpha.get_batch_converter())
    em, alpha, bc = _cache["esm"]
    _, _, toks = bc([("q", seq)])
    with torch.no_grad():
        rep = em(toks.to(device), repr_layers=[em.num_layers])["representations"][em.num_layers]
        v = rep[0, 1:1 + len(seq)].mean(0).cpu().numpy().astype(np.float32)
    return v


def build_retrieval(qseq, embs, donors, fn2gene, fn2pwm, K, ML, exclude_genes=()):
    """Gene-deduplicated top-K retrieval for a query DBD sequence."""
    q = esm_dbd_embedding(qseq); q = q / (np.linalg.norm(q) + 1e-8)
    M = np.stack([embs[fn] / (np.linalg.norm(embs[fn]) + 1e-8) for fn in donors])
    sims = M @ q
    order = np.argsort(-sims)
    excl = {g.upper() for g in exclude_genes}
    top, seen, nbrs = [], set(), []
    for di in order:
        g = fn2gene.get(donors[di], "?")
        if g in excl or g in seen:
            continue
        seen.add(g); top.append(di)
        nbrs.append((g, float(sims[di]), donors[di]))
        if len(top) == K:
            break
    rp = torch.full((1, K, 4, ML), 0.25); rm = torch.zeros((1, K, ML)); rs = torch.zeros((1, K))
    for ki, di in enumerate(top):
        pwm = fn2pwm[donors[di]]; L = min(pwm.shape[1], ML)
        rp[0, ki, :, :L] = torch.from_numpy(pwm[:, :L].copy())
        rm[0, ki, :L] = 1.0; rs[0, ki] = float(sims[di])
    return rp, rm, rs, nbrs, sims, order


def infer(model, tokens, mask, fam_id, ret=None):
    """Single forward; ret=(rp,rm,rs) for RAG else None. Returns gate, pwm, attn."""
    fam = torch.tensor([fam_id], dtype=torch.long, device=device)
    kw = dict(retrieved_pwms=None, retrieved_masks=None, retrieved_sims=None)
    if ret is not None:
        kw = dict(retrieved_pwms=ret[0].to(device), retrieved_masks=ret[1].to(device),
                  retrieved_sims=ret[2].to(device))
    with torch.no_grad():
        gl, pl, aux = model(tokens, mask, fam, **kw)
        gate = gl.sigmoid()[0].cpu().numpy()
        pwm = F.softmax(pl, 1)[0].cpu().numpy()
    attn = aux["attn"][0].cpu().numpy() if aux.get("attn") is not None else None
    return gate, pwm, attn


def confidence_score(cfg, mean_ic_bits, gate_conf):
    """Calibrated decisiveness score = 0.5*IC_norm + 0.5*gate_norm.

    Features chosen because they are the held-out predictors of motif-recovery
    accuracy on cluster40-test known TFs (gate Spearman +0.32, IC +0.29);
    the previously-used rag/noRAG-similarity and retrieval-cosine terms were
    dropped — they were anti-predictive (rho -0.35 and -0.14) on held-out data.
    """
    g0, g1 = cfg.get("gate_norm_range", [0.75, 0.97])
    ic_norm = float(np.clip(mean_ic_bits / 2.0, 0, 1))
    gate_norm = float(np.clip((gate_conf - g0) / (g1 - g0), 0, 1))
    score = 0.5 * ic_norm + 0.5 * gate_norm
    cls = ("High" if score >= cfg["high_confidence_min"]
           else "Medium" if score >= cfg["medium_confidence_min"] else "Low")
    comps = dict(motif_information_content=ic_norm, gate_confidence=gate_norm,
                 mean_IC_bits=float(mean_ic_bits), gate_prob=float(gate_conf))
    return score, cls, comps


def write_meme(path, name, pwm):
    L = pwm.shape[1]
    with open(path, "w") as f:
        f.write("MEME version 4\n\nALPHABET= ACGT\n\nstrands: + -\n\n")
        f.write("Background letter frequencies\nA 0.25 C 0.25 G 0.25 T 0.25\n\n")
        f.write(f"MOTIF {name}\nletter-probability matrix: alength= 4 w= {L} nsites= 20 E= 0\n")
        for j in range(L):
            c = pwm[:, j] / pwm[:, j].sum()
            f.write(" " + " ".join(f"{x:.6f}" for x in c) + "\n")


def write_pwm_tsv(path, pwm):
    import pandas as pd
    pd.DataFrame(pwm.T, columns=list("ACGT")).rename_axis("pos").to_csv(path, sep="\t", float_format="%.6f")


def ebox_pwm(consensus="CACGTG"):
    iupac = {"A":"A","C":"C","G":"G","T":"T","N":"ACGT","R":"AG","Y":"CT"}
    L = len(consensus); m = np.zeros((4, L))
    for j, ch in enumerate(consensus):
        for b in iupac[ch]:
            m["ACGT".index(b), j] = 1.0
    return m / m.sum(0, keepdims=True)
