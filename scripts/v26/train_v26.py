#!/usr/bin/env python
"""v26 Phase-5 training.

Matches v24's schedule for comparability: 225 epochs, early-stop patience 30 on a validation
metric, LoRA-only encoder adaptation, bf16.

Hard rules enforced here:
  * NO metadata inputs. The batch dict passed to the model is screened by
    model.assert_no_metadata_inputs every step; family/source/provenance are used ONLY for the
    hierarchical sampler and for reporting.
  * application_holdout examples (Barrera, MyoD1, designed DBPs) are DROPPED at dataset
    construction -- they cannot reach training, validation, early stopping or checkpoint selection.
  * test is never read.
  * 2-D contact loss weight is 0 by default (docs/v26_contact_2d_decision.md); the
    v24style_contact2d config turns it on as a diagnostic.

Target-first hierarchical sampling (brief Phase 5): sample target unit uniformly -> construct ->
motif source -> record. Uniform sampling over 5,966 rows would over-weight TFs with many motif
records (the dataset has 1,355 target units but up to dozens of records each).

  python scripts/v26/train_v26.py --config configs/v26/core.yaml --seed 42 --out <dir>
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, "src")
from tfscope.models.alignment import align_pwm                      # noqa: E402
from tfscope.v26.config import V26Config                            # noqa: E402
from tfscope.v26.model import TFScopeV26, assert_no_metadata_inputs  # noqa: E402

AA = {"L": 4, "A": 5, "G": 6, "V": 7, "S": 8, "E": 9, "R": 10, "T": 11, "I": 12, "D": 13,
      "P": 14, "K": 15, "Q": 16, "N": 17, "F": 18, "Y": 19, "M": 20, "H": 21, "W": 22, "C": 23}
PAD = 1
MANIFEST = "data/processed/splits/v26/manifest.parquet"


# ------------------------------------------------------------------------- config
def load_cfg(path):
    raw = {}
    for line in open(path):
        line = line.split("#")[0].strip()
        if not line or ":" not in line:
            continue
        k, v = line.split(":", 1)
        try:
            raw[k.strip()] = json.loads(v.strip())
        except Exception:
            raw[k.strip()] = v.strip()
    cfg = V26Config()
    for k, v in raw.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    return cfg, raw


# ------------------------------------------------------------------------ dataset
class V26Data:
    """Holds one split. Chains are packed per batch; nothing is precomputed on GPU."""

    def __init__(self, dataset: str, split: str, cfg: V26Config, contacts=None,
                 recog=None, shuffle_flanks=False, seed=0, manifest=None):
        ex = pd.read_parquet(f"data/processed/v26/v26_{dataset}.parquet")
        man = pd.read_parquet(manifest or MANIFEST)[
            ["target_unit_id", "split", "application_holdout"]].drop_duplicates("target_unit_id")
        ex = ex.merge(man, on="target_unit_id", how="inner")
        # application sets can never be seen, in any split
        ex = ex[~ex.application_holdout]
        ex = ex[ex.split == split].reset_index(drop=True)
        self.ex = ex
        self.cfg = cfg
        self.split = split
        self.shuffle_flanks = shuffle_flanks
        self.rng = np.random.default_rng(seed)

        # 1-D contact labels: residues within 4.5 A of DNA, in CROP coordinates, eval_only excluded
        self.contact = defaultdict(set)
        if contacts is not None:
            c = contacts[(contacts.in_crop) & (~contacts.eval_only)
                         & (contacts.chain_role == "primary")]
            for eid, g in c.groupby("example_id"):
                self.contact[eid] = set(int(i) for i in g.crop_residue_idx)
        self.recog = defaultdict(set)
        if recog is not None:
            for eid, g in recog.groupby("example_id"):
                self.recog[eid] = set(int(i) for i in g.crop_residue_idx)

        # hierarchical index: target unit -> sequence hash -> motif source -> row ids
        self.tree = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
        for i, r in enumerate(ex.itertuples()):
            self.tree[r.target_unit_id][r.primary_sequence_hash][r.motif_source].append(i)
        self.units = sorted(self.tree)

    def __len__(self):
        return len(self.ex)

    def sample_indices(self, n):
        """Target unit -> construct -> motif source -> record, each uniform at its level."""
        out = []
        for _ in range(n):
            u = self.units[self.rng.integers(len(self.units))]
            hs = list(self.tree[u]); h = hs[self.rng.integers(len(hs))]
            ss = list(self.tree[u][h]); s = ss[self.rng.integers(len(ss))]
            rows = self.tree[u][h][s]
            out.append(rows[self.rng.integers(len(rows))])
        return out

    def target_pwm(self, r):
        b = r.pwm
        a = np.frombuffer(b, dtype=np.float32).reshape(4, -1).astype(np.float32)
        L = min(int(r.motif_length), self.cfg.max_motif_length, a.shape[1])
        t = np.full((4, self.cfg.max_motif_length), 0.25, np.float32)
        t[:, :L] = a[:, :L]
        m = np.zeros(self.cfg.max_motif_length, np.float32)
        m[:L] = 1.0
        return t, m, L

    def _maybe_shuffle(self, seq, d0, d1):
        if not self.shuffle_flanks:
            return seq
        pre, core, post = list(seq[:d0]), seq[d0:d1], list(seq[d1:])
        self.rng.shuffle(pre); self.rng.shuffle(post)
        return "".join(pre) + core + "".join(post)

    def collate(self, idxs):
        toks, dms, cidx, prim = [], [], [], []
        pwms, masks, lens, c1d, cmask, rprior = [], [], [], [], [], []
        for b, i in enumerate(idxs):
            r = self.ex.iloc[i]
            d0, d1 = int(r.dbd_start), int(r.dbd_end)
            seq = self._maybe_shuffle(str(r.sequence), d0, d1)
            toks.append([AA.get(c, 4) for c in seq])
            dm = np.zeros(len(seq), bool); dm[d0:min(d1, len(seq))] = True
            dms.append(dm); cidx.append(b); prim.append(True)
            for p in json.loads(r.partner_entities or "[]")[:self.cfg.max_partners]:
                ps = str(p["sequence"])
                toks.append([AA.get(c, 4) for c in ps])
                dms.append(np.ones(len(ps), bool)); cidx.append(b); prim.append(False)
            t, m, L = self.target_pwm(r)
            pwms.append(t); masks.append(m); lens.append(L)
            cset = self.contact.get(r.example_id, set())
            rset = self.recog.get(r.example_id, set())
            c1d.append((len(seq), cset)); cmask.append(1.0 if cset else 0.0)
            rprior.append((len(seq), rset))

        Lmax = max(len(t) for t in toks)
        T = torch.full((len(toks), Lmax), PAD, dtype=torch.long)
        D = torch.zeros((len(toks), Lmax), dtype=torch.bool)
        for i, (t, d) in enumerate(zip(toks, dms)):
            T[i, :len(t)] = torch.tensor(t)
            D[i, :len(d)] = torch.from_numpy(d)
        B = len(idxs)
        C1 = torch.zeros(B, Lmax); RP = torch.zeros(B, Lmax)
        for b, (n, s) in enumerate(c1d):
            for j in s:
                if 0 <= j < Lmax:
                    C1[b, j] = 1.0
        for b, (n, s) in enumerate(rprior):
            for j in s:
                if 0 <= j < Lmax:
                    RP[b, j] = 1.0
        return {
            "sequence_tokens": T, "dbd_mask": D,
            "chain_index": torch.tensor(cidx), "is_primary": torch.tensor(prim),
            "target_pwm": torch.from_numpy(np.stack(pwms)),
            "pwm_mask": torch.from_numpy(np.stack(masks)),
            "target_length": torch.tensor(lens, dtype=torch.long),
            "contact1d": C1, "contact1d_has_label": torch.tensor(cmask),
            "recog_prior": RP,
        }


# --------------------------------------------------------------------------- loss
RC_IDX = [3, 2, 1, 0]          # A<->T, C<->G


def _masked_pwm_terms(pwm, tgt, mm):
    """(L1, 1-pearson) between pwm and tgt over the masked columns."""
    n = mm.sum().clamp(min=1)
    l1 = ((pwm - tgt).abs() * mm).sum() / (n * 4)
    p = (pwm * mm).flatten(1); q = (tgt * mm).flatten(1)
    p = p - p.mean(1, keepdim=True); q = q - q.mean(1, keepdim=True)
    pear = (p * q).sum(1) / (p.norm(dim=1) * q.norm(dim=1)).clamp(min=1e-8)
    return l1, pear


def registered_pwm_loss(pwm, tgt, mask, max_shift=6, use_rc=True):
    """PWM loss evaluated at the BEST offset+orientation, per batch.

    v24 trained with --latent-registration; v26 did not, and the plain column-wise loss assumes
    predicted column j corresponds to target column j. If the model places the motif a few
    columns off, that loss punishes a correct motif for being mis-registered, and the gradient
    pushes toward a blurred average instead of a sharp shifted motif. This is the single most
    likely source of the v26-vs-v24 gap (v24 0.5828 vs v26 0.3507 cov_r on the same clean split).

    Hard selection over alignments: gradient flows through the winning alignment only.
    """
    mm = mask.unsqueeze(1)
    best = None
    best_pear = None
    for rc in ((False, True) if use_rc else (False,)):
        p0 = pwm[:, RC_IDX].flip(-1) if rc else pwm
        for sh in range(-max_shift, max_shift + 1):
            p = torch.roll(p0, shifts=sh, dims=-1)
            if sh > 0:
                p = torch.cat([p.new_full((*p.shape[:-1], sh), 0.25), p[..., sh:]], dim=-1)
            elif sh < 0:
                k = -sh
                p = torch.cat([p[..., :-k], p.new_full((*p.shape[:-1], k), 0.25)], dim=-1)
            l1, pear = _masked_pwm_terms(p, tgt, mm)
            loss = l1 + (1.0 - pear).mean()
            if best is None or loss.item() < best.item():
                best, best_pear = loss, pear
    return best, best_pear


def topbase_hinge(pwm, tgt, mask, margin=2.0, ic_thresh=0.5):
    """Hinge encouraging the correct dominant base on INFORMATIVE target columns (v24 term)."""
    p = torch.clamp(tgt, 1e-9, 1.0)
    ic = 2.0 + (p * torch.log2(p)).sum(1)                      # (B,W)
    sel = (mask > 0) & (ic > ic_thresh)
    if not sel.any():
        return pwm.new_zeros(())
    tb = tgt.argmax(1)                                         # (B,W) target dominant base
    lg = torch.log(pwm.clamp(min=1e-9))
    correct = lg.gather(1, tb.unsqueeze(1)).squeeze(1)          # (B,W)
    other = lg.masked_fill(
        F.one_hot(tb, 4).permute(0, 2, 1).bool(), -1e9).max(1).values
    return (F.relu(margin - (correct - other)) * sel.float()).sum() / sel.float().sum()


def covr_term(gate, mask):
    """Reward predicted span length matching the target motif length (v24 pwm_cov_r_weight)."""
    pred_len = gate.sum(-1).clamp(min=1e-3)
    true_len = mask.sum(-1).clamp(min=1e-3)
    cov = torch.minimum(pred_len, true_len) / torch.maximum(pred_len, true_len)
    return (1.0 - cov).mean()


def ic_terms(pwm, tgt, mask):
    """v24's three sharpness terms, ported from src/tfscope/losses/tfscope_loss.py.

    v26 omitted all three (combined weight 1.1 in v24). The multi-metric backfill showed v26's
    ic_mae at 0.81-1.03 vs v24's 0.49-0.62 -- i.e. v26 predicts PWMs that are far too FLAT, which
    drags down pearson, top-base, MAE and IC simultaneously. These are the terms that suppress
    flat solutions.

      ic        |IC_target - IC_pred| per column                (_pwm_ic)
      ic_pcc    per-COLUMN Pearson over the 4-vector (A,C,G,T),
                weighted by target IC and IC-normalised          (_pwm_ic_pcc)
      entropy   mean H(pred); minimising it sharpens the output  (_pwm_entropy)

    Note ic_pcc is a *per-column* correlation, unlike the flattened whole-PWM Pearson v26
    already had -- a model can score well on the flattened version while every individual
    column is uninformative.
    """
    LOG4 = float(np.log(4.0))
    p = pwm.clamp(1e-8, 1.0)
    t = tgt.clamp(1e-8, 1.0)
    m = mask
    denom = m.sum().clamp(min=1)

    ic_pred = (p * p.log()).sum(1) + LOG4                      # (B,L) nats
    ic_tgt = (t * t.log()).sum(1) + LOG4
    l_ic = ((ic_tgt - ic_pred).abs() * m).sum() / denom

    pc = pwm - pwm.mean(1, keepdim=True)
    tc = tgt - tgt.mean(1, keepdim=True)
    r = (pc * tc).sum(1) / (pc.norm(dim=1) * tc.norm(dim=1) + 1e-8)   # (B,L)
    w = ic_tgt.clamp(min=0) * m
    w = w / w.sum(1, keepdim=True).clamp(min=1e-8)
    l_ic_pcc = ((1.0 - r) * w).sum(1).mean()

    ent = -(p * p.log()).sum(1)                                 # (B,L)
    l_ent = (ent * m).sum() / denom
    return l_ic, l_ic_pcc, l_ent


def compute_loss(cfg, out, batch, aux):
    pwm, gate = out
    tgt, m = batch["target_pwm"], batch["pwm_mask"]
    mm = m.unsqueeze(1)
    n = mm.sum().clamp(min=1)

    if getattr(cfg, "w_registration", 0.0) > 0:
        l_pwm, pear = registered_pwm_loss(
            pwm, tgt, m, max_shift=getattr(cfg, "registration_max_shift", 6),
            use_rc=bool(getattr(cfg, "registration_rc", True)))
    else:
        l1, pear = _masked_pwm_terms(pwm, tgt, mm)
        l_pwm = l1 + (1.0 - pear).mean()

    l_tb = (topbase_hinge(pwm, tgt, m, getattr(cfg, "topbase_margin", 2.0))
            if getattr(cfg, "w_topbase", 0.0) > 0 else pwm.new_zeros(()))
    l_cov = (covr_term(gate, m) if getattr(cfg, "w_covr", 0.0) > 0
             else pwm.new_zeros(()))

    w_ic = getattr(cfg, "w_ic", 0.0); w_icp = getattr(cfg, "w_ic_pcc", 0.0)
    w_ent = getattr(cfg, "w_entropy", 0.0)
    if w_ic or w_icp or w_ent:
        l_ic, l_icp, l_ent = ic_terms(pwm, tgt, m)
    else:
        l_ic = l_icp = l_ent = pwm.new_zeros(())

    tl = (batch["target_length"] - cfg.min_motif_length).clamp(
        0, cfg.max_motif_length - cfg.min_motif_length)
    l_len = F.cross_entropy(aux["length_logits"], tl)

    l_c1 = pwm.new_zeros(())
    if cfg.w_contact1d > 0:
        logit = aux["contact1d_logit"]
        lab = batch["contact1d"][:, :logit.shape[1]]
        valid = (~aux["core_ignore_primary"]).float()
        has = batch["contact1d_has_label"].unsqueeze(1)
        w = valid * has
        if w.sum() > 0:
            l_c1 = (F.binary_cross_entropy_with_logits(logit, lab, reduction="none")
                    * w).sum() / w.sum()

    l_rp = pwm.new_zeros(())
    if cfg.w_recognition_prior > 0:
        logit = aux["contact1d_logit"]
        pr = batch["recog_prior"][:, :logit.shape[1]]
        valid = (~aux["core_ignore_primary"]).float()
        if valid.sum() > 0:
            l_rp = (F.binary_cross_entropy_with_logits(logit, pr, reduction="none")
                    * valid).sum() / valid.sum()

    total = (cfg.w_pwm * l_pwm + cfg.w_length * l_len + cfg.w_contact1d * l_c1
             + cfg.w_recognition_prior * l_rp + cfg.balance_loss_weight * aux["balance_loss"]
             + getattr(cfg, "w_topbase", 0.0) * l_tb + getattr(cfg, "w_covr", 0.0) * l_cov
             + w_ic * l_ic + w_icp * l_icp + w_ent * l_ent)
    return total, {"pwm": float(l_pwm), "len": float(l_len), "c1d": float(l_c1),
                   "rprior": float(l_rp), "bal": float(aux["balance_loss"]),
                   "tb": float(l_tb), "cov": float(l_cov),
                   "ic": float(l_ic), "icp": float(l_icp), "ent": float(l_ent),
                   "pearson": float(pear.mean())}


# --------------------------------------------------------------------- validation
@torch.no_grad()
def validate(model, data, cfg, device, batch=8):
    """Target-unit-level content Pearson r with oracle registration + coverage (v24 PanelA-like).

    Statistical unit is the TARGET UNIT, not the row (brief Phase 6), so a TF with 30 motif
    records counts once.
    """
    model.eval()
    per_unit = defaultdict(list)
    idxs = list(range(len(data)))
    for s in range(0, len(idxs), batch):
        sel = idxs[s:s + batch]
        b = data.collate(sel)
        bb = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in b.items()}
        pwm, gate, aux = model(bb["sequence_tokens"], bb["dbd_mask"],
                               bb["chain_index"], bb["is_primary"])
        pwm = pwm.float().cpu().numpy(); gate = gate.float().cpu().numpy()
        for j, i in enumerate(sel):
            r = data.ex.iloc[i]
            t, m, L = data.target_pwm(r)
            span = max(1, int(round(gate[j].sum())))
            pred = pwm[j][:, :span]
            gt = t[:, :L]
            try:
                al, _, _, ov = align_pwm(pred, gt, max_shift=10, consider_revcomp=True,
                                         min_overlap=3)
                a = al.flatten(); g = gt.flatten()
                if a.size == g.size and a.std() > 0 and g.std() > 0:
                    r_ = float(np.corrcoef(a, g)[0, 1])
                    cov = min(span, L) / max(span, L)
                    per_unit[r.target_unit_id].append((r_, cov))
            except Exception:
                continue
    if not per_unit:
        return {"content_r": 0.0, "cov_r": 0.0, "n_units": 0}
    rs, cs = [], []
    for u, v in per_unit.items():
        rs.append(np.mean([x[0] for x in v])); cs.append(np.mean([x[0] * x[1] for x in v]))
    return {"content_r": float(np.mean(rs)), "cov_r": float(np.mean(cs)),
            "n_units": len(per_unit)}


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=225)          # match v24
    ap.add_argument("--patience", type=int, default=30)         # match v24
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=4.5e-4)
    ap.add_argument("--lora-lr", type=float, default=7.5e-6)
    ap.add_argument("--steps-per-epoch", type=int, default=0)   # 0 = len(train)/batch
    ap.add_argument("--manifest", default=None,
                    help="split manifest; defaults to the v26 one. Use "
                         "data/processed/splits/v26/manifest_v23compat.parquet for the "
                         "same-split-as-v24 diagnostic.")
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--save-every", type=int, default=25)
    a = ap.parse_args()

    cfg, raw = load_cfg(a.config)
    ds = raw.get("dataset", "core")
    # BUGFIX: config-file epochs/patience were ignored, so reg_strong asked for 40 and ran 225
    # with a OneCycle schedule spanning 225 -- the LR never annealed. CLI still wins if given.
    import sys as _s
    if "--epochs" not in _s.argv and isinstance(raw.get("epochs"), int):
        a.epochs = int(raw["epochs"])
    if "--patience" not in _s.argv and isinstance(raw.get("patience"), int):
        a.patience = int(raw["patience"])
    print(f"schedule: epochs={a.epochs} patience={a.patience}", flush=True)
    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    contacts = recog = None
    cp = f"data/contacts_v26/projected_{ds}.parquet"
    if os.path.exists(cp):
        contacts = pd.read_parquet(cp)
    rp = f"data/contacts_v26/recognition_prior_{ds}.parquet"
    if os.path.exists(rp):
        recog = pd.read_parquet(rp)

    shuf = bool(raw.get("shuffle_flanks", False))
    tr = V26Data(ds, "train", cfg, contacts, recog, shuffle_flanks=shuf, seed=a.seed,
                 manifest=a.manifest)
    va = V26Data(ds, "val", cfg, contacts, recog, shuffle_flanks=shuf, seed=a.seed + 1,
                 manifest=a.manifest)
    print(f"config={a.config} dataset={ds} seed={a.seed} shuffle_flanks={shuf}", flush=True)
    print(f"train examples={len(tr)} units={len(tr.units)} | "
          f"val examples={len(va)} units={len(va.units)}", flush=True)
    print(f"contact-labelled train examples: {sum(1 for e in tr.ex.example_id if tr.contact.get(e))}",
          flush=True)

    model = TFScopeV26(cfg).to(device); model.build(device)
    print("param counts:", model.param_counts(), flush=True)
    lora = [p for n, p in model.named_parameters() if "lora_" in n and p.requires_grad]
    head = [p for n, p in model.named_parameters() if "lora_" not in n and p.requires_grad]
    opt = torch.optim.AdamW([{"params": head, "lr": a.lr},
                             {"params": lora, "lr": a.lora_lr}], weight_decay=0.01)
    spe = a.steps_per_epoch or max(1, len(tr) // a.batch_size)
    total_steps = spe * a.epochs
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=[a.lr, a.lora_lr], total_steps=total_steps, pct_start=0.05)
    json.dump({"config_file": a.config, "config": cfg.to_dict(), "raw": raw,
               "seed": a.seed, "epochs": a.epochs, "patience": a.patience,
               "batch_size": a.batch_size, "grad_accum": a.grad_accum,
               "steps_per_epoch": spe, "dataset": ds},
              open(f"{a.out}/config.json", "w"), indent=2)

    best, best_ep, bad = -1e9, -1, 0
    hist = []
    t0 = time.time()
    for ep in range(1, a.epochs + 1):
        model.train()
        agg = defaultdict(float); nb = 0
        for step in range(spe):
            idxs = tr.sample_indices(a.batch_size)
            b = tr.collate(idxs)
            assert_no_metadata_inputs(b)
            bb = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in b.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                pwm, gate, aux = model(bb["sequence_tokens"], bb["dbd_mask"],
                                       bb["chain_index"], bb["is_primary"])
                loss, parts = compute_loss(cfg, (pwm.float(), gate.float()), bb, aux)
            (loss / a.grad_accum).backward()
            if (step + 1) % a.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0)
                opt.step(); opt.zero_grad(set_to_none=True)
                if sched.last_epoch < total_steps - 1:
                    sched.step()
            for k, v in parts.items():
                agg[k] += v
            agg["total"] += float(loss); nb += 1

        line = " ".join(f"{k}={agg[k]/max(nb,1):.4f}" for k in
                        ("total", "pwm", "len", "c1d", "rprior", "pearson"))
        if ep % a.eval_every == 0 or ep == 1 or ep == a.epochs:
            m = validate(model, va, cfg, device)
            hist.append({"epoch": ep, **{k: agg[k] / max(nb, 1) for k in agg}, **m})
            improved = m["cov_r"] > best
            if improved:
                best, best_ep, bad = m["cov_r"], ep, 0
                torch.save({"model": model.state_dict(), "epoch": ep, "metric": m,
                            "config": cfg.to_dict()}, f"{a.out}/ckpt_best.pt")
            else:
                bad += 1
            print(f"[ep {ep:3d}/{a.epochs}] {line} | val content_r={m['content_r']:.4f} "
                  f"cov_r={m['cov_r']:.4f} units={m['n_units']} "
                  f"{'*BEST*' if improved else f'({bad}/{a.patience//a.eval_every} bad)'} "
                  f"{(time.time()-t0)/60:.1f}m", flush=True)
            json.dump(hist, open(f"{a.out}/history.json", "w"), indent=1)
            if bad >= max(1, a.patience // a.eval_every):
                print(f"early stop at epoch {ep}: no val improvement for {a.patience} epochs",
                      flush=True)
                break
        else:
            print(f"[ep {ep:3d}/{a.epochs}] {line} {(time.time()-t0)/60:.1f}m", flush=True)
        if ep % a.save_every == 0:
            torch.save({"model": model.state_dict(), "epoch": ep}, f"{a.out}/ckpt_ep{ep:03d}.pt")

    json.dump({"best_cov_r": best, "best_epoch": best_ep, "epochs_run": ep,
               "minutes": (time.time() - t0) / 60},
              open(f"{a.out}/DONE.json", "w"), indent=2)
    print(f"DONE best cov_r={best:.4f} @ epoch {best_ep} in {(time.time()-t0)/60:.1f}m", flush=True)


if __name__ == "__main__":
    main()
