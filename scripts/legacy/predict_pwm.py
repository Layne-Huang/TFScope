#!/usr/bin/env python
"""TFScope standalone predictor: protein (DBD) sequence -> predicted PWM + logo.

Self-contained de-novo prediction (no retrieval database needed) using the v18a
DeepPBS-split checkpoint, on which retrieval is inert (de-novo == RAG). Given a
DNA-binding-domain amino-acid sequence it writes the PWM (TSV + MEME) and a sequence
logo, and prints the consensus, information content and a calibrated confidence.

Examples
--------
  # single sequence, known family
  python scripts/predict_pwm.py --seq SCLRRNVISERERRKRMSLSCERLRALLPQFDGRRE... \
      --family bHLH --name MYTF --outdir results/predict

  # from a FASTA file, let the tool pick the best-fitting family
  python scripts/predict_pwm.py --fasta mytf_dbd.fasta --scan-families

Run from the repo root (so the tfscope package and pwm_rosetta resolve). Requires the
`tfscope` conda env.
"""
import os, sys, argparse, json
sys.path.insert(0, "scripts/case_study")          # cs_utils (handles src/ + pwm_rosetta paths)
sys.path.insert(0, "pwm_rosetta")
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from cs_utils import (load_model, column_ic, active_cols, tokens_from_seq, infer,
                      build_retrieval, BASES)
from pwm_hybrid.pwm.viz import makeLogo

DEFAULT_CKPT = "/n/holylabs/lpinello_lab/Lab/leihuang/TFScope/checkpoints/deeppbs_v18a_attnrepair/ckpt_best.pt"
RAG_CKPT = "/n/holylabs/lpinello_lab/Lab/leihuang/TFScope/checkpoints/cluster40_v18a_rag/ckpt_best.pt"
DEFAULT_SPLIT = "data/processed/splits/cluster40/split.json"
DEFAULT_PARQUET = "data/processed/tf_pwm_aug_dbd_canon_trim.parquet"
DEFAULT_EMB = "data/processed/tf_dbd_embeddings_aug.npz"
FAMILIES = {"C2H2_short": 0, "C2H2_medium": 1, "C2H2_long": 2, "bHLH": 3, "Homeodomain": 4,
            "bZIP": 5, "Nuclear_Receptor": 6, "Forkhead": 7, "ETS": 8, "Other": 9}
ID2FAM = {v: k for k, v in FAMILIES.items()}
# calibrated-confidence constants (held-out predictors of motif-recovery accuracy)
GATE_LO, GATE_HI = 0.75, 0.97
CONF_HIGH, CONF_MED = 0.70, 0.40


def read_seqs(args):
    """Return a list of (name, sequence). --seq accepts one or more sequences;
    --fasta reads every record in the file."""
    out = []
    if args.seq:
        for i, s in enumerate(args.seq):
            nm = args.name if len(args.seq) == 1 else f"{args.name}_{i+1}"
            out.append((nm, "".join(s.split()).upper()))
        return out
    name, seq = None, []
    for ln in open(args.fasta):
        ln = ln.strip()
        if ln.startswith(">"):
            if seq: out.append((name or args.name, "".join(seq))); seq = []
            name = ln[1:].split()[0] if len(ln) > 1 else None
        elif ln:
            seq.append(ln.upper())
    if seq: out.append((name or args.name, "".join(seq)))
    return out


def resolve_family(s):
    if s is None: return FAMILIES["Other"]
    if str(s).isdigit() and int(s) in ID2FAM: return int(s)
    for k in FAMILIES:                              # case-insensitive name match
        if k.lower() == str(s).lower(): return FAMILIES[k]
    raise SystemExit(f"unknown --family '{s}'. choose from: {list(FAMILIES)} or 0-9")


def confidence(core, gate_active):
    ic = float(column_ic(core).mean())
    ic_norm = float(np.clip(ic / 2.0, 0, 1))
    gate_norm = float(np.clip((float(gate_active.mean()) - GATE_LO) / (GATE_HI - GATE_LO), 0, 1))
    score = 0.5 * ic_norm + 0.5 * gate_norm
    cls = "High" if score >= CONF_HIGH else "Medium" if score >= CONF_MED else "Low"
    return ic, score, cls


def predict(model, seq, fam_id, gate_thr, width=0, ret=None):
    tok, mask = tokens_from_seq(seq)
    gate, pwm, _ = infer(model, tok, mask, fam_id, ret=ret)    # ret=None -> de-novo
    if width and width > 0:                                    # fixed-width: highest-IC window
        L = pwm.shape[1]; w = min(width, L)
        ic_cols = column_ic(pwm)
        s = int(np.argmax([ic_cols[i:i + w].sum() for i in range(L - w + 1)]))
        core, gsel = pwm[:, s:s + w], gate[s:s + w]
    else:                                                      # gate-active columns
        m = active_cols(gate, gate_thr); core, gsel = pwm[:, m], gate[m]
    ic, conf, cls = confidence(core, gsel)
    return core, ic, conf, cls


def write_pwm_tsv(path, core):
    with open(path, "w") as f:
        f.write("pos\tA\tC\tG\tT\n")
        for j in range(core.shape[1]):
            c = core[:, j] / core[:, j].sum()
            f.write(f"{j}\t" + "\t".join(f"{x:.6f}" for x in c) + "\n")


def write_meme(path, name, core):
    with open(path, "w") as f:
        f.write("MEME version 4\n\nALPHABET= ACGT\n\nstrands: + -\n\n")
        f.write("Background letter frequencies\nA 0.25 C 0.25 G 0.25 T 0.25\n\n")
        f.write(f"MOTIF {name}\nletter-probability matrix: alength= 4 w= {core.shape[1]} nsites= 20 E= 0\n")
        for j in range(core.shape[1]):
            c = core[:, j] / core[:, j].sum()
            f.write(" " + " ".join(f"{x:.6f}" for x in c) + "\n")


def save_logo(path, core, title):
    ppm = np.clip(core.T, 1e-9, 1); ppm = ppm / ppm.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(max(2.2, 0.42 * core.shape[1]), 1.8))
    makeLogo(ppm, ax); ax.set_ylim(0, 2); ax.set_yticks([0, 1, 2])
    ax.set_ylabel("bits", fontsize=8); ax.set_title(title, fontsize=8); ax.tick_params(labelsize=7)
    fig.savefig(path, bbox_inches="tight"); fig.savefig(path.replace(".pdf", ".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description="TFScope: protein DBD sequence -> predicted PWM + logo")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--seq", nargs="+", help="one or more DBD amino-acid sequences")
    g.add_argument("--fasta", help="FASTA file (all records processed)")
    ap.add_argument("--family", help="DBD family name or id 0-9 (default: Other). "
                    f"names: {', '.join(FAMILIES)}")
    ap.add_argument("--scan-families", action="store_true",
                    help="try all 10 families and pick the highest-confidence one")
    ap.add_argument("--name", default="query", help="label for outputs (default: query)")
    ap.add_argument("--outdir", default="results/predict", help="output directory")
    ap.add_argument("--checkpoint", default=DEFAULT_CKPT, help="model checkpoint (.pt)")
    ap.add_argument("--gate-threshold", type=float, default=0.5, help="position-gate threshold")
    ap.add_argument("--width", type=int, default=0,
                    help="fixed PWM width in columns (e.g. 9 = highest-IC window); 0 = gate-active columns")
    ap.add_argument("--rag", action="store_true",
                    help="enable retrieval (needed for the cluster40 checkpoint, whose de-novo path is "
                         "degenerate); auto-uses the cluster40 checkpoint unless --checkpoint is given")
    ap.add_argument("--split", default=DEFAULT_SPLIT, help="donor split json for --rag (train+val = donor pool)")
    args = ap.parse_args()

    seqs = read_seqs(args)
    os.makedirs(args.outdir, exist_ok=True)
    # --rag without an explicit checkpoint -> use the retrieval-trained cluster40 checkpoint
    ckpt_path = args.checkpoint
    if args.rag and args.checkpoint == DEFAULT_CKPT:
        ckpt_path = RAG_CKPT
    model, _ = load_model(ckpt_path, force_retrieval=True if args.rag else None)
    ckpt = os.path.basename(os.path.dirname(ckpt_path))

    # retrieval donor pool (only when --rag)
    retro = None
    if args.rag:
        import pandas as pd
        df = pd.read_parquet(args.donor_parquet if hasattr(args, "donor_parquet") else DEFAULT_PARQUET)
        df["g"] = df["gene_symbol"].astype(str).str.upper()
        fn2gene = dict(zip(df["filename"], df["g"]))
        fn2pwm = {r["filename"]: np.frombuffer(r["pwm"], np.float32).reshape(4, -1) for _, r in df.iterrows()}
        sp = json.load(open(args.split)); tv = set(sp["train"]) | set(sp.get("val", []))
        embs = np.load(DEFAULT_EMB); donors = [f for f in embs.files if f in tv]
        retro = (embs, donors, fn2gene, fn2pwm)

    print(f"# TFScope predictor | {len(seqs)} sequence(s) | ckpt {ckpt} | "
          f"mode {'RAG' if args.rag else 'de-novo'}"
          f"{f' | width {args.width}' if args.width else ''}\n")
    fixed_fid = None if args.scan_families else resolve_family(args.family)

    def get_ret(seq):
        if retro is None: return None
        embs, donors, fn2gene, fn2pwm = retro
        rp, rm, rs, *_ = build_retrieval(seq, embs, donors, fn2gene, fn2pwm, 3, 20)
        return (rp, rm, rs)

    for name, seq in seqs:
        if not seq or any(c not in "ACDEFGHIKLMNPQRSTVWYXBZUO" for c in seq):
            print(f"[skip] {name}: empty or non-amino-acid sequence", file=sys.stderr); continue
        if len(seq) > 200:
            print(f"[warn] {name}: {len(seq)} aa; TFScope expects a DNA-binding DOMAIN "
                  "(~50-160 aa). Pass just the DBD for best results.", file=sys.stderr)
        ret = get_ret(seq)                                     # retrieval once per sequence (or None)
        if args.scan_families:
            rows = []
            for fam, fid in FAMILIES.items():
                core, ic, conf, cls = predict(model, seq, fid, args.gate_threshold, width=args.width, ret=ret)
                rows.append((conf, fam, fid, "".join(BASES[core.argmax(0)]), ic, core))
            rows.sort(reverse=True, key=lambda r: r[0])
            print(f"=== {name} ({len(seq)} aa) — family scan ===")
            for conf, fam, fid, consensus, ic, _ in rows:
                print(f"   {fam:16s} {consensus:18s} IC {ic:4.2f}  conf {conf:4.2f}")
            conf, fam, fid, consensus, ic, core = rows[0]
            cls = "High" if conf >= CONF_HIGH else "Medium" if conf >= CONF_MED else "Low"
        else:
            fid = fixed_fid; fam = ID2FAM[fid]
            core, ic, conf, cls = predict(model, seq, fid, args.gate_threshold, width=args.width, ret=ret)
            consensus = "".join(BASES[core.argmax(0)])

        base = os.path.join(args.outdir, name)
        write_pwm_tsv(f"{base}.pwm.tsv", core)
        write_meme(f"{base}.meme", name, core)
        save_logo(f"{base}_logo.pdf", core, f"{name} ({fam})  {consensus}")
        json.dump(dict(name=name, family=fam, family_id=fid, length_aa=len(seq),
                       motif_length=int(core.shape[1]), consensus=consensus,
                       mean_IC_bits=round(ic, 3), confidence=round(conf, 3),
                       confidence_class=cls, checkpoint=ckpt),
                  open(f"{base}.json", "w"), indent=2)
        print(f"-> {name}: {consensus}  | {core.shape[1]} bp, IC {ic:.2f} | "
              f"conf {conf:.2f} ({cls}) [{fam}] -> {base}.pwm.tsv/.meme/_logo.pdf\n")


if __name__ == "__main__":
    main()
