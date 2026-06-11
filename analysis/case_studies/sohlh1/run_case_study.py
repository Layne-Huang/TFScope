#!/usr/bin/env python
"""SOHLH1 orphan-TF case study — sequence-only motif nomination with TFScope.

Pipeline (single driver; see plan/TFScope_SOHLH1_case_study_plan.md):
  1. metadata + sequence/DBD tables for SOHLH1 (orphan) and SOHLH2 (paralog)
  2. leakage audit  -> SOHLH1 absent from every TFScope table; SOHLH2 present
  3. retrieval neighbours (ESM2 L33 DBD-mean-pool query vs donor pool, LGO)
  4. TFScope inference in two modes: noRAG and RAG_LGO
  5. calibrated confidence (transparent rule-based score)
  6. reference-motif comparison: SOHLH1 prediction vs SOHLH2 motif & E-box

All numeric results are computed here and written to results/case_study_sohlh1/.
"""
import os, sys, json, yaml, glob
sys.path.insert(0, "src"); sys.path.insert(0, "pwm_rosetta")
os.environ.setdefault("TORCH_HOME", "/n/holylabs/lpinello_lab/Lab/leihuang/.cache/torch")
import numpy as np, pandas as pd, torch, torch.nn.functional as F
from scipy.stats import pearsonr
from tfscope.config import TFScopeConfig
from tfscope.models.tfscope import TFScopeModel
from tfscope.data.dataset import AA_TO_TOKEN
from tfscope.models.alignment import align_pwm

HERE = os.path.dirname(os.path.abspath(__file__))
CFG  = yaml.safe_load(open(os.path.join(HERE, "config.yaml")))
OUT  = CFG["case_study"]["output_dir"]
BASES = np.array(list("ACGT"))
device = "cuda" if torch.cuda.is_available() else "cpu"
ML   = CFG["motif_processing"]["max_motif_length"]
GTH  = CFG["motif_processing"]["active_gate_threshold"]


# ── helpers ───────────────────────────────────────────────────────────────────
def column_ic(pwm):                                   # (4,L) -> (L,) bits
    p = np.clip(pwm, 1e-9, 1.0); p = p / p.sum(0, keepdims=True)
    return 2.0 + (p * np.log2(p)).sum(0)

def active_cols(gate):                                # gate (ML,) -> bool mask
    m = gate > GTH
    if m.sum() < 2:
        m = np.zeros_like(gate, bool); m[:4] = True
    return m

def write_meme(path, name, pwm):                      # pwm (4,L) rows ACGT
    L = pwm.shape[1]
    with open(path, "w") as f:
        f.write("MEME version 4\n\nALPHABET= ACGT\n\nstrands: + -\n\n")
        f.write("Background letter frequencies\nA 0.25 C 0.25 G 0.25 T 0.25\n\n")
        f.write(f"MOTIF {name}\n")
        f.write(f"letter-probability matrix: alength= 4 w= {L} nsites= 20 E= 0\n")
        for j in range(L):
            col = pwm[:, j]; col = col / col.sum()
            f.write(" " + " ".join(f"{x:.6f}" for x in col) + "\n")
        f.write("\n")

def write_pwm_tsv(path, pwm):                          # rows = position, cols ACGT
    L = pwm.shape[1]
    df = pd.DataFrame(pwm.T, columns=list("ACGT")); df.index.name = "pos"
    df.to_csv(path, sep="\t", float_format="%.6f")

def aligned_r(pred, target):
    """oracle-aligned (offset +/-10, revcomp-aware) mean per-column Pearson r."""
    al, _, _, _ = align_pwm(pred, target, max_shift=10, consider_revcomp=True)
    Lc = min(al.shape[1], target.shape[1])
    rs = [pearsonr(target[:, j], al[:, j])[0] for j in range(Lc)]
    return float(np.nanmean(rs)), al[:, :Lc]


# ── load model ─────────────────────────────────────────────────────────────────
def load_model(ckpt):
    cfg = TFScopeConfig()
    cp = os.path.join(os.path.dirname(ckpt), "config.json")
    if os.path.isfile(cp):
        for k, v in json.load(open(cp)).items():
            if hasattr(cfg, k):
                try: setattr(cfg, k, type(getattr(cfg, k))(v))
                except Exception: pass
    cfg.use_retrieval = True
    m = TFScopeModel(cfg).to(device)
    m.load_state_dict(torch.load(ckpt, map_location=device, weights_only=False)["model"], strict=False)
    m.eval()
    return m, cfg


# ── ESM2 query embedding for a DBD window (L33 mean-pool, matches donor pool) ───
def esm_dbd_embedding(seq):
    import esm
    em, alpha = esm.pretrained.esm2_t33_650M_UR50D()
    em = em.to(device).eval(); bc = alpha.get_batch_converter()
    _, _, toks = bc([("q", seq)])
    with torch.no_grad():
        rep = em(toks.to(device), repr_layers=[em.num_layers])["representations"][em.num_layers]
        v = rep[0, 1:1 + len(seq)].mean(0).cpu().numpy().astype(np.float32)
    del em; torch.cuda.empty_cache()
    return v


def main():
    os.makedirs(OUT, exist_ok=True)
    s1 = CFG["sohlh1"]; s2 = CFG["sohlh2"]
    seq1 = s1["dbd_sequence"]

    # ── 1. metadata + sequence/DBD tables ──────────────────────────────────────
    meta = pd.DataFrame([
        dict(gene_symbol="SOHLH1", uniprot_id=s1["uniprot_id"], family="bHLH",
             dbd_type="bHLH", full_length=s1["full_length"],
             uniprot_bhlh_domain="53-104", training_dbd_window=f'{s1["dbd_window_start"]}-{s1["dbd_window_end"]}',
             motif_status="no curated motif in TFScope tables",
             pdb_protein_dna_complex="none found", role="germ-cell / fertility bHLH (sperm/oogenesis)",
             case_role="primary orphan candidate"),
        dict(gene_symbol="SOHLH2", uniprot_id=s2["uniprot_id"], family="bHLH",
             dbd_type="bHLH", full_length=s2["full_length"],
             uniprot_bhlh_domain="201-252", training_dbd_window=f'{s2["dbd_window_start"]}-{s2["dbd_window_end"]}',
             motif_status="curated motif present (HOCOMOCO/CIS-BP/JASPAR)",
             pdb_protein_dna_complex="none found", role="germ-cell bHLH paralog of SOHLH1",
             case_role="paralog reference"),
    ])
    meta.to_csv(f"{OUT}/metadata/candidate_tf_metadata.tsv", sep="\t", index=False)

    seqtab = pd.DataFrame([
        dict(gene_symbol="SOHLH1", uniprot_id=s1["uniprot_id"], family="bHLH", dbd_type="bHLH",
             dbd_start=s1["dbd_window_start"], dbd_end=s1["dbd_window_end"],
             dbd_len=len(seq1), dbd_sequence=seq1,
             source_sequence="UniProt Q5JUK2", source_dbd="UniProt bHLH domain, SOHLH2-matched window",
             dbd_annotation_confidence="high"),
        dict(gene_symbol="SOHLH2", uniprot_id=s2["uniprot_id"], family="bHLH", dbd_type="bHLH",
             dbd_start=s2["dbd_window_start"], dbd_end=s2["dbd_window_end"],
             dbd_len=len(s2["dbd_sequence"]), dbd_sequence=s2["dbd_sequence"],
             source_sequence="UniProt Q9NX45", source_dbd="TFScope training DBD window",
             dbd_annotation_confidence="high"),
    ])
    seqtab.to_csv(f"{OUT}/metadata/sohlh1_sequence_and_dbd.tsv", sep="\t", index=False)

    # ── 2. leakage audit ───────────────────────────────────────────────────────
    df = pd.read_parquet(CFG["model"]["donor_parquet"])
    df["g"] = df["gene_symbol"].astype(str).str.upper()
    split = json.load(open(CFG["model"]["donor_split"]))
    train_val_fns = set(split["train"]) | set(split.get("val", []))
    fn2gene = dict(zip(df["filename"], df["g"]))
    fn2pwm  = {r["filename"]: np.frombuffer(r["pwm"], np.float32).reshape(4, -1) for _, r in df.iterrows()}

    audit = []
    for g in ["SOHLH1", "SOHLH2"]:
        rows = df[df["g"] == g]
        in_train = any(fn in train_val_fns for fn in rows["filename"])
        audit.append(dict(gene_symbol=g, n_rows_in_dataset=int(len(rows)),
                          in_model_training=bool(in_train),
                          status=("ORPHAN — never seen (safe prediction target)" if len(rows) == 0
                                  else "paralog present in training (RAG/encoder support, reported transparently)")))
    pd.DataFrame(audit).to_csv(f"{OUT}/metadata/sohlh1_leakage_audit.tsv", sep="\t", index=False)
    print("Leakage audit:")
    for a in audit: print("  ", a)

    # ── 3. retrieval neighbours (LGO: exclude same gene) ───────────────────────
    embs = np.load(CFG["model"]["embeddings"])
    donors = [fn for fn in embs.files if fn in train_val_fns and fn2gene.get(fn, "") != "SOHLH1"]
    qvec = esm_dbd_embedding(seq1)
    M = np.stack([embs[fn] / (np.linalg.norm(embs[fn]) + 1e-8) for fn in donors])
    q = qvec / (np.linalg.norm(qvec) + 1e-8)
    sims = M @ q
    order = np.argsort(-sims)
    K = CFG["retrieval"]["top_k"]
    # gene-level dedup: take the best-scoring file per distinct gene (a diverse,
    # robust prior; avoids retrieving one gene's multiple motif sources K times).
    top, seen_g = [], set()
    for di in order:
        g = fn2gene.get(donors[di], "?")
        if g in seen_g:
            continue
        seen_g.add(g); top.append(di)
        if len(top) == K:
            break
    top = np.array(top)
    nbrs, seen_n = [], set()
    for di in order:
        g = fn2gene.get(donors[di], "?")
        if g in seen_n:
            continue
        seen_n.add(g)
        fn = donors[di]
        nbrs.append(dict(rank=len(nbrs) + 1, neighbor_gene=g,
                         cos_sim=float(sims[di]), filename=fn,
                         family=df.loc[df["filename"] == fn, "family_name"].iloc[0]))
        if len(nbrs) == 12:
            break
    pd.DataFrame(nbrs).to_csv(f"{OUT}/metadata/sohlh1_retrieval_neighbors.tsv", sep="\t", index=False)
    print(f"\nTop retrieval neighbours (LGO, K shown=12):")
    for n in nbrs[:6]: print(f"   {n['rank']:>2} {n['neighbor_gene']:<10} cos={n['cos_sim']:.3f}  {n['family']}")

    # build retrieval tensors for RAG_LGO
    ret_pwms  = torch.full((1, K, 4, ML), 0.25)
    ret_masks = torch.zeros((1, K, ML)); ret_sims = torch.zeros((1, K))
    for ki, di in enumerate(top):
        pwm = fn2pwm[donors[di]]; L = min(pwm.shape[1], ML)
        ret_pwms[0, ki, :, :L] = torch.from_numpy(pwm[:, :L])
        ret_masks[0, ki, :L] = 1.0; ret_sims[0, ki] = float(sims[di])

    # ── 4. inference (noRAG + RAG_LGO) ─────────────────────────────────────────
    model, cfg = load_model(CFG["model"]["checkpoint"])
    tokens = torch.tensor([[AA_TO_TOKEN.get(a, 4) for a in seq1]], dtype=torch.long, device=device)
    dbd_mask = torch.ones(1, len(seq1), dtype=torch.bool, device=device)
    fam = torch.tensor([CFG["case_study"]["family_id"]], dtype=torch.long, device=device)

    def run(mode):
        kw = dict(retrieved_pwms=None, retrieved_masks=None, retrieved_sims=None)
        if mode == "RAG_LGO":
            kw = dict(retrieved_pwms=ret_pwms.to(device), retrieved_masks=ret_masks.to(device),
                      retrieved_sims=ret_sims.to(device))
        with torch.no_grad():
            gl, pl, aux = model(tokens, dbd_mask, fam, **kw)
            gate = gl.sigmoid()[0].cpu().numpy()
            pwm  = F.softmax(pl, 1)[0].cpu().numpy()          # (4,ML)
        attn = aux["attn"][0].cpu().numpy() if "attn" in aux and aux["attn"] is not None else None
        return gate, pwm, attn

    preds = {}
    for mode in ["noRAG", "RAG_LGO"]:
        gate, pwm, attn = run(mode)
        m = active_cols(gate)
        core = pwm[:, m]
        preds[mode] = dict(gate=gate, pwm=pwm, core=core, attn=attn, mask=m)
        write_pwm_tsv(f"{OUT}/predictions/SOHLH1_{mode}.pwm.tsv", core)
        write_meme(f"{OUT}/predictions/SOHLH1_{mode}.meme", f"SOHLH1_{mode}", core)
        cons = "".join(BASES[core.argmax(0)])
        ic = column_ic(core)
        print(f"\n[{mode}] active_len={core.shape[1]} consensus={cons} "
              f"meanIC={ic.mean():.2f} gate={gate[m].mean():.2f}")
    np.save(f"{OUT}/predictions/SOHLH1_attention.npy",
            preds["RAG_LGO"]["attn"] if preds["RAG_LGO"]["attn"] is not None else np.array([]))

    # ── 5. confidence (transparent rule-based) ─────────────────────────────────
    r_rag_norag, _ = aligned_r(preds["noRAG"]["core"], preds["RAG_LGO"]["core"])
    gate_conf = float(preds["RAG_LGO"]["gate"][preds["RAG_LGO"]["mask"]].mean())
    ic_norm   = float(np.clip(column_ic(preds["RAG_LGO"]["core"]).mean() / 2.0, 0, 1))
    retr_supp = float(np.clip(sims[top[0]], 0, 1))
    family_prior = 0.485     # bHLH leave-family-out macro oracle-r (documented prior)
    comps = dict(rag_noRAG_similarity=max(r_rag_norag, 0.0), gate_confidence=gate_conf,
                 motif_information_content=ic_norm, retrieval_support=retr_supp,
                 family_prior=family_prior)
    conf = (0.40 * comps["rag_noRAG_similarity"] + 0.20 * comps["gate_confidence"] +
            0.15 * comps["motif_information_content"] + 0.15 * comps["retrieval_support"] +
            0.10 * comps["family_prior"])
    cls = ("High" if conf >= CFG["confidence"]["high_confidence_threshold"]
           else "Medium" if conf >= CFG["confidence"]["medium_confidence_threshold"] else "Low")
    pd.DataFrame([{**comps, "confidence_score": conf, "confidence_class": cls}]
                 ).to_csv(f"{OUT}/predictions/SOHLH1_confidence.tsv", sep="\t", index=False)
    print(f"\nConfidence = {conf:.3f} ({cls});  components: " +
          ", ".join(f"{k}={v:.3f}" for k, v in comps.items()))

    # ── 6. reference-motif comparison ──────────────────────────────────────────
    sohlh2_pwm = fn2pwm[CFG["sohlh2"]["reference_motif_filename"]]
    write_meme(f"{OUT}/validation/SOHLH2_reference.meme", "SOHLH2_JASPAR_MA1560.1", sohlh2_pwm)

    def ebox_pwm(consensus):                       # IUPAC -> PWM (4,L)
        iupac = {"A":"A","C":"C","G":"G","T":"T","N":"ACGT"}
        L = len(consensus); m = np.zeros((4, L))
        for j, ch in enumerate(consensus):
            for b in iupac[ch]: m["ACGT".index(b), j] = 1.0
        return m / m.sum(0, keepdims=True)
    ebox = ebox_pwm(CFG["references"]["ebox_canonical"])

    cmp_rows = []
    for mode in ["noRAG", "RAG_LGO"]:
        core = preds[mode]["core"]
        r_s2, _ = aligned_r(core, sohlh2_pwm)
        r_eb, _ = aligned_r(core, ebox)
        cmp_rows.append(dict(prediction_mode=mode,
                             r_vs_SOHLH2_JASPAR=r_s2, r_vs_canonical_Ebox=r_eb,
                             pred_consensus="".join(BASES[core.argmax(0)])))
    cmpdf = pd.DataFrame(cmp_rows)
    cmpdf.to_csv(f"{OUT}/validation/sohlh1_vs_sohlh2_similarity.tsv", sep="\t", index=False)
    # ceiling: how well does SOHLH2's own E-box-likeness look (paralog vs E-box)
    r_par_eb, _ = aligned_r(sohlh2_pwm, ebox)
    print("\nReference comparison (oracle-aligned r):")
    print(cmpdf.to_string(index=False))
    print(f"   [context] SOHLH2 motif vs canonical E-box r = {r_par_eb:.3f}")

    # prediction summary
    summ = dict(gene_symbol="SOHLH1", checkpoint=os.path.basename(os.path.dirname(CFG["model"]["checkpoint"])),
                noRAG_consensus="".join(BASES[preds["noRAG"]["core"].argmax(0)]),
                RAG_LGO_consensus="".join(BASES[preds["RAG_LGO"]["core"].argmax(0)]),
                noRAG_len=int(preds["noRAG"]["core"].shape[1]),
                RAG_LGO_len=int(preds["RAG_LGO"]["core"].shape[1]),
                rag_noRAG_r=r_rag_norag, confidence=conf, confidence_class=cls,
                r_vs_SOHLH2=cmp_rows[1]["r_vs_SOHLH2_JASPAR"],
                top_neighbor=nbrs[0]["neighbor_gene"], top_neighbor_cos=nbrs[0]["cos_sim"])
    pd.DataFrame([summ]).to_csv(f"{OUT}/predictions/SOHLH1_prediction_summary.tsv", sep="\t", index=False)
    json.dump(summ, open(f"{OUT}/predictions/SOHLH1_prediction_summary.json", "w"), indent=2, default=float)
    print(f"\nAll outputs -> {OUT}/")
    print("Summary:", json.dumps(summ, indent=2, default=float))


if __name__ == "__main__":
    main()
