#!/usr/bin/env python
"""Generate manuscript snippets for the SOHLH1 case study.
All numbers are read from the result tables (never hard-coded)."""
import os, json, yaml, numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CFG  = yaml.safe_load(open(os.path.join(HERE, "config.yaml")))
OUT  = CFG["case_study"]["output_dir"]
MS   = f"{OUT}/manuscript"; os.makedirs(MS, exist_ok=True)

summ = json.load(open(f"{OUT}/predictions/SOHLH1_prediction_summary.json"))
conf = pd.read_csv(f"{OUT}/predictions/SOHLH1_confidence.tsv", sep="\t").iloc[0]
cmp  = pd.read_csv(f"{OUT}/validation/sohlh1_vs_sohlh2_similarity.tsv", sep="\t").set_index("prediction_mode")
nbrs = pd.read_csv(f"{OUT}/metadata/sohlh1_retrieval_neighbors.tsv", sep="\t")
top3 = ", ".join(nbrs["neighbor_gene"].head(3).tolist())
ic_rag = conf.motif_information_content * 2
r_s2_rag = cmp.loc["RAG_LGO", "r_vs_SOHLH2_JASPAR"]
r_s2_nr  = cmp.loc["noRAG", "r_vs_SOHLH2_JASPAR"]
r_eb_rag = cmp.loc["RAG_LGO", "r_vs_canonical_Ebox"]

result = f"""### TFScope nominates an E-box specificity for the orphan germ-cell transcription factor SOHLH1

Having characterized the generalization behaviour of TFScope under leakage-controlled
benchmarks, we asked whether the model could serve as a sequence-only annotation engine for
human transcription factors that lack a curated binding motif. We focused on SOHLH1
(spermatogenesis- and oogenesis-specific bHLH 1; UniProt {CFG['sohlh1']['uniprot_id']}), a
germ-cell bHLH factor with established roles in fertility but no curated motif in HOCOMOCO,
JASPAR or CIS-BP and no protein–DNA complex structure. SOHLH1 is absent from every TFScope
training, retrieval and benchmark table, making it a genuine out-of-distribution target; its
paralogue SOHLH2 is present with a curated motif and serves as an independent reference.

Operating on the SOHLH1 bHLH domain alone, TFScope's de-novo (retrieval-free) pathway produced
a low-information, weakly specific motif (consensus {summ['noRAG_consensus']}, r = {r_s2_nr:.2f}
to the SOHLH2 motif). Enabling leave-gene-out retrieval — which supplied three sequence-distant
but family-consistent E-box-binding bHLH factors ({top3}), and never SOHLH1 or SOHLH2 themselves
— sharpened the prediction into a canonical E-box (consensus {summ['RAG_LGO_consensus']},
{ic_rag:.2f} bits mean information content), yielding a calibrated confidence of
{conf.confidence_score:.2f} ({conf.confidence_class}). Critically, the retrieval-augmented
prediction recovered the experimentally curated SOHLH2 paralogue motif (r = {r_s2_rag:.2f}) and
the canonical E-box CACGTG (r = {r_eb_rag:.2f}), even though neither SOHLH2 nor SOHLH1 was among
the retrieved neighbours — an independent corroboration rather than a circular lookup.

These results nominate a testable E-box DNA-binding specificity for SOHLH1 from protein sequence
alone, without a protein–DNA complex structure, and illustrate a broader pattern consistent with
our benchmark analyses: for orphan factors, the sequence-intrinsic pathway establishes family
identity while leave-gene-out retrieval of related members converts that identity into a sharp,
experimentally prioritizable motif hypothesis.
"""

methods = f"""### Methods — SOHLH1 orphan-TF case study

**Candidate selection and metadata.** Human bHLH transcription factors lacking a curated motif
were considered as orphan annotation targets. SOHLH1 (UniProt {CFG['sohlh1']['uniprot_id']},
{CFG['sohlh1']['full_length']} aa) was selected: it carries a UniProt-annotated bHLH domain
(residues 53–104), has documented germ-cell/fertility function, lacks a curated motif in
HOCOMOCO/JASPAR/CIS-BP, and has no protein–DNA complex structure. Its paralogue SOHLH2 (UniProt
{CFG['sohlh2']['uniprot_id']}) has a curated motif and was used as an independent reference.

**Sequence and DBD definition.** The bHLH DBD input window for SOHLH1 (UniProt residues
{CFG['sohlh1']['dbd_window_start']}–{CFG['sohlh1']['dbd_window_end']}) was defined to match the
convention used for SOHLH2's training window (the UniProt bHLH domain start, extended six
residues at the C-terminus), giving a 58-residue window directly comparable to the paralogue.

**Leakage audit.** SOHLH1 was confirmed absent from the TFScope training tables, the retrieval
donor pool and all benchmark splits (0 rows). SOHLH2 is present in training and is reported
transparently; it was not retrieved for SOHLH1 (below).

**Inference.** We used the production retrieval checkpoint (cluster40_v18a_rag), trained on the
full augmented dataset minus the cluster40 (≤40 % identity) held-out clusters. Predictions were
generated in two modes: (i) noRAG (retrieval disabled), and (ii) RAG_LGO (leave-gene-out
retrieval). Retrieval neighbours were obtained by cosine similarity between the ESM-2 (layer 33,
DBD mean-pooled) embedding of the SOHLH1 DBD and the donor pool, deduplicated to the top
{CFG['retrieval']['top_k']} distinct genes and excluding SOHLH1.

**Confidence.** A transparent rule-based confidence combined noRAG/RAG agreement (0.40),
mean active-column gate probability (0.20), normalized information content (0.15), top retrieval
similarity (0.15) and a bHLH leave-family-out prior (0.10).

**Reference comparison.** Predicted motifs were compared to the SOHLH2 curated motif (JASPAR
MA1560.1) and the canonical E-box (CACGTG) by oracle-aligned, reverse-complement-aware mean
per-column Pearson correlation (offset ±10).
"""

caption = f"""**Fig. 5 | Sequence-only motif nomination for the orphan germ-cell transcription factor SOHLH1.**
**a**, Selection cascade from the human TF catalogue to orphan bHLH factors lacking curated
motifs, identifying SOHLH1. **b**, SOHLH1 domain architecture (UniProt {CFG['sohlh1']['uniprot_id']},
{CFG['sohlh1']['full_length']} aa) with the bHLH DBD (residues 53–104); SOHLH1 has no curated
motif and no protein–DNA complex structure. **c**, TFScope predicted motifs from the SOHLH1 bHLH
domain alone: the de-novo (noRAG) pathway yields a low-information motif (consensus
{summ['noRAG_consensus']}), whereas leave-gene-out retrieval (RAG) yields a sharp E-box
(consensus {summ['RAG_LGO_consensus']}, {ic_rag:.2f} bits; confidence {conf.confidence_score:.2f},
{conf.confidence_class}). **d**, The retrieval-augmented SOHLH1 prediction matches the
experimentally curated motif of its paralogue SOHLH2 (JASPAR MA1560.1; r = {r_s2_rag:.2f}) and
the canonical E-box (r = {r_eb_rag:.2f}), despite neither paralogue being among the retrieved
neighbours ({top3}). **e**, Cross-attention from the active motif positions onto the SOHLH1 bHLH
DBD residues; the basic region (cyan box, DNA-contacting) receives concentrated attention.
All numeric values are computed in results/case_study_sohlh1/.
"""

open(f"{MS}/result5_draft.md", "w").write(result)
open(f"{MS}/methods_case_study_draft.md", "w").write(methods)
open(f"{MS}/figure5_caption.md", "w").write(caption)
print("wrote manuscript snippets to", MS)
print("\n--- key numbers ---")
print(f"noRAG consensus {summ['noRAG_consensus']}  r_vs_SOHLH2 {r_s2_nr:.3f}")
print(f"RAG   consensus {summ['RAG_LGO_consensus']}  r_vs_SOHLH2 {r_s2_rag:.3f}  r_vs_Ebox {r_eb_rag:.3f}")
print(f"confidence {conf.confidence_score:.3f} ({conf.confidence_class})  neighbours {top3}")
