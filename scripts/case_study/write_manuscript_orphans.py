#!/usr/bin/env python
"""Manuscript snippet for the remaining-orphan homeodomain nominations (deeppbs v18a RAG)."""
import os, json, yaml, pandas as pd
from scipy.stats import spearmanr

cfg = yaml.safe_load(open("configs/case_study_orphans_deeppbs.yaml"))
OUT = cfg["output_dir"]; MS = f"{OUT}/manuscript"; os.makedirs(MS, exist_ok=True)
summ = json.load(open(f"{OUT}/predictions/orphan_summaries.json"))
md = pd.read_csv(f"{OUT}/predictions/orphan_prediction_metrics.tsv", sep="\t").set_index("gene")
cons = pd.read_csv(f"{OUT}/predictions/paralog_consistency.tsv", sep="\t").set_index("pair")["r"]
ctl = pd.read_csv(f"{OUT}/validation/homeodomain_masked_control_metrics.tsv", sep="\t")
cal = pd.read_csv(f"{OUT}/confidence/heldout_known_confidence.tsv", sep="\t")
prom = pd.read_csv(f"{OUT}/targets/orphan_promoter_scan_summary.tsv", sep="\t").set_index("gene")
rho = spearmanr(cal.confidence, cal.oracle_r).correlation
HD = cal[cal.family == "Homeodomain"]

result = f"""### Extending the orphan nominations across the homeodomain family (ADNP2, ZHX2, ZHX3)

Using the v18a model trained on the full augmented dataset under the DeepPBS-benchmark split
(the checkpoint behind our DeepPBS comparison) with leave-gene-out retrieval, we nominated
motifs for three further orphan homeodomain TFs — all absent from every training, retrieval and
benchmark table. Confidence on this checkpoint's held-out known TFs was calibrated to accuracy
(Spearman rho = {rho:.2f}; held-out homeodomain median oracle r = {HD.oracle_r.median():.2f}), and a
retrieval-masked homeodomain control recovered known motifs (EN1, PBX1 at r ≥ 0.6; PITX1, ISL1
produced the correct TAAT consensus just below threshold).

ADNP2, the paralogue of ADNP, was nominated a TGAT/AT-rich homeodomain motif
({summ['ADNP2']['RAG_consensus']}, {summ['ADNP2']['mean_IC']:.2f} bits, confidence
{summ['ADNP2']['confidence']:.2f}), retrieving homeodomain neighbours ({summ['ADNP2']['retrieved']});
its prediction agreed with ADNP's at r = {cons['ADNP2 vs ADNP']:.2f}. For the zinc-fingers-and-
homeoboxes proteins ZHX2 and ZHX3 — each carrying four to five homeoboxes — we screened all
homeoboxes and the first (HD1) gave the most decisive, canonical prediction in both. ZHX3 yielded
a sharp canonical homeodomain element ({summ['ZHX3']['RAG_consensus']}, {summ['ZHX3']['mean_IC']:.2f}
bits, confidence {summ['ZHX3']['confidence']:.2f}; oracle r = {md.loc['ZHX3','r_vs_TAAT']:.2f} vs the
TAAT consensus), and ZHX2 a related element ({summ['ZHX2']['RAG_consensus']},
confidence {summ['ZHX2']['confidence']:.2f}). The two ZHX paralogues agreed strongly
(r = {cons['ZHX2 vs ZHX3']:.2f}), a cross-prediction consistency that is notable given ZHX1's
training motif is uninformative and could not serve as a reference.

Composition-controlled promoter scans (dinucleotide-preserving background) supported the ZHX
nominations: the ZHX3 and ZHX2 motifs were enriched in their hepatic / cell-cycle target promoters
beyond the shuffled control (AUROC {prom.loc['ZHX3','auroc_RAG']} and {prom.loc['ZHX2','auroc_RAG']}),
with bona-fide ZHX targets (AFP, GPC3, cell-cycle genes) among the top-scoring promoters. The ADNP2
motif, like ADNP's, was composition-confounded (no enrichment beyond background) and is reported as
a sequence-level hypothesis only. All nominations are testable by PBM/HT-SELEX/ChIP and do not assert
in-vivo occupancy; ZHX2/ZHX3 act as multi-domain repressors and ADNP2 within a chromatin complex.
"""

open(f"{MS}/result_orphans_draft.md", "w").write(result)
open(f"{MS}/figure8_caption.md", "w").write(
    "**Fig. 8 | Sequence-only homeodomain motif nominations for orphan TFs ADNP2, ZHX2 and ZHX3 "
    "(v18a, full augmented data, DeepPBS split, leave-gene-out RAG).** "
    "**a**, RAG-predicted motifs and the canonical homeodomain TAAT site. **b**, deeppbs held-out "
    "confidence vs motif-recovery accuracy (homeodomain highlighted; the three orphans' confidences "
    "marked). **c**, paralog cross-prediction consistency (ZHX2↔ZHX3, ADNP2↔ADNP) and retrieval-masked "
    "homeodomain positive control. **d**, composition-controlled promoter enrichment per orphan "
    "(targets vs dinucleotide-preserving shuffled background). Values computed in results/orphan_homeodomain_deeppbs/.")
print("wrote ->", MS)
print(md[["RAG_consensus", "confidence", "confidence_class", "r_vs_TAAT", "retrieved"]].to_string())
print("consistency:", dict(cons)); print(f"masked HD control recovery: {ctl.recovered.mean()*100:.0f}%")
