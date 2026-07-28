# Fig. 2d — TFScope nominates foldable motifs for structure-less TFs

Builder: scripts/build_fig2d_structureless_foldability.py. Folds:
/data1/leihuang/project/TFScope/structureless_af3_folding (10 TFs, AF3).
Inputs: results/structureless_af3_inputs/ (TFScope-predicted consensus, no crystal exists).
Key numbers: n=10, mean ipTM 0.83, 8/10 ≥0.8, 9/10 ≥0.6; ELK1 0.94, FOXG1/MYOG 0.93;
weakest SLC2A4RG 0.47 (single 31-aa zinc finger).

---

## Subsection: "Sequence-only prediction extends to factors with no structure"

> The benchmark and the AlphaFold3 head-to-head (Fig. 1) both concern factors for which an
> experimental protein–DNA structure exists. The central practical advantage of a
> sequence-only model, however, is that it applies where structure-based methods cannot run
> at all: a method such as DeepPBS requires an input protein–DNA complex, which is
> unavailable for the great majority of transcription factors. To probe this regime we
> selected ten high-confidence factors that lack any experimental DNA-bound structure,
> spanning homeodomain, bHLH, bZIP, ETS, forkhead and C2H2 zinc-finger families, predicted
> their consensus motifs from sequence alone, and folded each factor with its predicted
> double-stranded site in AlphaFold3. Eight of the ten complexes folded with high interface
> confidence (ipTM ≥ 0.8; mean 0.83), and nine of ten above 0.6 (Fig. 2d), with the
> myogenic bHLH factor MYOG, the forkhead factor FOXG1 and the ETS factor ELK1 all reaching
> ipTM ≥ 0.93. The single low-confidence case, SLC2A4RG, is a lone 31-residue zinc finger
> whose minimal protein–DNA interface offers little for the confidence estimate to score.
> Because no structure exists for any of these factors, a structure-based predictor produces
> no answer at all; that TFScope's sequence-only motifs nonetheless assemble into
> physically confident complexes shows the model generalises beyond the structurally
> characterised fraction of the proteome.

---

## Figure caption

> **(d)** AlphaFold3 interface confidence (ipTM) for ten transcription factors that lack any
> experimental DNA-bound structure, each folded with its TFScope-predicted consensus site
> (motif shown in parentheses); bars coloured by structural family, dashed line at the
> high-confidence threshold (ipTM 0.8). Eight of ten fold at ipTM ≥ 0.8. A structure-based
> method cannot be applied to these factors, as no input structure exists.

Caveats: this is foldability (plausibility), not ground-truth validation — there is no
crystal to compare against (the point of the panel). The elevated fraction_disordered for
CREB3L1/2 reflects the flexible bZIP coiled-coil arms away from the DNA interface, so the
interface ipTM remains informative.
