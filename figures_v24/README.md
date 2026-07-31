# Figures 1–4 reproduced with the ACTUAL v24 checkpoint

All model-dependent manuscript figures were originally built with the **combined**
checkpoint (`v19_combined_fm_deeppbs_contact/rag_seed42`), NOT the v24 model we keep.
This folder regenerates them with **v24** (`v24_contact/contact_v24_seed42/ckpt_best.pt`)
via `scripts/build_*_v24.py` (auto-generated: checkpoint repointed, output → figures_v24/).

## Regenerated with v24
| figure | script | status | v24 vs combined |
|---|---|---|---|
| Fig4a MyoD1 switch | build_fig4a_switch_v24.py | done | **Δ_switch +1.70 (v24) vs +9.18 (combined)** — much weaker |
| Fig1c best logos | build_fig1c_best_logos_v24.py | done | regenerated |
| DBP design heatmap | build_dbp_tfscope_heatmap_v24.py | done | weaker CAC recovery (DBP006 no longer clean CAC) |
| Fig3a heldout recovery | build_fig3a_heldout_clean_v24.py | done | v24 <40%id median r=0.785, ≥40% 0.653 |
| Fig1e per-family | build_fig1e_perfamily_benchmark84_v24.py | done | regenerated |
| Fig3d DNA evolution | build_fig3d_dna_evolution_v24.py | running (bg) | — |
| Fig3bc CRE enrichment | build_fig3bc_cre_enrichment_v24.py | running (bg) | — |

## NOT reproducible with v24
- **Fig2e prototypes** — architecture-incompatible. v24 uses **residue-MoE**
  (`moe_granularity=residue`), so it has no protein-level prototype dictionary
  (`m.moe.proto`); the prototype panel is specific to the combined model's protein-MoE.

## Model-agnostic (unchanged — do NOT depend on the checkpoint)
Fig1d ladder, Fig1f AF3 foldability, Fig2a ablation, Fig2b mutagenesis,
Fig2c recognition code, Fig2d structureless foldability. These read precomputed
data / structures, not the PWM model, so "v24 version" == original.

## Headline
The manuscript figures use the stronger **combined** model; the audited/kept **v24**
gives systematically weaker results (MyoD1 switch +1.70 not +9.17; DBP CAC weaker).
Any v24-consistent paper must either (a) re-cut all figures on v24 (weaker story) or
(b) state clearly that the showcased predictions are from the combined contact model.
