# TFScope Case-Study Plan (Nature Methods–style biological vignettes)

**Author:** design pass, 2026-06-08 (rev. architecture-first)
**Goal:** Pair TFScope's aggregate benchmarks with 3–5 sharp, interpretable biological case studies.

**Organizing principle (IMPORTANT): each case study demonstrates a specific TFScope
architectural component in action — we do NOT mirror DeepPBS's figure list.** TFScope is a
sequence-only model with a distinct architecture (ESM-2/LoRA encoder → dual-stream gated DBD
pooling → family-conditioned MoE decoder → PWM-column↔DBD-residue cross-attention head → gate +
PWM heads, optional RAG, optional Module-2 structure calibration). Every vignette is chosen to
expose what one of these components buys you, as a *biological* story. DeepPBS appears only as a
**contrast/positioning** anchor (sequence-only vs structure-required), never as the spine. Two of
the candidate vignettes (MoE expert specialization, RAG retrieval) have **no DeepPBS analogue at
all** — they are pure TFScope-architecture stories.

### Architecture → case-study map

| TFScope component | What it does | Case study that demonstrates it | DeepPBS analogue? |
|---|---|---|---|
| ESM-2/LoRA encoder + dual-stream DBD pooling | Reads specificity from sequence, no 3D | **CS1** — sequence matches/beats structure (CTCF) | contrast only (DeepPBS needs the complex) |
| PWM-column ↔ DBD-residue cross-attention head (v18 contact branch) | Localizes which residue encodes each base preference | **CS2** — model "reads the recognition residue" (KLF4 K409 / MyoD L122) | loose (DeepPBS Fig.4 RI, but 3D-based) |
| Family-conditioned semantic embedding + generalization behavior | Lets an unseen family still get a meaningful prior; sets the transfer floor | **CS3** — C2H2 long-array frontier (why family-consensus transfer breaks) | none (can't hold out a family from a structure method) |
| MoE decoder (12 experts, top-2 routing) | Specializes capacity by recognition mode | **CS-MoE** (candidate) — expert routing specializes by TF family/DBD class | **none** — pure TFScope story |
| RAG retrieval branch | Pulls nearest-neighbor donor PWMs as a prior | **CS-RAG** (candidate) — what retrieval adds, and its leakage risk | **none** — pure TFScope story |
| Module-2 structure calibration (Boltz + MM-GBSA) | Optional physics re-scoring | **CS5** (defer) — currently negative; honest limitation only | DeepPBS Fig.5 (design) — but ours doesn't work yet |

---

## PART A — DeepPBS case-study playbook (CONTRAST/POSITIONING ONLY, not a template)

> This section is reference material to position TFScope against the structure-based SOTA — NOT a
> figure list to copy. TFScope's case studies are driven by its own architecture (see the
> Architecture→case-study map above). We cite DeepPBS only where it sharpens the sequence-only vs
> structure-required contrast.

Source: Mitra et al., *Nature Methods* 21:1674–1683 (2024), `papers/deeppbs.pdf`.

DeepPBS is a geometric-deep-learning model: input = one protein–DNA **structure** (heavy-atom
protein graph + symmetrized DNA "sym-helix"), output = PWM. Its figures pair an aggregate
cross-family benchmark (Fig. 2) with four mechanistic vignettes:

| DeepPBS fig | TF / system | Biological point | Figure type |
|---|---|---|---|
| **Fig. 1f, 2a–d** | 130-chain benchmark, family-stratified | Works across protein families; "groove readout" > "shape readout"; family-agnostic | Box plots (MAE/RMSE) + scatter vs alignment score; family abundance bar |
| **Fig. 3a–d** | bHLH proteins **Q4H376 (Max), TCF21 (O43680), OJ1581_H09.2 (Q6H878)** — no PDB, RFNA-folded | DeepPBS works on *predicted* (AlphaFold/RFNA) structures, not just crystals; recovers CACGTG E-box; corrects a wrong assumed motif | Predicted-structure cartoon → predicted logo vs JASPAR logo |
| **Fig. 3e–g** | TGIF2LY (Q8IUE0) homeodomain | DeepPBS **feedback loop**: argmax of output → refold → better complex → better PWM (vdW→H-bond switch at Arg57) | Iteration rounds; zoom-in of one H-bond; logo per round |
| **Fig. 3h,i** | HD monomers vs rCLAMPS | Competitive with a family-*specific* method while being family-agnostic | Head-to-head MAE scatter, colored by pLDDT |
| **Fig. 4 (full)** | **p53 tetramer (PDB 3Q05)** | **Interpretability flagship**: relative-importance (RI) scores per heavy atom → aggregate to residue → validate against alanine-scanning ΔΔG (PCC 0.605). Names Lys120, Arg280, Cys277, Arg248 as specificity drivers | Structure overlay (spheres sized by RI) + zoom panels of single residues + residue-importance bar + logo + RI-vs-ΔΔG scatter |
| **Fig. 5 (full)** | **Designed HTH scaffolds DBP5/6/9/35** (Glasscock et al.) | DeepPBS predicts specificity of *de novo designed* proteins; matches yeast-display single-bp competition assays | Design cartoon + predicted logo + RI spheres + experimental mutation heatmap |

**DeepPBS's interpretability axis is atom→residue RI scores validated by mutagenesis ΔΔG.**
**Its scalability axis is "works on predicted structures, seconds per call."**
**Its design axis is de novo HTH scaffolds.**

### How TFScope differentiates
- **Sequence-only**: DeepPBS *requires* a docked protein–DNA complex (and shows, Fig 3e–g, that it
  must iteratively refold the complex to get the register right). TFScope skips the structure
  entirely. The clean win is a **head-to-head logo where TFScope (from sequence) matches the
  experiment better than DeepPBS (from structure)** — we have exactly this (CTCF/5kkq, see CS1).
- **Interpretability without a structure**: DeepPBS reads importance off heavy-atom geometry.
  TFScope's analogue is **cross-attention from each PWM column to DBD residues** — we can show the
  model learns to *read the recognition residue* (KLF4 K409 / MyoD L122) directly from sequence
  (CS2). This is a different, arguably more surprising interpretability claim: a language model with
  no 3D input localizes the specificity-switching residue.
- **Honest generalization frontier**: DeepPBS never quantifies an out-of-family transfer floor.
  TFScope's leave-family-out + cluster-40 experiments do, and they tell a *biological* story about
  why C2H2 long zinc-finger arrays are the hard frontier (CS3). This is a strength of the
  aggregate-to-vignette pairing, not a weakness to hide.

---

## PART B — Inventory of existing TFScope assets

| Asset | Path | What it is | Case-study value |
|---|---|---|---|
| CTCF head-to-head logo | `figures/5kkq_comparison_logo.pdf` | Target vs DeepPBS-ensemble vs TFScope-v8 logos for 5kkq-A & -D (CTCF). **TFScope r=0.926/0.915, MAE 0.075/0.076; DeepPBS r=0.698/0.726, MAE 0.218/0.220** (DeepPBS logo renders near-flat) | **CS1 flagship** — ready now |
| DeepPBS-only 5kkq logo | `figures/5kkq_deeppbs_logo.pdf` | DeepPBS logo alone for 5kkq | supporting |
| v17→v18a attention repair | `results/v18_attn/attn_v17_vs_v18a.png` | 2×2 heatmap: KLF4 & MyoD, v17 (rank-1 stripe/sink, mass@K409=0.000) vs v18a (spread, reads residue: mass@K409 0.000→0.094, MyoD@L122 retained) | **CS2 flagship** — ready now |
| KLF4 cross-attn panels | `results/klf4_attn/cross_attn_compare.png`, `cross_attn.png`, `cross_attn_testset.png`, `raw.npz` | KLF4 attention with/without RAG, raw arrays | CS2 supporting |
| KLF4 noRAG attn | `results/klf4_attn_noRAG/cross_attn.png`, `raw.npz` | Proves collapse is RAG-independent | CS2 supporting (rebuts "it's a retrieval artifact") |
| MyoD WT-vs-mut logo | `results/myod1_mut/wt_vs_mut_logo.{png,pdf}` | WT vs L12R MyoD logos **identical, r=0.990** | CS5 honesty panel (output is mutation-blind) |
| MyoD LGO logo | `results/myod1_lgo/logo.{png,pdf}` | MyoD predicted logo (ACCATCT core) leave-gene-out | CS4/CS5 supporting |
| LFO per-TF oracle-r | `results/lofo/per_tf_oracle_r.json` | 4241 TFs, 10 families, each `{fn, oracle_r}` — gives best/worst per family | **CS3 + CS4 data** — ready now |
| cluster40 metrics | memory `cluster40-honest-benchmark.md`; logos `figures/pred_vs_gt_cluster40.pdf` | Per-family OOD oracle-r; C2H2_long=0.43 bottleneck (28% of test) | **CS3 data** — ready now |
| DeepPBS per-sample metrics | `results/deeppbs_eval/per_sample.json`, `metrics.json`, `gene_preds.npz` | DeepPBS r/MAE per TF (incl. CTCF) for fair head-to-head | CS1/CS4 |
| pred-vs-gt grids | `figures/pred_vs_gt_deeppbs.pdf`, `figures/pred_vs_gt_cluster40.pdf` | Multi-TF logo grids | source for picking CS4 winners/losers |
| MyoD de-novo ΔΔG scan | `results/myod1_denovo/results.json` (Boltz+MM-GBSA, 16 cores × WT/L122R) | CACCTG-family ΔΔG per core; **but WT vs L122R ΔΔG barely differ** | CS5 — NEGATIVE/honesty only |
| MyoD/KLF4 mutant Boltz | `results/myod1_cacctg/`, `klf4_gcbox/` (WT vs K409Q / L12R) | structures + ppm.npy | CS5 supporting (structure side) |

**Scripts that generate per-TF visuals:**
`scripts/visualize_pred_vs_gt.py` (logo grids), `viz_attn_compare.py`, `viz_attn_testset.py`,
`attn_v18.py`, `viz_attn_v18.py` (attention heatmaps), `scripts/plot_logo_comparison.py`
(target/DeepPBS/ours triple logos — the 5kkq figure lineage), `scripts/predict_v14_lgo.py`,
`scripts/predict_klf4_wtmut.py`, `scripts/predict_v17_wtmut.py`.

---

## PART C — Candidate case studies (ranked)

### CS1 — Encoder: "the sequence encoder alone reads specificity" (CTCF) ★★★★★  BUILD FIRST
- **Component demonstrated:** ESM-2/LoRA backbone + dual-stream gated DBD pooling — the claim that
  the protein-language-model encoder, with no 3D input, captures the recognition information a
  structure method extracts from an atomic complex.
- **(a) Point:** TFScope, from protein sequence alone, predicts the CTCF motif *better* than
  structure-based DeepPBS predicts it from the crystal complex — the existence proof that you do
  not need a structure to read specificity.
- **(b) Parallels / differentiates:** Parallels DeepPBS Fig. 3a–d (predicted-structure logo vs
  JASPAR) but inverts the punchline — TFScope needs **no** structure and still wins. Direct
  head-to-head against DeepPBS's own output on the same PDB.
- **(c) Example:** **CTCF**, PDB 5kkq (chains A and D), motif CTCF_HUMAN.H11MO.0.A.
- **(d) Evidence we HAVE:** `figures/5kkq_comparison_logo.pdf` (Target | DeepPBS r=0.698/0.726 |
  TFScope-v8 r=0.926/0.915). DeepPBS numbers corroborated by `results/deeppbs_eval/per_sample.json`
  (5kkq_A r=0.743, 5kkq_D r=0.761). **Caveat / TO GENERATE:** the figure uses **TFScope v8**, an old
  checkpoint, and CTCF is L2-leakage (same H11MO accession seen in training under a different PDB).
  For the paper, regenerate with the headline model (**v18a, honest LGO index**) and label the
  leakage tier honestly; ideally add a CTCF-aware OOD caveat or pick a second, leakage-clean partner
  TF. Script: `scripts/plot_logo_comparison.py` (already produces this exact triple-logo layout).
- **(e) Figure type:** 3-row stacked sequence logos (experimental target / DeepPBS / TFScope) with
  r + MAE annotations; optionally a small inset structure of 5kkq to underline "DeepPBS needed this,
  TFScope did not."

### CS2 — Cross-attention head: "the model reads the recognition residue" (KLF4 K409 / MyoD L122) ★★★★★  BUILD FIRST
- **Component demonstrated:** the PWM-column ↔ DBD-residue cross-attention head (v18 contact
  branch) — TFScope's native interpretability surface, read directly off attention over the
  language-model embedding (no structure, unlike DeepPBS's atom-graph relative-importance).
- **(a) Point:** TFScope's PWM-column→DBD-residue cross-attention, after the v18 repair, places
  attention mass directly on the experimentally known specificity-determining residue — the model
  learns *where* in the protein the base preference is encoded, from sequence alone.
- **(b) Parallels / differentiates:** This is TFScope's answer to DeepPBS Fig. 4 (RI scores →
  residue importance). DeepPBS reads importance off **3D heavy-atom geometry**; TFScope reads it off
  **attention over a language-model embedding** — no structure. The surprise: a degenerate rank-1
  attention (every column attending the same 3 residues, a terminal sink, **zero** mass on K409 →
  model mutation-blind, WT-vs-mut r=1.000) is *repaired* so the model now "looks at" the causal
  residue (KLF4 K409 row-constancy 0.81→0.25, entropy 1.47→2.36, mass@K409 0.000→0.094).
- **(c) Example:** **KLF4 K409** (C2H2 ZF, GC-box recognition) and **MyoD L122** (bHLH, E-box).
- **(d) Evidence we HAVE:** `results/v18_attn/attn_v17_vs_v18a.png` (the 2×2 repair heatmap, all
  numbers above are on it). RAG-independence shown by `results/klf4_attn_noRAG/`. **TO GENERATE for
  publication quality:** vectorize as PDF, overlay the known recognition-residue position, and add a
  small alanine-scanning / known-contact annotation track to mirror DeepPBS's mutagenesis
  validation. Scripts: `attn_v18.py`, `viz_attn_v18.py` (already produce the heatmap).
- **(e) Figure type:** paired attention heatmaps (PWM position × DBD residue) for v17 vs v18a, with
  the causal residue marked; a small bar of "attention mass on known recognition residues."
- **HONESTY NOTE (must state in caption):** v18a repairs *attention* but the *output* is still
  mutation-blind (WT-vs-mut PWM r≈0.9998). Frame CS2 strictly as "the model learns to read the
  residue," NOT as "the model predicts mutation effects." Output mutation-sensitivity is a stated
  limitation / future v18c.

### CS3 — Family conditioning: "why long zinc-finger arrays are the frontier" (C2H2-long, ZNF) ★★★★☆  BUILD (analysis ready)
- **Component demonstrated:** the family-conditioned semantic-embedding prior + MoE — it transfers
  cleanly when a family shares one recognition grammar, but long C2H2 arrays violate the
  one-family-one-consensus assumption the conditioning relies on, exposing the architecture's
  honest transfer floor (only possible to probe *because* family embeddings are semantic, so a
  held-out family still gets a meaningful prior).
- **(a) Point:** TFScope generalizes well to families with a shared consensus grammar (bZIP,
  homeodomain, NR) but its honest OOD floor is set by **long C2H2 zinc-finger arrays**, where each
  finger has its own recognition code and there is no family-level consensus to transfer — naming
  the real biological frontier of sequence-only specificity prediction.
- **(b) Parallels / differentiates:** DeepPBS Fig. 2c/2d shows zf-C2H2 is its most abundant but
  hardest family too (shape-readout fails; zf-C2H2 "scans with minimal conformational change"). CS3
  *agrees with and extends* that observation into a quantified transfer story DeepPBS never tells.
- **(c) Example:** family-level — C2H2_long (181 TFs, 28% of cluster40 test, oracle-r 0.43, the
  bottleneck) vs the easy families; anchor TF examples from `results/lofo/per_tf_oracle_r.json`
  (e.g. C2H2_long best ZNF76 r=0.997 vs worst ZNF649 r=−0.302 — same family, opposite outcomes
  because finger composition differs).
- **(d) Evidence we HAVE:** cluster40 per-family table (memory `cluster40-honest-benchmark.md`); LFO
  per-family + per-TF (`results/lofo/per_tf_oracle_r.json`); logos `figures/pred_vs_gt_cluster40.pdf`.
  **TO GENERATE:** one summary panel (per-family oracle-r bar, C2H2_long highlighted) + 2 example
  logos (a ZNF the model nails vs one it misses) — both extractable from existing JSON + the logo
  script `scripts/visualize_pred_vs_gt.py`. No new training/GPU needed.
- **(e) Figure type:** per-family oracle-r bar chart (ordered, C2H2_long flagged) + 2 example logo
  triples + a cartoon of a multi-finger array vs a single DBD to explain the mechanism.

### CS-RAG — Retrieval branch: "what the retrieval prior adds" (NFATC2 + RAG ablation) ★★★★☆  BUILD (data exists), pure TFScope story
- **Component demonstrated:** the optional RAG branch — retrieving nearest-neighbor donor PWMs in
  sequence/embedding space and using them as a prior. **No DeepPBS analogue** (a structure method
  has nothing to retrieve). Shows TFScope can *borrow* from a related, experimentally-characterized
  TF when it exists, and degrades gracefully when it does not.
- **(a) Point:** for a query TF, TFScope's retrieved neighbor(s) carry most of the recoverable
  signal; turning RAG off vs on quantifies the prior's contribution, and a worked example (NFATC2)
  shows the retrieved-neighbor PWM next to the prediction and target.
- **(b) Contrast:** purely TFScope — frames retrieval as the model's "memory," and is also where
  the **leakage discipline** lives (the honest LGO index vs the leaky normal index, the central
  evaluation-hygiene theme of the paper).
- **(c) Example:** NFATC2 neighbor comparison; RAG-on vs RAG-off aggregate ablation.
- **(d) Evidence we HAVE:** `results/nfatc2_neighbor_pwm_comparison.pdf` (query vs retrieved
  neighbor vs target); RAG-off vs RAG-on models `results/tfscope_v14_noRAG_inference/`,
  `tfscope_v14_noRAG_trained/`, and `eval_v18a_ragoff.py`; the LGO vs normal index runs
  (`tfscope_v17_tf_nn_index*`). `build_nn_index.py`, `evaluate_nn_baseline.py` build/score the
  index. **TO GENERATE:** a clean RAG-on vs RAG-off delta bar (honest LGO index) + the NFATC2 panel
  re-rendered on the headline model. No GPU if checkpoints exist.
- **(e) Figure type:** query/neighbor/target logo triple (NFATC2) + a paired RAG-off→RAG-on Δr bar.
- **HONESTY NOTE:** retrieval is the main leakage vector — any RAG number MUST use the leave-gene-out
  (honest) index; the leaky "normal" index inflates results and is the wrong thing to showcase.

### CS-MoE — MoE decoder: "experts specialize by recognition mode" ★★★☆☆  CANDIDATE (needs new analysis, no GPU), pure TFScope story
- **Component demonstrated:** the 12-expert top-2 MoE decoder — the hypothesis that routing
  partitions capacity along biologically meaningful lines (DBD class / family / recognition mode),
  i.e. the MoE isn't just extra parameters but learns an interpretable division of labor. **No
  DeepPBS analogue.**
- **(a) Point:** if expert assignment correlates with TF family/DBD class, the MoE is an
  unsupervised discovery of recognition modes — a clean architecture-interpretability vignette.
- **(c) Example:** per-TF top-2 expert routing across the test set, colored by family.
- **(d) Evidence we HAVE:** the MoE itself (`src/tfscope/models/moe.py`, load-balance loss
  `losses/balance.py`) but **NO routing-analysis output exists yet** — this is honest: we have not
  extracted or visualized routing. **TO GENERATE (new analysis, inference-only, no training):** add
  a hook to log per-TF expert indices/weights during an eval pass, then (i) expert-vs-family
  contingency heatmap, (ii) UMAP of the joint representation colored by dominant expert. Only build
  if routing turns out to be family-structured — if it's uniform/degenerate, DROP it (do not force a
  null result into a figure).
- **(e) Figure type:** expert×family heatmap + representation UMAP colored by expert.
- **VERDICT:** highest-upside *new* architecture vignette, but **contingent** on the analysis
  showing real specialization. Run the routing extraction first as a quick diagnostic before
  committing it to the figure list.

### CS4 — "Transfer to a held-out family" (LFO winner vs loser) ★★★☆☆  OPTIONAL / merges into CS3
- **(a) Point:** Even with an *entire family held out of training*, TFScope predicts some TFs well
  (transfer floor ≈0.48) but the families with the strongest conserved grammar (bZIP/homeo/NR)
  crash hardest (−0.20) — revealing that part of their in-distribution score was family
  memorization, not sequence→specificity transfer.
- **(b) Parallels / differentiates:** No DeepPBS analogue; this is uniquely a sequence-model story
  (you can't "hold out a family" from a structure method the same way).
- **(c) Example:** LFO best vs worst per family from `results/lofo/per_tf_oracle_r.json`
  (e.g. Homeodomain Ubx r=1.0 vs SEBOX r=−0.19; bZIP CREB3 r=0.98 vs CEBPE r=0.10).
- **(d) Evidence we HAVE:** full per-TF LFO JSON + memory `lofo-v2-experiment.md` (macro 0.479,
  per-family deltas). **TO GENERATE:** scatter/slope plot of in-dist vs LFO per family. No GPU.
- **(e) Figure type:** paired-dot "in-distribution → leave-family-out" slope plot per family + 2
  example logos.
- **VERDICT:** Strong data, but the message **overlaps CS3** and is more "ML generalization" than
  "biological vignette." Recommend folding the best LFO example *into* CS3 rather than a standalone
  figure, unless a reviewer asks for a dedicated generalization figure.

### CS5 — MyoD CACCTG de-novo / mutation vignette ★★☆☆☆  DEFER (honesty risk)
- **(a) Point (as ASPIRED):** rewiring a single bHLH recognition residue (MyoD L122R) should shift
  the predicted E-box preference (CAGGTG vs CACCTG family).
- **(b) Parallels:** DeepPBS Fig. 5 (de novo design) + Fig. 3e–g (mutation/feedback).
- **(c) Example:** MyoD1 L122R, CACCTG core; Boltz+MM-GBSA ΔΔG scan over 16 dinucleotide cores.
- **(d) Evidence we HAVE — and its problem:** `results/myod1_denovo/results.json`,
  `results/myod1_cacctg/`, `results/myod1_dimer_ensemble/`, `results/myod1_mut/wt_vs_mut_logo.png`.
  **The data say the effect is NOT there:** (i) TFScope output is **mutation-blind** — WT vs L12R
  predicted logos are identical, r=0.990 (`myod1_mut/wt_vs_mut_logo.png`); (ii) the physics side
  (MM-GBSA ΔΔG in `myod1_denovo/results.json`) shows WT vs L122R per-core ΔΔG differences within
  noise (e.g. CACGTG WT −100.1 vs L122R −94.6; rankings barely move), and MM-GBSA was overall
  *worse* than the seed in the 28-TF pilot. So we cannot honestly claim TFScope predicts this
  mutation effect.
- **(e) Figure type (if used):** strictly as a **limitation/outlook panel**, not a success vignette.
- **VERDICT: DEFER.** Do not present as a positive result. Either (i) omit, or (ii) use the
  WT-vs-mutant identical-logo panel as the *honest limitation figure* that motivates future
  mutation-contrastive training (v18c). Building it as a "design success" would be misleading.

---

## PART D — Ranked recommendation (architecture-first: each maps to a component)

The three core vignettes cover three distinct TFScope components (encoder, cross-attention head,
family conditioning) — together they let the reader watch the architecture work end to end. The two
pure-TFScope candidates (RAG, MoE) are the differentiators with **no DeepPBS analogue**; build
CS-RAG if the ablation is clean, and run the CS-MoE diagnostic before committing it.

| Rank | Case study | Component | Why | Ship a figure THIS week? | What's missing |
|---|---|---|---|---|---|
| 1 | **CS1 CTCF: encoder reads specificity** | ESM-2/DBD-pooling encoder | Cleanest, most quotable; sequence-only vs structure existence proof; figure exists | **Yes** — figure exists | Regenerate with headline **v18a/LGO** model (not old v8); annotate CTCF L2-leakage honestly OR add a leakage-clean second TF; vectorize. Script: `plot_logo_comparison.py` |
| 2 | **CS2 reads the recognition residue** | PWM↔DBD cross-attention head | TFScope's native interpretability surface (no 3D); heatmap exists | **Yes** — heatmap exists | Vectorize to PDF; mark causal residue; add known-contact track; **caption must state output is still mutation-blind**. Scripts: `attn_v18.py`, `viz_attn_v18.py` |
| 3 | **CS3 C2H2-long frontier** | Family-conditioned prior + MoE | Turns the honest OOD limit into a biological story; only probeable because conditioning is semantic; data computed | **Yes** — JSON exists, plotting only | Per-family oracle-r bar + 2 ZNF example logos + finger-array cartoon. `visualize_pred_vs_gt.py` + small plot from `results/lofo/per_tf_oracle_r.json` |
| 4 (diff.) | **CS-RAG retrieval prior** | RAG branch | Pure-TFScope, no DeepPBS analogue; carries the leakage-discipline theme | **Yes** — NFATC2 fig + ablation exist | RAG-on/off Δr bar on **honest LGO index**; re-render NFATC2 on headline model. `eval_v18a_ragoff.py` |
| 5 (diff.) | **CS-MoE expert specialization** | MoE decoder | Highest-upside *new* architecture vignette; pure-TFScope | **Contingent** — needs new routing extraction | Log per-TF top-2 expert indices in an eval pass → expert×family heatmap + UMAP. **Build only if routing is family-structured.** |

**Hold / fold:** CS4 (fold its best LFO example into CS3, or hold for a reviewer-requested
generalization figure). **Defer / reframe:** CS5 / Module-2 (only as an honest limitation panel
using the identical WT/L12R logo; never as a design success).

**Suggested figure budget:** 3 core component vignettes (CS1–CS3) as main-text figures; CS-RAG as a
main-text or supplementary differentiator; CS-MoE promoted to main text only if the diagnostic shows
real specialization; CS5 as a limitations/outlook panel.

### Brutal-honesty flags
- CS1's existing figure is **v8** (stale) and CTCF is **L2 leakage** (same motif accession in
  training) — the win is real but must be re-run on the honest model and labeled, or it invites a
  "leakage" reviewer hit.
- CS2 repairs *attention*, not *output*. The model does **not** yet predict mutation effects. The
  caption must say so.
- CS5's underlying claim is **not supported by our data** — both the model (mutation-blind logos)
  and the physics (ΔΔG within noise) say the effect isn't captured. Do not imply we have it.
- CS3/CS4 oracle-r numbers are **oracle-offset+RC-aligned** (upper bounds, applied equally to all
  methods for fair ranking) — state this; the deployable register cost is the dominant remaining
  error for every method including DeepPBS.
