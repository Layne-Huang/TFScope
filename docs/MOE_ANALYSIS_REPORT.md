# Mixture-of-Experts in TFScope — Analysis Report

*Last updated: 2026-07-09. All numbers are reproduced from saved artifacts; scripts are named inline.*

---

## Executive summary

TFScope's original Mixture-of-Experts (MoE) block was **inert**: it routed once per protein and
collapsed to uniform expert usage, contributing capacity but no specialization. Attempts to force
specialization (CE-supervised routing over 4/5/12 experts) succeeded mechanically but **cost ~0.05
accuracy**.

We redesigned the block as a **per-residue, DeepSeekMoE-style MoE** (`ResidueMoE`). It routes every
DBD residue rather than the pooled protein vector. The result is the first TFScope MoE that

1. **does not collapse** — expert usage is uneven, routing is decisive (mean top-1 gate 0.35 vs 0.125 uniform);
2. **organizes along real DNA-recognition chemistry** — an arginine major-groove expert, a Cys/His
   zinc-coordination expert, a glutamine (homeodomain Q50) expert;
3. survives a **family-label ablation**, proving the routing has a genuine residue-chemistry backbone
   rather than merely re-encoding the family label.

**Important correction (2026-07-09).** We previously described this as "specialization at *zero accuracy
cost*," based on validation-set parity. That validation set was **contaminated** (~10% of its records
also appear in train) and was additionally used for checkpoint selection. On a **clean, zero-overlap
test set**, `moe_base` scores **0.633** gate-oracle-r versus **0.657** for the `combined` model — a real
**0.024 deficit**. The correct claim is *interpretable specialization at a modest accuracy cost*, and it
is not yet established whether that 0.024 gap is statistically significant (n = 84; no bootstrap CI yet).

---

## 1. Background: why the original MoE did nothing

The legacy `MOEBlock` (`src/tfscope/models/moe.py`) sits on the **pooled** per-protein vector and adds
its output as one residual term: `out = x + shared + routed + proto`. Three consequences:

- **One routing decision per protein** (~881 decisions across the whole dataset). Real MoE systems
  (Mixtral, DeepSeek-V2/V3, AIDO.Protein) route **per token**, making millions of decisions. There is
  simply not enough routing signal here for specialization to emerge.
- **No task pressure.** The shared expert always fires and the cross-attention PWM head reads ESM
  embeddings directly, so `moe_out` is not a bottleneck — gradients flow around the router.
- **Losses that actively fight specialization.** `family_diversity_loss` *maximizes* routing entropy,
  and Switch-style `load_balance_loss` pushes toward uniform usage — while the family distribution is
  **41× imbalanced** (Other 1,963 records vs Forkhead 47).

Diagnostics confirmed the collapse: per-TF top-expert weight ≈ **0.084 ≈ 1/12**, i.e. exactly uniform.

### Prior attempts and why they failed

| variant | experts | routing outcome | benchmark panel-r |
|---|---|---|---|
| pooled `combined` | 12 | **collapsed** (uniform) | 0.683 |
| `coarse12_4moe` (diversity off) | 4 | still collapsed (exactly 0.250 each) | 0.558 |
| `mode5_nonresidual` (CE route supervision) | 5 | **perfectly specialized** (argmax = mode, 100%) | 0.567 |
| `coarse12_specialized` | 12 | perfectly specialized | 0.618 → 0.537 (overfits) |
| no-MoE ablation | 1 | n/a | 0.672 (val panel) |

The lesson: whether experts **collapsed** or were **forced to specialize**, accuracy stayed in the
0.56–0.62 band, far below the collapsed `combined` at 0.683. Specialization could be *imposed*, but it
was never *useful*. And the direct no-MoE ablation showed the collapsed MoE's entire edge was
**capacity/ensemble (+0.02)**, not routing — it has 49M trainable params against no-MoE's 11M.

---

## 2. The redesign: `ResidueMoE`

`src/tfscope/models/moe.py::ResidueMoE`, enabled by `--moe-granularity residue`.

| property | value |
|---|---|
| granularity | **per DBD residue** (~50–70 decisions/protein, not 1) |
| routed experts | 8 fine-grained SwiGLU (`expert_hidden = 512`) |
| shared experts | 2, always active (DeepSeek "shared-expert isolation") |
| top-k | 2 |
| load balance | **token-level**, weight 0.01 |
| diversity loss | **off** (it was anti-specialization) |
| route supervision | **none** — specialization must be emergent |
| bottleneck | refined residue reps feed **both** pooling **and** the cross-attention PWM-head keys |
| trainable params | ~20 M (vs 41 M for the pooled 12-expert block) |

Design rationale, per the modern MoE literature: fine-grained expert segmentation plus shared-expert
isolation (DeepSeekMoE), routing at token granularity (Mixtral/AIDO.Protein), and removing the losses
that fight an imbalanced label distribution.

---

## 3. Result 1 — routing does not collapse

`scripts/diagnose_residue_moe_routing.py` — 555 proteins, **84,693 DBD tokens**, `moe_base` `ckpt_best`.

| statistic | value | uniform/collapse baseline |
|---|---|---|
| marginal top-1 usage | `[.117, .057, .099, .119, .120, .152, .188, .149]` | all 0.125 |
| usage std | **0.036** | 0 |
| max/min usage ratio | **3.3×** | 1.0× |
| mean per-token entropy | 1.751 / 2.079 max (0.84) | 1.0 of max |
| **mean top-1 gate weight** | **0.350** | 0.125 |
| NMI(expert; family) | **0.186** | 0 |

**Verdict: SPECIALIZED.** Routing is uneven and decisive, in sharp contrast to the pooled block's exact
1/12 uniformity.

### Per-family routing, `P(top-1 expert | family)`

| family | dominant expert(s) |
|---|---|
| Forkhead | **e6 (0.78)** |
| C2H2_medium | **e4 (0.66)** |
| C2H2_short | **e7 (0.59)** |
| bHLH | e6 (0.47) + e4 (0.41) |
| ETS | e6 (0.39) |
| Other | e2 (0.33) + e3 (0.28) |
| Homeodomain | e5 (0.30) + e7 (0.26) |
| bZIP | e6 (0.30) + e0 (0.26) |
| Nuclear_Receptor | e6 (0.26) |
| C2H2_long | e5 (0.23) + e0 (0.20) |

---

## 4. Result 2 — experts encode DNA-recognition chemistry

`scripts/analyze_moe_experts.py`. For tokens routed to expert *e*, log2 enrichment of each amino acid
relative to background:

| expert | top enriched residues (log2) | interpretation |
|---|---|---|
| **e0** | **R +1.72**, I, L, A | **arginine** — the canonical major-groove base reader |
| e1 | A +1.90, S +1.79, P | small/flexible backbone |
| e2 | D +2.08, P +1.80, K +1.65 | mixed acidic/basic |
| e3 | T +2.32, C +1.60 | threonine + cysteine |
| **e4** | **C +1.00**, M, N, **H +0.44** | **Cys₂His₂ zinc coordination** — and e4 *is* the C2H2_medium expert |
| e5 | F +1.72, H +1.70, M +1.61 | aromatic + His |
| **e6** | **Q +1.41**, E +1.32 | **glutamine** — the homeodomain Q50 base contact |
| e7 | Y +1.62, W +1.44, S, G | aromatics (Tyr/Trp) |

The correspondence is biologically coherent: the zinc-coordination expert (e4) is the one that C2H2
zinc fingers route to; the arginine expert (e0) serves the basic domains.

---

## 5. Result 3 — the decisive control: family-label ablation

The router receives the family embedding as an input, so `NMI(expert; family) = 0.186` could be
circular — the router might simply re-encode the label it was handed. We tested this by re-routing
every protein with the **family label held constant**:

| quantity | true family label | constant family label |
|---|---|---|
| NMI(expert; **family**) | 0.186 | **0.030** — collapses |
| NMI(expert; **amino acid**) | 0.115 | **0.152** — holds, even rises |
| tokens keeping the same expert | — | **58.5 %** |

**Interpretation — "chemistry backbone + family-label sharpening."**
Most of the *clean per-family* structure does come from the family embedding (family-NMI falls to 0.03
without it). But a substantial **residue-chemistry backbone is independent of the label**: amino-acid
NMI is undiminished and **~59 % of tokens route identically** with the label removed.

So the correct claim is neither "experts emerge to match families" (too strong) nor "the router just
re-encodes the family id" (too weak). Experts organize by **recognition residues**, and the family
embedding sharpens that into per-family routing.

---

## 6. Accuracy in context (the honest part)

Evaluated on the **clean held-out test set** `deeppbs_cluster40` (n = 84 records, **zero overlap** with
any TFScope training set), scored with `scripts/eval_oracle_r_testset.py`:

| model | gate-oracle-r | panel-r |
|---|---|---|
| **combined** (pooled, collapsed MoE) | **0.657** | **0.683** |
| **moe_base** (`ResidueMoE`) | 0.633 | 0.643 |
| contact_bias | 0.629 | 0.596 |
| deep_tune (deeper LoRA) | 0.618 | 0.595 |
| **DeepPBS** (structure-based) | **0.626** | 0.626 |

`gate_oracle_r` is the fair metric: each method uses its **own** predicted motif core, with no
ground-truth extent supplied. (`panel_r` hands TFScope the ground-truth mask window but not DeepPBS, so
it is asymmetric and should not be the headline.)

**Consequences:**

- `moe_base` **loses 0.024** to `combined` on the honest test set. The earlier "parity" (0.703 vs 0.714)
  came from a **contaminated validation set** (42 of 402 val filenames also appear in train) that was
  *also* used for checkpoint selection. It is not a valid basis for a parity claim.
- `moe_base` (0.633) only **barely clears DeepPBS** (0.626). The published "sequence-only TFScope beats
  structure-based DeepPBS" result rests on **`combined`** (+0.031), not on the MoE variant.
- Whether the 0.024 gap is significant is **unresolved** — with n = 84 a paired bootstrap CI is required
  and has not yet been computed.

---

## 7. What can and cannot be claimed

**Supported**
- The per-residue MoE **does not collapse**, unlike every previous TFScope MoE (usage std 0.036, top-1 gate 0.35, NMI 0.186).
- Its experts carry **interpretable recognition chemistry** (arginine reader e0, zinc-coordination e4, Q50 glutamine e6).
- The chemistry organization is **partly label-independent** (AA-NMI holds; 59% of routing unchanged without the family label).
- It achieves this with **~half the trainable MoE parameters** of the pooled block (20 M vs 41 M).

**Not supported (and previously overstated)**
- ❌ "Emergent specialization at **zero accuracy cost**." On the clean test it costs **0.024** gate-oracle-r.
- ❌ "Experts **emerge** to match TF families." Most family-level structure comes from the family embedding.
- ❌ Any claim that `moe_base` beats DeepPBS convincingly — the margin is +0.007.

---

## 8. Limitations

1. **No confidence intervals.** All comparisons are point estimates on n = 84. A **paired bootstrap** on
   per-TF r values is required before any "beats" claim.
2. **Checkpoint selection was flawed.** `ckpt_best` is chosen by oracle-r on only the **first 40**
   validation records, checked every 5 epochs. It is noisy (read 0.808 at ep30 where the full eval gave
   0.660) and demonstrably picked a degraded checkpoint for `deep_tune`.
3. **Diagnostics run on one checkpoint.** The routing/chemistry analyses used `moe_base` `ckpt_best`.
   NMI drifted 0.233 → 0.186 across nearby checkpoints, so the exact value is checkpoint-sensitive
   (the *verdict* SPECIALIZED is stable).
4. **The deploy model is unanalyzed.** The all-data `moe_base` run has not yet been passed through the
   routing diagnostic.

---

## 9. Reproduction

| artifact | command |
|---|---|
| routing collapse | `python scripts/diagnose_residue_moe_routing.py <ckpt_dir> ckpt_best.pt 60` |
| expert chemistry + ablation | `python scripts/analyze_moe_experts.py <ckpt_dir> ckpt_best.pt 60` |
| clean-test accuracy | `python scripts/eval_oracle_r_testset.py <ckpt_dir> data/processed/splits/deeppbs_cluster40/split.json data/processed/tf_pwm_deeppbs_only_canon_trim.parquet ckpt_best.pt` |
| training | `scripts/run_v19_residue_moe.sh` (`--moe-granularity residue`) |

Saved outputs: `results/residue_moe_cases/routing_diagnostic.json`,
`results/residue_moe_cases/expert_analysis.json`.

---

## 10. Recommended next steps

1. **Paired bootstrap CIs** for `combined − moe_base` and `combined − DeepPBS` on the 84 test TFs. This
   single computation determines whether the parity claim survives and whether the DeepPBS win is real.
2. **Rebuild a clean, protein-disjoint train/val/test split** and re-score all models; report selection
   on val and final numbers on test.
3. **Re-run the routing diagnostic on the all-data deploy model** to confirm specialization persists.
4. For the manuscript, present the MoE as the **interpretability contribution** (Fig. 2e: experts =
   recognition chemistry) and `combined` as the **accuracy model** — with the cost stated plainly, not
   as a free lunch.
