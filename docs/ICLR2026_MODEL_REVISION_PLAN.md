# TFScope ICLR 2026 model-revision plan

**Status:** candidate research plan; **v24 remains the production and paper
baseline until the replacement gate in §7 is passed.**

**Primary task:** sequence-only prediction of a monomeric or multimeric
transcription factor's DNA-binding PWM from its DBD sequence(s).

**Out of scope for this revision:** pathogenicity classification, general
mutation-effect prediction, reliable mutant-PWM prediction, protein design, and a
"structured recognition code" claim. The mutation audit remains an honest
limitation and is not the criterion for promoting a new WT-PWM model.

---

## 1. Why revise v24

v24 is the current best complete TFScope model, but the current evidence does not
show that its recognition-energy decoder, explicit amino-acid channel, or MoE is
necessary:

- on the controlled bHLH audit, a simple frozen-ESM mean-pool head matched or
  exceeded the structured variants for WT PWM prediction;
- removing the explicit amino-acid channel did not hurt;
- the MyoD1 L112R effect was carried by ESM's distributed re-embedding of the
  mutant sequence, and no tested model recovered the full `CAGCTG -> CACGTG`
  consensus switch;
- reverse, path, and identity losses on differences of a state function telescope
  algebraically and therefore provide no useful constraint.

The ICLR revision must therefore answer a narrower question:

> Can an architecture designed for unordered TF partner sets and
> residue-by-DNA-position interfaces improve leakage-free WT PWM prediction beyond
> both v24 and a simple ESM readout?

The intended novelty is **multimer equivariance plus training-only interface
distillation**, not mutation prediction.

---

## 2. Non-negotiable experimental rules

1. Keep `v24` and its checkpoint/config immutable.
2. Run every baseline and candidate on the identical MMseqs2 cluster40 split and
   the identical 291-row structure test set.
3. Use identical preprocessing, PWM registration, maximum output length,
   optimization budget, early-stopping rule, and evaluation code.
4. Use at least three fixed seeds. Report each seed, mean, standard deviation, and
   paired bootstrap confidence intervals.
5. Use gene-balanced coverage-aware correlation (`gene_covR`) as the primary
   endpoint. Also report row `covR`, PWM MAE, top-base accuracy, coverage,
   gate-length error, family-stratified scores, monomer scores, and multimer scores.
6. Report parameter count, trainable parameter count, peak memory, and inference
   time. A larger model must be compared with a parameter-matched simple head.
7. Contact maps and structures may supervise training, but the promoted model must
   require **sequence only at inference**.
8. Do not use the test set for architecture selection or threshold tuning. Select
   on validation; evaluate the frozen choice on test once.

---

## 3. Phase I — necessity audit before architecture work

Train or re-evaluate the following variants on the full benchmark:

| ID | Variant | Purpose |
|---|---|---|
| B0 | family-average PWM | family prior floor |
| B1 | nearest training sequence/PWM | memorization and retrieval control |
| B2 | frozen ESM + mean pool + MLP | strongest simple baseline |
| B3 | frozen ESM + attention pool + MLP | pooling control |
| B4 | ESM + span gate | isolates variable-length prediction |
| B5 | v24 without MoE | tests MoE necessity |
| B6 | v24 without contact losses/bias | tests contact contribution |
| B7 | v24 N-chain, otherwise minimal head | tests chain input contribution |
| B8 | complete v24 | frozen reference |

For B2–B7, match the trainable parameter budget where practical. In addition to
aggregate metrics, report paired per-gene differences against B8.

### Phase-I decision

- If B2/B3 statistically matches B8 on the complete benchmark and neither N-chain
  nor contacts provide a reproducible subgroup gain, **stop architecture work**.
  Keep v24 as the historical model and frame the ICLR work as a benchmark/analysis
  paper only if the benchmark itself is sufficiently novel.
- If N-chain helps multimers, proceed to §4.
- If contact supervision helps sequence-only inference, proceed to §5.
- If both help, build the unified candidate in §6.
- If MoE has no reproducible benefit, remove it from every candidate. Specialised
  router statistics alone do not justify retaining it.

---

## 4. Candidate A — permutation-equivariant chain-set encoder

The current N-chain representation is order-aware because chains are concatenated
and receive chain-ID embeddings. Replace this only in a candidate implementation:

1. Encode each DBD chain with the same frozen ESM backbone.
2. Apply the same residue projection to every chain.
3. Treat chains as a set and exchange information using a shared-parameter
   Set Transformer or equivalent permutation-equivariant attention block.
4. Preserve residue-level states for PWM decoding; do not collapse each chain to a
   single vector before inter-chain interaction.
5. Use a shared embedding for equivalent homomer chains. Do not assign semantic
   meaning to arbitrary chain indices.
6. Train with random chain permutations, chain dropout, and valid partner swaps.
7. Add a permutation-consistency loss between predictions for two orderings of the
   same complex.

Required stress tests:

- every permutation of chains for dimers and tractable higher-order complexes;
- unseen partner combinations;
- held-out stoichiometry;
- homomer versus heteromer performance;
- monomer preservation.

This candidate proceeds only if it improves multimer performance without degrading
monomers and is empirically invariant to chain order.

---

## 5. Candidate B — residue × DNA-position interface distillation

Replace the opaque global recognition-energy path with an explicit pair
representation between chain residues and latent DNA positions.

For chain \(k\), residue \(i\), and latent motif position \(j\):

```text
Z[k,i,j]   = W_h h[k,i] + W_q q[j] + W_c c[k]
C[k,i,j]   = sigmoid(contact_head(Z[k,i,j]))
E[k,i,j,b] = base_energy_head(Z[k,i,j], b)
logit[j,b] = prior[j,b] + sum(k,i) C[k,i,j] * E[k,i,j,b]
```

where:

- `h[k,i]` is the context-aware residue representation;
- `q[j]` is a learned latent DNA-position query;
- `c[k]` is an equivariant chain context, not a fixed chain identity;
- `C` predicts residue–DNA-position occupancy;
- `E` predicts base-specific contributions.

Training-only structural supervision:

- distill the 2D residue × DNA-base-position contact map into `C`;
- supervise a 1D recognition-residue marginal only as an ablation;
- mask missing contact labels rather than treating them as negatives;
- evaluate 0%, 25%, 50%, and 100% of available contact labels;
- include shuffled-contact and wrong-family-contact negative controls.

Required causal ablations:

- no contact labels;
- 1D recognition labels only;
- true 2D contact distillation;
- shuffled 2D contacts;
- predicted contacts at inference;
- oracle contacts as a diagnostic upper bound, never as the headline
  sequence-only result.

The contact story is supported only if true 2D distillation outperforms both no
contact and 1D labels at sequence-only inference, and shuffled controls do not.

---

## 6. Unified candidate architecture

Build the unified candidate only when Phase I supports both multimer and contact
signals:

```text
DBD chain sequences
  -> shared frozen ESM encoder
  -> per-chain residue projections
  -> permutation-equivariant chain-set interaction
  -> residue × latent-DNA-position pair mixer
  -> contact occupancy × base-energy aggregation
  -> continuous span gate
  -> variable-length PWM
```

Implementation order:

1. freeze ESM and train only the new encoder/head;
2. compare against parameter-matched B2/B3;
3. only then test projection-layer tuning or the existing LoRA recipe;
4. retain MoE only if a pre-registered ablation shows independent improvement.

Recommended temporary name: `tfscope_interface_set_candidate`. Do not call it
`v25` until §7 is passed.

---

## 7. Pre-registered replacement gate

Promote the candidate to the next TFScope version only if **all** conditions hold:

1. validation-selected candidate beats complete v24 on test
   `gene_covR` by at least **+0.02 absolute**;
2. the paired hierarchical-bootstrap 95% confidence interval for
   candidate − v24 is above zero;
3. all three seeds have positive candidate − v24 `gene_covR`;
4. the candidate also beats the best parameter-matched simple ESM baseline;
5. monomer `gene_covR` decreases by no more than **0.01 absolute**;
6. multimer `gene_covR` improves by at least **0.03 absolute** if the chain-set
   module is part of the claimed contribution;
7. chain permutations change `gene_covR` by less than **0.005** and produce
   numerically equivalent predictions within a documented tolerance;
8. gains are not explained by one dominant family and remain positive under
   leave-one-family-out summaries;
9. sequence-only inference is used for the headline result;
10. all metrics, configs, seeds, checkpoints, and failed variants are recorded.

If any condition fails:

- keep **v24** as the current TFScope model;
- label the new implementation an unsuccessful candidate/ablation;
- do not tune repeatedly on the 291-row test set;
- report the negative result internally and return to the original TFScope paper.

Passing a mutation benchmark is explicitly **not** part of this promotion gate.
Mutation-specific PWM prediction remains separate future work.

---

## 8. Deliverables

The implementing agent should produce:

1. one immutable manifest for v24 and one manifest per candidate;
2. a single apples-to-apples results table containing B0–B8 and all candidates;
3. per-seed and paired per-gene prediction files;
4. aggregate, family, monomer, multimer, and permutation stress-test tables;
5. parameter/runtime accounting;
6. contact-label scaling and shuffled-control figures;
7. a machine-readable `promotion_decision.json` listing every §7 condition as
   `pass` or `fail`;
8. an update to `docs/TFSCOPE_ARCHITECTURE.md` only after the decision:
   - pass: add the promoted architecture as a new version while retaining v24
     history;
   - fail: leave v24 as current and add only a brief candidate-ablation note.

---

## 9. Immediate next action

Do **not** begin with a full architecture rewrite. First complete Phase I on the
full 291-row benchmark. That audit determines whether Candidate A, Candidate B,
both, or neither is scientifically justified.

