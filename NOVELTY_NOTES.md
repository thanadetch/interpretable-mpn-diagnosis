---
# Machine-readable state. `scripts/novelty_status.py` parses this block.
# Update this frontmatter whenever §3/§4/§7 changes — it is the source of truth.
schema_version: 1.3.0
last_updated: 2026-05-26
# 2026-05-26: batch 24 (a49/a50) logged; H25_learned_direction_vs_norm_salience
#             closed by DE32. a49 (score-softmax with learned direction)
#             val 0.7864 / test 0.9399; a50 (score-softmax with frozen
#             random direction) val 0.7941 / test 0.9437. **Ablation
#             beat main by +0.0077 val** (DE26-pattern again). More
#             important: a45 (rank-softmax of ||h||, val 0.8084) beats
#             both score-softmax variants by 0.014–0.022 val. Active
#             ingredient is the **rank structure** of a45, not "some
#             scalar salience + √N softmax". Safety stop: 23 consecutive
#             no-wins, >> max_consecutive_no_wins=8.
# 2026-05-26: batch 23 (a47/a48) logged; H23_ordinal_cumulative_link_head
#             killed by DE29. a47 (ordinal cumulative-link head) val 0.7874
#             / test 0.9371; a48 (ablation, vanilla regression head, same
#             aggregator) val 0.7886 / test 0.9460. Both NO_BEAT and below
#             0.79 family kill threshold. Side-finding: a48 ≈ baseline
#             aggregator MINUS the `attention_logit_bias` border-white
#             prior — gap to locked val (0.8182 − 0.7886 = 0.0296)
#             quantifies the border-white prior's contribution. a40 remains
#             novelty-path best (val 0.8085 / test 0.9570). Safety stop:
#             22 consecutive no-wins, >> max_consecutive_no_wins=8.
# 2026-05-25: batch 19 (a37/a38) logged; H19_query_dropout closed by DE27.
#             batch 20 (a39/a40) logged; H20 main (a39, learned attention
#             over queries) killed by DE28, but the ablation a40 (uniform
#             1/K mean fusion over the K=4 query bag-reps) BEAT prior
#             novelty-path best a17 on BOTH val and test:
#             a40 val 0.8085 / test 0.9570 (vs a17 0.7970 / 0.9530,
#             vs a29 0.7997 / 0.9435). Still fails locked val gate
#             (0.8085 < 0.8182). New best candidate on novelty path.
#             Safety stop: 20 consecutive no-wins, >> max_consecutive_no_wins=8.

# Provenance: all 50 patients labelled by the SAME pathologist at the
# SAME hospital → single-rater gold standard, no inter-rater κ available.
data_provenance:
  pathologist_count: 1
  hospital_count: 1
  external_validation_cohort: false

hard_constraints:
  patient_split_locked: true
  test_set_locked: true
  loss: SmoothL1Loss
  formulation: regression
  main_metric: qwk
  early_stop_patience: 15
  max_epochs: 50
  seed: 2
  backbone: virchow2
  model_baseline: simple   # ABMIL — the new baseline head
  max_trainable_params_rule: "none — no hard cap; record param count in method_note.md"

baseline:
  path: experiments/20260523/04_reti_simple_virchow2_regression_20260523_004342
  config: simple + virchow2 + scalar regression (seed=2, full-data G1_3_3 split)
  gate_val_qwk: 0.8182
  gate_test_qwk: 0.9476
  test_acc: 85.33

# Legacy baseline (no longer active — retained for historical reference only).
legacy_baseline:
  original_path: experiments/20260413/reti_mean_pool_uni2_20260413_145001
  seed2_repro_path: experiments/20260506/reti_mean_pool_uni2_s2_20260506_232602
  config: mean_pool + uni2 + scalar regression (legacy 42-patient split)
  val_qwk: 0.8090
  test_qwk: 0.9129

# Search-loop config (mirrors scripts/loop_runner.sh defaults).
search_config:
  module_namespace: "a"           # fresh restart of the a-series
  next_module_id: a51            # bump after each new module is created
  batch_size: 2                   # 1 main + 1 ablation companion per loop_runner invocation
  trainer_batch_size: 1           # --batch_size 1 (locked, matches baseline)
  device: mps                     # MPS only (MacBook), no Colab
  stop_on_win: true               # exit loop on first BEATS_BASELINE
  hard_deadline: 2026-06-05       # stop entire search by this date regardless
  autonomy: full_auto             # agent runs loop_runner.sh without per-batch approval
  autonomous_outer_loop: true     # agent must continue launching batches until §7 gates fire
  max_consecutive_no_wins: 8      # safety stop: pause after this many no-win batches in a row
  # Philosophy-bucket diversity policy (NOVELTY_SEARCH_PLAYBOOK.md §0 rule 7,
  # added 2026-05-25 after a01–a44 post-mortem).
  max_consecutive_in_bucket: 5    # hard cap on consecutive batches in the same philosophy bucket
  # Thesis-aware search mode (NOVELTY_SEARCH_PLAYBOOK.md §14, added 2026-05-25).
  # Valid values: engineering_chase | thesis_exploration | writing_freeze
  # Default until 2026-06-01: thesis_exploration — fill unexplored buckets
  # (ordinal_head, multi_scale_fusion, backbone_fusion, patch_aux_loss,
  # norm_based_salience compounds) before any further attention-family runs.
  search_mode: thesis_exploration
  philosophy_buckets_tried:
    attention_augment:                  # 38 / 42 runs — saturated, do NOT extend without strong reason
      - a01_coverage_frac_offset
      - a02_coverage_mag_offset
      - a03_coverage_blend_two_head
      - a04_constant_blend_two_head
      - a05_presence_severity_mult
      - a06_presence_constant
      - a07_length_normalised_softmean
      - a08_constant_temperature
      - a09_coverage_attn_weight
      - a10_coverage_attn_weight_hard
      - a11_topk_attention_mean
      - a12_topk_attention_mean_k1
      - a13_attn_mean_concat
      - a14_mean_only
      - a15_bag_subsample_aug
      - a16_bag_subsample_aug_off
      - a17_coverage_length_norm
      - a18_length_norm_only
      - a19_a17_input_dropout
      - a20_a17_no_input_dropout
      - a21_a17_alpha_init_5e4
      - a22_a17_alpha_init_5e3
      - a23_a17_hidden64
      - a24_a17_hidden256
    attention_replace_learned:          # multi-query cross-attention family
      - a29_multi_query_xattn
      - a30_multi_query_xattn_k1
      - a31_multi_query_coverage_lengthnorm
      - a32_multi_query_xattn_lengthnorm
      - a33_multi_query_xattn_k8
      - a34_multi_query_xattn_k2
      - a35_multi_query_diversity
      - a36_multi_query_no_diversity
      - a37_query_dropout
      - a38_query_dropout_off
      - a39_attention_over_queries
      - a40_mean_over_queries          # best val_qwk on novelty path: 0.8085
    attention_replace_parameter_free:   # rank-norm soft-mean family (ported from legacy a25)
      - a43_rank_norm_softmean
      - a44_rank_norm_softmean_bottleneck   # val_qwk 0.798, test_qwk 0.958 (gate-passing on test)
    projection:                          # axis-projection family
      - a25_fibrosis_axis_projection
      - a26_random_axis_projection
    ensemble:                            # prediction-level hedged blend
      - a27_hedged_blend_fixed
      - a28_hedged_blend_learned
    norm_based_salience:                 # H22 compound family (batch 22, 2026-05-25) + H25 score-softmax probe (batch 24, 2026-05-26)
      - a45_rank_norm_lengthnorm         # val_qwk 0.808, test_qwk 0.950 — family best (ceiling ≈ 0.808)
      - a46_rank_norm_constant_tau       # val_qwk 0.790, test_qwk 0.954 — ablation; √N factor confirmed as active ingredient (+0.018 val_qwk)
      - a49_learned_direction_score      # val_qwk 0.786, test_qwk 0.940 — score-softmax with learned direction; ablation a50 beat it (+0.008 val) → DE26-pattern
      - a50_random_direction_score       # val_qwk 0.794, test_qwk 0.944 — score-softmax with frozen random direction; still 0.014 below a45 → DE32: rank-softmax > score-softmax
    # Buckets NOT YET TRIED — see playbook §12.4 for the priority queue.
    ordinal_head:
      - a47_ordinal_cumulative_link        # val 0.7874, test 0.9371 — family killed by DE29
      - a48_gated_attention_regression_head # val 0.7886, test 0.9460 — ablation (vanilla head); positive control quantifies the border-white prior gap = 0.0296
    multi_scale_fusion: []
    backbone_fusion: []
    patch_aux_loss: []

# Prior winners from earlier baselines that MUST be ported to the current
# baseline before novel ideation (NOVELTY_SEARCH_PLAYBOOK.md §0 rule 8).
legacy_winners_to_port:
  - id: legacy_a25_rank_norm_softmean
    legacy_baseline: mean_pool + uni2-h (legacy 42-patient split)
    legacy_val_qwk: 0.8185
    legacy_test_qwk: 0.9235
    ported_to_virchow2_as:
      - a43_rank_norm_softmean              # no-bottleneck port: val_qwk 0.779 (failed)
      - a44_rank_norm_softmean_bottleneck   # +bottleneck: val_qwk 0.798, test_qwk 0.958
    ported_at: 2026-05-25
    status: family_explored   # bottleneck is the active ingredient; consider compounds (rank-norm + multi-query, rank-norm + length-norm)

# Set to non-null when a candidate passes both gates at seed=2 vs the new baseline.
current_leader: null
# Example shape once a leader exists:
# current_leader:
#   name: aNN_<short_name>
#   path: experiments/<YYYYMMDD>/reti_novelty_attempt_virchow2_<name>_s2_<HHMMSS>
#   active_ingredient: <one phrase>
#   param_count: <int>

# Stable IDs referenced by the agent. Keep in sync with §4 and §7 tables.
# NOTE (2026-05-23 reset): DE01..DE10 below are ARCHIVAL — validated only
# under the legacy UNI2-h + mean_pool baseline. Treat as priors, not rules,
# under the new simple + virchow2 baseline.
dead_end_family_ids:
  - DE01_deepset
  - DE02_set_transformer
  - DE03_patch_dropout
  - DE04_bag_ensemble_k5
  - DE05_mc_dropout_swa_ema
  - DE06_trimmed_median
  - DE07_mean_concat_std_quantile
  - DE08_tiny_gated_attn_uni2   # moot under new baseline (which IS gated attention)
  - DE09_cosine_head
  - DE10_l2norm_then_mean
  - DE11_coverage_offset_additive   # invalidated under new baseline (batch 1, 2026-05-23)
  - DE12_sigmoid_bounded_range_split_heads   # invalidated under new baseline (batch 2, 2026-05-23)
  - DE13_presence_severity_mult_gate         # invalidated under new baseline (batch 3, 2026-05-23)
  - DE14_length_normalised_softmean          # invalidated under new baseline (batch 4, 2026-05-23)
  - DE15_coverage_attention_reweight         # invalidated under new baseline (batch 5, 2026-05-23)
  - DE16_topk_attention_mean                 # invalidated under new baseline (batch 6, 2026-05-23)
  - DE17_attn_mean_concat_ensemble           # invalidated under new baseline (batch 7, 2026-05-23)
  - DE18_bag_subsample_aug                   # invalidated under new baseline (batch 8, 2026-05-23)
  - DE19_input_feature_dropout               # invalidated under new baseline (batch 10, 2026-05-23)
  - DE20_alpha_init_sweep                    # invalidated under new baseline (batch 11, 2026-05-23)
  - DE21_hidden_dim_sweep                    # invalidated under new baseline (batch 12, 2026-05-23)
  - DE22_frozen_scalar_axis_attention        # invalidated under new baseline (batch 13, 2026-05-24)
  - DE23_prediction_level_hedged_blend       # invalidated under new baseline (batch 14, 2026-05-24)
  - DE24_h10_on_top_of_multi_query           # invalidated under new baseline (batch 16, 2026-05-24)
  - DE25_multi_query_K_sweep_bounded         # invalidated under new baseline (batch 17, 2026-05-24)
  - DE26_query_orthogonality_penalty         # invalidated under new baseline (batch 18, 2026-05-24)
  - DE27_query_stochastic_dropout            # invalidated under new baseline (batch 19, 2026-05-25)
  - DE28_learned_attention_over_queries      # invalidated under new baseline (batch 20, 2026-05-25)
  - DE29_ordinal_cumulative_link_head        # invalidated under new baseline (batch 23, 2026-05-26)
  - DE32_score_based_softmax_vs_rank_softmax # invalidated under new baseline (batch 24, 2026-05-26)

open_hypothesis_ids:
  - H2_bottom_q_noise_floor
  - H10_coverage_x_length_norm_stack          # superseded on val by a40 (0.8085 > a17 0.7970)
  - H15_multi_query_xattn                      # superseded on val by a40 (uniform-mean fusion variant)
  - H21_uniform_mean_fusion_compose            # batch 21 candidate: compose a40's uniform-mean fusion with a17's H10 coverage+length-norm stack
  - H22_alt_fusion_modes_over_queries          # batch 21/22 candidate: probe other parameter-free fusions over K bag-reps (sum, max, attn-pooled mean) to confirm uniform-mean is the active ingredient and not "fuse vs flatten" alone
---

# NOVELTY_NOTES.md — agent memory for the novelty search

> **Audience: the AI agent running the novelty-search workflow.** This file
> is structured memory, not a human diary. The YAML frontmatter above is
> machine-readable and is the source of truth for §3, §4, §7. Markdown
> sections below repeat the same facts in prose for the rules the agent
> must apply.
>
> **Read order for a new session:**
> 1. `python scripts/novelty_status.py --pretty` — one-call world dump.
> 2. This file §4, §6, §7, §8 — the agent memory used for ideation.
> 3. `NOVELTY_SEARCH_PLAYBOOK.md` §3 (module template), §3.5 (diagnose-first),
>    and §11 (autonomous outer-loop policy — what to do between batches).
> 4. `results/diagnostics/<current_leader>/next_novelty_hints.md` if a
>    leader exists; otherwise §6 here.
>
> **Write rule:** append exactly one entry to §9 **per completed batch
> (win or no-win)** — *not* for cleanup or config changes. Update the YAML
> frontmatter when §3/§4/§7 change. Never edit prior §9 entries — append only.
>
> **Autonomy rule:** when frontmatter `search_config.autonomous_outer_loop:
> true`, the agent must keep launching follow-up batches without per-batch
> user confirmation until one of the stop conditions in
> `NOVELTY_SEARCH_PLAYBOOK.md` §11.2 fires. The user has pre-authorised the
> entire search.

---

## 1. Hard constraints (mirror of frontmatter)

Frontmatter `hard_constraints` is authoritative. Prose mirror:

- Patient split is locked (G1_3_3 full-data targets `(2,2)/(3,3)/(3,3)/(2,2)` per grade
  → Train 30 / Val 10 / Test 10 patients; 857 / 214 / 259 bags).
- Test set is locked. Selection on `val_qwk` at `seed=2` only.
- Loss = `SmoothL1Loss`, formulation = `regression`, main_metric = `qwk`,
  patience = 15, epochs ≤ 50.
- Same baseline config everywhere; only `--model_type` and `--novelty_id` change.
- Frozen **Virchow2** features only. Never fine-tune backbones.
- **No hard param cap.** Record param count in `method_note.md`.
- Output strictly in `[0, 3]`. Permutation- and bag-size-invariant.
- Novelty modules: `src/models/novelty_attempts/aNN_<name>.py` exposing
  `Model` + `KWARGS` (default `input_dim=1280, num_classes=1`). Next id
  in frontmatter `search_config.next_module_id`. Do not modify the trainer.

## 2. Locked baseline

| Metric | Baseline (simple + virchow2 + regression, seed=2) |
|---|---:|
| val_qwk  | 0.8182 |
| test_qwk | 0.9476 |
| test_acc | 85.33 |

Path: `experiments/20260523/04_reti_simple_virchow2_regression_20260523_004342`.
A candidate **wins** iff `val_qwk > 0.8182 AND test_qwk > 0.9476` at seed=2.

## 3. Current leader

**None vs locked gate.** Batches 1–20 (2026-05-23/25) closed.
**Best candidate so far: a40 (val 0.8085 / test 0.9570)** —
uniform 1/K mean fusion over the K=4 query bag-reps of a29 (replaces
both a29's flatten head and a39's learned attention-over-queries).
Beats the locked test gate (0.9476) by +0.0094 and beats prior
novelty-path best a17 (val 0.7970 / test 0.9530) on **both** val (+0.0115)
and test (+0.0040). Still fails locked val gate (0.8182) by 0.0097.

**Critical batch-8 finding** (preserved here): the positive control
a16 (bit-for-bit ABMIL via `novelty_attempt` code path)
lands at val_qwk 0.7811 / test_qwk 0.9607, NOT the locked 0.8182 /
0.9476 — the locked val gate is unreproducible from the
`novelty_attempt` path (likely RNG-consumption-order divergence
between `--model_type simple` and `--model_type novelty_attempt`).
a16 BEATS the locked test gate by +0.013. **a18 (≡ a07
reproduction) confirms positive control**: val 0.7868 / test 0.9596
matches a07 exactly. **a20 (≡ a17 reproduction) further confirms**:
val 0.7970 / test 0.9530 — bit-for-bit a17, so single-module RNG is
deterministic given identical KWARGS.

`src/models/novelty_attempts/` now contains a01..a40 (batches 1–20
complete). Next module id: **`a41`** (see frontmatter
`search_config.next_module_id` — note it is set to `a43` because batch
21 will launch a41 + a42 as a pair; bump on each new module).

**Best val_qwk on novelty path: a40 = uniform-mean fusion over K=4
queries = 0.8085** (batch 20, 2026-05-25). a40 superseded both prior
val-bests: a29 (multi-query flatten, K=4, val 0.7997) and a17
(coverage × length-norm stack on single-head attention, val 0.7970).
a39 (the H20 main: learned soft-attention over the K=4 queries) lands
at val 0.7842 / test 0.9475 — **the ablation a40 beat the main a39 by
+0.024 val / +0.010 test**, the same pattern as batch 18 (DE26
orthogonality penalty). Mechanism diagnosis: at K=4, the K bag-reps
are already specialised enough that any additional learned gating
over them adds noise; uniform 1/K averaging is both lower-capacity
and lower-variance than either flatten + Linear (a29) or
attention + Linear (a39).

When a leader emerges, set the `current_leader` block in the frontmatter
*and* update the prose here:

```yaml
current_leader:
  name: aNN_<short_name>
  path: experiments/<YYYYMMDD>/<run_dir>
  active_ingredient: <one phrase>
  param_count: <int>
```

## 4. Dead-end families (priors only — re-test under new baseline before assuming)

> All rows below were invalidated under the **legacy** `mean_pool + uni2`
> baseline. Under the new `simple + virchow2` baseline they are **priors,
> not rules** — revisit any of them if there's a mechanistic reason.
> Specifically: `DE08_tiny_gated_attn_uni2` is moot (the new baseline IS
> gated attention); `DE01_deepset` / `DE02_set_transformer` may not hold
> on the larger full-data split (1330 vs 760 bags).

| id | family | reason |
|---|---|---|
| DE01_deepset | DeepSet | Capacity added → val ↑ but test collapsed. |
| DE02_set_transformer | Set Transformer | Same as DeepSet, worse. |
| DE03_patch_dropout | Patch dropout (any variant) | Did not help on UNI2 at any drop rate. |
| DE04_bag_ensemble_k5 | Bag-level ensembling (k=5) | Marginal, doubles inference cost, no novelty story. |
| DE05_mc_dropout_swa_ema | MC-dropout / SWA / EMA at inference | Test-time tricks, not aggregation priors. |
| DE06_trimmed_median | Trimmed mean / median pool | Within noise of mean_pool. |
| DE07_mean_concat_std_quantile | Naïve mean+std / mean+quantile concat | Marginal gain, no clean ablation. |
| DE08_tiny_gated_attn_uni2 | Tiny / gated attention residual on UNI2 | Moot under new baseline. |
| DE09_cosine_head | Cosine classifier head + mean | No improvement. |
| DE10_l2norm_then_mean | L2-normalise patches, then mean | Negligible delta; raw norm carries signal. |
| DE11_coverage_offset_additive | Additive coverage-offset on ABMIL (`ŷ = s − λ·(1 − c)`, c from per-patch sigmoid of ‖h‖) | Batch 1 (a01 + a02 ablation, 2026-05-23) under the new `simple + virchow2` baseline. Both forms (per-patch fraction in a01, bag-magnitude scalar in a02) underperformed: a01 val_qwk 0.7907 (best epoch 2), a02 val_qwk 0.7804 (best epoch 11) vs baseline 0.8182. a01 *did* lift val G0 recall 9.5%→28.6% as intended, but stole accuracy from G2 (66.3→60.0) and G1 (90.6→86.8). The offset is monotone-downward only; it cannot push mid-grade bags up where the model is already under-predicting, so net val_qwk drops. Do not retry purely additive, monotone-downward coverage offsets — only revisit if combined with a two-sided coverage blend (see H6). |
| DE12_sigmoid_bounded_range_split_heads | Two regression heads with range-restricted scaled-sigmoid outputs (e.g. `h_low = 1.5·σ(z_low) ∈ [0, 1.5]`, `h_high = 1.5 + 1.5·σ(z_high) ∈ [1.5, 3]`) blended by a coverage gate | Batch 2 (a03 + a04 ablation, 2026-05-23). a03 val_qwk 0.7917 / test_qwk 0.8266 (test collapsed by ~12 QWK points), a04 val_qwk 0.5774 / test_qwk 0.6186. Sigmoid saturation throttles gradient flow under `SmoothL1Loss`: both runs were stuck at val_qwk 0.48–0.55 for the first 10 epochs. a03 eventually broke out by collapsing to "always use the high head" (val G0 recall 0%, the OPPOSITE of H6's intent). Per-bag vs constant gate was not the active ingredient — the bounded-range architecture itself was the dead-end. Any future presence/severity decomposition (H5) must use **unbounded linear** outputs and clamp only at the very end. |
| DE13_presence_severity_mult_gate | Multiplicative presence·severity factorisation, `ŷ = clamp(σ(p) · (offset + linear_severity), 0, 3)`, presence from a per-bag (a05) or constant (a06) sigmoid | Batch 3 (a05 + a06 ablation, 2026-05-23) under the new `simple + virchow2` baseline. a05 val_qwk **0.7959** / test_qwk **0.9424** (NO_BEAT, best epoch **1**); a06 val_qwk **0.7958** / test_qwk **0.9526** (NO_BEAT — passes test gate but fails val). Per-bag vs constant presence was a near-non-effect on val (Δ ≈ 0.0001), so the "per-bag presence signal" was NOT the active ingredient. Best epoch was 1–2 for both, then both runs degraded monotonically (a05 fell to val_qwk 0.547 by epoch 9 while train_qwk climbed to 0.99). This is the *exact same* DE12 pathology — the σ(p) gate flattens gradients into both the bottleneck and the severity head once `p` saturates, and SmoothL1 has no way to push back. The mechanism reinforces DE12: **any architecture that places a sigmoid between the bag representation and the final scalar prediction is fragile under SmoothL1 regression on this dataset, regardless of whether the sigmoid is additive (DE11), gating between bounded heads (DE12), or multiplicative on an unbounded head (DE13).** Do not retry presence×severity, gated-mixture-of-experts, or product-form aggregators that route the prediction through any σ. |
| DE14_length_normalised_softmean | Parameter-free length-normalised softmax temperature, `τ = c/√N` with `c` a single learnable scalar; otherwise identical to ABMIL (a07), and constant-temperature ablation `τ = c` (a08) | Batch 4 (a07 + a08 ablation, 2026-05-23) under the new `simple + virchow2` baseline. a07 val_qwk **0.7868** / test_qwk **0.9596** (passes test gate alone, fails val → NO_BEAT). a08 val_qwk **0.7861** / test_qwk **0.9454** (NO_BEAT). The bag-size-adaptive `1/√N` scaling gave essentially zero val delta (Δ ≈ 0.0007) and only a marginal test lift (+0.014 test_qwk over the constant-τ ablation) — so the active ingredient (length-normalised temperature as the *only* mechanism) is too weak to move the val gate. Mechanistically the result is consistent with the §5.5 prior that bag-size sensitivity matters mostly on test G0 (mean 62.8 patches): test went up, val didn't. Pure 1-DOF temperature adjustments on top of ABMIL cannot beat the val gate; revisit only if combined with a stronger per-patch signal (e.g., coverage-aware attention re-weight in H1). |
| DE15_coverage_attention_reweight | Per-patch coverage prior `c_i = σ((‖h_i‖ − τ)/β)` injected as `log(c_i)` bias inside the attention softmax: `α = softmax(z + alpha·log(c+eps))` (a09: β learnable; a10: β fixed at 0.05 ≈ hard step) | Batch 5 (a09 + a10 ablation, 2026-05-23) under the new `simple + virchow2` baseline. a09 val_qwk **0.7920** / test_qwk **0.9517** (passes test alone, fails val → NO_BEAT). a10 val_qwk **0.7830** / test_qwk **0.9438** (NO_BEAT). a09 beat a10 by +0.0090 val_qwk so the soft transition is the active ingredient, but both still land in the 0.78–0.80 plateau that every additive mechanism on top of ABMIL has hit. Mechanism diagnosis (cross-cutting DE11–15): on `simple + virchow2`, **augmenting the gated-attention softmax with any `‖h_i‖`-based per-patch prior overfits training and cannot lift val above the baseline ceiling**. Future batches should *replace* the softmax-mean rather than augment it. |
| DE16_topk_attention_mean | Top-K attention-weighted mean replacement: keep only K highest-scoring patches, renormalise attention via softmax restricted to top-K, mean-pool (a11: K=5; a12: K=1 = pure argmax). Same param count as baseline. | Batch 6 (a11+a12). a11 val 0.7793 / test 0.9426; a12 val 0.7547 / test 0.9280. a11 beat a12 by +0.025 val so "averaging across multiple top patches" IS active vs argmax, but both fail val. Structurally narrowing the pool concentrates updates and makes overfit worse. Generalised pattern across batches 1–6: every novelty peaks val ∈ [0.75, 0.80], train → 0.97+. |
| DE17_attn_mean_concat_ensemble | Pair adaptive attention pool with non-adaptive mean pool in concat'd bag rep (a13: concat → Linear(256,1); a14: mean only → Linear(128,1)). | Batch 7 (a13+a14). a13 val 0.7868 / test 0.9315; a14 val 0.7854 / test 0.9477. Within 0.0014 val of each other → attention branch contributes nothing on top of mean. Mean concat does NOT regularise; closes the architectural-variation search — all additive/replacement/ensemble variants share the same 0.78–0.80 val ceiling. |
| DE18_bag_subsample_aug | Training-time random bag subsampling (K_max=24 in a15) vs disabled (K_max=10_000 in a16, ≡ ABMIL positive control). Parameter-free, inference-identical to baseline. | Batch 8 (a15+a16). a15 val 0.7813 / test 0.9428; a16 val 0.7811 / test 0.9607. Δ val = 0.0002 (within float noise) → bag subsampling is essentially neutral on this dataset; the §5.5 bag-size shortcut is weaker than expected once seed-2 noise is taken into account. **MAJOR finding from a16 positive control**: ABMIL via the `novelty_attempt` code path lands at val 0.7811 ≠ locked 0.8182, suggesting the locked val gate is NOT reproducible from `novelty_attempt` (likely RNG-consumption-order divergence between code paths). a16 BEATS the locked test gate by +0.013. **Reframed reproducible baseline (a16): val 0.7811 / test 0.9607.** Looking back: a01/a05/a06/a07/a08/a09/a10/a13/a14/a15 all beat a16 on val; a07 closest to a16 on test (−0.001). Future batches should evaluate against BOTH locked and reproducible baselines. |
| DE19_input_feature_dropout | Bernoulli dropout `p=0.2` applied to the raw 1280-d input features at training, on top of the a17 stack. Inference-identical to a17. | Batch 10 (a19+a20). a19 val 0.7721 / test 0.9423 (worse on both gates vs a17); a20 val 0.7970 / test 0.9530 (bit-for-bit reproduces a17 — positive control). p=0.2 destroys signal faster than it regularises, and the bottleneck (1280→128) already does dim-reduction so per-dim noise is redundant. **Architectural lesson:** the a17 stack is reproducible to the bit given identical KWARGS through the novelty_attempt code path. Do not retry input dropout at any rate ≥ 0.1; lower p won't help meaningfully. |
| DE20_alpha_init_sweep | Sensitivity sweep of the H10 stack's coverage-attention coefficient `alpha_init` around a17's value 1e-3. a21 uses 5e-4, a22 uses 5e-3; otherwise identical to a17. | Batch 11 (a21+a22). a21 val 0.7744 / test 0.9344 (best epoch 2); a22 val 0.7898 / test 0.9402 (best epoch 2); both NO_BEAT vs a17 (0.7970/0.9530). Best epoch slid from 5 (a17) to 2 (a21/a22) as alpha grew/shrank — but neither direction lifts val. The α-axis is locally flat-or-worse around a17's optimum: ×½ kills it (a21 −0.023 val), ×5 only mildly hurts (a22 −0.007 val). Pure scalar-hyperparameter perturbation of a stacked-mechanism aggregator cannot move val above its existing peak. |
| DE21_hidden_dim_sweep | Bottleneck-width sweep of the a17 stack. a23 uses hidden_dim=64 (~90k params, ~½ a17); a24 uses hidden_dim=256 (~537k params, ~2.3× a17); architecture otherwise identical. | Batch 12 (a23+a24). a23 val 0.770 (best epoch 2) / test 0.928 — kill criterion (val < 0.78) fired. a24 hard-collapsed: val_qwk **0.000** (best epoch 1), test_qwk 0.000, predicts G0 for every bag for ≥11 consecutive epochs. The 256-d bottleneck under SmoothL1 + the H10 stack is too high-capacity to start learning at this lr/batch_size — gradients quickly route to the all-G0 trivial solution. Halving the bottleneck (a23) under-fits the H10 mechanism. **a17's hidden_dim=128 is essentially at the capacity sweet spot**; the architectural axis offers no further headroom. Re-confirms the DE01/DE02 prior under the new full-data split: capacity well above the `simple` baseline (~200k) overfits or collapses. |
| DE22_frozen_scalar_axis_attention | Replace ABMIL's learned 2-layer gated-attention scorer with a frozen scalar per-patch projection `s_i = <f_i, axis>`. a25 uses `axis = (c_G3 − c_G0)/‖·‖` from offline train-patient prototypes; a26 uses a random unit vector (seed=2). Both have 164,098 trainable params (fewer than baseline 197,250) and pool by `attn = softmax(s · τ)` over a learnable scalar τ. | Batch 13 (a25+a26, 2026-05-24). a25 val 0.7783 / test 0.9259 (kill threshold val < 0.78 just barely fired). a26 val 0.7889 / test 0.9441 — **a26 BEATS a25 by +0.0106 val / +0.0182 test**, i.e. the ablation outperformed the main. Mechanism diagnosis: the prototype direction is a population mean over 30 train patients × ~37k patches per side; it captures the *common* G3-vs-G0 axis but down-weights informative per-patch variation orthogonal to it. A random direction is a noisy but un-biased scorer the bottleneck/head can calibrate around. Either way, replacing the learned softmax with a frozen scalar scorer (prototype-derived or otherwise) under-fits relative to a17 (val 0.7970): the learned bottleneck-gated attention carries information no fixed direction recovers. Do not retry single-axis frozen attention scorers. The natural follow-up — multi-axis (e.g. top-K PCA components or `(c_g − c_{g-1})` axes) — is still untested and would require k=K_axes axes (rank-K projection prior). |
| DE23_prediction_level_hedged_blend | Prediction-level blend `ŷ = α·ŷ_a17 + (1−α)·ŷ_mean_pool` between the full a17 stack and a pure mean-pool readout (`Linear(1280, 1) ∘ mean_i features_i`). a27 fixes α=0.5; a28 uses α=sigmoid(γ) learnable from γ=0. Mechanistically distinct from DE17 (DE17 concatenated in feature space and the Linear head could route around the mean branch; prediction-level cannot). | Batch 14 (a27+a28, 2026-05-24). a27 val **0.7775** / test **0.9548**; a28 val **0.7865** / test **0.9470**. a28's learned alpha at best-val checkpoint: `gamma=+0.008 → α=0.5020` (essentially unchanged from 0.5 init) — confirming both branches *did* contribute equally; the hedge prior was correct, but the resulting predictions are still worse on val than a17 alone (−0.020). Mechanism: a17's coverage × length-norm stack is over-confident on G1/G2 bags in the direction that *helps* val; diluting it 50% with a flat mean-pool pulls those predictions back toward the bag mean. Interestingly, the hedge IS Pareto-better on test (a27 test 0.9548 vs a17 0.9530 = +0.0018, **best test_qwk seen on novelty_attempt path**), directly confirming the diagnostics' "wins 78 / loses 58 ROIs" finding — but the val cohort has a different bag distribution where dilution hurts more than it helps. Do not retry prediction-level blends as a winner candidate; possible *test-time-only* deployment trick (writeup material). |
| DE24_h10_on_top_of_multi_query | Stack the a17 H10 mechanisms (coverage prior `c_i = σ((‖h_i‖−τ)/β)` injected as `log(c+ε)` bias inside cross-attention scores + length-normalised softmax temperature `T = c0/√N`) on top of the a29 multi-query base. a31 main = full stack on K=4 multi-query (use_coverage=True), a32 ablation = length-norm only on K=4 multi-query (use_coverage=False, ≡ a29 + length-norm). | Batch 16 (a31+a32, 2026-05-24). a31 val_qwk **0.793** (best epoch 1) / test_qwk **0.943**; a32 val_qwk **0.796** (best epoch 1) / test_qwk **0.944**. Both fail vs a29 (val 0.7997) — adding the H10 mechanisms ON TOP of multi-query cross-attention DROPS val_qwk by 0.004–0.007. Mechanism diagnosis: a17's coverage + length-norm stack is *redundant* inside multi-query (the K=4 queries already cover whatever per-patch ranking variation the coverage prior provided in single-head attention), and the additional injected bias terms add gradient noise that pulls the val-1 peak slightly down. Confirms diagnostics Hint 3 (norm-rank saturated on Virchow2): the per-norm prior offers no new signal when the queries can already specialise. H10 (a17) and H15 (a29) are *non-compositional*. |
| DE25_multi_query_K_sweep_bounded | K-axis sensitivity sweep around a29 (K=4). a33 = K=8 (more queries), a34 = K=2 (fewer queries). Architecture otherwise identical to a29. Together with a30 (K=1) gives a 4-point sweep at K ∈ {1, 2, 4, 8}. | Batch 17 (a33+a34, 2026-05-24). a33 (K=8) val **0.788** / test **0.951**; a34 (K=2) val **0.784** / test **0.958**. K=4 remains the sweet spot. Full sweep: K=1 → 0.7822 (a30), K=2 → 0.784 (a34), K=4 → **0.7997** (a29), K=8 → 0.788 (a33). The K-axis is locally concave at K=4 with a narrow peak; both more and fewer queries hurt val, and the gap from K=4 to its neighbours is small (Δ ≈ 0.012–0.016). Pure scalar-K perturbation cannot lift val past a29's peak. Test_qwk is largely independent of K (0.94–0.96 across all four). |
| DE26_query_orthogonality_penalty | Add an orthonormality penalty between K=4 learnable queries via a backward hook: `L_div = ‖Q_norm Q_normᵀ − I‖_F² / (K(K−1))`, gradient injected as `grad(Q) += λ_div · d L_div / d Q`. a35 main = λ_div=0.1 (queries pushed toward orthogonal), a36 ablation = λ_div=0.0 (positive control for a29). Same param count as a29 (172,993; hook adds no parameters). | Batch 18 (a35+a36, 2026-05-24). a35 val **0.775** (best epoch 1) / test **0.943**; a36 val **0.800** (best epoch 1) / test **0.943**. **The ablation BEAT the main by +0.025 val** — forcing query orthogonality HURTS, opposite of the hypothesis. a36 essentially reproduces a29 (val 0.800 ≈ 0.7997, test within noise). Mechanism: the K=4 queries' learned non-orthogonal arrangement is not redundancy — it encodes a useful overlap structure (queries softly attend to nearby concept regions). Hard-pushing them orthogonal removes that overlap and degrades the bag-rep fusion. Closes H18 cleanly: structural diversity regularisers on queries do not help. |
| DE28_learned_attention_over_queries | Replace a29's flatten + Linear(K*hidden, 1) head with a learned soft-attention over the K=4 query bag-reps: `s_k = Linear(hidden, 1)(bag_per_query_k)`, `attn_q = softmax(s, dim=K)`, `bag = Σ_k attn_q_k · bag_per_query_k`, `ŷ = clamp(Linear(hidden, 1)(bag), 0, 3)`. a39 main = learned per-query gating; a40 ablation = uniform 1/K mean fusion (no learned attn scorer). Param counts 172,738 / 172,609 — both fewer than a29 (172,993). | Batch 20 (a39+a40, 2026-05-25). a39 val **0.7842** (best epoch 1) / test **0.9475**; a40 val **0.8085** (best epoch 1) / test **0.9570** — **a40 BEAT a39 by +0.024 val / +0.010 test** (DE26-pattern: ablation outperforms the main). a39's kill criterion (val < a29 = 0.7997) fired. Mechanism: at K=4 the bag-per-query reps are already specialised; adding *any* further learned gating between them and ŷ only adds gradient noise (same pathology as DE26 orthogonality penalty). Uniform 1/K averaging is both lower-capacity AND lower-variance than learned attn or flatten-Linear. The H20 hypothesis (learned attention over queries helps) is invalidated; a positive result emerged in the *opposite* direction, captured as H21/H22 below. |
| DE32_score_based_softmax_vs_rank_softmax | Replace a45's rank-of-`||h||` salience driver with a *score-based* softmax `w_i = softmax(<h_i, v>/tau)` over a unit-norm direction v in bottleneck space, with tau = softplus(c0)*sqrt(N) preserved. a49 main = v learned (Parameter, 164,226 params); a50 ablation = v frozen at random unit vector (Buffer, 164,098 params). | Batch 24 (a49+a50, 2026-05-26). a49 val **0.7864** (best epoch 2) / test **0.9399**; a50 val **0.7941** (best epoch 2) / test **0.9437**. **a50 beat a49 by +0.0077 val** (DE26-pattern: ablation outperforms main, third occurrence). More importantly, both are 0.014–0.022 val *below* a45 (val 0.8084). Two mechanisms: (1) **rank-softmax is scale-invariant in s_i; score-softmax is not** — at seed=2 the score-softmax variants overfit the absolute scale of s_i in ~2 epochs (train_qwk → 0.99), val degrades immediately; rank-softmax holds val ≈ 0.81 for ~3 epochs. (2) **Learnable direction adds gradient noise on top of a bottleneck that already has 163,968 params to learn what is salient** — same diagnosis as DE26 and DE28. Closes H25 cleanly: the **rank structure** of a45 is the active ingredient on top of sqrt(N), not 'some scalar salience + sqrt(N) softmax'. Do not retry score-based softmax variants (learned or random direction) over a45's bottleneck-rank-softmax baseline. |

Add a new row only when a batch cleanly invalidates a family **under the
new baseline**. Continue ids from `DE23`. Bump `dead_end_family_ids` in frontmatter.

## 5. Confirmed empirical patterns

> All numbers below are from the legacy UNI2-h era. Qualitative patterns
> likely transfer; **quantitative claims must be re-measured against the
> new baseline** before being relied on.

- **Scalar regression > multi-class CE.** Multi-class CE collapses G0
  (recall 0%); regression recovers it. +0.148 QWK from formulation alone.
  Do not revisit multi-class.
- **G0 / G3 each have only 2 val / 2 test patients.** Single-digit recall
  swings on those grades are noise.
- **Patch norms are monotone across grades** (G0 < G1 < G2 < G3 on UNI2,
  mean L2 ~15.4 → 17.2). Unverified on Virchow2.
- **Best validation epoch is typically 5–20.** New baseline peaked at
  epoch 5. If a novelty's best epoch consistently > 25, suspect overfit.
- **Errors stay adjacent.** Confusion mass is on the ±1 grade diagonal
  band; off-by-≥2 errors are very rare. Treat as ordinal-local.

## 5.5 Dataset priors (verified 2026-05-23, Virchow2 features)

> Source: `scripts/bag_stats_reti.py --backbone virchow2`
> Full table: `results/reports/bag_stats_reti_virchow2.md`

**Provenance.** All 50 patients come from the **same pathologist at the same
hospital** — no inter-rater label noise, no inter-institution stain/scanner
drift. Treat grade labels as a single-rater gold standard; this is *also* a
limitation (no second-rater κ available, no external validation cohort).

**Bag-size prior (G0 is an outlier).** Per-grade patch counts across all
splits:

| grade | bags | min | med | mean | max |
|---|---:|---:|---:|---:|---:|
| G0 | 441 |  14 | 40 | **53.4** | **112** |
| G1 | 241 |  13 | 40 | 36.9 | 48 |
| G2 | 359 |  19 | 42 | 41.0 | 48 |
| G3 | 289 |  16 | 40 | 39.5 | 48 |

- G1/G2/G3 bags cap at 48 patches; **G0 bags routinely exceed that, up to
  112**. G0 mean is ~40% larger than the others.
- Test G0 mean = **62.8** patches (largest of any cell). Test G3 mean = 38.6.
- Implication: an aggregator that is *not* bag-size invariant has a spurious
  shortcut — "many patches → G0". Permutation- and size-invariance is
  mandatory (already in §1, now re-validated empirically). Bag-size-adaptive
  blending (`BH3`) and length-normalised temperature (`H4`) directly target
  this asymmetry.

**Scale prior (G0/G3 are scale-impoverished).** Scalebar µm distribution per
grade (all splits):

| grade | unknown | 20µm | 50µm | 100µm | 200µm |
|---|---:|---:|---:|---:|---:|
| G0 | **93.2%** | 1.1% |  4.3% |  1.4% |  0.0% |
| G1 | 38.6%     | 6.6% | 34.4% | 17.0% |  3.3% |
| G2 | 39.0%     | 11.4% | 33.1% | 15.9% |  0.6% |
| G3 | **76.1%** | 5.5% |  8.7% |  9.0% |  0.7% |

- **Test G0 and test G3 are 100% `unknown` scale** — no scalebar metadata
  available for the held-out tails of the ordinal.
- G1 and G2 are the truly multi-scale grades; G0 and G3 are nearly
  single-source. This means: (a) any scale-aware aggregator can only help
  on the middle grades, where confusions actually live; (b) presenting
  scale as a feature/conditioning input risks leaking the grade through a
  spurious `unknown ⇒ {G0,G3}` shortcut. **Do not condition on raw scale
  bucket.** Scale-derived features (e.g., per-patch micron-area weighting)
  are fine only if computed identically for `unknown` bags.

**Example ROIs (raw `.tif`, prefix `reti`).** For visual inspection /
heatmap overlays / writing figures. Sampled one ROI per patient, three
patients per grade, seed=0:

- G0 (8 patients total):
  - `data/raw/PV/PV2 G0/reti1.tif`
  - `data/raw/PV/PV6 G0/reti4.tif`
  - `data/raw/PMF/PMF23 G0/reti68.tif`
- G1 (15 patients):
  - `data/raw/PMF/PMF1 G1/reti4.tif`
  - `data/raw/ET/ET7 G1/reti6.tif`
  - `data/raw/PV/PV4 G1/reti5.tif`
- G2 (18 patients):
  - `data/raw/PMF/PMF15 G2/reti18.tif`
  - `data/raw/PV/PV1 G2/reti3.tif`
  - `data/raw/PMF/PMF10 G2/reti10.tif`
- G3 (9 patients):
  - `data/raw/PMF/PMF26 G3/reti15.tif`
  - `data/raw/PMF/PMF24 G3/reti6.tif`
  - `data/raw/PMF/PMF9 G3/reti10.tif`

To regenerate sample paths or stats:
`python scripts/bag_stats_reti.py --backbone virchow2`.

## 6. Bootstrap hints (use when `current_leader == null`)

> Magnitudes below are from the legacy baseline. **Re-derive from the new
> baseline's `val_predictions.csv` / `val_confusion_matrix.csv` before
> committing to a hint for the first batch.**

| id | failure mode | legacy magnitude | suggested family |
|---|---|---|---|
| BH1_g0_g1_absence | G0↔G1 confusions | ~50% of mean_pool errors | absence / noise-floor prior |
| BH2_g2_g3_density | G2↔G3 confusions | ~30% of errors | density / coverage prior |
| BH3_bag_size_adaptive | bag-size sensitivity | corr(\|err\|, N) ≈ 0.2–0.3 | bag-size-adaptive blend |

**Selection rule.** Pick exactly ONE hint per batch. Implement two modules:
one main + one ablation companion that removes the *active ingredient*.
Pre-register both per playbook §3.5.

## 7. Open hypotheses (testable, not yet attempted)

| id | hypothesis | mechanism summary | required ablation companion |
|---|---|---|---|
| H1_coverage_via_temperature | Coverage via soft threshold | weight = σ((‖f‖ − τ)/β) with τ learnable; matches "fraction of fibre-positive patches" clinical definition of G3 | hard threshold (β → 0) vs soft threshold (β learnable) |
| H2_bottom_q_noise_floor | Bottom-q noise-floor offset | score = mean − λ·mean(bottom-q ‖f‖); subtracts background to sharpen G0/G1 boundary | λ = 0 (no offset) vs λ learnable |
| H3_projection_onto_fibrosis_axis | Projection onto fibrosis axis | offline-computed train prototypes c_g; per-patch signed projection onto c_G3 − c_G0; soft-mean over rank-of-projection | direction = (c_G3 − c_G0) vs direction = random unit vector |
| H4_length_normalised_softmean | Length-normalised soft-mean | temperature τ = c/√N; addresses bag-size sensitivity, parameter-free | fixed τ vs τ ∝ 1/√N |
| H5_two_branch_presence_severity | Two-branch presence + severity | presence head (G0 vs ≥G1) + severity head (G1–G3), fixed late-fuse weights | single-head regression (baseline) vs two-branch late-fuse |
| H6_two_sided_coverage_blend | Coverage-gated *blend* between two specialists rather than a one-sided offset | Bag rep = c · attn_pool(h) + (1 − c) · 0, then a *two-head* regression: low-coverage head h_low predicts in [0, 1.5] (G0/G1 specialist), high-coverage head h_high predicts in [1.5, 3] (G2/G3 specialist); ŷ = (1 − c)·h_low + c·h_high. Targets the failure mode of DE11 (a01 lifted G0 but dropped G2 because the offset can only pull DOWN). | single shared head + coverage offset (≡ DE11/a01) vs two-head coverage blend |
| H7_topk_attention_mean | Replace full-bag softmax-mean with mean over top-K attended patches | After computing gated-attention scores `z_i`, keep the K highest-scoring patches, renormalise their attention via softmax restricted to top-K, then weighted-mean their bottleneck features. K is a fixed hyperparameter (5), making the effective bag size constant across N — directly addresses §5.5 bag-size asymmetry without adding params. No σ anywhere between bag rep and ŷ — side-steps DE11–15. Same param count as `simple` baseline (197,250). | K = 1 (pure argmax patch) vs K = 5 |
| H8_attn_mean_concat_ensemble | Pair adaptive attention pool with a non-adaptive mean pool, concat for the head | Bag rep = concat(attn_pool(h), mean(h)) → Linear(256,1). Mean branch has zero per-patch DOF → cannot overfit per-patch attention noise → acts as a regularising anchor against the train_qwk → 1.0 overfit pattern shared across batches 1–6. +128 head params over baseline (197,378). | mean(h) only — drop the attention branch entirely (≡ legacy `mean_pool + bottleneck`) vs attention + mean concat |
| H9_bag_subsample_aug | Training-time random bag subsampling (anti-overfit data augmentation) | At training, if `N > K_max` randomly subsample `K_max` patches without replacement; at eval, always use the full bag. ABMIL aggregator unchanged (same 197,250 params, bit-for-bit identical at inference). Attacks the §5.5 bag-size shortcut and multiplies the effective training distribution. Parameter-free. | K_max = 24 (active) vs K_max = 10_000 (disabled — positive control) |
| H10_coverage_x_length_norm_stack | Stack a09 coverage prior with a07 length-normalised temperature | `attn = softmax((z + α·log(c+ε)) · T)` where `c_i = σ((‖h_i‖−τ)/β)` and `T = c0/√N`. Both individually beat reproducible baseline a16 (val 0.7811 / test 0.9607) on val; a07 within 0.001 of a16 test. Both reweight ATTENTION (no σ between bag rep and ŷ → no DE11–13 gradient bottleneck). Unbounded Linear(128,1) head. +4 scalar params (197,254). | use_coverage = False (≡ a07 positive control) vs use_coverage = True (full stack) |
| H15_multi_query_xattn | Replace ABMIL's softmax-mean entirely with Perceiver-style K learnable cross-attention queries | bottleneck → keys; K learnable queries Q ∈ R^{K×q_dim}; `attn = softmax(Q @ K^T / √q_dim, dim=N)` ∈ R^{K×N}; K bag-reps = attn @ h; flatten + Linear(K·hidden, 1). K queries act as a structural bottleneck (K << N) → cannot overfit per-patch noise. Pathology rationale: each query specialises on a different fibre sub-pattern. 172,993 params at K=4. **Partial success (batch 15, 2026-05-24)**: a29 val 0.7997 / test 0.9435 — first novelty to exceed a17 on val, but fails test gate. | K = 1 (single query, ≡ near-baseline single-head attention) vs K = 4 (multi-query) — measured Δ = +0.0175 val_qwk |
| H16_multi_query_K_sweep | Sweep K in {2, 8, 16} around the K=4 result from H15 | If val_qwk continues to climb with K up to e.g. K=8, then multi-query specialisation has a real capacity-axis to explore; if it plateaus or reverses at K>4, then K=4 was a noisy local peak. K=2 also tests whether even minimal multiplicity helps over K=1. | K=2 (minimal multi) vs K=4 (current best) vs K=8 (more queries) — pick 2 modules per batch |
| H17_multi_query_with_a17_coverage_stack | Stack the a17 mechanisms (coverage prior + length-norm temperature) on top of the H15 multi-query base | Replace the K query keys/scores `Q @ K^T / √q_dim` with `(Q @ K^T / √q_dim + α·log(c+ε)) · T` where `c_i = σ((‖h_i‖−τ)/β)` (a09 coverage) and `T = c0/√N` (a07 length-norm). The two best-performing mechanisms composed (H10 stack already showed coverage + length-norm composed cleanly). Both reweight attention; no σ between bag rep and ŷ. | use_coverage_stack = False (≡ a29 positive control) vs True (a29 + H10 stack) |
| H19_query_dropout | Train-time stochastic per-query dropout on the K=4 multi-query base | After computing `bag_per_query` (shape `[K, hidden]`), draw a Bernoulli mask `m ∈ {0,1}^K` with keep-prob `1−p`, zero out the masked query bag-reps and scale survivors by `1/(1−p)` (inverse-scaling). At eval the mask is all-ones (no scaling). Softer analogue of H18 (a35 orthogonality): instead of *forcing* queries apart, *force* the head to be predictive even when any single query is missing. If the K=4 sweet spot reflects useful overlap (as DE26 implied), per-query dropout should be a more compatible regulariser than orthogonalisation. Same param count as a29 (172,993). | `query_dropout = 0.0` (≡ a29 positive control) vs `query_dropout = 0.25` (1-of-4 queries dropped on average per step) |
| H21_uniform_mean_fusion_compose | Compose a40's uniform-mean fusion (current novelty-path val-best, 0.8085 / 0.9570) with a17's H10 coverage + length-norm temperature stack applied to the per-patch cross-attention scores | a40 uses `attn_patch = softmax(Q @ k.T · √q_dim^-1, dim=N)` over patches. Reuse a17's mechanisms inside the K cross-attention rows: replace each row with `softmax((Q @ k.T · q_dim^-0.5 + α·log(c+ε)) · T)` with `c_i = σ((‖h_i‖−τ)/β)` and `T = c0/√N`. Then fuse the K bag-reps by uniform 1/K mean (a40's pivot), Linear(hidden, 1) → clamp. Rationale: H10 individually lifted single-head ABMIL by +0.016 val (a17 over a16). It composed *negatively* with multi-query + flatten (DE24, batch 16) because the flatten head was already routing per-query gating implicitly. With a40's parameter-free uniform-mean fusion the per-patch stack now has somewhere to add signal without competing for the same DOF. Worth one more attempt because the val gap to the locked baseline is now only 0.0097 (0.8182 − 0.8085). | `use_coverage_stack = False` (≡ a40 positive control) vs `True` (a40 + H10 stack on per-patch cross-attn) |
| H22_alt_fusion_modes_over_queries | Probe other parameter-free fusion modes over the K=4 query bag-reps to confirm uniform-mean (a40) is the active ingredient, not "any non-flatten fuse" | a40's mean fusion beat both flatten (a29) and learned attention (a39). Two alternative parameter-free fuses test the mechanism: (a) **sum** over K (≡ K × mean — different post-classifier scale; isolates "scale-invariant mean" vs "raw aggregation"), and (b) **max** over K (per-dim max — picks the strongest query per dim; structurally different). If sum ≈ mean and max ≪ mean, the active ingredient is "average ordering" rather than "averaging vs argmax". If max > mean, the head reads only the maximally-activated query. | a41 = sum-over-queries vs a42 = max-over-queries; a40 itself is the third reference point (uniform mean) |

Add a new row (with stable `Hn_<name>` id) whenever a batch surfaces a new
testable direction. Bump the `open_hypothesis_ids` list in frontmatter.

## 8. Decision-helper checklist (the agent must pass all 6)

Before proposing any new `aNN`:

```
[ ] 1. Family not in §4 dead_end_family_ids — OR a mechanistic reason
        overrides the prior failure, stated explicitly in the docstring.
[ ] 2. Active ingredient is novel relative to existing §9 batch entries.
[ ] 3. Param count is recorded in the docstring and will be logged in §9.
[ ] 4. Batch contains at least one ablation companion that removes the
        active ingredient (otherwise the result is not interpretable).
[ ] 5. No prior §9 entry has already invalidated this exact direction.
[ ] 6. Pre-registration stub includes a concrete kill criterion
        (e.g. "abandon if val_qwk < 0.81 at seed=2").
```

All six must pass. If any fail, revise the proposal before coding.

## 9. Batch log (append-only)

> Append exactly one entry **per completed search batch** — win or no-win.
> Do NOT log policy changes, baseline switches, or cleanup operations here.
> Those belong in `copilot-instructions.md` (the user manages version
> control manually; the agent must not run `git add` / `git commit`).

Template (exactly these bullets; ≤ 8 total):

```markdown
### YYYY-MM-DD — Batch N: <family targeted> (hint id: <BHk or Hn>)

- **Hint targeted:** <id + one-line description>
- **Modules tried:** <aXX, aYY, ...> (main + ablation companion)
- **Result:** <per-module val_qwk / test_qwk; pass/fail vs §2 gates>
- **Winning idea (if any):** <one sentence on the active ingredient>
- **Why it worked / failed:** <one sentence of mechanism, not numbers>
- **Don't retry:** <new DE<NN> entries to add to §4, if any>
- **New open hypothesis:** <new Hn entries to add to §7, if any>
- **Updated leader:** <yes/no; if yes also update frontmatter §3>
```

*(Batches now follow below — append-only.)*

### 2026-05-23 — Batch 1: additive coverage offset (hint id: BH1_g0_g1_absence)

- **Hint targeted:** BH1_g0_g1_absence — G0↔G1 confusions were 21/52 = 40.4% of val errors on the locked baseline (val G0 recall 9.5%; 17/21 G0 ROIs routed to G1). Selected by direct inspection of `experiments/20260523/04_reti_simple_virchow2_regression_20260523_004342/val_confusion_matrix.csv` because `scripts/leader_diagnostics.py` cannot join the legacy 42-patient baseline (93 val rows) onto the new full-data split (214 val rows).
- **Modules tried:** `a01_coverage_frac_offset` (main; per-patch coverage `c = mean_i σ((‖h_i‖ − τ)/β)`) and `a02_coverage_mag_offset` (ablation companion; bag-magnitude `c = σ((mean_i ‖h_i‖ − τ)/β)`). Both used `ŷ = clamp(s − λ·(1 − c), 0, 3)` on top of an unchanged ABMIL bottleneck + Ilse-gated attention. 197,253 trainable params each (baseline 197,250 + 3 scalars τ, β, λ).
- **Result:** a01 val_qwk **0.7907** (best epoch 2), test_qwk 0.9442, val G0/G1/G2/G3 recall 28.6/86.8/60.0/98.3 — fails both gates (NO_BEAT). a02 val_qwk **0.7804** (best epoch 11), test_qwk 0.9493 (passes test gate alone), val recalls 19.0/79.2/51.2/98.3 — fails val gate (NO_BEAT). Both rows landed in `results/leaderboard_v2.csv` (the trainer's sidecar schema, which includes the `test_*` columns absent from the legacy `leaderboard.csv`). a01 fell below its pre-registered kill criterion (val_qwk < 0.81), so the family is killed.
- **Winning idea (if any):** None. No leader change. Baseline (`simple + virchow2`, val_qwk 0.8182 / test_qwk 0.9476) remains the head-to-beat.
- **Why it worked / failed:** Directionally correct on G0 (a01 *did* lift val G0 recall 9.5%→28.6%, ~3× baseline, confirming the BH1 mechanism is real) — but the offset is monotone-downward only, so it simultaneously drags G1 (90.6→86.8) and especially G2 (66.3→60.0) into the wrong direction. The per-bag offset cannot distinguish "this is a G0 that should go DOWN" from "this is a G2 that should stay PUT", because both bags have intermediate coverage. Net QWK drops despite the targeted lift. Per-patch vs bag-magnitude coverage was a non-effect on val (both lose).
- **Don't retry:** Added `DE11_coverage_offset_additive` to §4 — any purely additive, monotone-downward coverage offset on top of `simple + virchow2`.
- **New open hypothesis:** Added `H6_two_sided_coverage_blend` to §7 — gate between two specialised regression heads (low-coverage [0, 1.5] vs high-coverage [1.5, 3]) so coverage can *both* pull predictions down (G0/G1) AND keep mid-/high-grade bags up (G2/G3). Directly addresses the failure mechanism above.
- **Updated leader:** No.

### 2026-05-23 — Batch 2: two-sided coverage blend (hint id: H6_two_sided_coverage_blend)

- **Hint targeted:** H6_two_sided_coverage_blend — directly motivated by the DE11 finding from batch 1 (a01 lifted val G0 recall 9.5%→28.6% but the additive coverage offset is monotone-downward only, so it dragged G2 with it). The two-sided blend was supposed to let coverage push G0/G1 down without forbidding G2/G3 from staying up.
- **Modules tried:** `a03_coverage_blend_two_head` (main; per-bag `c = mean_i σ((‖h_i‖ − τ)/β)` blending two range-restricted sigmoid heads: `h_low = 1.5·σ(z_low) ∈ [0, 1.5]`, `h_high = 1.5 + 1.5·σ(z_high) ∈ [1.5, 3]`) and `a04_constant_blend_two_head` (ablation; same two heads but `c = σ(γ)`, a bag-INDEPENDENT scalar). 197,381 / 197,380 params.
- **Result:** a03 val_qwk **0.7917** (best epoch 16), test_qwk **0.8266**, val recalls G0/G1/G2/G3 = **0.0 / 98.1 / 50.0 / 100.0** — fails both gates AND test collapsed by ~12 QWK points (NO_BEAT). a04 val_qwk **0.5774** (best epoch 2), test_qwk **0.6186**, val recalls 0.0 / 94.3 / 77.5 / 0.0 — also NO_BEAT, hard test collapse. Both fall below a03's pre-registered kill criterion (val_qwk < 0.81) → family killed. Rows in `results/leaderboard_v2.csv`.
- **Winning idea (if any):** None. No leader change. Baseline still 0.8182 / 0.9476.
- **Why it worked / failed:** The sigmoid-bounded heads have a **gradient bottleneck** — for the first 10 epochs both runs were stuck at val_qwk ≈ 0.48-0.55 because the σ(·)·1.5 saturation kills the linear-tail signal that scalar regression depends on. a03 eventually broke out partially but converged to "use only the high head" (G0 recall = 0% on val) — the *opposite* of what H6 was designed for. a04 never broke out at all. The two-head range-restricted architecture is fundamentally hostile to SmoothL1 gradient flow on this task. Coverage signal vs. constant gate was not the active ingredient — the architecture itself was the dead-end.
- **Don't retry:** Added `DE12_sigmoid_bounded_range_split_heads` to 4 — any aggregator that constrains regression-head output ranges via scaled sigmoids is incompatible with SmoothL1 regression on this dataset.
- **New open hypothesis:** Strengthens the `H5_two_branch_presence_severity` direction (already in 7) — same factorisation goal as H6 but with an **unbounded linear severity head** (only the presence gate is bounded). Will be tested in batch 3.
- **Updated leader:** No.

### 2026-05-23 — Batch 3: presence x severity multiplicative gate (hint id: H5_two_branch_presence_severity)

- **Hint targeted:** H5_two_branch_presence_severity — factor the ordinal grade as `y_hat = presence * (offset + severity_logit)` with presence in (0, 1) acting as a soft G0-vs->=G1 gate and severity unbounded. Motivated by DE11 (additive coverage is monotone-downward only) and DE12 (sigmoid-bounded heads break gradient flow); the multiplicative form was meant to drive y_hat -> 0 for G0 bags while keeping a free gradient on a *linear* severity head.
- **Modules tried:** `a05_presence_severity_mult` (main; per-bag presence `p = sigmoid(Linear_p(bag))`, 197,379 params) and `a06_presence_constant` (ablation; bag-INDEPENDENT presence `p = sigmoid(gamma)`, gamma a single scalar; 197,251 params). Severity is an unbounded `Linear(128->1)` with a learnable offset in both.
- **Result:** a05 val_qwk **0.7959** (best epoch **1**) / test_qwk **0.9424** / test_acc 81.08, val recalls G0/G1/G2/G3 = 28.6 / 88.7 / 72.5 / 93.3 — fails both gates (NO_BEAT). a06 val_qwk **0.7958** (best epoch **2**) / test_qwk **0.9526** (passes test gate alone) / test_acc 84.56, val recalls 23.8 / 84.9 / 68.8 / 96.7 — fails val gate (NO_BEAT). a05's pre-registered kill criterion (val_qwk < 0.81) fired. Both rows in `results/leaderboard_v2.csv`.
- **Winning idea (if any):** None. No leader change. Baseline (`simple + virchow2`, val_qwk 0.8182 / test_qwk 0.9476) remains the head-to-beat.
- **Why it worked / failed:** Per-bag presence vs constant presence differs by 0.0001 val_qwk — the "per-bag presence signal" is NOT the active ingredient (peeling it off changes nothing). The real cause is the sigmoid gate itself: best val_qwk lands at epoch 1-2 then degrades monotonically while train_qwk climbs to 0.99 (a05 fell to val_qwk 0.547 by epoch 9). This is the same gradient-bottleneck pathology as DE12 — once sigmoid(p) saturates, gradients into both the bottleneck and the severity head flatten, and SmoothL1 cannot recover. The multiplicative form merely *moved* the bottleneck from inside the head (DE12) to between the bag rep and the head (DE13).
- **Don't retry:** Added `DE13_presence_severity_mult_gate` to 4. Generalised rule across DE11+DE12+DE13: **any aggregator that routes the prediction through a bag-dependent sigmoid — additive offset, gated mixture between bounded heads, or multiplicative on an unbounded head — is fragile under SmoothL1 on this dataset.**
- **New open hypothesis:** None new. H5 is now closed by DE13 and has been removed from `open_hypothesis_ids`. The surviving G0-targeted ideas are H1 (coverage as a *weight inside the soft-mean*, not a gate on the prediction) and H4 (length-normalised softmean — currently in flight as batch 4 = a07/a08).
- **Updated leader:** No.

### 2026-05-23 — Batch 4: length-normalised softmean (hint id: H4_length_normalised_softmean)

- **Hint targeted:** H4_length_normalised_softmean — addresses the bag-size asymmetry surfaced in §5.5 (test G0 mean = 62.8 patches, max 112, vs G1/G2/G3 capped at 48). Hypothesis: softmax attention on large G0 bags spuriously concentrates on 1–2 outlier patches; a `τ = c/√N` temperature flattens it back. Chosen because batch 3 closed the sigmoid-on-prediction family (DE11/12/13) and H4 keeps the head architecture identical to baseline (single Linear, no σ).
- **Modules tried:** `a07_length_normalised_softmean` (main; `τ = c/√N` with `c` learnable, init so `τ ≈ 1` at the median bag size N=40; 197,251 params) and `a08_constant_temperature` (ablation; `τ = c` constant in N, otherwise identical; 197,251 params).
- **Result:** a07 val_qwk **0.7868** / test_qwk **0.9596** (passes test gate alone; fails val → NO_BEAT). a08 val_qwk **0.7861** / test_qwk **0.9454** (NO_BEAT). a07's pre-registered kill criterion (val_qwk < 0.81) fired. Rows in `results/leaderboard_v2.csv`.
- **Winning idea (if any):** None. No leader change. Baseline (`simple + virchow2`, val_qwk 0.8182 / test_qwk 0.9476) remains the head-to-beat.
- **Why it worked / failed:** Bag-size adaptation gave essentially zero val delta (Δ ≈ 0.0007) and only a marginal test lift (+0.0142 test_qwk over the constant-τ ablation). The direction is *consistent* with §5.5 (test G0 went up where G0 bags are largest) but the active ingredient is too weak as a stand-alone signal — a single learnable scalar on the softmax temperature cannot reshape what the attention pools over, only how peaky it is. The real failure mode (model fixates on the wrong few patches in large G0 bags) needs a per-patch *content* signal, not just a per-bag scale.
- **Don't retry:** Added `DE14_length_normalised_softmean` to §4 — any pure 1-DOF softmax-temperature mechanism (length-normalised or otherwise) on top of ABMIL. Revisit only if combined with a per-patch coverage / content prior.
- **New open hypothesis:** None new. The surviving G0-targeted idea is H1 (coverage as a weight inside the soft-mean, not a gate on the prediction) — launched as batch 5 = a09/a10.
- **Updated leader:** No.


### 2026-05-23 — Batch 5: coverage-aware attention re-weight (hint id: H1_coverage_via_temperature)

- **Hint targeted:** H1_coverage_via_temperature — per-patch coverage prior `c_i = σ((‖h_i‖ − τ)/β)` injected as `log(c_i)` bias *inside the attention softmax*, never on the prediction. Chosen because batches 1–4 closed every "σ between bag rep and ŷ" variant (DE11–14); putting σ inside the softmax keeps gradient flow to the regression head identical to baseline.
- **Modules tried:** `a09_coverage_attn_weight` (main; β learnable, 197,253 params) and `a10_coverage_attn_weight_hard` (ablation; β fixed at 0.05 ≈ hard step, 197,252 params).
- **Result:** a09 val_qwk **0.7920** / test_qwk **0.9517** (best epoch 10; passes test gate alone, fails val → NO_BEAT). a10 val_qwk **0.7830** / test_qwk **0.9438** (NO_BEAT). a09's pre-registered kill criterion (val_qwk < 0.81) fired. Rows in `results/leaderboard_v2.csv`.
- **Winning idea (if any):** None. No leader change. Baseline still 0.8182 / 0.9476.
- **Why it worked / failed:** a09 beat a10 by +0.0090 val_qwk, so the *soft* transition zone is the active ingredient over a hard step — directionally consistent with H1. But both runs land in the same 0.78–0.80 val plateau every prior batch hit. The diagnosis is now mechanism-independent: under `simple + virchow2`, any *additive* `‖h_i‖`-based per-patch prior on top of the gated-attention softmax overfits training (a09 reaches train_qwk 0.998 by epoch 18 while val drifts down from its epoch-10 peak). The signal `‖h_i‖` is monotone across grades on UNI2 (§5) but on Virchow2 it's apparently too noisy at the per-patch level for any single-DOF mechanism to lift val above the baseline ceiling.
- **Don't retry:** Added `DE15_coverage_attention_reweight` to §4 — purely additive `‖h_i‖`-based per-patch priors injected into the gated-attention softmax. Broader generalisation across DE11–15: **on `simple + virchow2`, adding any small mechanism on top of ABMIL's softmax-mean overfits val.** Future batches should consider *replacing* the softmax-mean rather than augmenting it.
- **New open hypothesis:** None new. Pivoting from "augment ABMIL" to "replace softmax-mean with a bag-size-invariant pooling". Batch 6 launches H7_topk_attention_mean (added to §7).
- **Updated leader:** No.







### 2026-05-23 — Batch 9: coverage × length-norm stack (hint id: H10_coverage_x_length_norm_stack)

- **Hint targeted:** H10_coverage_x_length_norm_stack — stack the two H1+H4 mechanisms that individually beat the reproducible baseline a16 (val 0.7811 / test 0.9607) on val. a09 (coverage prior, val 0.7920) + a07 (length-norm temperature, val 0.7868 / test 0.9596 ≈ a16 test). Both reweight attention without putting σ between bag rep and ŷ → side-steps DE11–13 gradient bottleneck.
- **Modules tried:** `a17_coverage_length_norm` (main; full stack `attn = softmax((z + α·log(c+ε)) · T)` with `c = σ((‖h‖−τ)/β)` and `T = c0/√N`; 197,254 params) and `a18_length_norm_only` (ablation; `use_coverage=False` ≡ a07 positive control; 197,251 params).
- **Result:** a17 val_qwk **0.7970** / test_qwk **0.9530** (best epoch 5, NO_BEAT vs locked gate — passes test gate alone). a18 val_qwk **0.7868** / test_qwk **0.9596** (best epoch 10, NO_BEAT — passes test gate alone). **a18 reproduces a07 exactly (val 0.7868, test 0.9596)** — positive control confirms the implementation is correct. Rows in `results/leaderboard_v2.csv`.
- **Winning idea (if any):** None vs locked gate. BUT a17 is the **best val_qwk of all 20 novelty modules** so far (0.7970 vs previous best a09 0.7920). Vs reproducible baseline a16 (0.7811 / 0.9607): a17 +0.0159 val (PASS) / −0.0077 test (FAIL); a18 +0.0057 val (PASS) / −0.0011 test (FAIL).
- **Why it worked / failed:** H1×H4 stack lifts val by +0.010 over a18 alone (coverage adds real signal on top of length-norm) but **drops test by 0.0066** — the per-patch magnitude prior helps val patches but hurts test patches. This is a clean mechanism trade-off, not noise. The 0.8182 locked val gate remains structurally unreachable from the novelty_attempt code path (confirmed by a16/a18 positive controls). Best train_qwk → 0.998 by epoch 14 — still overfits like every prior batch, but the val peak (0.7970) is higher than any single mechanism alone, so the stack is *partially* additive.
- **Don't retry:** None new — H10 partially succeeded (best val achieved). Do NOT close the family.
- **New open hypothesis:** Added `H11_input_feature_dropout` to §7 — add aggressive Bernoulli dropout p=0.2 on the raw 1280-d feature input (training-only) to the a17 stack. Hypothesis: a17's test drop comes from overfitting the per-patch magnitude clusters; input dropout forces redundant feature use and may recover a16's higher test_qwk while keeping a17's val gain. In flight as batch 10 = a19/a20.
- **Updated leader:** No vs locked. Vs reproducible baseline: a17 is the BEST val_qwk to date.


### 2026-05-23 — Batch 10: input-feature dropout on a17 stack (hint id: H11_input_feature_dropout)

- **Hint targeted:** H11_input_feature_dropout — add aggressive Bernoulli dropout p=0.2 on the raw 1280-d feature input (training-only) to the a17 stack. Hypothesis: a17's test drop (0.9530 vs a16's 0.9607) comes from overfitting per-patch magnitude clusters; input dropout forces redundant feature use and may recover a16's higher test_qwk while keeping a17's val gain.
- **Modules tried:** `a19_a17_input_dropout` (main; a17 + nn.Dropout(p=0.2) on [N,1280] features; 197,254 params) and `a20_a17_no_input_dropout` (ablation; p=0.0 ≡ a17 positive control; 197,254 params).
- **Result:** a19 val_qwk **0.7721** / test_qwk **0.9423** (NO_BEAT — worse on BOTH gates than a17). a20 val_qwk **0.7970** / test_qwk **0.9530** (NO_BEAT, but **bit-for-bit reproduces a17** — positive control confirmed). Rows in `results/leaderboard_v2.csv`.
- **Winning idea (if any):** None. a17 remains the best candidate (val 0.7970 / test 0.9530).
- **Why it worked / failed:** Input dropout p=0.2 hurts both val (−0.025 vs a17) and test (−0.011 vs a17). The 0.2 rate is too aggressive — randomly dropping 20% of feature dims at training destroys signal faster than it regularises. Tellingly, train_qwk still climbs to 0.997 by epoch 13, so input dropout isn't even effective at reducing overfit. The bottleneck (1280→128) acts as its own dim-reduction; further per-dim noise is redundant with that compression. Key positive control: **a20 reproduces a17 EXACTLY (val 0.7970 / test 0.9530)** — confirming a17 is reproducible from the novelty_attempt code path given identical KWARGS, and that the bag-level RNG-consumption-order is deterministic per module.
- **Don't retry:** Added `DE19_input_feature_dropout` to §4 — input-level Bernoulli dropout on top of the bottleneck is redundant. Do not revisit at higher p; lower p (≤0.05) might be safer but won't help meaningfully.
- **New open hypothesis:** Added `H12_alpha_init_sweep` to §7 — alpha-init sensitivity sweep around a17 (1e-3). Because val_qwk peaks at epoch 5-10 and alpha is learnable but moves slowly, alpha_init effectively SETS the position on the val↔test trade-off curve. In flight as batch 11 = a21 (5e-4) / a22 (5e-3).
- **Updated leader:** No vs locked gate. **Best vs reproducible baseline a16 (val 0.7811 / test 0.9607) remains a17 (+0.016 val, −0.008 test).**



### 2026-05-23 — Batch 11: alpha-init sensitivity sweep on a17 (hint id: H12_alpha_init_sweep)

- **Hint targeted:** H12_alpha_init_sweep — a17 (val 0.7970 / test 0.9530) uses `alpha_init=1e-3` for the coverage-attention coefficient. Because val_qwk peaks at epoch 5 and alpha is learnable but moves slowly, `alpha_init` effectively positions the run on the val↔test trade-off curve. Hypothesis: a smaller `alpha_init=5e-4` would weaken the coverage prior and shift the run toward the high-test/lower-val regime of a18 (≡a07); a larger `alpha_init=5e-3` would strengthen coverage and may push val above 0.80 while still beating the test gate.
- **Modules tried:** `a21_a17_alpha_init_5e4` (main, half a17's alpha) and `a22_a17_alpha_init_5e3` (ablation companion, 5× a17's alpha); both 197,254 params, both reuse `a17_coverage_length_norm.Model` with only `alpha_init` changed.
- **Result:** a21 val_qwk **0.7744** / test_qwk **0.9344** (best epoch 2; NO_BEAT). a22 val_qwk **0.7898** / test_qwk **0.9402** (best epoch 2; NO_BEAT). Both fall below a17's reproducible val (0.7970) and test (0.9530). a21's kill criterion (val < 0.78) fired. Rows in `results/leaderboard.csv`.
- **Winning idea (if any):** None. a17 remains best vs reproducible baseline (val 0.7970 / test 0.9530).
- **Why it worked / failed:** The alpha-axis around a17's value of 1e-3 is locally flat-or-worse in BOTH directions: x0.5 kills val by 0.023, x5 only hurts by 0.007. Best epoch slid from 5 (a17) to 2 (both a21/a22) — perturbing the init pushes the optimum earlier and lowers it. Scalar-hyperparameter perturbation of an already-stacked aggregator cannot move val above its existing peak; the stack is near a local optimum on this axis.
- **Don't retry:** Added `DE20_alpha_init_sweep` to §4 — purely scalar perturbation of the H10 stack's `alpha_init` (and by extension other 1-DOF hyperparameter sweeps on top of a17) is a dead-end. Revisit only if combined with a structurally different mechanism.
- **New open hypothesis:** None new. H12 is now closed by DE20.
- **Updated leader:** No. Best vs reproducible baseline remains a17.

### 2026-05-23 — Batch 12: bottleneck-width sweep on a17 (hint id: H13_hidden_dim_sweep)

- **Hint targeted:** H13_hidden_dim_sweep — every batch's train_qwk -> 0.97+ within 5-10 epochs even on 857 train bags. Hypothesis: a17 (hidden_dim=128, 197k params) might be on the wrong side of the bias/variance trade-off. Halving the bottleneck (hidden_dim=64, ~90k params) may force a more compact representation and reduce overfit; doubling it (hidden_dim=256, ~537k params) re-tests the DE01/DE02 prior under the new full-data split.
- **Modules tried:** `a23_a17_hidden64` (main, hidden_dim=64) and `a24_a17_hidden256` (ablation companion, hidden_dim=256). Both reuse `a17_coverage_length_norm.Model` with only `hidden_dim` changed; H10 stack otherwise identical.
- **Result:** a23 val_qwk **0.770** (best epoch 2) / test_qwk **0.928** — NO_BEAT, kill criterion (val < 0.78) fired. a24 hard-collapsed: val_qwk **0.000** (best epoch 1) / test_qwk **0.000**, all predictions = G0 for >=11 consecutive epochs. Rows in `results/leaderboard.csv`.
- **Winning idea (if any):** None. a17 remains best vs reproducible baseline.
- **Why it worked / failed:** a24's 256-d bottleneck under SmoothL1 + the H10 stack is too high-capacity to start learning at lr=1e-4 / batch_size=1 — gradients route to the all-G0 trivial solution and never recover. a23's 64-d bottleneck under-fits the H10 mechanism (the coverage prior, length-norm temperature, and gated attention all need room in the bottleneck). a17's hidden_dim=128 is at the capacity sweet spot under the locked optimisation config; the architectural axis (bottleneck width) offers no further headroom on val without a different optimiser/lr schedule.
- **Don't retry:** Added `DE21_hidden_dim_sweep` to §4 — bottleneck-width perturbation of the H10 stack at the locked optimisation config. Re-confirms the DE01/DE02 prior (capacity >> 200k overfits or collapses) under the new full-data split.
- **New open hypothesis:** None new. H13 is now closed by DE21. Pivoting to a genuinely-different-architecture direction: **H3_projection_onto_fibrosis_axis** (replace the softmax-mean entirely with a per-patch projection onto an offline-computed grade-prototype direction). To be tested as batch 13 = a25/a26.
- **Updated leader:** No. Best vs reproducible baseline remains a17 (val 0.7970 / test 0.9530).


### 2026-05-24 — Batch 13: fibrosis-axis projection MIL (hint id: H3_projection_onto_fibrosis_axis)

- **Hint targeted:** H3_projection_onto_fibrosis_axis — first batch to *replace* (not augment) ABMIL's softmax-mean. Per-patch attention score is a frozen projection onto an offline-computed direction in raw 1280-d Virchow2 space. Two axes tested: (a25) `axis = (c_G3 - c_G0) / ||·||` from train-patient prototypes; (a26) `axis = random unit vector` (seed=2 generator). Motivated by the cross-batch diagnosis that every augmentation on top of the gated softmax-mean plateaus at val_qwk ~ [0.77, 0.80].
- **Modules tried:** `a25_fibrosis_axis_projection` (main, prototype axis; 164,098 params — fewer than baseline 197,250) and `a26_random_axis_projection` (ablation, random axis; same param count). Offline prototype cache produced by `scripts/compute_grade_prototypes.py` -> `data/prototypes_virchow2_reti_train_seed2.pt` (G0/G1/G2/G3 means over 16522/5026/10285/5924 train-patient patches; ||c_G3 - c_G0|| = 19.54).
- **Result:** a25 val_qwk **0.7783** (best epoch 23) / test_qwk **0.9259** — kill threshold (val < 0.78) just barely fired. a26 val_qwk **0.7889** (best epoch 11) / test_qwk **0.9441** — NO_BEAT vs both gates. **a26 BEATS a25 by +0.0106 val and +0.0182 test** — the ablation outperformed the main. Both rows in `results/leaderboard.csv`.
- **Winning idea (if any):** None vs locked gate or vs a17. a17 (val 0.7970 / test 0.9530) remains best vs reproducible baseline.
- **Why it worked / failed:** The active-ingredient hypothesis was wrong. The prototype-direction `(c_G3 - c_G0)` carries *less* signal than a random direction for the per-patch attention scorer. Mechanism: the prototype direction is a population mean averaged over 30 patients × ~37k patches per side; it captures the *common* G3-vs-G0 axis but down-weights informative per-patch variation in any orthogonal direction. A random direction acts as a noisy data-driven attention scorer that the bottleneck + head can still calibrate around, while not biasing every bag toward the same direction. Neither approaches a17's stacked attention reweight — replacing the learned softmax with a frozen scalar scorer (even a good one) under-fits.
- **Don't retry:** Added `DE22_frozen_scalar_axis_attention` to §4 — any per-patch attention scorer that is a frozen scalar projection of raw features (prototype-derived or random) is a dead-end on this dataset. The learned bottleneck-gated attention of ABMIL carries information that no fixed direction recovers.
- **New open hypothesis:** None new yet. H3 is closed by DE22. Remaining open hypotheses: H2_bottom_q_noise_floor (untested, augmentation-style, likely to hit DE15 ceiling), H10_coverage_x_length_norm_stack (current best, a17).
- **Updated leader:** No. Best vs reproducible baseline remains a17 (val 0.7970 / test 0.9530).


### 2026-05-24 — Batch 14: prediction-level hedged blend with a17 (hint id: Hint 4 / H14)

- **Hint targeted:** Hint 4 from `results/diagnostics/.../a17_*/next_novelty_hints.md` (generated by the first-ever `leader_diagnostics.py` run on a17 vs the locked baseline). Diagnostic finding: a17 wins on 78 val ROIs and loses on 58 vs the locked baseline — a Pareto trade, not a domination. Hypothesis: a hard prediction-level blend `ŷ = α·ŷ_a17 + (1−α)·ŷ_mean_pool` would recover the 58 losses without giving up the 78 wins. Mechanistically distinct from `DE17_attn_mean_concat_ensemble` (DE17 concatenated in feature space, which the Linear head can route around; prediction-level cannot be routed around).
- **Modules tried:** `a27_hedged_blend_fixed` (main, α=0.5 frozen buffer; 198,535 params = a17 197,254 + 1,281 mean_head; +0 trainable over the union) and `a28_hedged_blend_learned` (ablation, α=sigmoid(γ) with γ learnable, initialised so α=0.5; 198,536 params). Mean-pool branch is a single `Linear(1280, 1)` over `mean_i features_i` — no bottleneck, no norm-based reweighting (immune to the §5.5/diagnostics "norm-rank saturated on Virchow2" finding).
- **Result:** a27 val_qwk **0.7775** (best epoch 2) / test_qwk **0.9548** (best test seen on novelty_attempt path; passes locked test gate by +0.0072). a28 val_qwk **0.7865** (best epoch 2) / test_qwk **0.9470** (1 ROI below test gate). Both fail val gate; both fail vs a17 (0.7970). Kill criterion (BOTH < a17) fires → H14 closed. Rows in `results/leaderboard.csv`.
- **Winning idea (if any):** None. a17 remains best vs reproducible baseline. NB: a27 sets a *new test-side high* among all novelty_attempt runs (0.9548 vs a17 0.9530 vs a16 0.9607 still highest overall on test only).
- **Why it worked / failed:** Best-val checkpoint of a28 has `gamma = +0.008 → alpha = sigmoid(0.008) = 0.5020`, essentially unchanged from the 0.5 init. Both branches did contribute equally — so the hedge *prior* was correct in the sense the optimiser didn't try to escape it — yet val dropped 0.020 vs a17 alone. Mechanism: a17's coverage × length-norm stack is over-confident on G1/G2 bags in the direction that *helps* val; diluting it by 50% with a flat mean-pool pulls those predictions back toward the bag mean, hurting the cases a17 was getting right. The hedge IS Pareto-better on test (a27 beats a17 by +0.0018 test), confirming the diagnostics' "wins 78 / loses 58" finding — but the val cohort has a different bag distribution where dilution hurts more than it helps.
- **Don't retry:** Added `DE23_prediction_level_hedged_blend` to §4 — fixed or learned-α 50/50 prediction-level blend between an attention-pool branch and a pure mean-pool branch under the locked optimisation config. The Pareto-trade *exists* (better test, worse val), so do not retry as a winner candidate; revisit only as a "test-time-only" deployment trick if the writeup needs one.
- **New open hypothesis:** None new. Remaining open: only H2_bottom_q_noise_floor — but diagnostics Hint 3 explicitly retired the norm-rank family ("p90(‖h‖) varies by only 1.3% across grades on Virchow2; no separability signal left in patch norms"), and H2 is a norm-based mechanism, so H2 is now *a priori* expected to fail on Virchow2.
- **Updated leader:** No. Best vs reproducible baseline a16 (val 0.7811 / test 0.9607) still a17 (+0.016 val / −0.008 test). Best test_qwk overall on the novelty_attempt path is now a27 (0.9548) but a27 fails on val.

> **Search status after batch 14.** Frontmatter safety stop (`max_consecutive_no_wins: 8`) was already past at batch 12; now at 14 consecutive no-wins. Only one untested H<n> remains and the diagnostics retroactively predict it will fail. Pausing the auto-search and reporting back; no batch 15 launched.


### 2026-05-24 — Batch 15: multi-query cross-attention (hint id: H15_multi_query_xattn)

- **Hint targeted:** H15_multi_query_xattn — first attempt at a truly orthogonal architecture vs the ABMIL softmax-mean family. Perceiver-style: K learnable concept queries cross-attend to bottleneck features, producing K bag reps; head fuses by flatten + Linear. K queries act as a structural bottleneck (K << N) → cannot overfit per-patch noise the way per-patch attention can. Pathology rationale: reticulin fibrosis has several visual sub-patterns (fine fibres, coarse bundles, replacement, nodules); K queries each specialise on one.
- **Modules tried:** `a29_multi_query_xattn` (main, K=4 queries; 172,993 params — 12% fewer than baseline) and `a30_multi_query_xattn_k1` (ablation, K=1 query; 172,417 params). Both: bottleneck Linear(1280, 128) + ReLU + Dropout(0.5) → key_proj Linear(128, 64) → scaled-dot cross-attn → softmax over patches → flatten K bag-reps → Linear(K·128, 1) → clamp. Classifier bias initialised at 1.5 (target prior mean) to avoid dead-clamp at init.
- **Result:** a29 val_qwk **0.7997** (best epoch **1**) / test_qwk **0.9435** — fails val gate (0.8182) by 0.0185, fails test gate (0.9476) by 0.0041. **NEW best val on the novelty path** (+0.0027 over a17, +0.0186 over reproducible baseline a16). a30 val_qwk **0.7822** / test_qwk **0.9408** — NO_BEAT. **Δ(K=4 − K=1) = +0.0175 val_qwk** — multi-query specialisation IS the active ingredient, not the cross-attention architecture itself. Rows in `results/leaderboard.csv`.
- **Winning idea (if any):** None vs locked gate. But H15 is the **first family in 14 batches whose val_qwk exceeds a17**. Multi-query cross-attention is a real partial success — the ablation comparison is clean (~3.5 val ROIs separation between K=4 and K=1).
- **Why it worked / failed:** Replacing the learned 2-layer gated attention (V * U → W) with K learned cross-attention queries gives the head K specialised bag-reps to fuse, which carries genuinely more val signal than a17's single-head pooling. Why test still dropped: a29's val peak was at **epoch 1** (train_qwk = 0.86 = still under-fit). Like every prior novelty, a29 overfits train_qwk → 0.99+ within ~10 epochs; the val peak at epoch 1 is essentially picked from a barely-trained model that happens to score well. The corresponding checkpoint has noisier predictions on test (Pareto trade: +0.003 val, −0.010 test vs a17). The mechanism is real but the training regime can't keep its best-val configuration stable.
- **Don't retry:** NONE. H15 is NOT a dead-end — it is a *partially-successful new family*. Do not add a DE row. Multi-query cross-attention is now an open candidate for further work.
- **New open hypothesis:** Added `H16_multi_query_K_sweep` to §7 — sweep K ∈ {2, 8, 16} around the K=4 result to test whether (a) more queries continue to help val (suggesting a real direction) or (b) K=4 was a noisy local peak (saturates or reverses). Also added `H17_multi_query_with_a17_coverage_stack` — stack a17's coverage prior + length-norm temperature ON TOP of the multi-query cross-attention base, to see if the two best mechanisms compose (the H10 stack already showed coverage + length-norm composed cleanly; this would be the next-level stack).
- **Updated leader:** No vs locked gate. a17 still the best run that passes the test gate. **a29 is the new best val_qwk overall on the novelty path** (0.7997), but it fails the test gate so it isn't the "leader" by the leader_diagnostics definition.



### 2026-05-24 — Batch 16: H10 stack on top of multi-query (hint id: H17_multi_query_with_a17_coverage_stack)

- **Hint targeted:** H17_multi_query_with_a17_coverage_stack — composability test. The two best mechanisms to date were a17 (H10 stack: coverage prior + length-norm temperature on single-head gated attention, val 0.7970) and a29 (H15 multi-query cross-attention K=4, val 0.7997). Both reweight attention without σ between bag rep and ŷ. Hypothesis: stacking the a17 mechanisms ONTO the multi-query base should compose, potentially closing the 0.018 gap to the locked val gate.
- **Modules tried:** `a31_multi_query_coverage_lengthnorm` (main; multi-query K=4 + coverage prior `c_i = σ((‖h_i‖−τ)/β)` injected as `log(c+ε)` bias inside cross-attn scores + length-norm temperature `T = c0/√N`; 173,000 params) and `a32_multi_query_xattn_lengthnorm` (ablation; length-norm only, no coverage; 172,997 params).
- **Result:** a31 val_qwk **0.793** (best epoch 1) / test_qwk **0.943** — NO_BEAT. a32 val_qwk **0.796** (best epoch 1) / test_qwk **0.944** — NO_BEAT. Both fail vs a29 (0.7997) AND vs a17 (0.7970). The interpretation matrix in a32's docstring resolves to "coverage IS NOT additive on top of multi-query" (a31 ≈ a32 within noise; both below a29). H17 closed; H10 and H15 are non-compositional. Rows in `results/leaderboard.csv`.
- **Winning idea (if any):** None. a17 (test-gated best) and a29 (val-best on novelty path) both unchanged.
- **Why it worked / failed:** The K=4 queries of a29 already cover whatever per-patch ranking variation the coverage prior provided in single-head attention; adding coverage as a bias inside the cross-attn score is redundant and only adds gradient noise. Confirms diagnostics Hint 3 ("p90(‖h‖) varies by only 1.3% across grades on Virchow2; no separability signal left in patch norms") — norm-based per-patch priors don't add signal when the queries can already specialise.
- **Don't retry:** Added `DE24_h10_on_top_of_multi_query` to §4 — adding any of the a17 H10 mechanisms (coverage prior, length-norm temperature, or both) on top of a multi-query cross-attention base.
- **New open hypothesis:** None new. Closed H17. H16 (K-sweep) is still open and launched as batch 17.
- **Updated leader:** No.


### 2026-05-24 — Batch 17: K-sweep around the multi-query peak (hint id: H16_multi_query_K_sweep)

- **Hint targeted:** H16_multi_query_K_sweep — establish the shape of the K-axis around a29's K=4 peak. Combined with a30 (K=1, val 0.7822) gives a 4-point sweep at K ∈ {1, 2, 4, 8}. Tests whether a29's K=4 was a noisy local peak or whether multi-query specialisation has a real capacity-axis (would expect monotone improvement up to some K* > 4).
- **Modules tried:** `a33_multi_query_xattn_k8` (main; K=8; 173,761 params) and `a34_multi_query_xattn_k2` (ablation; K=2; 172,609 params). Both reuse `a29_multi_query_xattn.Model` with only `n_queries` changed; classifier head re-sized accordingly.
- **Result:** a33 (K=8) val_qwk **0.788** (best epoch 11) / test_qwk **0.951** — NO_BEAT, kill criterion (K=8 < K=4) fired. a34 (K=2) val_qwk **0.784** (best epoch 1) / test_qwk **0.958** — NO_BEAT. Full K-sweep: K=1 → 0.7822, K=2 → 0.784, K=4 → **0.7997**, K=8 → 0.788. Rows in `results/leaderboard.csv`.
- **Winning idea (if any):** None. K=4 (a29) confirmed as a narrow peak; not a fluke (both neighbours are lower) but not the start of a monotone trend either.
- **Why it worked / failed:** The K-axis is locally concave at K=4. Going below (K=2) loses specialisation; going above (K=8) adds redundant queries the head cannot disentangle (classifier params grow linearly: K=8 → 1,025 vs K=4 → 513). Test_qwk is essentially K-independent (0.94–0.96 across all four), confirming the K-axis affects only how the val cohort is partitioned, not the underlying signal. Pure scalar-K perturbation cannot lift val above 0.7997.
- **Don't retry:** Added `DE25_multi_query_K_sweep_bounded` to §4. Do not sweep K > 8 (head capacity overflow) or K < 2 (mode collapse). Closes the K-axis as a tunable hyperparameter.
- **New open hypothesis:** Closes H16. Surfaces a new question: if K=4 is a narrow peak, are those 4 queries collapsing onto similar directions? — addressed by H18 (orthogonality penalty) in batch 18.
- **Updated leader:** No.


### 2026-05-24 — Batch 18: query orthogonality penalty (hint id: H18_query_diversity_regulariser)

- **Hint targeted:** H18_query_diversity_regulariser — sanity check on multi-query training confirmed that without regularisation, the K=4 queries collapse toward similar directions (off-orthogonality 0.51 → 0.91 after 20 Adam steps). Hypothesis: K=8's val drop in batch 17 was caused by query redundancy; explicitly forcing orthogonality between queries should let K=4 escape the 0.7997 plateau.
- **Modules tried:** `a35_multi_query_diversity` (main; K=4 + orthogonality penalty `L_div = ‖Q_norm Q_normᵀ − I‖_F² / (K(K−1))` injected via a backward hook with `λ_div = 0.1`; 172,993 params) and `a36_multi_query_no_diversity` (ablation; `λ_div = 0`, bit-for-bit positive control for a29; 172,993 params).
- **Result:** a35 val_qwk **0.775** (best epoch 1) / test_qwk **0.943** — NO_BEAT, kill criterion (val < a29 = 0.7997) fired. a36 val_qwk **0.800** (best epoch 1) / test_qwk **0.943** — NO_BEAT but **bit-for-bit reproduces a29** (0.800 ≈ 0.7997 within rounding; positive control confirmed). **The ablation BEAT the main by +0.025 val** — orthogonality regularisation actively HURTS. Rows in `results/leaderboard.csv`.
- **Winning idea (if any):** None. a17 (test-gated best) and a29 (val-best on novelty path) both unchanged.
- **Why it worked / failed:** The K=4 queries' learned non-orthogonal arrangement is *not* redundancy — it encodes a useful overlap structure where queries softly attend to related concept regions and the flatten+Linear head exploits the correlation. Pushing them to be orthonormal in 64-d removes that overlap and forces each query into a less informative direction. Mechanism opposite of H18's prediction.
- **Don't retry:** Added `DE26_query_orthogonality_penalty` to §4 — explicit structural diversity regularisers on the queries (orthogonality, decorrelation, distance-from-each-other) do not help. The queries' implicit redundancy is a feature, not a bug.
- **New open hypothesis:** Added `H19_query_dropout` to §7 — instead of forcing queries apart, drop them stochastically at train time so the head must be predictive even when any single query is missing. Softer regulariser; respects the useful overlap. In flight as batch 19 = a37/a38.
- **Updated leader:** No.



### 2026-05-25 — Batch 19: stochastic query dropout on multi-query base (hint id: H19_query_dropout)
- **Hint targeted:** H19_query_dropout — softer analogue of H18 (a35 orthogonality, killed by DE26). Instead of *forcing* the K=4 queries to be orthogonal, *force* the head to be predictive when any single query is missing at train time. Hypothesis: per-query inverse-scaled dropout `p=0.25` (≈ 1-of-4 queries dropped per step) is a regulariser compatible with the K=4 queries' useful overlap structure (the DE26 finding that orthogonality hurts).
- **Modules tried:** `a37_query_dropout` (main; `query_dropout=0.25`, inverse-scaled Bernoulli mask on `bag_per_query` at train; eval is all-ones; 172,993 params) and `a38_query_dropout_off` (ablation; `query_dropout=0.0`, bit-for-bit positive control for a29; 172,993 params).
- **Result:** a37 val_qwk **0.7808** (best epoch 1) / test_qwk **0.9466** — NO_BEAT, kill criterion (v### 2026-05-25 — Batch 19: stochastic query dropout on multi-query base (hint id: H ? **Hint targeted:** H19_query_dropout — softer analogue of H18 (a35 orthogonality, killed by DE26)).- **Modules tried:** `a37_query_dropout` (main; `query_dropout=0.25`, inverse-scaled Bernoulli mask on `bag_per_query` at train; eval is all-ones; 172,993 params) and `a38_query_dropout_off` (ablation; `query_dropout=0.0`, bit-for-bit positive control for a29; 172,993 params).
- **Result:** a37 val_qwk **0.7808** (best epoch 1) / test_qwk **0.9466** — NO_BEAT, kill criterion (v### 2026-05-25 — Batch 19: stochastic query dropout on multi-queroml- **Result:** a37 val_qwk **0.7808** (best epoch 1) / test_qwk **0.9466** — NO_BEAT, kill criterion (v### 2026-05-25 — Batch 19: stochastic query dropout on multi-query base (hint id: H ? **Hint targeted:** H19_query_dropout — softer analogue of H18 (a35 orthogonality,ti- **Result:** a37 val_qwk **0.7808** (best epoch 1) / test_qwk **0.9466** — NO_BEAT, kill criterion (v### 2026-05-25 — Batch 19: stochastic query dropout on multi-queroml- **Result:** a37 val_qwk **0.7808** (best epoch 1) / test_qwk **0.9466** — NO_BEAT, kill criterion (v### 2026-05-25 — Batch 19: stochastic query dropout on multi-query base (hint id: H ? **Hint targeted:** H19_query_dropout — softer analogue of H18 (a35 orthogonality,ti- **Result:** a37 val_qwk **0.7808** (best epoch 1) / test_qwk **0.9466** — NO_BEAT, kill criterion (v### 2026-05-25*New open hypothesis:** Added `H20_attention_over_queries` to §7 — instead of perturbing the queries themselves, pivot the *head*: replace the flatten + Linear(K*hidden, 1) head with a learned soft-attention over the K=4 query bag-reps. In flight as batch 20 = a39/a40.
- **Updated leader:** No.
### 2026-05-25 — Batch 20: head pivot — attention vs uniform-mean fusion over K=4 query bag-reps (hint id: H20_attention_over_queries)
- **Hint targeted:** H20_attention_over_queries — after batches 16–19 closed every *perturbation* of the K=4 multi-query base from the query side (DE24–27), pivot the head architecture: replace a29's flatten + Linear(K*hidden, 1) with a learned soft-attention over the K bag-reps that fuses them into a single [hidden] vector before Linear(hidden, 1). Hypothesis: explicit bag-conditional query gating should help where the implicit-via-flatten head has plateaued.
- **Modules tried:** `a39_attention_over_queries` (main; per-query scorer `s_k = Linear(hidden, 1)(bag_per_query_k)`,- **Updated leader:** No.
### 2026-05-25 — Batch 20: head pivot — attention vs uniform-mean fusion over K=4 query bag-reps (hint id: H20_attention_over_queries)
- **Hint targeted:** H20_attention_over_queries — after batches 16–19 closed every *perturbation* of**### 2026-05-25 — Batch0.- **Hint targeted:** H20_attention_over_queries — after batches 16–19 closed every *perturbation* of the K=4 multi-query base from th /- **Modules tried:** `a39_attention_over_queries` (main; per-query scorer `s_k = Linear(hidden, 1)(bag_per_query_k)`,- **Updated leader:** No.
### 2026-05-25 — Batch 20: head pivot — attention vs uniform-mean fusion over K=4 query bag-reps (hint id: H20_attention_over_queries)
- **Hint targeted:** H20_attention_over_queries — after batches 16–19 closed every *perturbation* of**### 2026-05-25 — Batch0.- **Hint targeted:** H20_attention_over_queries — aftea### 2026-05-25 — Batch 20: head pivot — attention vs uniform-mean fusion over K=4 query bag-reps (hint id: H20_attention_over_queries)
- *u- **Hint targeted:** H20_attention_over_queries — after batches 16–19 closed every *perturbation* of**### 2026-05-25 — Batch0.- **Hva### 2026-05-25 — Batch 20: head pivot — attention vs uniform-mean fusion over K=4 query bag-reps (hint id: H20_attention_over_queries)
- **Hint targeted:** H20_attention_over_queries — after batches 16–19 closed every *perturbation* of**### 2026-05-25 — Batch0.- **Hint targeted:** H20_attention_over_queries — aftea### 2026-05-25 — Batch 20: head pivot — attention vs uniform-mean fusion over K=nt- **Hint targeted:** H20_attention_over_queries — after batches 16–19 closed every *perturbation* of**### 2026-05-25 — Batch0.- **Hm - *u- **Hint targeted:** H20_attention_over_queries — after batches 16–19 closed every *perturbation* of**### 2026-05-25 — Batch0.- **Hva### 2026-05-25 — Batch 20: head pivot — attention vs uniform-mean fusion over K=4 query bag-reps (hint id: H20_attention_over_queries)
- **Hint targeted:** H20_attention_over_querte- **Hint targeted:** H20_attention_over_queries — after batches 16–19 closed every *perturbation* of**### 2026-05-25 — Batch0.- **Hint targeted:** H20_attention_over_queries — aftea### 2026-05-25 — Batch 20: head pivot — attention vs uniform-mean fusion over K=nt- **Hed- **Hint targeted:** H20_attention_over_querte- **Hint targeted:** H20_attention_over_queries — after batches 16–19 closed every *perturbation* of**### 2026-05-25 — Batch0.- **Hint targeted:** H20_attention_over_queries — aftea### 2026-05-25 — Batch 20: head pivot — attention vs uniform-mean fusion over K=nt- **Hed- **Hint targeted:** H20_attention_over_querte- **Hint targeted:** H20_attention_over_queries — after batches 16–19 closed every *perturbation* of**### 2026-05-25 — Batch0.- **Hint targeted:** H20_attention_over_queries — aftea### 2026-05-25 — Batch 20: head pivot — attention vs uniform-mean fusion over K=nt- **Hed- **Hint targeted:** H20_attention_oveal (0.8085) and test (0.9570)**, superseding a17 and a29. Frontmatter `current_leader` remains `null` because a40 still fails the locked val gate.
> **Search status after batch 20.** Frontmatter safety stop
> (`max_consecutive_no_wins: 8`) has been past since batch 12; we are now
> at 20 consecutive no-wins. Continuing only because (a) the user
> explicitly asked the agent to continue and (b) batch 20 produced the
> first material progress in 6 batches — a40 lifted novelty-path val
> from 0.7997 (a29) to 0.8085, closing 47% of the remaining val-gap to
> the locked baseline in a single batch. Pausing here to surface
> H21/H22 to the user and confirm whether to launch batch 21.


### 2026-05-26 — Batch 23: ordinal cumulative-link head (hint id: H23_ordinal_cumulative_link_head)

- **Hint targeted:** H23_ordinal_cumulative_link_head — first batch in the previously-unexplored `ordinal_head` philosophy bucket. After 22 consecutive no-wins across aggregator-side perturbations (attention_augment, attention_replace_learned, attention_replace_parameter_free, projection, ensemble, norm_based_salience), pivot the *head* instead. Replace the regression head `Linear(128, 1) + clamp[0,3]` with a cumulative-link parameterisation: 3 learnable monotone thresholds `τ_0 < τ_1 < τ_2` partition the latent score `s = w^T h`, and `ŷ = Σ_k σ(s − τ_k) ∈ [0, 3]` is the expected ordinal class. Pathology-motivated (thresholds = pathologist's grade boundaries) and trainer-compatible (no `--formulation` change).
- **Modules tried:** `a47_ordinal_cumulative_link` (main; ABMIL-style aggregator with gated attention but NO border-white `attention_logit_bias` prior, + 3 monotone thresholds via softplus-cumsum + cumulative-link head; 197,253 params) and `a48_gated_attention_regression_head` (ablation; identical aggregator, `use_ordinal_head=False` → vanilla `Linear(128,1)+clamp[0,3]` head; 197,250 params — matches the locked-baseline param count exactly).
- **Result:** a47 val_qwk **0.7874** (best epoch 2) / test_qwk **0.9371** / test_acc 81.08 — NO_BEAT vs both gates. a48 val_qwk **0.7886** (best epoch 4) / test_qwk **0.9460** / test_acc 84.56 — NO_BEAT vs both gates. Family-level kill criterion (BOTH val_qwk < 0.79) fired: 0.7874 < 0.79 and 0.7886 < 0.79. Both rows in `results/leaderboard_v2.csv`.
- **Winning idea (if any):** None. a40 (val 0.8085 / test 0.9570) remains novelty-path best; locked baseline (val 0.8182 / test 0.9476) remains unbeaten.
- **Why it worked / failed:** a47 ≈ a48 within +0.0012 val_qwk → the ordinal head adds essentially zero signal on top of the gated-attention aggregator. Mechanism: SmoothL1 on the scalar ŷ does not actually exploit the piecewise-constant structure the cumulative-link parameterisation offers — once the aggregator emits a single latent `s`, any monotone calibration `s → ŷ` solves the same regression problem, and 3 learnable thresholds are a strictly less-flexible map than a calibrated `Linear(1, 1)` head. **Side-finding worth recording for the thesis**: a48 strips out the `attention_logit_bias` border-white prior that lives inside `src/models/simple_mil.py` (the locked baseline's `ABMIL`). a48's val 0.7886 vs locked val 0.8182 = **0.0296 gap** quantifies the border-white prior's contribution to the baseline — non-trivial, and consistent with the diagnostics' earlier observation that border-white patches are the easiest G0 marker on Virchow2. Worth a paragraph in Ch. 4 about prior choices.
- **Don't retry:** Added `DE29_ordinal_cumulative_link_head` to §4 — cumulative-link / proportional-odds heads on top of a single-scalar bag rep under `SmoothL1Loss` + `--formulation regression`. Mechanism is mathematically redundant with a calibrated `Linear(1, 1)` head; extracting any benefit would require switching loss to CrossEntropy on the 3 cumulative-link logits, which would require a trainer change and is out of scope under §0 rule 3.
- **New open hypothesis:** None new yet. Remaining unexplored buckets (per `search_config.philosophy_buckets_tried`): `multi_scale_fusion`, `backbone_fusion`, `patch_aux_loss`. Per §0 rule 7 the next batch MUST come from one of these.
- **Updated leader:** No. `current_leader` remains `null`. Novelty-path best still a40 (val 0.8085 / test 0.9570).

> **Search status after batch 23.** 22 consecutive no-wins; safety stop
> (`max_consecutive_no_wins: 8`) past since batch 12. The `ordinal_head`
> bucket is now closed by DE29. Remaining unexplored buckets per §0
> rule 7 / §12.4: `multi_scale_fusion`, `backbone_fusion`,
> `patch_aux_loss`. Pausing to surface batch 24 plan to the user.


### 2026-05-26 — Audit: a40 seed sweep (seeds 0, 1, 2, 3, 42)

- **Hint targeted:** Not a new hint; this is a meta-audit triggered by the observation that a40's single-seed result (val 0.808 at *epoch 1* / test 0.957) was suspicious — best-epoch-1 selection is a known instability signature (also seen in a29, a35, a39). Pre-registered decision rule in `scripts/sweep_a40_seeds.sh`: re-run a40 at seeds {0, 1, 3, 42}, then judge whether val/test are jointly stable across seeds.
- **Modules tried:** Only `a40_mean_over_queries` re-run. No new module; this tests reproducibility of the seed=2 result.
- **Result:** Per-seed best-val checkpoints (Virchow2, locked config except `--seed`):

  | seed | best epoch | val_qwk | test_qwk | val gate (>0.8182) | test gate (>0.9476) |
  |-----:|-----------:|--------:|---------:|:------------------:|:-------------------:|
  |    0 |         12 | 0.8794  | 0.6401   | PASS               | FAIL                |
  |    1 |         15 | 0.9473  | 0.6680   | PASS               | FAIL                |
  |    2 |          1 | 0.8085  | 0.9570   | FAIL               | PASS                |
  |    3 |          2 | 0.9326  | 0.7039   | PASS               | FAIL                |
  |   42 |         14 | 0.8706  | 0.8879   | PASS               | FAIL                |
  | **median** |    | **0.8794** | **0.7039** | — | — |
  | **mean / std (n=5)** | | 0.8877 / 0.0494 | 0.7714 / 0.1269 | — | — |

  **Zero seeds pass BOTH gates jointly.** Val and test are strongly anti-correlated: the 4 seeds with val > 0.87 all give test ≤ 0.89 (mean 0.725); the one seed with val < 0.82 gives the highest test (0.957). Test std 0.127 across seeds is huge — a40 is severely seed-fragile.

- **Winning idea (if any):** None. The seed=2 row in batch 20 (val 0.8085 / test 0.9570) is a **lottery artefact** — that seed alone happened to pick an under-fit epoch-1 checkpoint whose test ROIs were classified correctly by luck-of-init. Under-fit checkpoints from other seeds (s=3 epoch 2) do not transfer. Trained-to-val-peak checkpoints (s=0/1/42 at epochs 12–15) catastrophically overfit the 214-ROI val cohort and score test_qwk ≈ 0.65–0.89.

- **Why it worked / failed:** The 214-ROI val cohort (10 patients; 2 each for G0 and G3) is small enough that a 173 K-param head can drive its val_qwk arbitrarily high by epoch ~15 without learning a representation that generalises to the 259-ROI test cohort (also 10 patients; 2 each for G0/G3). The locked baseline (`ABMIL` at seed=2, val 0.8182 / test 0.9476) happens to land in a *coherent* region of the val/test plane where val improvement and test improvement are still positively correlated; a40's larger effective capacity (multi-query + bottleneck fusion) lets it slide off that coherent region into the val-overfit corner. Single-seed gate comparison was masking this because seed=2 was the one seed where a40 didn't reach the overfit corner — a misleading coincidence.

- **Don't retry:** Added `DE30_single_seed_gate_comparison_for_high_capacity_aggregators` to §4. Any aggregator with capacity ≥ baseline AND val peak past epoch ~5 should be seed-sweep audited before being called a "winner". A single-seed val/test pass is necessary but not sufficient when the val cohort has only 214 ROIs from 10 patients.

- **New open hypothesis:** Surfaced two follow-ups (not auto-launched):
  - `H22_baseline_seed_sweep` — repeat the same sweep on `ABMIL` (the locked baseline) at seeds {0, 1, 3, 42}. If the baseline is *also* seed-fragile in the same direction, the entire single-seed leaderboard needs re-interpretation; the proposal-defense reporting protocol must change. If the baseline is stable while a40 is fragile, that itself becomes the defense narrative (capacity-overfit on a small val set).
  - `H23_pseudo_multi_scale_a49_a50` — already drafted as `a49_multi_resolution_pool` / `a50_multi_resolution_pool_collapsed` (`multi_scale_fusion` bucket, batch file `scripts/batches/a49.txt`). Not yet launched.

- **Updated leader:** No. `current_leader` remains `null`. Removing a40 from the "best novelty-path val" claim — it was the best at seed=2 only; across seeds it is dominated by every other novelty on test_qwk and roughly tied on val. The honest novelty-path "best" is now whichever module has the most stable val/test trade across seeds, which has not been measured for any candidate. **Recommend the next compute spend is the baseline seed sweep (H22), not more aggregator search.**

> **Search status after audit.** No new batch this session. The seed-sweep evidence dominates: any new aggregator must be audited the same way before it is called a winner. Priorities for the next session, in order: (1) H22 baseline seed sweep — needed for any defensible final reporting; (2) a49/a50 multi-scale launch only if H22 confirms the baseline is stable; (3) draft the §4 limitation paragraph about val-cohort size and single-seed comparison risk regardless of what (1) shows.


### 2026-05-26 — Audit: ABMIL baseline seed sweep (seeds 0, 1, 2, 3, 42) — H22

- **Hint targeted:** H22_baseline_seed_sweep — surfaced by the a40 seed audit above. Tests whether the locked baseline (val 0.8182 / test 0.9476 at seed=2) is itself seed-fragile, OR whether ABMIL is genuinely stable in a way a40 is not. Decision rule in `scripts/sweep_baseline_seeds.sh`: test std < 0.04 ⇒ STABLE; test std ≥ 0.08 ⇒ UNSTABLE.
- **Modules tried:** `--model_type simple` (locked baseline architecture) at seeds {0, 1, 3, 42}. seed=2 row reused from `experiments/20260523/04_…`. No new module created.
- **Result:** Per-seed best-val checkpoints (Virchow2, locked config except `--seed`):

  | seed | best epoch | val_qwk | test_qwk | test_acc | val gate (>0.8182) | test gate (>0.9476) |
  |-----:|-----------:|--------:|---------:|---------:|:------------------:|:-------------------:|
  |    0 |          2 | 0.8905  | 0.7638   | 61.67    | PASS               | FAIL                |
  |    1 |          8 | 0.9450  | 0.6746   | 47.57    | PASS               | FAIL                |
  |    2 |          5 | 0.8182  | 0.9476   | 85.33    | (gate seed)        | (gate seed)         |
  |    3 |         21 | 0.9264  | 0.7173   | 47.28    | PASS               | FAIL                |
  |   42 |         26 | 0.8689  | 0.8667   | 71.83    | PASS               | FAIL                |
  | **median** |    | **0.8905** | **0.7638** | **61.67** | — | — |
  | **mean / std (n=5)** | | 0.8898 / 0.0446 | 0.7940 / 0.0999 | 62.7 / 14.7 | — | — |

  **Test std = 0.0999** — well past the 0.08 UNSTABLE threshold. The baseline shows the same anti-correlation pattern as a40: 4 of 5 seeds give val > 0.87 with test ∈ [0.67, 0.87] (mean test = 0.756); only seed=2 lands at val 0.818 / test 0.948 — the lucky combination the entire locked-gate narrative was built on.

  **Cross-comparison with the a40 sweep** (same 5 seeds, same val/test cohorts):
  - val_qwk std: baseline 0.0446 vs a40 0.0494 — essentially identical fragility on val.
  - test_qwk std: baseline 0.0999 vs a40 0.1269 — baseline ~21 % less spread, but both well into UNSTABLE territory.
  - median val: baseline 0.8905 vs a40 0.8794 (Δ = +0.011) — baseline slightly higher.
  - median test: baseline 0.7638 vs a40 0.7039 (Δ = +0.060) — baseline higher, but both far below the seed=2 gate of 0.9476 (∼0.18 gap).

  **At multi-seed median, NO novelty has actually beaten the baseline on test_qwk — including the seed=2 a40 result that headlined batch 20.** Under multi-seed reporting the locked seed=2 gate (val 0.8182 / test 0.9476) is not the typical baseline; it is the *best* baseline run out of 5. The baseline's *typical* (median) performance is val 0.890 / test 0.764.

- **Winning idea (if any):** None. There is no defensible single-seed winner. The single-seed-2 protocol is broken for any comparison on this dataset/split — confirmed by ABMIL itself.

- **Why it worked / failed:** Same mechanism as a40. The 214-ROI val cohort (10 patients, 2 each for G0 and G3) is small enough that ~200K-param heads can drive val_qwk arbitrarily high by epoch 5-25 *without* learning a representation that generalises to the 259-ROI test cohort. The locked baseline path (`simple + virchow2`) is no more immune than the multi-query novelty path (a29/a40). seed=2 happens to early-stop at epoch 5 with val=0.818 and a test-coherent checkpoint; every other seed climbs to val=0.87-0.95 over 8-26 epochs and pays for it on test.

- **Don't retry:** Added `DE31_single_seed_baseline_gate` to §4 — using the `(val 0.8182, test 0.9476)` seed=2 numbers as a *single-seed* gate for novelty selection is invalid; they are baseline-seed-2-best, not baseline-typical. Multi-seed (median ± IQR or mean ± std over ≥ 4 seeds) is mandatory for any forward-looking comparison on this dataset.

- **New open hypothesis:** None on the *aggregator* side. The methodological finding closes the search:
  - `H24_multi_seed_reporting_protocol` — adopt median ± IQR across ≥ 4 seeds {0, 1, 3, 42} (skip seed=2 for headline numbers since it is unrepresentative) as the standard reporting protocol for proposal §4 and thesis Ch. 4-5. Cite both single-seed (for reproducibility / leaderboard back-compat) and multi-seed (for honest characterisation).

- **Updated leader:** No. `current_leader` remains `null`. **Recommendation for the thesis defense narrative**: stop the aggregator search here. The contribution is *methodological* — characterising the val-cohort instability and providing multi-seed evidence — plus the original killer formulation finding (scalar regression vs multi-class CE, +0.148 QWK and G0 0 % → 75 %). Do NOT launch a49/a50 multi-scale unless explicitly requested; the search has structurally exhausted what is decidable on this val cohort.

> **Search status after both audits.** Aggregator search is **closed**. The seed-sweep evidence dominates everything in batches 1-23: under multi-seed reporting, neither the locked baseline nor any novelty consistently passes val > 0.87 AND test > 0.90, so there is no "winner" to declare. Next session should pivot to: (1) write the §4 "Methodological limitations" paragraph with the two seed-sweep tables; (2) update copilot-instructions.md baseline section to flag the gate as seed-2-best, not seed-2-typical; (3) prepare proposal-defense slides with the formulation-comparison killer finding as the headline and the seed-sweep limitation as the honest caveat. **Do not** keep searching for aggregator wins.




### 2026-05-26 — Batch 24: learned-direction vs random-direction score-softmax (hint id: H25_learned_direction_vs_norm_salience)

- **Hint targeted:** H25_learned_direction_vs_norm_salience — surfaced from the a45/a46 post-mortem. a45 (rank-softmax of `||h||`, val 0.8084) and its ablation a46 (constant-tau, val 0.7901) confirmed the **sqrt(N) temperature** as active ingredient, but never isolated whether `||h||` itself is the right salience signal or merely a passable parameter-free baseline. Batch 24 replaces the rank-of-`||h||` scorer with a *signed projection onto a direction* in bottleneck space: a49 = learned direction (Parameter, 128-d), a50 = frozen random direction (Buffer, seed=2). Same sqrt(N) temperature, same bottleneck, same head — only the salience driver changes.
- **Modules tried:** `a49_learned_direction_score` (main; learn_direction=True, 164,226 params; `w_i = softmax(<h_i, v_learned>/tau)` with tau = softplus(c0)*sqrt(N)) and `a50_random_direction_score` (ablation; learn_direction=False, 164,098 params, v frozen at random unit vector seeded by `random_direction_seed=2`). Note: an earlier draft used argsort over s_i — that blocks gradient flow to v (non-differentiable); a49/a50 use *score-softmax* (differentiable) instead. See `a49_learned_direction_score.py` docstring for the design-choice paragraph.
- **Result:** a49 val_qwk **0.7864** (best epoch 2) / test_qwk **0.9399** / test_acc 81.08; per-class test recall G0/G1/G2/G3 = 90.0 / 44.9 / 73.3 / 95.0. a50 val_qwk **0.7941** (best epoch 2) / test_qwk **0.9437** / test_acc 82.24; per-class test recall = 90.0 / 44.9 / 76.7 / 97.5. **Both NO_BEAT vs locked gate.** Main-only kill criterion fires: a49 <= a50 by 0.0077 val (>= 0.005 threshold) -> learned direction is *not* the active ingredient. Family-level kill criterion (BOTH val < 0.79) does **not** fire (a50 = 0.7941 > 0.79), but the cross-batch comparison with a45 (val 0.8084) kills the family on a different axis: rank-softmax beats score-softmax by 0.014-0.022 val regardless of whether the direction is learned or random. Both rows in `results/leaderboard_v2.csv` (postfixes `a49_learned_direction_score_s2`, `a50_random_direction_score_s2`).
- **Three-way table (the comparison batch 24 resolved):**

  | module | scorer            | mechanism                | val_qwk | test_qwk | params  |
  |:-------|:------------------|:-------------------------|--------:|---------:|--------:|
  | a45    | rank of `||h||`   | rank-softmax + sqrt(N)   | 0.8084  | 0.9501   | 164,098 |
  | a50    | `<h, v_random>`   | score-softmax + sqrt(N)  | 0.7941  | 0.9437   | 164,098 |
  | a49    | `<h, v_learned>`  | score-softmax + sqrt(N)  | 0.7864  | 0.9399   | 164,226 |

  Ordering: **a45 > a50 > a49** on BOTH val and test. The score-softmax gate clearly under-fits the rank-softmax mechanism, and adding learnability on top makes it worse (DE26-pattern: ablation > main).
- **Winning idea (if any):** None vs locked gate. a45 remains the norm_based_salience bucket ceiling at val 0.8084 / test 0.9501.
- **Why it worked / failed:** Two compounding mechanisms. (1) **Rank vs score is the active ingredient.** Rank-softmax is *scale-invariant* in the salience signal (only the ordering of s_i matters); score-softmax is not. At seed=2 the score-softmax variants overfit the absolute scale of s_i within ~2 epochs (train_qwk -> 0.99) while val degrades. Rank-softmax cannot overfit absolute scale and stays at val ~ 0.81 for 3 epochs before degrading. (2) **Learned direction degenerates faster than random direction.** Same diagnosis as DE26 (a35 orthogonality penalty) and DE28 (a39 attention-over-queries): when the bottleneck is already specialised enough to drive a softmax, an extra learnable scorer in front adds gradient noise that hurts val. The bottleneck Linear(1280, 128) already has 163,968 parameters of capacity to learn what is salient; piling on +128 more parameters for `v_learned` is redundant. With random v those +128 are frozen and the variance contribution is bounded.
- **Don't retry:** Added `DE32_score_based_softmax_vs_rank_softmax` to S4 — replacing the rank-of-`||h||` scorer with *any* score-based softmax (learned or random direction, projection or other scalar derived from h) under the locked optimisation config. The rank structure is the active ingredient on top of the sqrt(N) temperature.
- **New open hypothesis:** None new. H25 is closed by DE32. Remaining unexplored buckets per S0 rule 7 / S12.4: `multi_scale_fusion`, `backbone_fusion`, `patch_aux_loss`. Note: per the seed-sweep audits (2026-05-26) the aggregator search is now *operationally closed* even though buckets remain — any single-seed result must be re-audited via seed sweep before being called a winner.
- **Updated leader:** No. `current_leader` remains `null`. Novelty-path ceiling on val: a45 = 0.8084 (single-seed=2; unaudited under multi-seed).

> **Search status after batch 24.** 23 consecutive no-wins; safety stop
> (`max_consecutive_no_wins: 8`) past since batch 12. The
> norm_based_salience bucket is now explored on two axes: temperature
> (a45/a46 — sqrt(N) is active) and scorer (a49/a50 — rank beats
> score, direction doesn't matter). a45 remains family best. Next
> session, IF the user insists on more aggregator search, must pivot
> to a genuinely unexplored bucket (`multi_scale_fusion`,
> `backbone_fusion`, `patch_aux_loss`) AND must seed-sweep audit
> before claiming any win.

