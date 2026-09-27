export const meta = {
  name: 'mpn-build-gaps',
  description: 'Build 3 genuinely-untried grading-aligned MIL aggregators (soft-quantile / pairwise-contrast / distribution-matching) + ablations for MPN reticulin grading',
  phases: [{ title: 'Build' }],
}

const HARD = `
HARD CONSTRAINTS (the trainer CANNOT be edited):
- Drop-in module at src/models/novelty_attempts/a<NN>_<name>.py exposing class Model(nn.Module) + a KWARGS dict.
- Trainer calls: logits,_,_ = model(features)  (features = ONE bag, float tensor [N,1280] Virchow2; N varies).
  forward MUST return a 3-tuple (logits, aux, aux); logits shape [1] or [1,1]; clamp output to [0,3]. Permutation- & bag-size-invariant; deterministic at inference.
- Loss (SmoothL1) is in the trainer — NO aux-loss. KWARGS default input_dim=1280, num_classes=1.
- forward must use ONLY 'features' (NO label, NO val/test data) — assert no leakage.
`;

const GRADING = `
GRADING PRINCIPLE: grade = OVERALL/DIFFUSE reticulin density across the whole bag (bag-wide, not a few patches). DO NOT weight by feature-norm ||h|| (grade-uninformative, Spearman ~0). The grade signal is coverage/density along a learned fibrosis direction v = mean(G2/G3) - mean(G0/G1). A warm-start axis is at data/prototypes_virchow2_reti_train_seed2.pt (keys: 'axis' [1280], 'prototypes' {0..3}). Built from seed=2 TRAIN patients (no leakage).
KEY DESIGN PRINCIPLE for these candidates: the per-patch map is a FROZEN/warm-started LINEAR projection s_i=<f_i,v>; the nonlinearity lives ONLY in the BAG-LEVEL statistic (quantiles / pairwise spread / distribution distance). This is WHY they cannot collapse to mean-pool the way the a56-a63 nonlinear-severity pools did (ablation beat main 6 times). Each must ship an ablation that removes exactly the active ingredient.
`;

const TEMPLATES = `Read these for the exact interface + idioms before writing: src/models/novelty_attempts/a52_fibrosis_coverage.py (how to load the warm-start axis defensively, fall back to random if missing), a66_a64_fibrosis_moment_field.py (a recent moment-readout module), and a57_perpatch_linear_pool.py (how an ablation imports the main Model and flips a flag). Match style + type hints.`;

phase('Build')

const BUILD_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['main_file', 'ablation_file', 'param_count', 'sanity_pass', 'leakage_ok', 'sanity_detail', 'notes'],
  properties: {
    main_file: { type: 'string' },
    ablation_file: { type: 'string' },
    param_count: { type: 'integer' },
    sanity_pass: { type: 'boolean' },
    leakage_ok: { type: 'boolean' },
    sanity_detail: { type: 'string' },
    notes: { type: 'string' },
  },
}

const GAPS = [
  {
    ids: ['a70', 'a71'],
    name: 'fibrosis_quantiles',
    spec: `SOFT-QUANTILE VECTOR of the fibrosis projection.
Project s_i = <f_i, v> (v = trainable Parameter warm-started from the prototype 'axis' [1280], raw input space, NO bottleneck). Compute a small set of DIFFERENTIABLE bag quantiles of {s_i} at FIXED levels {0.1, 0.5, 0.9} (use a soft/differentiable quantile, e.g. softmax-weighted order statistic: for level L, weight_i = softmax(-(soft_rank_i/N - L)^2 / tau) and q_L = sum_i weight_i * s_i; soft_rank via differentiable ranking or sorting). y = clamp(Linear(3->1)([q0.1,q0.5,q0.9]) , 0,3). Rationale: G0|G1 decided by the upper tail rising off the clean-marrow floor; G2|G3 by the median/upper-mid rising; reading multiple quantiles of ONE axis acts like per-boundary axes without 3 learned directions.
MAIN = a70_fibrosis_quantiles (3 quantiles). ABLATION = a71_fibrosis_quantiles_median_only: keep ONLY the median quantile -> Linear(1->1)(q0.5) (≈ robust mean of the projection), isolating 'do the tail quantiles add over the central density?'. Ablation imports a70's Model with a flag (quantile_levels=[0.5]).`,
  },
  {
    ids: ['a72', 'a73'],
    name: 'fibrosis_pairwise_spread',
    spec: `PAIRWISE WITHIN-BAG DENSITY-CONTRAST (relational / 2nd-order).
Project s_i = <f_i, v> (v trainable, warm-started from prototype axis, raw space). Compute the bag scalar D = mean_{i,j} |s_i - s_j| (mean absolute pairwise difference = Gini-style SPREAD; O(N^2) but cheap for N<=~200; compute efficiently). y = clamp(Linear([mean_s, D]) , 0,3). Rationale: a truly DIFFUSE high-density bag (every patch moderately fibrotic = high grade) has LOW pairwise contrast; a FOCAL bag (few hot patches in clean marrow) has HIGH contrast at the SAME mean. mean-abs-pairwise-difference is NOT a function of the mean (genuinely 2nd-order) and directly measures 'is fibrosis uniform across the marrow' = the diffuseness the advisor cares about.
MAIN = a72_fibrosis_pairwise_spread (phi=|s_i-s_j|). ABLATION = a73_fibrosis_pairwise_product: replace phi with phi=s_i*s_j (so D = (mean_s)^2, a pure function of the mean -> collapses to mean-pool), isolating 'does relational density-CONTRAST add over the mean?'. Ablation imports a72's Model with a flag (kernel='product'). IMPORTANT: this is distinct from a59/a62 consensus (those contrast each patch to the bag MEAN = 1st-order centroid; this is the FULL pairwise spread).`,
  },
  {
    ids: ['a74', 'a75'],
    name: 'fibrosis_dist_match',
    spec: `DISTRIBUTION-MATCHING to per-grade reference distributions (1-D Wasserstein / quantile-matching).
Project s_i = <f_i, v> (v warm-started from prototype axis). Build per-grade REFERENCE quantile vectors R_g[1..Q] (g in 0..3) OFFLINE from seed=2 TRAIN patches ONLY (see leakage rule). At inference compute the bag's own soft quantiles Q_bag[1..Q] (as in a70) and distance d_g = mean_q (Q_bag[q] - R_g[q])^2 to each grade reference. Readout: y = clamp( sum_g g * softmax(-d_g / tau) , 0,3) (soft-argmin expected grade), tau learnable. Grade by which grade's DENSITY DISTRIBUTION the bag resembles (matching distributions, not means).
*** LEAKAGE RULE (critical) ***: References R_g MUST be computed from the seed=2 TRAIN split ONLY. Write a small offline precompute script scripts/build_fibrosis_refs.py that: (1) loads the full dataset the SAME way the trainer does (MPNBagDatasetFull over data/features_virchow2_reti), (2) calls patient_split(full_dataset, seed=2) from src/train_grading_reti.py to get train_idx, (3) projects ALL patches of TRAIN bags onto the axis, (4) per grade computes fixed quantiles R_g, (5) saves {'levels':..., 'R': {0:..,1:..,2:..,3:..}, 'seed':2} to data/fibrosis_refs_seed2.pt. The Model loads this file defensively (random ref fallback if missing). Add an assertion/comment that val_idx/test_idx patches are NEVER used to build R_g. Confirm in leakage_ok.
MAIN = a74_fibrosis_dist_match (full reference quantile distributions). ABLATION = a75_fibrosis_dist_match_meanonly: collapse each reference to its MEAN R_g_mean -> d_g = (mean_s - R_g_mean)^2 (nearest-prototype-mean 1-D classifier on the bag mean), isolating 'does matching the full DISTRIBUTION beat matching just the mean?'.`,
  },
]

const builds = await parallel(GAPS.map(g => () => agent(
  `You are building a new grading-aligned MIL aggregator + its ablation for the MPN reticulin GRADING novelty search, then sanity- and leakage-checking them. Then I (the orchestrator) will screen them at seed=2.
${HARD}${GRADING}
${TEMPLATES}
CANDIDATE TO IMPLEMENT (ids ${g.ids[0]}=main, ${g.ids[1]}=ablation):
${g.spec}
STEPS:
1. Implement the MAIN module at src/models/novelty_attempts/${g.ids[0]}_${g.name}.py (class Model + KWARGS, input_dim=1280, num_classes=1). Warm-start the projection direction from data/prototypes_virchow2_reti_train_seed2.pt['axis'] defensively (random fallback if missing). Keep capacity LOW (a few params beyond the 1280-d axis) — do NOT add a bottleneck unless the spec says so.
2. Implement the ABLATION at src/models/novelty_attempts/${g.ids[1]}_<...>.py importing the main Model and flipping the documented flag (like a57 imports a56).
3. SANITY (Bash, no training): import both, Model(**KWARGS), forward on random [N,1280] with N=37 and N=120; assert first return elem is a scalar (view(-1).numel()==1) within [0,3]; count params; confirm loss.backward() gives grads. For a74, also run scripts/build_fibrosis_refs.py and confirm it writes data/fibrosis_refs_seed2.pt using ONLY seed=2 train (grep your code to prove val/test indices are excluded).
4. LEAKAGE: confirm forward consumes ONLY 'features' (+ precomputed train-only refs for a74). Set leakage_ok accordingly.
Return main_file, ablation_file, param_count (main), sanity_pass, leakage_ok, sanity_detail, notes (exact --novelty_id to screen, and for a74 whether refs are seed=2-specific). Do NOT run training, do NOT edit the trainer, do NOT run git.`,
  { label: `build:${g.ids[0]}_${g.name}`, phase: 'Build', schema: BUILD_SCHEMA }
)))

return { builds: builds.filter(Boolean) }
