export const meta = {
  name: 'mpn-build-round3',
  description: 'Build 3 ROBUSTNESS-targeted grading-aligned aggregators (robust-trimmed-pool / calibrated-ordinal / heavy-reg) + ablations',
  phases: [{ title: 'Build' }],
}

const HARD = `
HARD CONSTRAINTS (trainer CANNOT be edited):
- Drop-in module src/models/novelty_attempts/a<NN>_<name>.py exposing class Model(nn.Module) + KWARGS.
- Trainer calls logits,_,_ = model(features); features = ONE bag [N,1280] Virchow2 (N varies). Return 3-tuple (logits,aux,aux); logits.view(-1).numel()==1. Output RAW logits (no in-model clamp; trainer rounds+clips at eval). DETERMINISTIC at inference (no test-time randomness/MC-dropout). Permutation- & bag-size-invariant. Loss=SmoothL1 in trainer (NO aux loss). KWARGS input_dim=1280, num_classes=1. forward uses ONLY 'features'.
`;
const GRADING = `
GRADING PRINCIPLE: grade = OVERALL/DIFFUSE reticulin density; NOT a few patches; do NOT weight by ||h|| (grade-uninformative). Read src/models/simple_mil.py (baseline gated attention) + a83_graph_smooth_pool.py (bottleneck->diffuse-pool->Linear, warm-start idiom). Ship an ablation removing EXACTLY the active ingredient (import main Model + flip a flag, like a84 imports a83). Capacity ~ baseline 197K or below.
`;
const STATUS = `
CONTEXT: 86 aggregators + 3 backbones + fusion ALL tie baseline on PAIRED cross-fold Δ≈0. The bottleneck is VARIANCE / generalisation, NOT signal. Even a84 cleared BOTH seed=2 gates but was a seed=2 LOTTERY (lost on 4/5 folds paired). So the WIN BAR is PAIRED cross-fold robustness, and these candidates explicitly target generalisation/robustness (reduce val<->test gap), not seed=2 val. Do NOT reproduce tried mechanisms (rank/norm-salience, top-k, plain coverage/moments/quantiles/pairwise/dist-match, per-patch severity, consensus, PMA, plain mean/attention, graph-smooth, L2-norm, ensemble).
`;

phase('Build')

const BUILD_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['main_file','ablation_file','param_count','sanity_pass','leakage_ok','sanity_detail','notes'],
  properties: {
    main_file:{type:'string'}, ablation_file:{type:'string'}, param_count:{type:'integer'},
    sanity_pass:{type:'boolean'}, leakage_ok:{type:'boolean'}, sanity_detail:{type:'string'}, notes:{type:'string'},
  },
}

const SPECS = [
  { ids:['a87','a88'], name:'robust_trimmed_pool', spec:`ROBUST diffuse location instead of mean-pool. Compute bottleneck h_i, then a SOFT-TRIMMED mean over patches that DOWN-WEIGHTS outlier patches (lone bone fragments / stain/edge artefacts that don't belong to the diffuse tissue manifold) — e.g. weight_i = softmax(-dist_i^2 / s) where dist_i = ||h_i - median_j h_j|| (or distance to the geometric median / bag centroid), s learnable; y = Linear(sum_i weight_i h_i). This is a ROBUST (outlier-resistant) diffuse density that should reduce per-fold variance. MAIN a87 = robust-trimmed pool (down-weight outliers). ABLATION a88 = uniform weights = plain mean-pool. Isolates 'does outlier-robust diffuse pooling generalise better than the mean?'. NOTE distinct from a58 (soft-median of per-patch SEVERITY scalars) and a59/a62 (consensus weight then weighted-mean): here we robustly locate the FEATURE centroid (down-weight feature-space outliers) — and it must be DETERMINISTIC (no sampling).` },
  { ids:['a89','a90'], name:'calibrated_ordinal', spec:`Reduce variance via a CALIBRATED ORDINAL readout on a diffuse-pooled scalar. h = bottleneck; z = mean_i h_i; s = <z, w> (scalar fibrosis score). Map s -> grade via a LEARNABLE MONOTONE calibration: y = sum_{k=1..3} sigmoid((s - b_k)/t) with monotone thresholds b_1<b_2<b_3 (b_k = b_1 + cumsum(softplus(d_k))), t learnable >0 -> y in [0,3] is the expected ordinal count. This proportional-odds-style readout is smoother/better-calibrated than a free Linear and should be more stable across folds. MAIN a89 = monotone cumulative-sigmoid ordinal readout. ABLATION a90 = plain Linear(z->1) (no ordinal calibration). Isolates 'does an ordinal-calibrated readout reduce val<->test variance vs a linear head?'. (a47/a48 tried a cumulative-link head on a learned scalar and tied — distinguish by: here the scalar is an explicit diffuse-pool projection z, and the focus is variance/calibration, but if it merely reproduces a47/a48 say so honestly.)` },
  { ids:['a91','a92'], name:'heavyreg_diffuse', spec:`Pure VARIANCE-REDUCTION via heavy regularisation of a diffuse grading model. bottleneck Linear(1280->128)+ReLU+Dropout(p=0.7, higher than baseline 0.5) with TRAIN-TIME Gaussian feature noise (std~0.1*feature-scale, applied to input features in training only, OFF at eval for determinism) -> mean-pool -> small Linear head. The hypothesis: the val<->test collapse is overfitting the tiny cohort; stronger regularisation shrinks the gap and yields a small but ROBUST paired gain. MAIN a91 = dropout 0.7 + train-time input noise. ABLATION a92 = dropout 0.5, no input noise (= standard diffuse mean-pool baseline). Isolates 'does heavier regularisation reduce the val<->test gap / improve paired generalisation?'. Ensure eval is deterministic (noise + dropout disabled in eval).` },
]

const builds = await parallel(SPECS.map(sp => () => agent(
  `Build a robustness-targeted grading-aligned MIL aggregator + ablation, then sanity/leakage-check. I (orchestrator) screen at seed=2 then auto-audit paired.
${HARD}${GRADING}${STATUS}
CANDIDATE (ids ${sp.ids[0]}=main, ${sp.ids[1]}=ablation):
${sp.spec}
STEPS: 1. MAIN -> src/models/novelty_attempts/${sp.ids[0]}_${sp.name}.py (Model + KWARGS). 2. ABLATION -> src/models/novelty_attempts/${sp.ids[1]}_<...>.py importing main Model + flipping the flag.
3. SANITY (Bash, no training): import both, Model(**KWARGS), forward random [N,1280]*6 for N in {37,120,1}; assert logits.view(-1).numel()==1 + FINITE; params (~<=197K); loss.backward finite grads; permutation-invariant + DETERMINISTIC at eval (model.eval(); two passes identical — critical for a87/a91 which must disable any randomness at eval). 4. LEAKAGE: forward uses ONLY 'features'.
Return main_file, ablation_file, param_count, sanity_pass, leakage_ok, sanity_detail, notes (exact --novelty_id). Do NOT train, edit trainer, or run git.`,
  { label:`build:${sp.ids[0]}_${sp.name}`, phase:'Build', schema:BUILD_SCHEMA }
)))

return { builds: builds.filter(Boolean) }
