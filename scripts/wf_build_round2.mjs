export const meta = {
  name: 'mpn-build-round2',
  description: 'Build 3 fresh principled grading-aligned aggregators (L2-norm / graph-smooth / diffuse-attention) + ablations',
  phases: [{ title: 'Build' }],
}

const HARD = `
HARD CONSTRAINTS (trainer CANNOT be edited):
- Drop-in module src/models/novelty_attempts/a<NN>_<name>.py exposing class Model(nn.Module) + KWARGS.
- Trainer calls logits,_,_ = model(features); features = ONE bag [N,1280] Virchow2 (N varies). Return 3-tuple (logits, aux, aux); logits view(-1).numel()==1. Output RAW logits (do NOT clamp in-model; the trainer rounds+clips at eval, matching the baseline). Permutation- & bag-size-invariant; deterministic at inference. Loss=SmoothL1 in trainer (NO aux loss). KWARGS input_dim=1280, num_classes=1. forward uses ONLY 'features'.
`;
const GRADING = `
GRADING PRINCIPLE: grade = OVERALL/DIFFUSE reticulin density across the whole bag; NOT a few patches. DO NOT weight by feature-norm ||h|| (grade-uninformative, Spearman ~0, nuisance). Warm-start fibrosis axis available at data/prototypes_virchow2_reti_train_seed2.pt['axis'] (1280-d, seed=2 train-only). Read src/models/simple_mil.py for the baseline gated-attention idiom and a52_fibrosis_coverage.py for warm-axis loading + coverage. Each candidate ships an ablation that removes EXACTLY its active ingredient (import the main Model + flip a flag, like a57 imports a56). Keep capacity ~ baseline 197K (higher overfits this 214-ROI val cohort).
`;
const STATUS = `
CONTEXT: 80 aggregators (a01-a80) + 3 backbones + fusion all TIE the baseline at the seed=2 fold; the bottleneck is VARIANCE (val<->test decoupling), and the baseline itself swings ±0.01 val at fixed seed. These 3 candidates are principled FRESH angles, each motivated by a specific diagnostic finding. Do NOT reproduce tried mechanisms (rank/norm-salience, top-k, plain coverage/moments/quantiles/pairwise/dist-match, per-patch severity, consensus, PMA).
`;

phase('Build')

const BUILD_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['main_file','ablation_file','param_count','sanity_pass','leakage_ok','sanity_detail','notes'],
  properties: {
    main_file: { type: 'string' }, ablation_file: { type: 'string' },
    param_count: { type: 'integer' }, sanity_pass: { type: 'boolean' }, leakage_ok: { type: 'boolean' },
    sanity_detail: { type: 'string' }, notes: { type: 'string' },
  },
}

const SPECS = [
  { ids:['a81','a82'], name:'l2norm_coverage', spec:`L2-NORMALIZE each patch feature f_i -> f_i/||f_i|| BEFORE aggregation (||h|| is a grade-irrelevant nuisance per the diagnostic; removing the magnitude lets the aggregator focus on DIRECTION, which carries the grade signal). Then run the baseline gated-attention bottleneck (Linear(1280->128)+ReLU+Dropout -> Ilse gated attention -> weighted mean -> Linear head) on the L2-normalized features. MAIN a81 = L2-normalize then gated-attention. ABLATION a82 = identical but NO L2-normalization (= the plain baseline). Isolates 'does explicitly removing the grade-irrelevant norm dimension help/stabilise?'. Import baseline idiom from simple_mil.py.` },
  { ids:['a83','a84'], name:'graph_smooth_pool', spec:`Build on a59 (feature-graph smoothness was the session-best, val 0.8128). Compute a patch-similarity graph (row-normalised cosine affinity A in BOTTLENECK space h, [N,N]), do K=2 steps of graph smoothing h <- (1-a)*h + a*A@h (a learnable in [0,1]) so fibrosis evidence that is CONTIGUOUS/diffuse is reinforced and isolated outlier patches (incl. bone/artifacts) are smoothed away, then DIFFUSE-pool: mean of smoothed h -> Linear head (or coverage along a warm-started direction). MAIN a83 = K=2 graph smoothing then diffuse pool. ABLATION a84 = K=0 (no smoothing) = plain diffuse pool. Read a59_spatial_contiguity_pool.py first and genuinely IMPROVE on it (it already exists; do NOT duplicate — extend with multi-step learnable smoothing). Keep O(N^2) affinity (N<=~200, fine).` },
  { ids:['a85','a86'], name:'diffuse_attention', spec:`The grading principle says grade = DIFFUSE density, NOT a few standout patches — so bias the attention to be SPREAD. Parameterise the pooling weights as w = (1-lambda)*uniform(1/N) + lambda*softmax(scores), with lambda a learnable scalar in [0,1] INITIALISED SMALL (~0.1) so the model STARTS at mean-pooling (fully diffuse) and may only sharpen toward gated attention if the data demands it. Use the baseline gated-attention scorer for 'scores'. MAIN a85 = uniform-anchored diffuse attention (lambda init 0.1). ABLATION a86 = lambda fixed = 1 (= standard gated attention). Isolates 'does an explicit diffuseness prior on the pooling weights (start diffuse, sharpen only if needed) help/stabilise on this cohort?'.` },
]

const builds = await parallel(SPECS.map(sp => () => agent(
  `Build a fresh grading-aligned MIL aggregator + ablation, then sanity/leakage-check. I (orchestrator) will screen at seed=2.
${HARD}${GRADING}${STATUS}
CANDIDATE (ids ${sp.ids[0]}=main, ${sp.ids[1]}=ablation):
${sp.spec}
STEPS:
1. MAIN -> src/models/novelty_attempts/${sp.ids[0]}_${sp.name}.py (class Model + KWARGS). 2. ABLATION -> src/models/novelty_attempts/${sp.ids[1]}_<...>.py importing the main Model and flipping the documented flag.
3. SANITY (Bash, no training): import both, Model(**KWARGS), forward on random [N,1280]*6 with N in {37,120,1}; assert logits.view(-1).numel()==1 and FINITE (no NaN/inf — this round a prior candidate died from NaN, so test with feature scale ~6 like real norms); count params (~197K target); loss.backward() gives finite grads; permutation-invariant at eval. 4. LEAKAGE: forward uses ONLY 'features' (+ warm axis if used). Set leakage_ok.
Return main_file, ablation_file, param_count, sanity_pass, leakage_ok, sanity_detail, notes (exact --novelty_id to screen). Do NOT train, edit the trainer, or run git.`,
  { label:`build:${sp.ids[0]}_${sp.name}`, phase:'Build', schema:BUILD_SCHEMA }
)))

return { builds: builds.filter(Boolean) }
