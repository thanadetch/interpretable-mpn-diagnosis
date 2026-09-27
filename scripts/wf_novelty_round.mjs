export const meta = {
  name: 'mpn-novelty-round',
  description: 'Audit + design + build next-round grading-aligned MIL aggregator candidates (feasibility-aware) for MPN reticulin fibrosis grading',
  phases: [
    { title: 'Audit' },
    { title: 'Design' },
    { title: 'Build' },
  ],
}

const HARD = `
HARD CONSTRAINTS (violating any makes the candidate unusable — the trainer CANNOT be edited):
- The novelty is a drop-in module at src/models/novelty_attempts/a<NN>_<name>.py exposing class Model(nn.Module) and a KWARGS dict.
- The trainer calls: logits, _, _ = model(features)  (and sometimes model(features, return_attention=False)).
  * 'features' is ONE bag: a float tensor of shape [N, 1280] (Virchow2 frozen features; N patches vary per bag).
  * forward MUST return a 3-tuple (logits, aux1, aux2). logits has shape [1, 1] for regression. aux1/aux2 may be None or attention.
  * The model receives ONLY 'features'. NO patch coordinates, NO scale tags, NO second backbone are passed in.
- Loss is SmoothL1Loss computed INSIDE the trainer on logits — you CANNOT add an auxiliary loss term (that would require editing the trainer). Any "aux loss" idea is INFEASIBLE.
- Output must be clamped to [0,3] (regression target range G0..G3). Permutation-invariant, bag-size-invariant, deterministic at inference. Frozen features only.
- KWARGS default: input_dim=1280, num_classes=1, and whatever hyperparams the design needs.
`;

const GRADING = `
GRADING PRINCIPLE (pathology prior the design MUST respect — this is what the advisor cares about):
- Grade = OVERALL / DIFFUSE density of the reticulin fibre meshwork across the whole bag. It is a bag-wide property, NOT a few standout patches.
- DO NOT weight patches by feature-norm ||h||. A diagnostic (results/diag/norm_vs_grade.md) proved ||h|| is grade-uninformative on Virchow2 (Spearman ~0; it tracks tissue-vs-background, not fibrosis). Any norm-ranking/norm-weighting design is RULED OUT.
- The signal that DOES track grade is coverage/density along a learned fibrosis direction v = mean(G2/G3 features) - mean(G0/G1 features) (held-out coverage Spearman +0.84). A warm-start prototype axis exists at data/prototypes_virchow2_reti_train_seed2.pt (keys: axis[1280], prototypes{0..3}).
- ROBUSTNESS over single-seed peak: a candidate counts only if it can plausibly improve ACROSS seeds, not a one-seed lottery (the a40 failure mode). Always include an ablation companion that removes the active ingredient.
`;

const STATUS = `
SEARCH STATUS: ~25 aggregators a01-a51 (logged in NOVELTY_NOTES.md section 9) plus this session's a52-a63 have all FAILED to beat the locked baseline (val_qwk 0.8182 / test_qwk 0.9476 at seed=2; same-env reproduction baseline ~0.7888 val). RECURRING PATTERN: nonlinear per-patch designs collapse to (or lose to) their mean-pool linear ablation (a56 nonlinear val 0.784 < a57 linear val 0.799). The single-backbone aggregator space appears near-exhausted with a val ceiling ~0.78-0.79. Do NOT re-propose anything already tried.
`;

phase('Audit')

const AUDIT_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['tried', 'gaps'],
  properties: {
    tried: {
      type: 'array',
      items: {
        type: 'object', additionalProperties: false,
        required: ['mechanism', 'example_ids', 'outcome'],
        properties: {
          mechanism: { type: 'string' },
          example_ids: { type: 'string' },
          outcome: { type: 'string' },
        },
      },
    },
    gaps: {
      type: 'array',
      items: {
        type: 'object', additionalProperties: false,
        required: ['name', 'mechanism', 'grading_rationale', 'why_untried', 'distinct_from'],
        properties: {
          name: { type: 'string' },
          mechanism: { type: 'string' },
          grading_rationale: { type: 'string' },
          why_untried: { type: 'string' },
          distinct_from: { type: 'string' },
        },
      },
    },
  },
}

const FUSION_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['feasible', 'patch_correspondence', 'preprocessing_plan', 'aggregator_idea', 'thesis_framing', 'risks', 'recommend'],
  properties: {
    feasible: { type: 'boolean' },
    patch_correspondence: { type: 'string' },
    preprocessing_plan: { type: 'string' },
    aggregator_idea: { type: 'string' },
    thesis_framing: { type: 'string' },
    risks: { type: 'string' },
    recommend: { type: 'boolean' },
  },
}

const auditPrompt = `You are auditing the novelty search for an MPN reticulin fibrosis GRADING thesis so the next design round does not repeat past work.
${HARD}${GRADING}${STATUS}
TASK:
1. Read NOVELTY_NOTES.md (especially section 9, the batch log) and list the source files: run "ls src/models/novelty_attempts/" and skim the docstrings of a52-a63 (this session's batch) plus a representative sample of a01-a51.
2. Produce a deduplicated map of DISTINCT mechanisms already tried (top-k, rank-softmax, norm-ranking, multi-resolution, coverage/extent, per-patch severity, soft-rank, PMA, consensus, spatial-contiguity, gated-attention baseline, etc.) with example ids and one-line outcomes.
3. Then identify genuinely UNTRIED, FEASIBLE (over [N,1280] only), grading-aligned aggregator gaps — ideas that express DIFFUSE bag-wide fibrosis density and do NOT weight by ||h||. Be concrete (the math). Mark how each differs from the closest tried mechanism. Aim for 4-6 high-quality gaps; quality over quantity. If you believe the space is genuinely exhausted, say so in the gaps' rationale but still propose the least-explored directions.`;

const fusionPrompt = `You are assessing whether MULTI-BACKBONE FEATURE FUSION is a feasible, thesis-worthy new axis for the MPN reticulin GRADING novelty search. This is the one direction that could raise the ~0.78-0.79 val ceiling because it adds NEW signal, while staying within the hard rules (fusion done in PREPROCESSING, not in the trainer).
${HARD}${GRADING}
TASK:
1. Confirm feature dirs exist: data/features_virchow2_reti, data/features_uni2_reti, data/features_titan_reti (dims 1280 / 1536 / 768).
2. VERIFY patch correspondence: write a tiny python snippet (run via Bash, read-only) that, for ONE matching slide (same .pt filename across two backbone dirs), loads both .pt files and checks whether the feature tensors have the SAME number of patches N and plausibly the same patch ORDER (same extraction pipeline -> same OD-filtered patch grid). Report what you find. (Look at how src/data/bag_dataset.py loads a .pt and which key holds the feature tensor.)
3. If correspondence holds, spec a preprocessing approach to build a FUSED feature dir (e.g. concat virchow2+uni2 per patch -> [N, 2816]) that the EXISTING trainer can consume by pointing --backbone/--data_root at it WITHOUT any trainer edit. Confirm the trainer's input_dim is driven by KWARGS/the module, so a fused module just sets input_dim=2816.
4. Propose a grading-aligned aggregator (coverage/density along a fibrosis direction learned on fused features) and a thesis framing that keeps the GRADING concept central (not "just more features").
5. Give a clear recommend: true/false with risks (disk, RAM, whether fusion is thesis-defensible as a grading contribution).
Do NOT build anything or write feature files. Read-only investigation + plan only.`;

const auditResults = await parallel([
  () => agent(auditPrompt, { label: 'audit:tried+gaps', phase: 'Audit', schema: AUDIT_SCHEMA }),
  () => agent(fusionPrompt, { label: 'audit:fusion-feasibility', phase: 'Audit', schema: FUSION_SCHEMA }),
])
const audit = auditResults[0]
const fusion = auditResults[1]

const nGaps = audit && audit.gaps ? audit.gaps.length : 0
log(`Audit: ${nGaps} untried gaps. Fusion feasible=${fusion && fusion.feasible} recommend=${fusion && fusion.recommend}`)

phase('Design')

const DESIGN_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['name', 'one_line', 'mechanism', 'grading_alignment', 'novelty_vs_tried', 'ablation_companion', 'feasibility_certain', 'robustness_argument', 'est_params'],
  properties: {
    name: { type: 'string' },
    one_line: { type: 'string' },
    mechanism: { type: 'string' },
    grading_alignment: { type: 'string' },
    novelty_vs_tried: { type: 'string' },
    ablation_companion: { type: 'string' },
    feasibility_certain: { type: 'boolean' },
    robustness_argument: { type: 'string' },
    est_params: { type: 'integer' },
  },
}

const LENSES = [
  'distributional: model the DISTRIBUTION of per-patch fibrosis-projection scores (e.g. learnable soft-histogram / quantile-coverage of <f,v>) as the bag descriptor — diffuse density, not a few patches',
  'thresholded-extent: a smooth, calibrated fraction-of-bag-above-fibrosis-threshold with a LEARNED soft threshold and temperature, aggregated as mean coverage (distinct from a52-a55 coverage — justify the difference precisely)',
  'second-order/dispersion: use the MEAN and SPREAD of the fibrosis-projection across patches (diffuse density has characteristic mean+variance signatures per grade) without any ||h|| term',
  'prototype-similarity field: per-patch soft assignment to grade prototypes (data/prototypes_virchow2_reti_train_seed2.pt) then a bag-wide diffuse aggregation of grade-evidence — distinct from gated attention and from coverage',
]

const gapsJson = JSON.stringify(audit && audit.gaps ? audit.gaps : [])

const designThunks = LENSES.map((lens, i) => () => agent(
  `You are a design agent proposing ONE new grading-aligned MIL aggregator for MPN reticulin fibrosis GRADING. Propose a candidate through THIS lens:
LENS: ${lens}
${HARD}${GRADING}${STATUS}
Untried gaps surfaced by the audit (use/refine these, do not contradict them): ${gapsJson}
Requirements: feasible over [N,1280] only; grading-aligned (diffuse density, NO ||h|| weighting); concretely distinct from every tried mechanism; include an ablation companion that removes exactly the active ingredient; argue seed-robustness; keep capacity modest (historically >>197K params overfit this 214-ROI val cohort). Return the full forward-pass math.`,
  { label: `design:${i}`, phase: 'Design', schema: DESIGN_SCHEMA }
))

if (fusion && fusion.feasible && fusion.recommend) {
  const fusionHard = HARD.replace('[N, 1280]', '[N, D_fused]').replace('input_dim=1280', 'input_dim=D_fused')
  const fusionDesignPrompt = `You are a design agent proposing ONE grading-aligned aggregator that runs on FUSED multi-backbone features (the feasibility audit approved this path).
Fusion plan: ${fusion.preprocessing_plan}
Aggregator idea seed: ${fusion.aggregator_idea}
Thesis framing to preserve: ${fusion.thesis_framing}
${fusionHard}${GRADING}
The module is an aggregator over fused features [N, D_fused] (e.g. 2816 = 1280+1536). Keep the GRADING concept central: coverage/density along a fibrosis direction learned on the fused space. Include an ablation companion (single-backbone version of the SAME aggregator, to isolate the fusion contribution). Return full forward math + note that KWARGS must set input_dim=D_fused and the fused feature dir must be built first.`
  designThunks.push(() => agent(fusionDesignPrompt, { label: 'design:fusion', phase: 'Design', schema: DESIGN_SCHEMA }))
}

const designsRaw = await parallel(designThunks)
const designs = designsRaw.filter(Boolean)
log(`Design: ${designs.length} candidates proposed`)

phase('Build')

const JUDGE_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['ranking', 'picked_indices', 'reasoning'],
  properties: {
    ranking: {
      type: 'array',
      items: {
        type: 'object', additionalProperties: false,
        required: ['index', 'name', 'score', 'verdict'],
        properties: {
          index: { type: 'integer' },
          name: { type: 'string' },
          score: { type: 'number' },
          verdict: { type: 'string' },
        },
      },
    },
    picked_indices: { type: 'array', items: { type: 'integer' } },
    reasoning: { type: 'string' },
  },
}

const judgePrompt = `You are the judge selecting which candidates to BUILD for the MPN reticulin GRADING novelty search. Score each on: grading-alignment (diffuse density, no ||h||), GENUINE novelty vs a01-a63, feasibility certainty (uses only allowed inputs / approved fusion), and seed-robustness plausibility. Penalize anything resembling a tried mechanism or anything one-seed-lottery-ish or over-capacity (>~250K params on a 214-ROI val cohort).
Candidates (indexed): ${JSON.stringify(designs.map((d, i) => Object.assign({ index: i }, d)))}
Pick the 2 strongest to build.`

const judge = await agent(judgePrompt, { label: 'judge', phase: 'Build', schema: JUDGE_SCHEMA })
const pickedIdx = judge && judge.picked_indices ? judge.picked_indices : [0, 1]
log(`Judge picked ${JSON.stringify(pickedIdx)}`)

const ID_PAIRS = [['a64', 'a65'], ['a66', 'a67']]
const picks = pickedIdx.slice(0, 2).map(i => designs[i]).filter(Boolean)

const BUILD_SCHEMA = {
  type: 'object', additionalProperties: false,
  required: ['main_file', 'ablation_file', 'param_count', 'sanity_pass', 'sanity_detail', 'notes'],
  properties: {
    main_file: { type: 'string' },
    ablation_file: { type: 'string' },
    param_count: { type: 'integer' },
    sanity_pass: { type: 'boolean' },
    sanity_detail: { type: 'string' },
    notes: { type: 'string' },
  },
}

const buildThunks = picks.map((d, k) => () => {
  const pair = ID_PAIRS[k]
  const mainId = pair[0]
  const ablId = pair[1]
  const rawName = d.name || ('cand' + k)
  const safe = rawName.toLowerCase().replace(/[^a-z0-9]+/g, '_').replace(/^_+|_+$/g, '').slice(0, 28)
  const buildPrompt = `You are building a new novelty module + its ablation companion for the MPN reticulin GRADING search, then sanity-checking them.
${HARD}${GRADING}
DESIGN TO IMPLEMENT: ${JSON.stringify(d)}
STEPS:
1. Read existing modules as the exact template for the Model/KWARGS interface and 3-tuple return: src/models/novelty_attempts/a56_perpatch_severity_pool.py and a52_fibrosis_coverage.py (and a57 to see how an ablation imports the main Model). Match their style, type hints, and the (logits, aux, aux) return contract. logits must be [1,1] and clamped to [0,3].
2. Write the MAIN module to: src/models/novelty_attempts/${mainId}_${safe}.py — expose class Model and KWARGS (input_dim per the design, num_classes=1). If the design warm-starts from data/prototypes_virchow2_reti_train_seed2.pt, load it defensively (fall back to random init if missing).
3. Write the ABLATION companion to: src/models/novelty_attempts/${ablId}_${safe}_ablation.py — identical except the active ingredient is removed (per the design's ablation_companion). Prefer importing the main Model and flipping a flag (like a57 imports a56).
4. SANITY CHECK via Bash (do NOT train): run a python snippet that imports both modules, instantiates Model(**KWARGS), runs forward on a random float tensor of shape [N, input_dim] with N=37 and N=120, asserts the first return element has shape [1,1] and values within [0,3], counts parameters, and confirms loss.backward() populates grads. Report param_count and whether it passed. Also grep your new files to confirm forward uses ONLY 'features' (NO label leakage).
5. Return file paths, param_count, sanity_pass, a short sanity_detail, and notes (incl. whether a fused feature dir must be built first, and the exact --novelty_id to screen it).
Do NOT run training, do NOT touch the trainer, do NOT write to data/ or results/, do NOT run any git command.`
  return agent(buildPrompt, { label: `build:${mainId}`, phase: 'Build', schema: BUILD_SCHEMA })
})

const buildsRaw = await parallel(buildThunks)
const builds = buildsRaw.filter(Boolean)

return { audit, fusion, designs, judge, builds }
