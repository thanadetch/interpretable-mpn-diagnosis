export const meta = {
  name: 'mpn-round4-design',
  description: 'Design panel for mechanisms orthogonal to all tried, then build top-2 + ablations (paired-robustness-targeted)',
  phases: [{ title: 'Design' }, { title: 'Build' }],
}

const HARD = `
HARD CONSTRAINTS (trainer CANNOT be edited):
- Drop-in module src/models/novelty_attempts/a<NN>_<name>.py: class Model(nn.Module) + KWARGS.
- Trainer calls logits,_,_ = model(features); features = ONE bag [N,1280] Virchow2. Return 3-tuple (logits,aux,aux); logits.view(-1).numel()==1; RAW logits (trainer rounds+clips at eval). DETERMINISTIC at inference. Permutation- & bag-size-invariant. Loss=SmoothL1 in trainer (NO aux loss). KWARGS input_dim=1280, num_classes=1. forward uses ONLY 'features'. Capacity <= ~197K.
`;
const GRADING = `GRADING: grade = OVERALL/DIFFUSE reticulin density; not a few patches; do NOT weight by ||h|| (Spearman ~0). Warm-start fibrosis axis at data/prototypes_virchow2_reti_train_seed2.pt['axis'] (seed=2 train-only, init-only use OK). Read simple_mil.py + a83_graph_smooth_pool.py for idioms.`;
const TRIED = `ALREADY TRIED & TIED (paired Δ≈0) — your mechanism MUST be orthogonal to ALL of these: gated-attention (baseline), mean-pool, top-k, rank/norm-salience, single+cascade coverage, per-patch severity (linear/nonlinear/robust-soft-median), consensus/self-agreement, spatial-contiguity, PMA/set-transformer, soft-rank, distribution moments (mean/var/skew), soft-quantiles, pairwise Gini-spread, distribution-matching to per-grade refs, 2/3-way backbone fusion, multi-mechanism ensemble, L2-norm, iterative graph-smoothing, robust-trimmed (geometric-median) pool, cumulative-link/ordinal head, heavy-dropout+input-noise. 8 times the simpler/mean ablation matched-or-beat the novel main.`;
const GOAL = `GOAL: a candidate that beats the gated-attention baseline on PAIRED cross-fold Δ (most of folds {0,1,2,3,42}, both val & test) — NOT a seed=2 lottery (a84 cleared both seed=2 gates yet lost 4/5 folds paired). Target GENERALISATION/variance, low capacity. Be honest if an idea is a relabel of something tried.`;

phase('Design')

const DESIGN_SCHEMA = {
  type:'object', additionalProperties:false,
  required:['name','one_line','mechanism','why_orthogonal','paired_robustness_arg','ablation','feasible','est_params'],
  properties:{
    name:{type:'string'}, one_line:{type:'string'}, mechanism:{type:'string'},
    why_orthogonal:{type:'string', description:'concretely why NOT any tried mechanism'},
    paired_robustness_arg:{type:'string', description:'why it should help PAIRED cross-fold (not just seed=2)'},
    ablation:{type:'string'}, feasible:{type:'boolean'}, est_params:{type:'integer'},
  },
}
const LENSES = [
  'feature-SUBSPACE: learn/select a low-rank grade-relevant subspace of the 1280-d features (e.g. a learned projection to k<<128 dims, or drop noisy dims) BEFORE diffuse pooling, to cut overfitting on grade-irrelevant directions — orthogonal to all pooling-mechanism changes (this changes the REPRESENTATION, not the pool).',
  'set-GEOMETRY / 2nd-moment: summarise the patch cloud by its SHAPE (e.g. the spread/covariance eigen-structure or the trace of within-bag scatter along the fibrosis axis) as a robust diffuse-density signature — distinct from mean/var-of-projection (a66) by using the multivariate scatter, not a 1-D moment.',
  'spectral-graph: pool the LOW-FREQUENCY (smooth) component of the patch-similarity graph via its leading eigenvector / Laplacian spectral filter (a CLOSED-FORM spectral readout, NOT a83 iterative smoothing) — captures the dominant coherent tissue mode, suppressing high-frequency outlier patches.',
  'wildcard: any mechanism you can argue is genuinely ABSENT from the tried list and targets PAIRED-fold generalisation (e.g. invariance/augmentation-consistency built into the forward, optimal-transport barycentre of patches, a calibrated prototype distance with shrinkage). Must be deterministic + low-capacity.',
]
const designsRaw = await parallel(LENSES.map((lens,i)=> ()=>agent(
  `Propose ONE grading-aligned MIL aggregator for MPN reticulin grading through THIS lens, genuinely ORTHOGONAL to everything tried.
LENS: ${lens}
${HARD}${GRADING}${TRIED}${GOAL}
Return the full forward-pass math + a precise ablation that removes EXACTLY the active ingredient. Keep capacity <=~197K. Be concrete and honest about novelty.`,
  {label:`design:${i}`, phase:'Design', schema:DESIGN_SCHEMA}
)))
const designs = designsRaw.filter(Boolean)
log(`Design: ${designs.length} proposed`)

phase('Build')
const JUDGE_SCHEMA={type:'object',additionalProperties:false,required:['picked_indices','reasoning','ranking'],properties:{
  picked_indices:{type:'array',items:{type:'integer'}},
  reasoning:{type:'string'},
  ranking:{type:'array',items:{type:'object',additionalProperties:false,required:['index','name','score','verdict'],properties:{index:{type:'integer'},name:{type:'string'},score:{type:'number'},verdict:{type:'string'}}}},
}}
const judge = await agent(
  `Pick the 2 designs most likely to give a GENUINE paired-cross-fold win for MPN grading. Score on: genuine orthogonality vs the tried list, paired-robustness plausibility (not seed=2 lottery), feasibility (only [N,1280], deterministic, no trainer edit), low capacity. Penalize relabels of tried mechanisms.
Designs: ${JSON.stringify(designs.map((d,i)=>Object.assign({index:i},d)))}`,
  {label:'judge', phase:'Build', schema:JUDGE_SCHEMA}
)
log(`Judge picked ${JSON.stringify(judge && judge.picked_indices)}`)

const IDP=[['a93','a94'],['a95','a96']]
const picks=((judge&&judge.picked_indices)||[0,1]).slice(0,2).map(i=>designs[i]).filter(Boolean)
const BUILD_SCHEMA={type:'object',additionalProperties:false,required:['main_file','ablation_file','param_count','sanity_pass','leakage_ok','sanity_detail','notes'],properties:{main_file:{type:'string'},ablation_file:{type:'string'},param_count:{type:'integer'},sanity_pass:{type:'boolean'},leakage_ok:{type:'boolean'},sanity_detail:{type:'string'},notes:{type:'string'}}}
const buildsRaw=await parallel(picks.map((d,k)=> ()=>{
  const safe=(d.name||('cand'+k)).toLowerCase().replace(/[^a-z0-9]+/g,'_').replace(/^_+|_+$/g,'').slice(0,28)
  const mainId=IDP[k][0], ablId=IDP[k][1]
  return agent(
    `Build this design + ablation for MPN grading, then sanity/leakage/determinism-check. I screen at seed=2 then auto-audit paired.
${HARD}${GRADING}
DESIGN: ${JSON.stringify(d)}
STEPS: 1. MAIN -> src/models/novelty_attempts/${mainId}_${safe}.py (Model + KWARGS; warm-start from prototypes...['axis'] defensively if used). 2. ABLATION -> src/models/novelty_attempts/${ablId}_${safe}_ablation.py importing main Model + flipping the documented flag.
3. SANITY (Bash, no training): import both, Model(**KWARGS), forward random [N,1280]*6 for N in {37,120,1}; assert logits.view(-1).numel()==1 + FINITE; params; loss.backward finite grads; permutation-invariant + DETERMINISTIC at eval (model.eval(); two passes identical). 4. LEAKAGE: forward uses ONLY 'features'.
Return main_file, ablation_file, param_count, sanity_pass, leakage_ok, sanity_detail, notes (exact --novelty_id). Do NOT train, edit trainer, or run git.`,
    {label:`build:${mainId}_${safe}`, phase:'Build', schema:BUILD_SCHEMA})
}))
const builds = buildsRaw.filter(Boolean)

return { designs, judge, builds }
