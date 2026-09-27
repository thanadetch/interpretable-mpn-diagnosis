export const meta = {
  name: 'mpn-round9-design',
  description: 'Design panel for mechanisms that escape the documented traps, build top-2 + ablations',
  phases: [{ title: 'Design' }, { title: 'Build' }],
}

const HARD = `
HARD CONSTRAINTS (trainer CANNOT be edited):
- Drop-in module src/models/novelty_attempts/a<NN>_<name>.py: class Model(nn.Module) + KWARGS.
- Trainer calls logits,_,_=model(features); features=ONE bag [N,1280] Virchow2. Return 3-tuple (logits,aux,aux); logits.view(-1).numel()==1; RAW logits. DETERMINISTIC at inference. Permutation- & bag-size-invariant. Loss=SmoothL1 (NO aux loss). KWARGS input_dim=1280,num_classes=1. forward uses ONLY 'features'. Capacity <=~197K.
`;
const GRADING = `GRADING: grade=OVERALL/DIFFUSE density; no ||h|| weighting. Warm axis at data/prototypes_virchow2_reti_train_seed2.pt['axis'] (seed=2 train-only, init/buffer use OK; note: anything seed=2-axis-derived is favorable-leaky cross-fold and washes out — a103 proved warm-start helps ONLY seed=2). Read simple_mil.py + a83_graph_smooth_pool.py.`;
const TRAPS = `THE TWO DOCUMENTED TRAPS your design MUST escape (this is why ~104 modules tied):
(1) UNDERFIT trap: low-DOF readouts (coverage 1.3K, scatter 6, frozen-axis 2-param, raw-projection quantiles) UNDERFIT (val 0.0-0.77).
(2) VAL-OVERFIT trap: richer/structured mechanisms at baseline capacity (subspace-proj a93 val 0.841, spectral a99 0.840, heavy-reg a91 0.839) LIFT seed=2 val but TANK test -> they val-overfit the hard seed=2 fold; paired Δ≈0 or negative.
The warm diffuse mean-pool / gated-attention baseline sits at the U-shape capacity OPTIMUM (reproducibly 0.827/0.959 at seed=2 but ties paired). EVERY tried mechanism (mean/attention/topk/rank/norm/coverage/severity/consensus/graph-smooth/moments/quantiles/sorted-curve/pairwise/dist-match/scatter/spectral/subspace/L2-norm/fusion/ensemble/heavy-reg/ordinal/warm-start/robust-trimmed) ties baseline on PAIRED cross-fold Δ.`;
const GOAL = `GOAL: beat the gated-attention baseline on PAIRED cross-fold Δ (most of folds {0,1,2,3,42}, BOTH val & test). A seed=2 gate pass is NOT enough (a84/a96 cleared seed=2 yet lost paired). Be HONEST if an idea is a relabel of a tried mechanism or falls into a trap.`;

phase('Design')
const DS={type:'object',additionalProperties:false,required:['name','one_line','mechanism','escapes_traps','why_orthogonal','paired_arg','ablation','feasible','est_params'],properties:{name:{type:'string'},one_line:{type:'string'},mechanism:{type:'string'},escapes_traps:{type:'string',description:'concretely how it avoids BOTH the underfit and val-overfit traps'},why_orthogonal:{type:'string'},paired_arg:{type:'string'},ablation:{type:'string'},feasible:{type:'boolean'},est_params:{type:'integer'}}}
const LENSES=[
  'CONSTRAINED-CAPACITY: a mechanism RICHER than mean-pool whose extra DOF are HARD-CONSTRAINED (weight-tied / monotone / non-negative / symmetry-locked) so they CANNOT fit fold-specific val noise — escaping the val-overfit trap while having more expressive power than the 2-6 param underfitters.',
  'MULTI-VIEW (one backbone, no fusion infra): combine two complementary DETERMINISTIC views of the SAME virchow2 features (e.g. the feature AND its within-bag RANK-transform, or raw AND L2-normalised) so grade evidence robust across both views is reinforced — distinct from a68 backbone-fusion and a79 multi-mechanism-ensemble; argue why it would not just average to a tied mean.',
  'WILDCARD-ESCAPE: any deterministic mechanism you can rigorously argue escapes BOTH traps and could give a PAIRED gain (e.g. a parameter-free closed-form statistic the head reads with <=baseline DOF, or a regularised-toward-mean rich pool). Bold but honest; reject if it is a known tied relabel.',
]
const designsRaw=await parallel(LENSES.map((lens,i)=> ()=>agent(
  `Propose ONE grading-aligned MIL aggregator for MPN reticulin grading through THIS lens, designed to ESCAPE the documented traps.
LENS: ${lens}
${HARD}${GRADING}${TRAPS}${GOAL}
Return full forward math + a precise ablation removing EXACTLY the active ingredient. Capacity <=~197K, deterministic. Explicitly state how it escapes BOTH traps.`,
  {label:`design:${i}`, phase:'Design', schema:DS}
)))
const designs=designsRaw.filter(Boolean)
log(`Design: ${designs.length} proposed`)

phase('Build')
const JS={type:'object',additionalProperties:false,required:['picked_indices','reasoning','ranking'],properties:{picked_indices:{type:'array',items:{type:'integer'}},reasoning:{type:'string'},ranking:{type:'array',items:{type:'object',additionalProperties:false,required:['index','name','score','verdict'],properties:{index:{type:'integer'},name:{type:'string'},score:{type:'number'},verdict:{type:'string'}}}}}}
const judge=await agent(
  `Pick the 2 designs most likely to give a GENUINE paired-cross-fold win, scoring HARD on: does it truly escape BOTH traps (underfit / val-overfit), genuine orthogonality vs the tried list, feasibility, low/constrained capacity. Heavily penalize trap-fallers and tied relabels.
Designs: ${JSON.stringify(designs.map((d,i)=>Object.assign({index:i},d)))}`,
  {label:'judge', phase:'Build', schema:JS}
)
log(`Judge picked ${JSON.stringify(judge&&judge.picked_indices)}`)
const IDP=[['a107','a108'],['a109','a110']]
const picks=((judge&&judge.picked_indices)||[0,1]).slice(0,2).map(i=>designs[i]).filter(Boolean)
const BS={type:'object',additionalProperties:false,required:['main_file','ablation_file','param_count','sanity_pass','leakage_ok','sanity_detail','notes'],properties:{main_file:{type:'string'},ablation_file:{type:'string'},param_count:{type:'integer'},sanity_pass:{type:'boolean'},leakage_ok:{type:'boolean'},sanity_detail:{type:'string'},notes:{type:'string'}}}
const buildsRaw=await parallel(picks.map((d,k)=> ()=>{
  const safe=(d.name||('cand'+k)).toLowerCase().replace(/[^a-z0-9]+/g,'_').replace(/^_+|_+$/g,'').slice(0,28)
  const mainId=IDP[k][0], ablId=IDP[k][1]
  return agent(
    `Build this design + ablation for MPN grading, then sanity/leakage/determinism-check.
${HARD}${GRADING}
DESIGN: ${JSON.stringify(d)}
STEPS: 1. MAIN -> src/models/novelty_attempts/${mainId}_${safe}.py. 2. ABLATION -> src/models/novelty_attempts/${ablId}_${safe}_ablation.py importing main Model + flipping the flag.
3. SANITY (Bash, no training): import both, Model(**KWARGS), forward random [N,1280]*6 for N in {37,120,1}; assert logits.view(-1).numel()==1 + FINITE; params; loss.backward finite grads; permutation-invariant + DETERMINISTIC at eval (two passes identical). 4. LEAKAGE: forward uses ONLY 'features'.
Return all schema fields incl exact --novelty_id. Do NOT train, edit trainer, or run git.`,
    {label:`build:${mainId}_${safe}`, phase:'Build', schema:BS})
}))
const builds=buildsRaw.filter(Boolean)
return { designs, judge, builds }
