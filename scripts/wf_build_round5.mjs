export const meta = {
  name: 'mpn-build-round5',
  description: 'Build the 2 unbuilt round-4 orthogonal designs (scatter-eigenstructure / spectral-lowfreq) + ablations',
  phases: [{ title: 'Build' }],
}

const HARD = `
HARD CONSTRAINTS (trainer CANNOT be edited):
- Drop-in module src/models/novelty_attempts/a<NN>_<name>.py: class Model(nn.Module) + KWARGS.
- Trainer calls logits,_,_ = model(features); features = ONE bag [N,1280] Virchow2. Return 3-tuple (logits,aux,aux); logits.view(-1).numel()==1; RAW logits (trainer rounds+clips at eval). DETERMINISTIC at inference. Permutation- & bag-size-invariant. Loss=SmoothL1 (NO aux loss). KWARGS input_dim=1280, num_classes=1. forward uses ONLY 'features'. Capacity <=~197K.
`;
const GRADING = `GRADING: grade=OVERALL/DIFFUSE density; no ||h|| weighting. Warm axis at data/prototypes_virchow2_reti_train_seed2.pt['axis'] (seed=2 train-only, init/buffer use OK). Read a83_graph_smooth_pool.py for the bottleneck+diffuse-pool+warm-start idiom. Ship the ablation that removes EXACTLY the active ingredient (import main Model + flip a flag).`;

phase('Build')
const BUILD_SCHEMA = {type:'object',additionalProperties:false,required:['main_file','ablation_file','param_count','sanity_pass','leakage_ok','sanity_detail','notes'],properties:{main_file:{type:'string'},ablation_file:{type:'string'},param_count:{type:'integer'},sanity_pass:{type:'boolean'},leakage_ok:{type:'boolean'},sanity_detail:{type:'string'},notes:{type:'string'}}}

const SPECS = [
  { ids:['a97','a98'], name:'scatter_eigenstructure', spec:`MEAN-FREE multivariate covariance-SHAPE readout. Frozen orthonormal basis Q in R^{1280 x K}, K=8 (buffer, no grad): column 0 = warm train-only fibrosis axis (seed=2 prototypes['axis']); columns 1..7 = deterministic random dirs orthonormalised via QR (fixed frame_seed=2). Forward: P = f@Q [N,K]; mu=mean_i P_i; Pc=P-mu (CENTER -> mean removed); C=(Pc^T@Pc)/N [K,K] within-bag covariance. Read invariants: tr=trace(C); along=C[0,0]; off=tr-along; aniso=along/(tr+eps); cross=||C[0,1:]||_2. Descriptor g=[log1p(tr),log1p(along),log1p(off),aniso,log1p(cross)] (log1p for ~6-scale stability). y=Linear(5->1)(g) RAW logit. Warm-start head weight=[0.2 on log1p(tr), 0.2 on log1p(along), 0 else], bias=1.5. N==1 -> C=0 -> g=0 -> y=bias (safe). ~6 trainable params (Q frozen). It is MEAN-FREE by construction (centroid removed) so it CANNOT collapse to mean-pool; reads OFF-diagonal cross-cov + off-axis spread that a 1-D moment (a64/a66) cannot. MAIN a97 = K=8. ABLATION a98 = subspace_dim=1 (K=1): Q=[axis] only, C is 1x1, g reads only single-axis variance (= a64 spread term) + constants — isolates 'does the multivariate covariance SHAPE beat a 1-D variance?'. a98 imports a97's Model with subspace_dim=1.` },
  { ids:['a99','a100'], name:'spectral_lowfreq', spec:`Closed-form SPECTRAL readout of the leading non-trivial graph mode, blended with the diffuse mean. h_i=Dropout(ReLU(Linear(1280->128) f_i)). z_dc=mean_i h_i (DC/diffuse mode = mean-pool). e_i=h_i.detach()/||.|| (cosine geometry); ec_i=e_i-mean(e) (CENTER => remove trivial DC eigenvector). Leading eigenvector u (||u||=1) of centered patch Gram G=Ec@Ec^T via T=8 fixed power-iteration steps WITHOUT materialising G (proj=Ec@v; u=proj/||proj||; v=Ec^T@u/||.||); u detached. membership m_i=sqrt(N)*u_i. Deterministic SIGN FIX: s=sign((1/N)sum_i m_i*(f_i . fib_axis)) [seed=2 train axis; fallback sign(mean m)]; m<-s*m. z_spec=(1/N)sum_i m_i*(h_i - z_dc) (NON-detached h so gradients reach encoder). gamma=sigmoid(gamma_logit) in (0,1) INIT 0.1. z = z_dc + gamma*z_spec. y=Linear(128->1)(z) RAW logit, warm-started along train axis. MAIN a99 = gamma learnable init 0.1 (starts ~mean-pool, can add the spectral mode). ABLATION a100 = gamma_fixed=0.0 (buffer, no gamma_logit) => z=z_dc = plain diffuse mean-pool + byte-identical head. Isolates 'does adding the leading centered-graph spectral mode beat the diffuse mean?'. ~164K params. a100 imports a99's Model with gamma_fixed=0.` },
]
const buildsRaw = await parallel(SPECS.map(sp => () => agent(
  `Build this round-4 design + ablation for MPN reticulin grading, then sanity/leakage/determinism-check.
${HARD}${GRADING}
CANDIDATE (ids ${sp.ids[0]}=main, ${sp.ids[1]}=ablation):
${sp.spec}
STEPS: 1. MAIN -> src/models/novelty_attempts/${sp.ids[0]}_${sp.name}.py. 2. ABLATION -> src/models/novelty_attempts/${sp.ids[1]}_${sp.name}_ablation.py importing main Model + flipping the flag.
3. SANITY (Bash, no training): import both, Model(**KWARGS), forward random [N,1280]*6 for N in {37,120,1}; assert logits.view(-1).numel()==1 + FINITE; params; loss.backward finite grads; permutation-invariant + DETERMINISTIC at eval (model.eval(); two passes identical — critical given power-iteration/eigenvector ops). 4. LEAKAGE: forward uses ONLY 'features'. Return all schema fields incl exact --novelty_id. Do NOT train, edit trainer, or run git.`,
  { label:`build:${sp.ids[0]}_${sp.name}`, phase:'Build', schema:BUILD_SCHEMA }
)))
const builds = buildsRaw.filter(Boolean)
return { builds }
