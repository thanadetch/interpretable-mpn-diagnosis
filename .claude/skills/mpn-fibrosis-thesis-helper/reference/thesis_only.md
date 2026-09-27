# Thesis-only material (not in the paper)

## 1. ASGAP — ABMIL with α-entmax attention

- Gated attention of ABMIL with the softmax replaced by α-entmax (Peters et al. 2019); α learnable in (1, 2), initialised at 1.5, converged to 1.506 (it stays at its initialisation from any start). Wording: "a learnable scalar that converged to ≈1.5 on this cohort". Parameters: ABMIL + 1.
- It was the proposed method in the June 2026 proposal; in the thesis it is the sparse/selection design, in the paper it is omitted.

| (val / test QWK) | Virchow2 | UNI2-h | TITAN |
|---|---|---|---|
| ABMIL | .8182 / .9476 | .7743 / .9418 | .7902 / .9584 |
| ASGAP | .7976 / .9588 | .7703 / .9262 | .7968 / .9600 |

- Beats ABMIL on both val and test only on TITAN; on test it wins every metric on Virchow2 and TITAN and loses every metric on UNI2-h (G1 → G0 18 ROIs). Clearest clinical effect: Virchow2 G2 over-grading 6 → 1 ROI (G2 recall 73 → 90).
- Fixed-α grid (α = 1.0 is ABMIL): best α 1.75 (Virchow2), 1.00 (UNI2-h), 1.875 (TITAN); never 2.0; the curve is rough — do not claim a universal optimum.
- Sparsity must be quantified with gini / effective N (α = 1.5 zeroes only ~3–8 of ~44 patches): gini ASGAP vs ABMIL .349/.249 (Virchow2), .471/.248 (UNI2-h), .442/.277 (TITAN).
- Never claim robustness: sparser attention is more sensitive to added foreign patches than ABMIL or mean pooling.

## 2. Averaging, not selection (mechanism analysis)

- Fibrosis axis from real labels: v = mean(train patches of G2, G3) − mean(G0, G1).
- The bag mean projected on v predicts grade (Spearman .834 / .860 / .866), but attention correlates negatively with v within bags (ABMIL −.322 / −.206 / −.115; ASGAP −.224 / −.099 / +.010), and the patches ASGAP zeroes out are more fibrotic than those it keeps (AUC .131 on Virchow2).
- Reading: at ROI scale grading behaves as averaging over the field; attention damps extreme patches rather than selecting fibrotic ones. Never write that attention "highlights fibrosis".
- Three learnable gates converged to "off": ASGAP α 1.5 → 1.506; tissue-group weights stayed ≈ 1/3; a re-injection gate for the whole-ROI view stayed ≈ 0.01.
- Patch features form a continuum (k-means silhouette ≈ .10 at every k, no peak at k = 3): no separable fat/bone/fibrosis groups.

## 3. Patching justification (advisor requirement)

Resizing the whole ROI to one 224×224 image instead of tiling lowers test QWK in all 12 MeanPool/ABMIL × encoder × formulation configurations; large under regression with ABMIL (e.g. Virchow2 .948 → .859, accuracy 85.3% → 63.7%), small under multi-class. Table in `results.md` §8.

## 4. Augmentation (a control variable)

- Main results use no augmentation. Settled position: augmentation is a control variable (same setting for every model, plus a no-augmentation column); named "within-grade CutMix + feature jitter" (CutMix, Yun et al. 2019); one setup paragraph, one results subsection, one discussion sentence; not in the abstract or contributions.
- Published MIL augmentations re-implemented from the authors' code (seed 2, test QWK Δ vs no-aug): ReMix (joint) helps ABMIL on Virchow2 (+.007) and UNI2-h (+.017) and ASGAP on UNI2-h (+.028), hurts TITAN; PseMix (defaults) hurts all 6 cells; image rotation (C4) 1/6. No augmentation is best in 4/6 cells.
- Never claim our augmentation beats ReMix; never justify skipping image augmentation by compute.

## 5. Advisor requirements — status

| requirement | status |
|---|---|
| replace mean pooling with a custom aggregator | ASGAP (thesis), WR-TransMIL (thesis + paper) |
| justify patching | done (§3) |
| holistic density, avoid norm/intensity weighting | respected; measured behaviour is averaging |
| gains should transfer across encoders | WR-TransMIL 3/3 vs TransMIL; UNI2-h favours ABMIL overall |
| test weighted cross-entropy | not done — multi-class runs used uniform weights |
