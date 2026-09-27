# Results (bundled with the mpn-fibrosis-thesis-helper skill)

All numbers are ROI-level, single locked patient-level split (seed 2), frozen encoders, no augmentation, copied from the stored run metrics. Test = 259 ROIs from 10 patients; val = 214 ROIs from 10 patients. "Far" = predictions two or more grades from the reference. All 56 reported checkpoints were re-evaluated with the current code on 2026-09-27 and reproduced exactly.

## 1. Dataset

| Split | Patients | ROIs | G0 | G1 | G2 | G3 |
|---|---:|---:|---:|---:|---:|---:|
| Train | 30 | 857 | 320 (4 pts) | 139 (9) | 249 (12) | 149 (5) |
| Val | 10 | 214 | 21 (2) | 53 (3) | 80 (3) | 60 (2) |
| Test | 10 | 259 | 100 (2) | 49 (3) | 30 (3) | 80 (2) |
| **Total** | **50** | **1,330** | 441 (8) | 241 (15) | 359 (18) | 289 (9) |

Subtypes: ET 13 patients / 279 ROIs, PMF 26 / 687, PV 11 / 364. Test patients: ET13 G0, PV6 G0, ET3 G1, ET4 G1, PMF21 G1, ET2 G2, PMF17 G2, PV1 G2, PMF24 G3, PMF27 G3.

## 2. Main comparison (scalar regression) — the paper's five models (+ ASGAP for the thesis)

**Virchow2**

| Model | Val QWK | Test QWK | MAE | Acc % | Macro F1 | Macro recall % | Correct /259 | Far |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| MeanPool | .8138 | .9209 | .2355 | 76.45 | .7321 | 77.61 | 198 | 0 |
| ABMIL | .8182 | .9476 | .1544 | 85.33 | .8162 | 81.94 | 221 | 2 |
| TransMIL | .7712 | .9604 | .1274 | 87.26 | .8477 | 85.07 | 226 | 0 |
| TransMIL + whole-ROI token | .8135 | .9631 | .1158 | 88.42 | .8615 | 88.49 | 229 | 0 |
| **WR-TransMIL** | .7957 | .9675 | .1042 | 89.58 | .8715 | 88.13 | 232 | 0 |
| *ASGAP (thesis only)* | .7976 | .9588 | .1236 | 88.03 | .8605 | 87.19 | 228 | 1 |

**UNI2-h**

| Model | Val QWK | Test QWK | MAE | Acc % | Macro F1 | Macro recall % | Correct /259 | Far |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| MeanPool | .7927 | .9067 | .2664 | 73.75 | .7124 | 74.80 | 191 | 1 |
| ABMIL | .7743 | .9418 | .1737 | 83.01 | .7865 | 79.82 | 215 | 1 |
| TransMIL | .7734 | .9283 | .2278 | 77.22 | .7232 | 74.57 | 200 | 0 |
| TransMIL + whole-ROI token | .7943 | .9347 | .1931 | 81.47 | .7691 | 77.12 | 211 | 2 |
| **WR-TransMIL** | .7822 | .9371 | .2046 | 79.54 | .7439 | 75.66 | 206 | 0 |
| *ASGAP (thesis only)* | .7703 | .9262 | .2317 | 76.83 | .7141 | 72.32 | 199 | 0 |

**TITAN**

| Model | Val QWK | Test QWK | MAE | Acc % | Macro F1 | Macro recall % | Correct /259 | Far |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| MeanPool | .7163 | .9253 | .2201 | 78.38 | .7413 | 76.19 | 203 | 1 |
| ABMIL | .7902 | .9584 | .1351 | 86.49 | .8325 | 82.88 | 224 | 0 |
| TransMIL | .7841 | .9576 | .1197 | 88.80 | .8627 | 87.56 | 230 | 2 |
| TransMIL + whole-ROI token | .7794 | .9496 | .1467 | 86.10 | .8170 | 81.34 | 223 | 2 |
| **WR-TransMIL** | .7942 | .9616 | .1236 | 87.64 | .8417 | 84.35 | 227 | 0 |
| *ASGAP (thesis only)* | .7968 | .9600 | .1197 | 88.42 | .8611 | 87.05 | 229 | 1 |

### Reserve (not in the paper): other published MIL baselines, same protocol

Decided 2026-09-27 not to tabulate these in the paper (the claim concerns TransMIL's readout; one reference per aggregation family is shown). Keep for a reviewer request. CLAM-SB = attention branch only (instance-clustering loss inactive); DTFD-MIL without its tier-1 loss; DSMIL complete except the regression head.

| Model | Virchow2 val / test | UNI2-h val / test | TITAN val / test | Far (v2/u2/ti) |
|---|---|---|---|---|
| CLAM-SB (attention branch) | .8056 / .9443 | .7710 / .9378 | .7839 / .9506 | 0/1/2 |
| DSMIL | .7631 / .9163 | .7368 / .8943 | .7242 / .9256 | 0/1/0 |
| DTFD-MIL | .7966 / .9517 | .7591 / .9218 | .7687 / .9417 | 0/0/2 |

## 3. Per-grade test recall % (G0 / G1 / G2 / G3)

| Model | Virchow2 | UNI2-h | TITAN |
|---|---|---|---|
| MeanPool | 88.0 / 63.3 / 96.7 / 62.5 | 70.0 / 77.6 / 76.7 / 75.0 | 88.0 / 75.5 / 70.0 / 71.3 |
| ABMIL | 88.0 / 71.4 / 73.3 / 95.0 | 90.0 / 75.5 / 70.0 / 83.8 | 88.0 / 71.4 / 73.3 / 98.8 |
| TransMIL | 88.0 / 71.4 / 83.3 / 97.5 | 90.0 / 42.9 / 86.7 / 78.8 | 89.0 / 79.6 / 86.7 / 95.0 |
| TransMIL + whole-ROI token | 89.0 / 81.6 / 93.3 / 90.0 | 85.0 / 73.5 / 60.0 / 90.0 | 89.0 / 75.5 / 63.3 / 97.5 |
| **WR-TransMIL** | 90.0 / 79.6 / 86.7 / 96.3 | 91.0 / 40.8 / 83.3 / 87.5 | 89.0 / 77.6 / 73.3 / 97.5 |
| *ASGAP (thesis only)* | 87.0 / 75.5 / 90.0 / 96.3 | 90.0 / 55.1 / 66.7 / 77.5 | 89.0 / 77.6 / 86.7 / 95.0 |

## 4. Test confusion matrices (rows = true G0..G3, columns = predicted G0..G3)

| Model | Encoder | true G0 | true G1 | true G2 | true G3 |
|---|---|---|---|---|---|
| ABMIL | Virchow2 | 88 12 0 0 | 7 35 5 2 | 0 2 22 6 | 0 0 4 76 |
| ABMIL | UNI2-h | 90 10 0 0 | 8 37 4 0 | 1 7 21 1 | 0 0 13 67 |
| ABMIL | TITAN | 88 12 0 0 | 10 35 4 0 | 0 4 22 4 | 0 0 1 79 |
| TransMIL | Virchow2 | 88 12 0 0 | 10 35 4 0 | 0 3 25 2 | 0 0 2 78 |
| TransMIL | UNI2-h | 90 10 0 0 | 23 21 5 0 | 0 3 26 1 | 0 0 17 63 |
| TransMIL | TITAN | 89 9 2 0 | 6 39 4 0 | 0 2 26 2 | 0 0 4 76 |
| TransMIL + whole-ROI token | Virchow2 | 89 11 0 0 | 4 40 5 0 | 0 0 28 2 | 0 0 8 72 |
| TransMIL + whole-ROI token | UNI2-h | 85 15 0 0 | 9 36 4 0 | 0 4 18 8 | 0 2 6 72 |
| TransMIL + whole-ROI token | TITAN | 89 10 1 0 | 8 37 3 1 | 0 9 19 2 | 0 0 2 78 |
| **WR-TransMIL** | Virchow2 | 90 10 0 0 | 5 39 5 0 | 0 1 26 3 | 0 0 3 77 |
| **WR-TransMIL** | UNI2-h | 91 9 0 0 | 24 20 5 0 | 0 4 25 1 | 0 0 10 70 |
| **WR-TransMIL** | TITAN | 89 11 0 0 | 6 38 5 0 | 0 4 22 4 | 0 0 2 78 |

## 5. Where the whole-ROI embedding is placed (same vector, different role)

| Placement | Virchow2 val / test | UNI2-h val / test | TITAN val / test | Far (v2/u2/ti) |
|---|---|---|---|---|
| not used (TransMIL) | .7712 / .9604 | .7734 / .9283 | .7841 / .9576 | 0/0/2 |
| extra token | .8135 / .9631 | .7943 / .9347 | .7794 / .9496 | 0/2/2 |
| added to the class token | .7781 / .9177 | .7852 / .9567 | .7727 / .9539 | 1/0/0 |
| **replaces the class token (WR-TransMIL)** | .7957 / .9675 | .7822 / .9371 | .7942 / .9616 | 0/0/0 |

WR-TransMIL beats TransMIL on both val and test on 3/3 encoders; the extra-token placement on 2/3 (loses on TITAN).

## 6. Controls for WR-TransMIL (test QWK; val in brackets)

| Variant | Virchow2 | UNI2-h | TITAN |
|---|---|---|---|
| WR-TransMIL (reference) | .9675 (.7957) | .9371 (.7822) | .9616 (.7942) |
| patch mean as readout (CLIP-style) | .9507 (.7960) | .9342 (.7835) | .9382 (.7803) |
| whole-ROI embedding of a different ROI | .9150 (.8192) | .9454 (.7840) | .9432 (.7798) |
| separate projection for the ROI embedding | .9181 (.7976) | .9397 (.8015) | .9493 (.7694) |
| ROI embedding from a different encoder (TITAN) | .9004 (.7730) | .9321 (.7546) | — |
| concatenated before the head (late fusion) | .9530 (.8124) | .9281 (.7935) | .9537 (.7626) |
| ROI embedding only, no patches | .8752 (.7912) | .8844 (.8066) | .8941 (.6310) |

"—": not applicable (on TITAN the ROI embedding already comes from TITAN). ROI-embedding-only G3 recall collapses (Virchow2 27.5, UNI2-h 42.5, TITAN 58.8) → patches carry most of the prediction.

## 7. Formulation: regression vs multi-class

Multi-class = cross-entropy + label smoothing 0.1 with **uniform class weights (all 1.0)**, checkpoint on val macro-recall. Regression = SmoothL1, rounded and clipped, checkpoint on val QWK.

| Aggregator | Virchow2 QWK reg / mc | Far reg / mc | UNI2-h QWK | Far | TITAN QWK | Far | G0 recall mc (v2/u2/ti) |
|---|---|---|---|---|---|---|---|
| MeanPool | .9209 / .9453 | 0 / 3 | .9067 / .9357 | 1 / 3 | .9253 / .9034 | 1 / 9 | 90.0 / 91.0 / 90.0 |
| ABMIL | .9476 / .9248 | 2 / 5 | .9418 / .9288 | 1 / 2 | .9584 / .9123 | 0 / 6 | 92.0 / 91.0 / 90.0 |
| TransMIL | .9604 / .9522 | 0 / 1 | .9283 / .9401 | 0 / 2 | .9576 / .8667 | 2 / 8 | 90.0 / 89.0 / 90.0 |
| WR-TransMIL | .9675 / .9319 | 0 / 5 | .9371 / .9250 | 0 / 2 | .9616 / .9073 | 0 / 8 | 90.0 / 91.0 / 90.0 |
| *ASGAP (thesis only)* | .9588 / .9367 | 1 / 2 | .9262 / .9303 | 0 / 3 | .9600 / .9220 | 1 / 5 | 93.0 / 90.0 / 90.0 |

Paper headline (MeanPool, ABMIL, TransMIL, WR-TransMIL × 3 encoders = 12 cells): errors ≥2 grades regression **7** vs multi-class **54**, regression fewer in **12/12** cells. On QWK alone regression wins only 9/12 — never use QWK as the formulation headline. Multi-class G0 recall is 89–93% for every aggregator here; the "multi-class → G0 = 0%" result came from the legacy 42-patient cohort and is not a current finding.

## 8. Patching justification (whole ROI resized to one 224×224 image vs tiling)

| Model | Formulation | Virchow2 tiled → resized | UNI2-h | TITAN |
|---|---|---|---|---|
| MeanPool | regression | QWK .921 → .855, acc 76.4 → 64.1 | QWK .907 → .882, acc 73.7 → 70.7 | QWK .925 → .880, acc 78.4 → 68.3 |
| MeanPool | multi-class | QWK .945 → .905, acc 85.3 → 84.2 | QWK .936 → .915, acc 81.5 → 83.8 | QWK .903 → .896, acc 79.5 → 80.3 |
| ABMIL | regression | QWK .948 → .859, acc 85.3 → 63.7 | QWK .942 → .882, acc 83.0 → 70.7 | QWK .958 → .888, acc 86.5 → 69.1 |
| ABMIL | multi-class | QWK .925 → .842, acc 82.6 → 82.6 | QWK .929 → .903, acc 78.8 → 83.8 | QWK .912 → .893, acc 80.7 → 78.8 |

Tiling helps QWK in all 12 configurations; the loss from resizing is large under regression with ABMIL and small under multi-class. Scope the claim to the regression + attention-MIL setting.

## 9. Parameters (effective, regression head)

| Model | Virchow2 | UNI2-h | TITAN |
|---|---:|---:|---:|
| MeanPool | 1,281 | 1,537 | 769 |
| ABMIL | 197,250 | 230,018 | 131,714 |
| TransMIL | 2,805,249 | 2,936,321 | 2,543,105 |
| TransMIL + whole-ROI token | 2,805,761 | 2,936,833 | 2,543,617 |
| **WR-TransMIL** | 2,804,737 | 2,935,809 | 2,542,593 |
| *ASGAP (thesis only)* | 197,251 | 230,019 | 131,715 |

WR-TransMIL keeps TransMIL's `cls_token` allocated but never uses it; the effective count excludes it (512 fewer than TransMIL). The extra-token variant has 512 more (modality embedding).

## 10. Other measured facts (analyses from earlier sessions; scripts not bundled)

- **ASGAP (thesis only)**: learnable α initialised at 1.5 converged to 1.506; beats ABMIL on both val and test only on TITAN; best fixed α per encoder 1.75 (Virchow2) / 1.00 = plain ABMIL (UNI2-h) / 1.875 (TITAN).
- **Averaging, not selection (thesis only)**: fibrosis axis v = mean(train patches of G2, G3) − mean(G0, G1). Bag-mean projection vs grade ρ = .834 / .860 / .866. Within-bag Spearman(attention, projection): ABMIL −.322 / −.206 / −.115; ASGAP −.224 / −.099 / +.010; patches ASGAP zeroes out are more fibrotic than kept ones (AUC .131 on Virchow2). Three learnable gates converged to inactive values: α → 1.506, tissue-group weights → ≈1/3, re-inject gate → .006–.019.
- **Patch features are a continuum**: silhouette ≈ .10 at every k = 2–5 (shuffled .002), no peak at k = 3 → no fat/bone/fibrosis clusters.
- **PPEG at ROI scale**: 91% of ROIs have a grid side S ≤ 7, so the 7×7 kernel spans the whole grid; six redesigns did not beat PPEG; a bag-mean-only replacement is within .0053 at 1/17 of the parameters.
- **Threshold fitting on val** (moving 0.5/1.5/2.5): worse on test in 9/9 cells (mean −.121); oracle thresholds on test gain only +.005 to +.038.
- **Acquisition confound**: raw image size is strongly associated with grade (e.g. 1223×627, 1231×635, 1903×888 are almost all G0). Size-only lookup from train predicts test grade poorly (QWK .22; val −.17). Test patient PMF27 (G3) has sizes that are 100% G0 in train, yet WR-TransMIL predicts G3 for 30/30, 28/30, 30/30 of its ROIs; no model predicts G0 for it.
- **UNI2-h weak cell**: WR-TransMIL G1 recall 40.8 (24/49 read as G0), inherited from TransMIL (42.9); the G1 scalar centre shifts to ≈0.65 on UNI2-h vs ≈0.93–0.97 on the other encoders.
- **Implementation notes**: TransMIL here uses full attention (bags ≈ 40 patches ≪ 256 Nyström landmarks); patch sequences are padded to a square grid by repeating the first patches (mean 14%, max 32% of tokens); median 40 patches per ROI (13–112).
