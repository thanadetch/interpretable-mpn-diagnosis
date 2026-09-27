# Methods and data — facts to write the Dataset and Methods sections

## 1. Cohort and labels

- 50 MPN patients (ET 13 / PMF 26 / PV 11), 1,330 reticulin-stained bone-marrow ROIs, 58,550 patches after filtering. Single institution.
- Grades G0–G3 from one expert pathologist following the European consensus (Thiele et al., Haematologica 2005; MF-0…MF-3). All ROIs of a patient carry the patient's grade. No second rater (limitation).
- Per-grade totals (patients / ROIs): G0 8 / 441 · G1 15 / 241 · G2 18 / 359 · G3 9 / 289. PMF dominates G2/G3; ET/PV concentrate in G0/G1.
- ROI sizes vary (≈ 663–1903 px wide) and magnifications vary: the measured scale bar gives 8 µm/px clusters from ≈ 0.27 to 2.70 (≈ 40× down to 4×), so one 224-px patch covers ≈ 61–605 µm; 384 ROIs (21 of 50 patients) have no readable scale annotation. Magnification was checked and does not bias the models (within-grade residual vs log µm/px ρ between −.07 and +.10). No scale normalisation.

## 2. Split (patient level, fixed)

| split | patients | ROIs | G0 | G1 | G2 | G3 |
|---|---:|---:|---:|---:|---:|---:|
| train | 30 | 857 | 320 (4 pts) | 139 (9) | 249 (12) | 149 (5) |
| validation | 10 | 214 | 21 (2) | 53 (3) | 80 (3) | 60 (2) |
| test | 10 | 259 | 100 (2) | 49 (3) | 30 (3) | 80 (2) |

Stratified by grade at the patient level; no patient appears in two splits. G0 and G3 have two patients each in validation and in test — per-grade recall on those grades is driven by two patients.

## 3. Preprocessing

1. Remove the scanner overlay (top 57 px) and the scale label (bottom 40 px) from each ROI.
2. Tile 224×224 patches with stride 112 (50% overlap), edge-anchored.
3. Optical-density filter: `tissue_threshold = 0.05`, `min_tissue_ratio = 0.10` (removes background/fat; bone is not removed). Median 40 patches per ROI (13–112).
4. Encode every patch with a frozen foundation encoder: Virchow2 (1280-d), UNI2-h (1536-d), TITAN's patch encoder CONCHv1.5 (768-d). No fine-tuning.
5. Whole-ROI embedding (WR-TransMIL and its controls only): the full, uncropped ROI resized to 224×224 and encoded by the same encoder.

## 4. Training and evaluation

- Scalar regression: SmoothL1 loss on one output; prediction = round and clip to [0, 3]; checkpoint selected on validation QWK.
- Multi-class comparison: cross-entropy with label smoothing 0.1 and **uniform class weights** (write exactly this — never "weighted CE"); checkpoint on validation macro recall.
- AdamW (lr 1e-4, weight decay 0.01), batch size 1 bag, at most 50 epochs, early stopping patience 15, cosine annealing to 1e-6, seed 2. Identical for every model. Hardware: Apple-silicon Mac (MPS).
- Metrics (all at ROI level): quadratically weighted kappa (primary), MAE, accuracy, macro F1, macro recall, per-grade recall, confusion matrices, number of predictions ≥ 2 grades from the reference.
- Why QWK: grades are ordinal and the clinical cost of an error grows with distance; QWK can rise while exact accuracy falls (seen on TITAN), so it is always reported with accuracy and far-error counts. Pathologist inter-rater κ for marrow fibrosis ≈ 0.76–0.83 in the literature (verify before citing; weighting differs from QWK).

## 5. Compared methods (main table)

MeanPool (mean aggregation); ABMIL (gated attention, Ilse et al. 2018); TransMIL (transformer, full attention) — one reference method per aggregation family — plus TransMIL + whole-ROI token and WR-TransMIL. All trained with the identical protocol above. Suggested sentence: *"We compare against one representative of each aggregation family — mean pooling, attention pooling (ABMIL) and transformer aggregation (TransMIL) — since the proposed change modifies the transformer's readout token."* CLAM-SB, DSMIL and DTFD-MIL are cited but not tabulated (decided 2026-09-27); their results exist (see results.md reserve section) and can be added in revision.

## 6. Disclosures to include in Methods / Limitations

- Model variants around WR-TransMIL were compared on the same test set (selection used validation and test); present them as ablations on this test set, never as validation-only selection.
- The whole-ROI view is uncropped (scanner overlay, slide-label thumbnail, scale bar); the image-size confound was tested (see wr_transmil.md §6).
- Single split, 10 test patients, no statistical significance claimed; single institution; single rater.
- Do not show uncropped ROIs in figures (the slide-label thumbnail may identify patients).
- Submission items (Elsevier): ethics/IRB statement and consent, CRediT, competing interests, funding, data and code availability, declaration of generative-AI use, highlights.
