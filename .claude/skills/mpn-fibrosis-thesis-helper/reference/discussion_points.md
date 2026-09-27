# Discussion points, limitations and likely reviewer questions (paper)

## 1. Points for the Discussion

1. **Why the readout role matters.** The same whole-ROI vector helps consistently only when it is the token the prediction is read from; as an extra token or added to the class token it helps inconsistently. The readout must also start in the patches' representation space (separate projection or a different encoder is worse). Do not go beyond this into internal mechanisms — they were not measured.
2. **Relation to image-conditioned queries.** Conditioning a query on the image is known outside MIL (CLIP's attention pooling uses the mean of the image's own features; Conditional/Efficient DETR in detection). Here the readout starts from a *separately encoded* whole-ROI embedding, which is only available because an ROI, unlike a WSI, fits in one forward pass; the CLIP-style patch-mean readout was worse on all three encoders.
3. **Encoder dependence.** The best aggregator differs by encoder: WR-TransMIL on Virchow2 and TITAN, ABMIL on UNI2-h. No encoder is treated as primary; the dependence itself is a finding. UNI2-h keeps TransMIL's G1 weakness (G1 read as G0: 24/49 ROIs; the G1 predictions centre near 0.65 instead of ≈0.95 on the other encoders).
4. **Formulation.** Scalar regression makes fewer distant errors than multi-class classification in every aggregator–encoder pair (7 vs 54 ROIs ≥ 2 grades off, 12/12); cross-entropy does not encode grade order. QWK alone favours regression only 9/12 — this is why error distance is the headline. Multi-class did not collapse G0 in this cohort (G0 recall 89–93%).
5. **PPEG at ROI scale.** TransMIL's positional generator was designed for gigapixel bags; at ROI scale 91% of bags have a grid side ≤ 7, so its 7×7 kernel spans the whole grid. Replacing it with a bag-mean-only operator comes within .0053 QWK at 1/17 of the parameters ⇒ at ROI scale PPEG mainly smooths at the bag level. (Answers "why PPEG on 40-patch bags?".)
6. **Clinically important boundary.** Under WHO criteria grade ≤ 1 vs ≥ 2 separates prefibrotic from overt PMF. WR-TransMIL does not reduce G1↔G2 confusions (TransMIL 7/8/6 vs WR 6/9/9 on Virchow2/UNI2-h/TITAN) — report these separately and say so.
7. **Acquisition confound handled.** Image size and scale annotation correlate with grade in this cohort, but image size alone predicts test grade poorly (QWK .22), and a G3 test patient whose images look like training G0 images in size is still graded G3.
8. **Data efficiency (optional sentence).** Reducing training to ~10 ROIs per patient left test QWK within noise; more patients (especially G0/G3) would help more than more ROIs per patient.

## 2. Limitations (write them plainly)

- 50 patients from one institution; one split with 10 test patients (2 per extreme grade); no external validation; one rater.
- Differences between the best models are small (≈ 6 ROIs per encoder); no significance testing; consistency across encoders is the evidence.
- Model variants were compared on the same test set.
- UNI2-h favours ABMIL overall.
- The whole-ROI view includes acquisition overlays.
- Frozen encoders only; no image-level augmentation (embedding-space augmentation is reported as on par with patch-level augmentation — Zaffar et al., ISBI 2023).

## 3. Likely reviewer questions → answers from existing results (no new runs)

| question | answer |
|---|---|
| Is the gain just more parameters? | No — WR-TransMIL has 512 fewer effective parameters than TransMIL. |
| Is it just extra information? | The same vector as an extra token or added to the class token is inconsistent; only the readout role helps on 3/3. |
| Would the patch mean do? | No — patch mean as the readout loses on all three encoders (CLIP-style). |
| Does it need the correct ROI? | Another ROI's embedding drops Virchow2 by .0525 (not UNI2-h, the unstable encoder). |
| Is the whole-ROI view enough on its own? | No — .875–.894 QWK, G3 recall collapses; patches carry most of the signal. |
| Why not Nyström attention? | Bags have ~40 tokens ≪ 256 landmarks; full attention is exact. |
| Why PPEG? | See §1.5. |
| Why not CLAM/DSMIL/DTFD-MIL? | The claim concerns TransMIL's readout; one reference per aggregation family is compared. Their results already exist on all three encoders (seed 2, same protocol) and can be added in the response letter: CLAM-SB (attention branch) .9443/.9378/.9506, DSMIL .9163/.8943/.9256, DTFD-MIL (no tier-1 loss) .9517/.9218/.9417 — none beats WR-TransMIL on Virchow2 or TITAN; CLAM-SB is .0007 above it on UNI2-h. |
| Why only TransMIL? | The change is to the class token, which attention-pooling MIL (ABMIL, CLAM, DTFD) does not have; CTMIL/MsCAMIL likewise modify TransMIL only. An earlier ABMIL-style test (whole-ROI view as the attention query) gained as much with another ROI's view, so no claim is made beyond TransMIL. |
| Is QWK appropriate? | Ordinal grades; reported with accuracy, MAE, far errors and per-grade recall. |
| Significance / CIs? | Not claimed; if asked, compute a patient-level bootstrap from the saved predictions (no retraining). Only if a reviewer asks. |
| Does the model exploit image size / scale overlays? | Tested: size alone QWK .22; PMF27 check. |
| Why no image augmentation? | Scope (frozen encoders, aggregator contribution) + Zaffar et al.; never the compute argument. |
| Multi-class with class weights? | The comparison used uniform weights; weighted CE was not run (be honest). |
