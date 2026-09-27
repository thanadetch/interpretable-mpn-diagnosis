# WR-TransMIL — the proposed method (for Methods, Results and the ablation section)

## 1. Definition (for the Methods section)

- **Name**: WR-TransMIL = *TransMIL with a Whole-ROI Readout*. Never write the module code (a445).
- **Idea in one sentence**: TransMIL reads its prediction from a learned class token that is identical for every bag, because a whole-slide image cannot be encoded in one pass; at ROI scale the whole image can, so the class token is replaced by the embedding of the whole ROI from the same frozen encoder.
- **Forward pass**: patch features → shared projection Linear(D→512)+ReLU → pad the patch sequence to S² tokens by repeating the first patches (for PPEG) → place the projected whole-ROI embedding at position 0 **instead of the learned class token** → attention layer (pre-LayerNorm, 8-head full self-attention, residual) → PPEG on the patch tokens only (position 0 held out) → attention layer → LayerNorm → read position 0 → Linear(512→1) → scalar grade, rounded and clipped to [0, 3].
- **Whole-ROI embedding**: the full ROI resized to 224×224 and encoded by the same frozen encoder as the patches (it is NOT cropped like the patches, so it still contains the scanner overlay and scale bar — disclose this).
- **Parameters (effective)**: 2,804,737 (Virchow2) / 2,935,809 (UNI2-h) / 2,542,593 (TITAN) = 512 fewer than TransMIL (the unused class token is allocated but receives no gradient). The extra-token comparator has 512 more than TransMIL.
- **Implementation notes to state**: full attention instead of Nyström (bags have ~40 tokens, median 40, range 13–112, far below the 256 landmarks, so Nyström would be equivalent); padding by repetition adds on average 14% of tokens (max 32%).

## 2. How the idea was reached (for the Introduction / motivation)

1. Pathologists grade the overall reticulin density of the whole ROI, while MIL only sees patches.
2. Adding the whole-ROI view to TransMIL as extra information (late fusion, extra token) helped inconsistently.
3. Asking which part of TransMIL exists only because of the WSI constraint points to the learned class token — a stand-in for a summary WSI MIL cannot compute. At ROI scale the real summary exists, so it replaces the stand-in.
4. Using the same vector in different roles shows that the benefit comes from the role (being the readout), not from the extra information.

## 3. Results (test QWK, 259 ROIs from 10 patients; val in brackets)

| | Virchow2 | UNI2-h | TITAN |
|---|---|---|---|
| TransMIL | .9604 (.7712) | .9283 (.7734) | .9576 (.7841) |
| TransMIL + whole-ROI token | .9631 (.8135) | .9347 (.7943) | .9496 (.7794) |
| **WR-TransMIL** | **.9675 (.7957)** | **.9371 (.7822)** | **.9616 (.7942)** |

- WR-TransMIL beats TransMIL on both val and test on **3/3** encoders; the extra token on 2/3 (loses TITAN).
- Errors ≥ 2 grades: WR-TransMIL 0/0/0 · TransMIL 0/0/2 · extra token 0/2/2.
- Correct ROIs /259: WR 232/206/227 vs TransMIL 226/200/230 — on TITAN QWK rises while accuracy falls; write "errors move closer to the true grade", not "more correct".
- Effect size: +.0071 / +.0088 / +.0040 QWK ≈ 6 ROIs per encoder.
- The two whole-ROI models behave differently, not just numerically: UNI2-h recall G1 41 vs 74, G2 83 vs 60.
- Full metric tables, per-grade recall and confusion matrices: `results.md`.

## 4. Placement and control experiments (for the ablation section)

| Variant | module | Virchow2 val / test | UNI2-h val / test | TITAN val / test | far (v2/u2/ti) | note |
|---|---|---|---|---|---|---|
| TransMIL (reference) | `a373_transmil` | .7712 / .9604 | .7734 / .9283 | .7841 / .9576 | 0/0/2 | learned class token |
| **replace class token (WR-TransMIL)** | `a445_tb_transmil_cls` | .7957 / .9675 | .7822 / .9371 | .7942 / .9616 | 0/0/0 | proposed |
| extra whole-ROI token (+ modality emb.) | `a381_tb_transmil_token` | .8135 / .9631 | .7943 / .9347 | .7794 / .9496 | 0/2/2 | comparator |
| whole-ROI emb. added to the class token | `a448_tb_transmil_cls_add` | .7781 / .9177 | .7852 / .9567 | .7727 / .9539 | 1/0/0 | inconsistent |
| concatenated before the head (late fusion) | `a380_tb_transmil_late` | .8124 / .9530 | .7935 / .9281 | .7626 / .9537 | 0/0/1 |  |
| patch mean as readout (CLIP-style) | `a447_tb_transmil_cls_mean` | .7960 / .9507 | .7835 / .9342 | .7803 / .9382 | 0/1/0 | content control |
| another ROI's emb. as readout (shuffle) | `a446_tb_transmil_cls_shuffle` | .8192 / .9150 | .7840 / .9454 | .7798 / .9432 | 3/0/1 | correspondence control |
| separate projection for the ROI emb. | `a456_tb_transmil_cls_sepproj` | .7976 / .9181 | .8015 / .9397 | .7694 / .9493 | 1/2/2 | same-space control |
| ROI emb. from another encoder (TITAN) | `a465_cls_titanfield` | .7730 / .9004 | .7546 / .9321 | — | 0/2/— | same-space control |

Reading:
- **Role, not information**: the same vector as an extra token (2/3) or added to the class token (inconsistent) does not match using it as the readout (3/3).
- **Content matters**: starting the readout from the patch mean (the CLIP-style query) loses on all three encoders; starting from another ROI's embedding drops Virchow2 by .0525 (UNI2-h improves — the unstable cell).
- **Same representation space matters**: on Virchow2, shared projection .9675 > separate projection .9181 > embedding from another encoder (TITAN) .9004.
- **Patches still carry most of the prediction**: the ROI embedding alone (no patches) gives .8752 / .8844 / .8941 with G3 recall 27.5 / 42.5 / 58.8.
- Other variants tried around WR-TransMIL (both roles at once, FiLM, several whole-ROI tokens, re-injecting the ROI view between layers, removing or redesigning PPEG, a single attention layer, a per-patch grade head) — none was better on all three encoders. Two facts worth keeping for reviewers: removing the second attention layer costs Virchow2 −.0755; replacing PPEG by a bag-mean-only operator lands within .0053 at 1/17 of PPEG's parameters (see discussion_points.md).

## 5. Multi-class check

| | test QWK reg → mc | far errors reg → mc |
|---|---|---|
| TransMIL (v2 / u2 / ti) | .9604→.9522 · .9283→.9401 · .9576→.8667 | 0→1 · 0→2 · 2→8 |
| WR-TransMIL | .9675→.9319 · .9371→.9250 · .9616→.9073 | 0→5 · 0→2 · 0→8 |

## 6. Verification done (can be stated or used in a response letter)

- Patient split: zero patient overlap; identical across encoders.
- Whole-ROI pairing: 1330/1330 bags matched to their own embedding on all encoders.
- Checkpoint selected on val only; all reported checkpoints reproduce exactly with the released code.
- Acquisition-format check: image size predicts test grade poorly (QWK .22); test patient PMF27 (G3), whose image sizes are all G0 in training, is still predicted G3 for 30/30 (Virchow2), 28/30, 30/30 ROIs.

## 7. What may and may not be claimed

- ✅ Improves TransMIL on all three encoders on both val and test; no errors ≥ 2 grades; fewer parameters; the gain comes from the readout role; the whole-ROI embedding beats the patch mean as the starting point.
- ⚠️ State with it: small margins; single split; on UNI2-h ABMIL (.9418) is better than every TransMIL variant; WR-TransMIL keeps TransMIL's UNI2-h G1 weakness (recall 40.8).
- ❌ "first", "novel mechanism", "state of the art", "significantly", "attention highlights fibrosis", internal-mechanism explanations beyond the controls, applicability to WSI.
