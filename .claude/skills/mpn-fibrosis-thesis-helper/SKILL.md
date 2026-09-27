---
name: mpn-fibrosis-thesis-helper
description: "Project context for Thanadetch's Master's thesis and BSPC paper on ROI-level reticulin fibrosis grading in MPN with frozen pathology foundation models (Virchow2, UNI2-h, TITAN) and MIL. Use for any thesis or paper task: WR-TransMIL (whole-ROI readout, module a445), ASGAP (ABMIL + alpha-entmax, a215), TransMIL / ABMIL / CLAM / DSMIL / DTFD baselines, scalar regression vs multi-class, QWK and far-error reporting, prior art and novelty wording, journal choice, advisor communication (Dr. Pittipol Kantavat), thesis chapters. Also triggers on ROIs, bone marrow, reticulin, G0-G3 grading, class token, whole-ROI embedding, even when 'thesis' is not mentioned."
---

# MPN Fibrosis Thesis Helper

> Project context for Claude. Mirrored in `.github/copilot-instructions.md`; keep the two in
> sync and update both when either changes. **Last reconciled: 2026-09-27.**
>
> **Bundled with this skill (self-contained — use them when the repository is not available, e.g. on claude.ai).**
> File paths elsewhere in this file only exist in the repo.

## Reference files — open the one that matches the task

| file | use it for |
|---|---|
| `results.md` | every number for the paper: main 5-model comparison (plus CLAM-SB/DSMIL/DTFD-MIL kept in reserve), per-grade recall, confusion matrices, placement and controls, formulation, patching ablation, parameter counts |
| `reference/wr_transmil.md` | the proposed method: definition, motivation, results, ablation/control table, verification, claims allowed |
| `reference/methods_and_data.md` | Dataset and Methods sections: cohort, labels, split table, preprocessing, training protocol, metrics, compared methods, disclosures |
| `reference/prior_art.md` | Related Work: what each related paper does and how WR-TransMIL differs, plus a ready sentence |
| `reference/discussion_points.md` | Discussion, Limitations, and answers to likely reviewer questions from existing results |
| `reference/thesis_only.md` | material for the thesis but not the paper: ASGAP, averaging-not-selection analysis, patching justification, augmentation, advisor requirements |
| `reference/rules.md` | how to work with the user, reporting rules, forbidden words, corrections already made, decisions |
| `reference/paper_writing_guide.md` | **how to write the paper**: story and contributions, writing order, section-by-section plan (each tagged with review criteria A–H), equations, figure/table list, highlights, graphical abstract, cover letter, declarations, pre-submission self-check |

---

## Project context

- **Student**: Thanadetch Lertwuthikarn — Chulalongkorn University, Computer Engineering, Faculty of Engineering
- **Advisors**: Dr. Pittipol Kantavat (main) + a second advisor (approval pending as of 2026-09)
- **Plan**: Ko1 (thesis-only) · **Target graduation**: end of 2026
- **Proposal**: presented June 2026 (the proposed method at that time was ASGAP; the story has since moved to WR-TransMIL, which the user considers acceptable — a proposal is not binding)
- **Paper venue**: the advisor shortlisted five Elsevier journals — Biomedical Signal Processing and Control (BSPC), Computers in Biology and Medicine, Computerized Medical Imaging and Graphics, Journal of Pathology Informatics, Informatics in Medicine Unlocked. **Submit to BSPC first.** BSPC is Scopus Q1 (CiteScore 13) but JCR Q2 (Engineering, Biomedical, rank 33/130) — whether it counts as Q1 depends on the graduation rule (Scopus vs WoS), which is still to be confirmed with the advisor.
- **Working condition**: full-time software engineer, limited daily bandwidth
- **Language**: the user writes in Thai; reply in Thai, keep technical terms (MIL, QWK, class token) in English. Manuscript text is English.

## Thesis

**Title**: Fibrosis Grading Diagnosis Using Image Recognition Techniques in Myeloproliferative Neoplasm Patients

**Task**: ROI-level reticulin fibrosis grading (ordinal, G0–G3) from frozen foundation-model features with MIL.

**Scope sentence** (filter every suggestion through it): *ROI-level reticulin grading with frozen foundation features; scalar regression is preferred over classification because it avoids errors of two or more grades; the whole-ROI embedding, used as the readout of TransMIL, improves it on all three encoders; sharper patch selection (sparse attention) does not help consistently.*

## Current stage (2026-09-27)

- ✅ **Experiments are closed.** Do not start training runs, new modules, new baselines, feature re-extraction, backbone fusion or extra controls unless the user explicitly says to run. ("ถ้าฉันไม่บอกให้ run ก็คือไม่ต้อง run")
- ✅ **Paper scope decided**: one paper, one proposed method — **WR-TransMIL** (*TransMIL with a Whole-ROI Readout*). ASGAP is **cut from the paper** and kept in full in the thesis.
- ✅ **Audit done**: patient split verified (no overlap), whole-ROI pairing verified (1330/1330, zero mismatches), checkpoint selection on validation verified, and **all 56 reported checkpoints reproduce their stored val/test QWK exactly with the current code** (strict state-dict load, zero changed predictions).
- ⏳ **Next**: email the advisor (ethics/IRB number, author list and order, graduation rule Scopus vs WoS and submitted vs accepted, BSPC first, ASGAP's new role, and that weighted cross-entropy was never actually tested); write Methods and results tables following `reference/paper_writing_guide.md`.
- `ADVISOR_BRIEF_2026-09.md` is the advisor-facing summary (it still uses module codes).

## Dataset

- 50 MPN patients, 1,330 reticulin ROIs (`data/features_{backbone}_reti/{Class}/{Patient G#}/{retiN}.pt`).
- **Locked patient-level split** (`patient_split(seed=2)` in `src/train_grading_reti.py`, per-grade targets `(2,2)/(3,3)/(3,3)/(2,2)`):
  - Train 30 patients / 857 ROIs (G0 320, G1 139, G2 249, G3 149)
  - Val 10 / 214 (G0 21, G1 53, G2 80, G3 60)
  - Test 10 / 259 (G0 100, G1 49, G2 30, G3 80)
  - Test patients: ET13 G0, PV6 G0, ET3 G1, ET4 G1, PMF21 G1, ET2 G2, PMF17 G2, PV1 G2, PMF24 G3, PMF27 G3.
- Per-grade totals (patients / ROIs): G0 8 / 441, G1 15 / 241, G2 18 / 359, G3 9 / 289. Subtypes: ET 13 / 279, PMF 26 / 687, PV 11 / 364 (PMF dominates G2/G3; ET/PV concentrate in G0/G1).
- **Labels**: one expert pathologist per ROI, European consensus grading (Thiele et al., Haematologica 2005; MF-0…MF-3 mapped to G0–G3); no inter-rater agreement measured → a limitation to state. Single institution.
- **How the grade is read** (user and advisor): holistic density of the reticulin meshwork across the whole ROI, excluding dark bone trabeculae ("intense ≠ fibrosis").
- G0 and G3 have only 2 val and 2 test patients each; treat single-digit recall swings as noise.
- ROI sizes vary (e.g. 705×911, 943×760, 1223×627, 1903×888 px); raw size is confounded with grade (see Key findings 7).
- Mixed magnifications / scale bars (20, 50, 100, 200 µm, unknown); no scale normalisation.

## Pipeline

1. **Patches**: crop top 57 px (overlay) and bottom 40 px (scale label), tile 224×224 at stride 112 (edge-anchored), OD filter (`tissue_threshold=0.05`, `min_tissue_ratio=0.10`), frozen encoder. Median 40 patches per ROI (13–112).
2. **Whole-ROI embedding** (`data/features_{backbone}_reti_no_patch/`): the **full ROI resized to 224×224, NOT cropped** — it still contains the slide-label thumbnail (barcode), "Magnification" text, scanner version and scale bar — passed through the same frozen encoder. Looked up per bag by a value fingerprint in `field_mil._FieldBank`.
3. **Encoders** (frozen, never fine-tuned; no primary backbone — always report all three): Virchow2 1280-d, UNI2-h 1536-d, TITAN 768-d.
4. **Formulations**: scalar regression (SmoothL1, `round` + clip to [0,3], checkpoint on val QWK) — the main formulation; multi-class (cross-entropy + label smoothing 0.1, checkpoint on val macro-recall) — comparison only. **The class weights are all 1.0** (`class_weights` in the trainer; every multi-class log prints `Weights: G0=1.0 G1=1.0 G2=1.0 G3=1.0`), so this is plain CE — never call it "weighted CE". The code comment claiming higher G0/G3 weights is stale.
5. **Training**: AdamW lr 1e-4, weight decay 0.01, batch 1, ≤50 epochs, early stop 15, cosine to 1e-6, seed 2. Trainer: `python3 src/train_grading_reti.py`.

## Models that matter

| Name in writing | Code | Notes |
|---|---|---|
| MeanPool | `--model_type mean_pool` | |
| ABMIL | `--model_type simple` (`SimpleGatedMIL`) | Ilse 2018 gated attention, 197,250 params. Always call it ABMIL. |
| CLAM-SB / DTFD-MIL / DSMIL | `a370_clam_sb` / `a371_dtfd_mil` / `a372_dsmil` | run on all encoders but **not in the paper** (decided 2026-09-27); kept in reserve for a reviewer request. CLAM-SB = attention branch only (no instance loss); DTFD-MIL without its tier-1 loss; DSMIL complete except the regression head |
| TransMIL | `a373_transmil` | **full attention, not Nyström** (bags ≈ 40 ≪ 256 landmarks); PPEG; learned class token |
| TransMIL + whole-ROI token | `a381_tb_transmil_token` | comparator: embedding added as an extra token, class token kept (+512 params) |
| **WR-TransMIL** (proposed) | `a445_tb_transmil_cls` (`two_branch_transmil.py`, `fuse="cls"`) | whole-ROI embedding, through the same `_fc1` projection, **replaces** the class token; `cls_token` stays allocated but gets no gradient → effective 2,804,737 params on Virchow2 (TransMIL 2,805,249) |
| ASGAP (thesis only) | `a215_learnable_entmax` | ABMIL with α-entmax; learnable α, init 1.5, converged to 1.506 |

Controls for WR-TransMIL: `a448` (embedding added to the class token), `a447` (patch mean as readout = CLIP-style), `a446` (another ROI's embedding), `a456` (separate projection), `a465` (embedding from another encoder), `a380` (concatenated before the head), `fldonly_*` (ROI view only, no patches).

## Canonical results (test QWK, seed-2 split, regression)

| Model | Virchow2 | UNI2-h | TITAN |
|---|---:|---:|---:|
| MeanPool | .9209 | .9067 | .9253 |
| ABMIL | .9476 | **.9418** | .9584 |
| *CLAM-SB (reserve, not in paper)* | *.9443* | *.9378* | *.9506* |
| *DSMIL (reserve)* | *.9163* | *.8943* | *.9256* |
| *DTFD-MIL (reserve)* | *.9517* | *.9218* | *.9417* |
| TransMIL | .9604 | .9283 | .9576 |
| TransMIL + whole-ROI token | .9631 | .9347 | .9496 |
| **WR-TransMIL** | **.9675** | .9371 | **.9616** |
| *ASGAP (thesis only)* | *.9588* | *.9262* | *.9600* |

Canonical run dirs: ABMIL/MeanPool `experiments/20260603_ablation/patch/` (see `results/baseline_reference.md`); CLAM/DTFD/DSMIL/TransMIL/token `experiments/20260808/puba370-373_*`, `tba381_*`; WR-TransMIL `experiments/20260831/a445_*`; ASGAP `experiments/a215_ablation/regression_seed2/`; multi-class `experiments/20260905/a373mc_*`, `a445mc_*` and `20260603_ablation/patch/*_multiclass_*`.

## Key findings (what the thesis and paper claim)

1. **WR-TransMIL vs TransMIL**: better on both val and test QWK on **3/3** encoders; **zero** errors of ≥2 grades on all three (TransMIL 0/0/2, token 0/2/2); 512 fewer effective params; correct ROIs 232/206/227 vs 226/200/230 (on TITAN QWK rises while exact accuracy falls — the gain there is fewer distant errors).
2. **Role, not information**: the same embedding as an extra token wins 2/3; added to the class token is inconsistent (.9177/.9567/.9539); patch mean as readout loses 3/3 (.9507/.9342/.9382); separate projection .9181 and other-encoder embedding .9004 (Virchow2) are worse; ROI view without patches .8752/.8844/.8941 (G3 recall collapses) → patches still carry most of the prediction.
3. **Formulation**: errors ≥2 grades, regression **7** vs multi-class **54** over MeanPool/ABMIL/TransMIL/WR-TransMIL × 3 encoders, **12/12 cells**; QWK alone favours regression in only 9/12. The old "multi-class → G0 recall 0%" finding came from the **legacy 42-patient cohort**; in the current 50-patient runs multi-class G0 recall is 89–93% for every aggregator. Do not cite the G0 collapse as a current result.
4. **UNI2-h is the weak cell**: ABMIL (.9418) beats every TransMIL variant (so does CLAM-SB .9378, which is not in the paper); WR-TransMIL G1 recall 40.8 (24/49 read as G0), inherited from TransMIL (42.9).
5. **Thesis-only analyses**: ASGAP wins both gates only on TITAN; best α differs by encoder (1.75 / 1.00 / 1.875); its zero-weighted patches look *more* fibrotic (AUC .131, Virchow2). A data-derived fibrosis axis (mean G2/G3 minus mean G0/G1 train patches) predicts grade from the bag mean (ρ .834/.860/.866) while ABMIL attention anticorrelates with it (ρ −.322/−.206/−.115) → ROI grading behaves as **averaging, not selection**. Three learnable gates (α, group weights, re-inject gate) all converged to their inactive values.
6. **Patching is justified (thesis)**: resizing the whole ROI to one 224×224 image instead of tiling lowers test QWK in all 12 MeanPool/ABMIL × encoder × formulation configs; the loss is large under regression with ABMIL (e.g. Virchow2 QWK .948 → .859, accuracy 85.3 → 63.7%) and small under multi-class. Scope the claim to the regression + attention-MIL setting. The paper's "ROI view only" control makes the same point for the TransMIL family.
7. **Acquisition confound, tested**: raw image size is strongly associated with grade, but size alone predicts test grade poorly (QWK .22; val −.17), and PMF27 G3 — whose image sizes are all-G0 in train — is still predicted G3 (30/30 on Virchow2).

## Advisor requirements (status)

| Requirement | Status |
|---|---|
| Replace mean pooling with a custom-designed aggregator (Type-2 contribution) | Done — ASGAP (thesis) and WR-TransMIL (thesis + paper) |
| Justify patching with a resize ablation | Done — see Key findings 6 |
| Do not weight patches by intensity / feature norm; follow holistic density | Respected; the measured behaviour is averaging, not selection |
| Gains should hold across encoders | WR-TransMIL: 3/3 vs TransMIL; UNI2-h still favours ABMIL overall |
| Test weighted cross-entropy before concluding regression is better | **Not done** — all multi-class runs used uniform weights. Raise with the advisor; do not run unless the user asks |

Benchmark tier: the advisor's previous student published in IEEE Access 2025 (papillary thyroid carcinoma segmentation, 60 cases, random split) — useful for scope and length, not a protocol to copy.

## Hard rules

1. **Split is locked** — never edit `patient_split()`.
2. **No new experiments unless the user says run.** No new baselines, re-extraction, backbone fusion, loss changes, external data or controls unasked. Inference-only checks need the user's go-ahead too.
3. **Report the single seed-2 split only.** No other seeds, no CV numbers, no patient-level metrics unless the user asks. All metrics are ROI-level.
4. **All three encoders, always together.** No primary backbone.
5. **Disclose test-set reuse**: the ~20 variants around WR-TransMIL were compared on the same test set with a val-and-test gate. Never write that selection used validation only.
6. **Prohibited**: zero-shot text-prompt concept scores (CONCH/TITAN/CLIP) to define bone or fibrosis; do not run or cite `src/tools/bone_vs_fibrosis_separability.py`.
7. **Frozen encoders only**; no new dependencies without asking; destructive operations need explicit confirmation; **no git writes** (the user manages version control).

## Writing rules (thesis and paper)

- **Paper structure**: proposed method WR-TransMIL; main table of **5 models** (MeanPool, ABMIL, TransMIL = reference methods for mean, attention and transformer aggregation; TransMIL + whole-ROI token and WR-TransMIL = ablation); placement table (none / extra token / added to class token / replaces class token); six controls; formulation table (far errors 7 vs 54, 12/12); per-grade recall; confusion matrices; PMF27 check. CLAM-SB, DSMIL, DTFD-MIL are cited in Related Work but not tabulated (decided 2026-09-27); their seed-2 results exist on all encoders and can be added in revision if a reviewer asks.
- **Not in the paper** (thesis only): ASGAP and the α analysis, the fibrosis-axis/attention analysis, the three gates, augmentation, PPEG redesigns, threshold fitting, tissue grouping, the full variant list (supplementary at most). Re-add ASGAP only if the fibrosis-axis analysis goes into the paper.
- **Claim ladder**: claim consistency (3/3 encoders, zero far errors, fewer params, role-not-information). State effect size (+0.004 to +0.009 QWK, ~6 ROIs per encoder), single centre, single split, ten test patients, no significance claim, UNI2-h counter-result. Never write *first*, *novel mechanism*, *state of the art*, *significantly*; use "to our knowledge" and "we show that". Do not explain internal mechanisms beyond the controls.
- **Terminology**: *class token* (TransMIL's own term; BERT calls it the classification token, [CLS]); *whole-ROI embedding* — avoid "global" (ambiguous with global attention/pooling/WSI). Comparators get descriptive names, no acronyms. Never let "a445", "a381" or other module codes appear in writing.
- **Prior art (methods sections read)**: nothing replaces a MIL readout token with a separately encoded whole-image embedding. TransMIL descendants keep the learned class token (CTMIL, MsCAMIL) or remove it (HAG-MIL); SEW uses a WSI thumbnail through its own learned class token plus a consistency loss; GMIC concatenates global features before the head; MEGT/PTCMIL/ViTAGG-MIL add context tokens. Related principles to cite: CrossViT (branch class token queries other branch, but starts from a learned constant), Conditional DETR and Efficient DETR (content-initialised queries in detection), CLIP (attention-pooling query = mean of the image's own features — our patch-mean control). ASGAP prior art: SMILE = top-N + softmax (no sparsemax), MINN-SA = sparsemax on a non-gated scorer over TCR sequences. Always open the PDF before stating what a paper does.
- **Never write** that ASGAP or any attention "highlights" or "attends to" fibrosis — the measurement shows the opposite. Never show uncropped ROIs in figures (the slide-label thumbnail may identify patients).
- **Disclose in Methods**: whole-ROI view is uncropped; full attention instead of Nyström; square padding by repeating patches; effective parameter count; variants compared on the same test set.
- **Submission items (Elsevier)**: ethics/IRB statement, CRediT, competing interests, funding, data and code availability, declaration of generative-AI use, highlights. Initial submission is format-free ("Your Paper Your Way"), but write in `elsarticle` so any of the five journals can take it without reformatting.
- Thesis format: Chulalongkorn (Thai title page and abstract, English body).

## What to avoid

- Proposing new aggregators, losses, extra baselines, external datasets, second raters or fusion — outside the scope sentence; the in-scope space is covered.
- Re-proposing ideas already closed: fat/bone/fibrosis clustering (features form a continuum), learnable α variants, PPEG redesigns, threshold fitting on val (worse on 9/9), more variants around WR-TransMIL.
- Asserting "nothing is left" as fact — say how many were tried and that no data-driven hypothesis remains.
- Narrating a mechanism before measuring it; interpreting images before measuring separability.
- Top-tier venue framing; whole-slide claims; patch-level labels; fine-tuning encoders.

## Workflow pointers (repo only)

- Canonical baseline numbers and run dirs: `results/baseline_reference.md`.
- Per-ROI image size and scale-bar table: `results/scale_um_per_px.csv`.
- WR-TransMIL implementation and all its variants/controls: `src/models/novelty_attempts/two_branch_transmil.py`; whole-ROI lookup: `field_mil.py`.
- The historical novelty-search playbook (`NOVELTY_SEARCH_PLAYBOOK.md`, `NOVELTY_NOTES.md`, `scripts/run_novelties_parallel.sh`) is **closed**; use it only if the user explicitly reopens experiments. Highest module id on disk: a471.
- Reproducing a reported number: rebuild the model exactly as the trainer does, load `best_<run_name>.pth` with `strict=True`, run `validate_and_evaluate` on the seed-2 split — the 2026-09-27 check reproduced all 56 reported checkpoints exactly.
