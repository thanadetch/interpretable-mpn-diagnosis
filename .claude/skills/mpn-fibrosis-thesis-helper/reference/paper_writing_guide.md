# Paper writing guide — BSPC manuscript on WR-TransMIL

How to write the paper, section by section. Facts and numbers live in the other files; this file says what goes where and what each section must achieve. Each section is tagged with the review criterion it satisfies (A–H from the manuscript-evaluation criteria), so a draft written from this guide can be self-checked with those criteria before submission.

## 0. Target and format

- Journal: Biomedical Signal Processing and Control (BSPC, Elsevier). Backups from the advisor's list, in order: Computers in Biology and Medicine, Computerized Medical Imaging and Graphics, Journal of Pathology Informatics, Informatics in Medicine Unlocked — all Elsevier, so one draft serves all.
- Write in LaTeX with the `elsarticle` class (initial submission is format-free, but revisions must follow the journal format).
- BSPC requirements (from a search summary of the Guide for Authors — confirm on the journal page before submitting): abstract < 250 words; 1–7 keywords; Highlights; graphical abstract expected; CRediT statement; data availability statement; every declaration listed in the guide (ethics, competing interests, funding, generative-AI use); figures and tables cited in order. Check the review type (single vs double anonymised) and length limits on the guide page.
- Language: formal, third person, conservative claims, numbered citations [N].

## 1. The story (every section must serve it)

> TransMIL reads its prediction from a learned class token that is identical for every bag, because a whole-slide image cannot be encoded in one pass. An ROI can. We therefore replace the class token with the ROI's own whole-image embedding (WR-TransMIL). Across three frozen pathology encoders this improves TransMIL on both validation and test, with no errors of two or more grades and fewer parameters, and controls show the gain comes from using the embedding as the readout rather than from the extra information. Scalar regression also avoids the distant errors that multi-class classification makes.

**Contributions (end of the Introduction, exactly three):**
1. WR-TransMIL: the whole-ROI embedding from the same frozen encoder replaces TransMIL's learned class token, exploiting that an ROI, unlike a WSI, can be encoded in a single pass.
2. Placement and control experiments showing the improvement arises from the readout role, not from the added information (extra token, addition to the class token, patch mean, another ROI's embedding, separate projection, another encoder).
3. A formulation comparison showing that scalar regression produces fewer errors of two or more grades than multi-class classification in all 12 aggregator–encoder pairs.

## 2. Writing order

1. Methods (facts are fixed; does not depend on the advisors' input) → 2. Results tables and figures → 3. Results text → 4. Discussion and Limitations → 5. Related Work → 6. Introduction (with the clinical co-advisor) → 7. Conclusion → 8. Abstract, title, keywords, highlights (last, so they match the final results) → 9. Declarations and cover letter → 10. Self-check (§6).

## 3. Section by section

### Title — A1
Name the task and the contribution, nothing more. E.g. *"Whole-ROI readout for transformer-based multiple instance learning in reticulin fibrosis grading of myeloproliferative neoplasms"*. Avoid "novel", "first", "state-of-the-art".

### Abstract (< 250 words) — A1
Five moves, one or two sentences each: (1) problem — reticulin grading in MPN is subjective and ordinal; MIL was designed for gigapixel slides; (2) method — WR-TransMIL, three frozen encoders, patient-level split, 50 patients / 1,330 ROIs; (3) results — test QWK .9675 / .9371 / .9616 vs TransMIL .9604 / .9283 / .9576, better on validation and test for all three encoders, no errors ≥ 2 grades, 512 fewer parameters; controls attribute the gain to the readout role; regression makes fewer distant errors than classification (7 vs 54 ROIs); (4) conclusion — one sentence, no overclaim; (5) optionally code availability. Numbers must match the tables exactly.

### Keywords — A2 (up to 7)
multiple instance learning; computational pathology; bone marrow fibrosis; myeloproliferative neoplasms; transformer; foundation models; ordinal regression.

### 1. Introduction — B1, B2, D1
Paragraphs, one idea each: (1) MPN and the clinical role of fibrosis grading (WHO: grade ≤ 1 vs ≥ 2 separates prefibrotic from overt PMF); inter-observer variability. (2) Computational pathology and MIL; MIL was built for WSIs. (3) The ROI setting: small bags (~40 patches) and the whole image is encodable — which TransMIL's learned class token ignores. (4) The idea (one sentence) and how it is tested (three encoders, placement and control experiments). (5) The three contributions. Keep BSPC's "applications-led methods for clinical diagnosis" scope visible. Sources: `wr_transmil.md` §2, `discussion_points.md` §1.

### 2. Related work — E1, E2, B1
Four short subsections: MIL aggregators in pathology (ABMIL, CLAM, DSMIL, DTFD-MIL, TransMIL and descendants CTMIL, MsCAMIL, HAG-MIL); global/context information in MIL (extra tokens MEGT/PTCMIL/ViTAGG-MIL, late fusion GMIC, thumbnail guidance SEW, multi-scale CS-MIL/ZoomMIL); input-conditioned queries outside MIL (CrossViT, Conditional/Efficient DETR, CLIP attention pooling); AI for marrow fibrosis (Virchows Archiv 2025 tool, CIF, BoMBR) and foundation encoders. End with the positioning sentence in `prior_art.md`. Only cite works whose content was checked; mark the rest to verify.

### 3. Materials and methods — C1, C2, G1
3.1 Dataset (cohort, labels, grading criteria, single rater, ethics statement — IRB number pending) · 3.2 Patient-level split (table) · 3.3 Preprocessing and frozen encoders · 3.4 Problem formulation (MIL, scalar regression, rounding) · 3.5 TransMIL recap · 3.6 WR-TransMIL (equations below; parameter count; full attention; padding) · 3.7 Compared models and controls (one reference per aggregation family; the placement variants; the six controls) · 3.8 Training and evaluation (identical protocol; metrics; why QWK + far errors; validation-based checkpointing; variants compared on the same test set) · 3.9 Implementation (hardware, software versions, runtime/parameters for efficiency, code availability). Source: `methods_and_data.md`, `wr_transmil.md` §1.

Equations for 3.6 (adapt notation):
- Patch embeddings `h_i = f(x_i)`, whole-ROI embedding `g = f(resize(X))` with the same frozen encoder `f`.
- Shared projection `φ(·) = ReLU(W·+b)`; tokens `Z⁰ = [φ(g); φ(h_1); …; φ(h_N)]` (padded to a square grid for PPEG). TransMIL instead uses `Z⁰ = [c; φ(h_1); …]` with a learned `c`.
- Two pre-norm self-attention layers with PPEG in between; `ŷ = wᵀ LN(Z²)₀ + b`; grade = clip(round(ŷ), 0, 3); loss SmoothL1(ŷ, y).

### 4. Results — C3, D2
4.1 Main comparison (5 models × 3 encoders; val and test QWK, MAE, accuracy, macro F1, far errors; `results.md` §2) · 4.2 Per-grade behaviour (recall table or confusion-matrix figure; state UNI2-h G1) · 4.3 Placement of the whole-ROI embedding (`results.md` §5) · 4.4 Controls (`results.md` §6) · 4.5 Formulation (`results.md` §7) · 4.6 Acquisition-format check (PMF27). Text reports numbers and direction only; interpretation goes to the Discussion. Report the UNI2-h counter-result (ABMIL best) in 4.1, not only in Limitations.

### 5. Discussion — F1
Use `discussion_points.md` §1: role vs information; relation to image-conditioned queries (CLIP, DETR); encoder dependence; formulation; PPEG at ROI scale; the G1/G2 boundary; the format check. Compare with prior fibrosis-grading work where comparable (different settings — say so). Implications: for ROI-scale MIL, reuse the whole-image summary the setting affords.

### 6. Limitations and future work — F2
From `discussion_points.md` §2, each with a specific future step: external validation (e.g. an independent cohort), a second rater, more patients per extreme grade, other token-based aggregators, weighted cross-entropy for the classification baseline.

### 7. Conclusion
Three or four sentences restating the contribution and the evidence; no new claims.

### Highlights (3–5 bullets; Elsevier usually limits each to 85 characters — confirm)
- Whole-ROI embedding replaces TransMIL's class token for ROI-level MIL
- Improves TransMIL on three frozen pathology encoders with fewer parameters
- Controls attribute the gain to the readout role, not added information
- Scalar regression avoids the distant grade errors of classification
- Evaluated on 1,330 reticulin ROIs from 50 MPN patients

### Graphical abstract
The three-row token diagram (Fig 2) plus the main result line (test QWK TransMIL → WR-TransMIL on the three encoders). Check the journal's size requirements.

### Cover letter (one page)
What is new (the readout idea in one sentence), why it fits BSPC (applications-led method for a clinical grading task), what evidence supports it (three encoders, controls, formulation), and a statement that the work is not under consideration elsewhere. Suggest 3–5 reviewers from computational pathology / MIL (verify each).

### Declarations — G1, G2
Ethics approval (number) and consent; CRediT; competing interests; funding; data availability (e.g. available on reasonable request subject to ethics approval); code availability (release a clean repository with WR-TransMIL and the evaluation script — not the ~500 exploratory modules); declaration of generative-AI use in writing.

## 4. Figures and tables

| # | content | criterion |
|---|---|---|
| Fig 1 | pipeline: ROI → patches + whole ROI → frozen encoder → WR-TransMIL → grade | D2 |
| Fig 2 | token sequences of TransMIL / + whole-ROI token / WR-TransMIL (three rows) | D2, B1 |
| Fig 3 | confusion matrices (TransMIL vs WR-TransMIL, 3 encoders) | C3 |
| Table 1 | dataset and split by grade | C1 |
| Table 2 | main comparison, 5 models × 3 encoders | C3 |
| Table 3 | placement of the whole-ROI embedding | C3 |
| Table 4 | controls | C3 |
| Table 5 | regression vs multi-class (QWK and far errors) | C3 |
| Graphical abstract | token diagram + main result | journal requirement |

Captions must be self-contained (split, metric, which number is best). Never show uncropped ROIs (slide-label thumbnail); if an example ROI is shown, crop the overlay and scale bar.

## 5. Claims and language

Follow `rules.md` §3 and `wr_transmil.md` §7. Claim consistency, not size; "to our knowledge"; no "first/novel/state-of-the-art/significant"; no module codes; no statements about what attention looks at.

## 6. Pre-submission self-check (run the manuscript-evaluation criteria on the draft)

| criterion | must be true before submission |
|---|---|
| A1 | title names task + contribution; abstract states objective, method, main numbers and a modest conclusion; numbers match the tables |
| A2 | 5–7 specific keywords |
| B1 | contribution stated as one idea + three bullets; related principles cited (CrossViT, CLIP, DETR, SEW); "to our knowledge" |
| B2 | clinical motivation (WHO grading boundary) and BSPC scope explicit |
| C1 | dataset, split table, preprocessing, protocol and metrics reproducible from the text; selection on validation stated; test-set reuse for variants disclosed |
| C2 | hyperparameters, hardware, parameter counts, full-attention note; efficiency mentioned (parameters; inference cost if measured) |
| C3 | all tables complete; UNI2-h counter-result reported; no significance claimed; limitations and biases acknowledged |
| D1 | IMRaD order; one idea per paragraph; Results report, Discussion interprets |
| D2 | every figure/table cited in order with self-contained captions |
| E1/E2 | related work covers the four areas; every citation checked against the source |
| F1 | findings interpreted against CLIP/DETR/context-MIL and the clinical boundary |
| F2 | limitations with concrete future steps |
| G1 | ethics number, consent, CRediT, funding, competing interests, data and code availability present |
| G2 | generative-AI declaration present |
| H | expected outcome for the current evidence: major or minor revision at BSPC; the blocking items are G1 (ethics number) and C1 (test-set reuse disclosure), not experiments |

**Known gaps that the writing alone cannot close** (state them; do not hide them):
- C3 "statistically validated": no significance testing by design (10 test patients). Claim consistency across encoders; if a reviewer asks, compute a patient-level bootstrap from the saved predictions (no retraining) — only after the user agrees.
- C2 "computational efficiency": parameter counts are available; inference time has not been measured. Measuring it needs the user's go-ahead (inference only).
- G1: the ethics approval number must come from the advisors.
- The multi-class baseline used uniform class weights; weighted cross-entropy was never run — say so if asked.
