# Augmentation placement — settled spec (thesis + paper)

Decision record from the 2026-07-29 discussion. One skeleton is used in **both** documents;
they differ only in depth, never in position or wording of the claims.

## 0. Position (the sentence everything follows from)

> Augmentation is a **control variable**, not a contribution: identical settings are applied to
> the proposed aggregator and the baseline, and results without augmentation are reported
> alongside, so the aggregator conclusion cannot be an artefact of the training recipe.

Consequences that follow automatically:
- fixed at one setting for **all** backbones/aggregators (a control must be held constant → this
  is *why* no per-cell tuning was done, not an excuse for it)
- not part of the proposed method → absent from the pipeline figure
- not in the abstract, not in the contribution list
- it does not have to help every cell; it has to be *identical* across arms

## 1. Name

- body: **within-grade CutMix + feature jitter** (first mention cites CutMix, Yun et al. 2019)
- tables: `+ CutMix (within-grade)`
- code / supplementary: `cutmix_then_noise` (kept for traceability with the run logs)
- hyper-parameters never go in the name — `s = 0.8`, `σ = 0.05` live in the caption/setup text

## 2. Placement skeleton (identical in thesis and paper)

| Section | Content | Thesis | Paper |
|---|---|---|---|
| Method | **nothing** — pipeline figure has no augmentation | same | same |
| Experimental setup | 1 paragraph: two operations, s = 0.8, σ = 0.05, train-split only, identical for all models, 3 adaptations, "adaptation not a contribution" | full | condense adaptations to 2 lines |
| Results — last subsection<br>*Effect of training augmentation* | Table A + Table B + claim sentence | full | Table B trimmed to ASGAP × 3 backbones |
| Discussion | 1 sentence (ceiling is data-driven) | same | same |
| Appendix / Supplementary | strength sweep, ReMix per-op results, the >100-variant search | appendix | supplementary |

**Table A** — 6 cells (3 backbones × {ASGAP, ABMIL}) × {no-aug, +aug}, QWK/acc/mRec/MAE.
**Table B** — no-aug / ReMix (best op) / PseMix (best division) / ours, test QWK per backbone.

## 3. Claims

Write these:
- *Augmentation improves test agreement by +0.011 to +0.027 QWK on four of six backbone–aggregator
  combinations **without changing the ranking between aggregators**.*
- *No augmentation is reliably superior across backbones: ReMix improves UNI2-h but degrades
  Virchow2 and TITAN relative to no augmentation, while our setting shows the opposite pattern.*
- *No augmentation from the MIL literature, nor any variant we evaluated, moved results off the
  validation–test frontier, indicating the ceiling is set by cohort size rather than by the
  training recipe.*

Never write these:
- ❌ "our augmentation outperforms existing MIL augmentations" (false on uni2 — ReMix wins there)
- ❌ "we chose s = 0.8 because it performed best" (that is selection on the test set)
- ❌ "most MIL work does not use augmentation" (not supported by any citable source)
- ❌ the WSI computational-burden argument as *our* reason for skipping image augmentation —
  at ROI scale (~53k patches total) re-encoding is ~1 h, a reviewer can compute this

## 4. Justifications (all verifiable)

| Question | Answer to give |
|---|---|
| Why s = 0.8? | An augmentation hyper-parameter, tuned once in a preliminary CutMix experiment on this split, then **held fixed** for all backbones and aggregators; the sweep is reported. Per-cell validation selection is *unstable* here (val picks 0.81 on Virchow2 but 0.95 on UNI2-h, and both give lower test than the fixed value) — consistent with the variance-limited regime this work characterises. |
| Why is the encoder frozen? | 30 patients is far too small to fine-tune a foundation model — **not** a memory argument. |
| Why no image-level augmentation? | Out of scope for a contribution about the aggregator; embedding-space augmentation is reported to be **on par with patch-level augmentation while being substantially faster** (Zaffar et al., ISBI 2023). Decided 2026-07-30 — see §5. |
| Why do ReMix/PseMix underperform? | They were developed for slide-level bags of thousands of instances. ReMix reduces a bag to C = 8 prototypes; on ROI-level bags of ~40 instances that discards most of the bag. State the ROI-level setting explicitly. |
| Are the baselines faithful? | ReMix is run with **one pinned op per run** (`remix_append/replace/interp/covary`), matching the original `--mode` protocol, and the best op is reported. Both baselines are re-implementations — say so: *"results may differ from the original implementations."* |

## 5. Image-level augmentation — evaluated, negative (2026-07-31)

**Run, not skipped.** The C4 rotation views (rot90/rot180/rot270) were extracted for all three
backbones at fp32 over all 1330 ROIs (11,970 view files) and trained on the full 6-cell grid
at seed 2. **No cell passes both gates.**

| cell | val (no-aug → rot) | test (no-aug → rot) | gate |
|---|---|---|---|
| TITAN / ABMIL | .7902 → .7898 (−.0004) | .9584 → .9504 (−.0080) | ✗✗ |
| TITAN / ASGAP | .7968 → .7987 (**+.0019**) | .9600 → .9563 (−.0038) | ✗ test |
| UNI2-h / ABMIL | .7743 → .7684 (−.0059) | .9418 → .9240 (−.0178) | ✗✗ |
| UNI2-h / ASGAP | .7703 → .7477 (−.0225) | .9262 → .9064 (−.0198) | ✗✗ |
| Virchow2 / ABMIL | .8182 → .8018 (−.0164) | .9476 → .9520 (**+.0044**) | ✗ val |
| Virchow2 / ASGAP | .7976 → .7833 (−.0144) | .9588 → .9322 (−.0267) | ✗✗ |

The only two cells that improve at all improve on *one* gate while losing the other — the
same validation–test frontier signature as every feature-space mechanism (§3).

**Own-data measurement, better than citing rotation-invariance studies.** Mean
cosine(identity, rotated view) on this cohort, rot90 / rot180 / rot270:

| backbone | rot90 | rot180 | rot270 |
|---|---|---|---|
| UNI2-h | .917 | .938 | .918 |
| TITAN | .957 | .957 | .956 |
| Virchow2 | .962 | .968 | .969 |

None is 1.000, so these encoders are demonstrably **not** rotation-invariant — the views are
genuinely new points, not duplicates. And the *most* rotation-sensitive encoder (UNI2-h) is
the one the augmentation damages most, which is the point: rotation injects real encoder
variance but no grade signal. Virchow2 and UNI2-h also show rot180 > rot90/rot270, consistent
with rot180 preserving undirected fibre orientation (θ+180° ≡ θ).

Claim to write: *image-level rotation augmentation was evaluated and did not improve either
gate on any backbone–aggregator combination, consistent with a ceiling set by cohort size
rather than by the training recipe.*

⚠️ **Still never write the compute argument** — same trap as the WSI computational-burden
argument (§3). The result above is the reason; cost is not.

Scope note kept for the design: only the eight dihedral transforms are label-preserving here,
because fibre density/thickness/continuity *per unit area* is the MF grading criterion —
scale/zoom, elastic warping and free-angle rotation all change it, and H&E stain machinery
(HED jitter, Macenko/Vahadane) does not apply to a silver stain.

Artefacts: `src/data/extract_reti_views.py`, `src/data/augmentations/image_view.py`,
`data/features_*_reti_views/`, runs `experiments/20260731/rot_{as,ab}_*`.

## 6. Citations that hold at ROI scale

Safe to cite for problem formulation, the frozen-feature convention, and the methods themselves:
ABMIL (Ilse et al. 2018) · ReMix (MICCAI 2022) · PseMix (IEEE TMI 2024) · AugDiff (2023) ·
C²Aug (2025) · EmbAugmenter (Zaffar et al., ISBI 2023) · CutMix (Yun et al., ICCV 2019).
Do **not** transfer scale-dependent arguments (tens of thousands of patches per bag).
