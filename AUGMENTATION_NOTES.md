# Augmentation Search Notes

Machine-readable analysis log for the **train-time feature-space augmentation** search on the
MPN reticulin cohort, mirroring `NOVELTY_NOTES.md` (the MIL-aggregator search). Each augmentation
is a pluggable module under `src/data/augmentations/<name>.py` (registry: `augmentation-registry`
memory), selected at train time with `--augmentation <name> --aug_strength <float>`. Default off =
byte-identical baseline.

- **Schema**: 1.0.0
- **last_updated**: 2026-06-16
- **anchor model**: a215 (`a215_learnable_entmax`), backbone TITAN, seed 2, regression, QWK selection.
- **runner**: `/tmp/aug_worker.sh` (parallel, `xargs -P 4`); canonical training config = the locked
  novelty-search config (epochs 50, lr 1e-4, batch 1, patience 15, regression, main_metric qwk).

## Acceptance bar (3 gates — current, user-set 2026-06-16)

A candidate is a **WIN** only if ALL THREE hold at seed-2 vs the a215 baseline:

| gate | baseline | rule |
|---|---|---|
| val_qwk  | 0.7968032916600728 | candidate > baseline |
| test_qwk | 0.9600366576113638 | candidate > baseline |
| **test G1 recall** | 77.55 | candidate > baseline (user added 2026-06-16) |

Baseline a215 (aug off, seed-2), all metrics:
- VAL : qwk 0.7968 / mae 0.327 / acc 70.56 / f1 0.652 / mRec 66.78 / G0-3 = 38/77/55/97 (ep10)
- TEST: qwk 0.9600 / mae 0.120 / acc 88.42 / f1 0.861 / mRec 87.05 / G0-3 = 89/78/87/95

> Honesty protocol (inherited from the MIL search): a seed-2 pass is necessary, not sufficient —
> flag near-init (low best_epoch), disclose every run, never claim a fake win, treat a single
> seed-2 pass on the −0.95 val↔test trap as a lucky-fold candidate until shown otherwise. User has
> opted OUT of multi-seed for now ("seed-2 ชนะก็พอ"), so seed-2 + the 3-gate bar is the working
> standard, but the caveats stay in the writeup.

## Results (seed-2, representative strength per family)

| run | val_qwk | test_qwk | test G1 | ep | v / t / G1 | verdict |
|---|---|---|---|---|---|---|
| **baseline a215** | 0.7968 | 0.9600 | 77.6 | 10 | — | anchor |
| mixup_inst α=1.0 | 0.8072 | 0.9366 | 73.5 | 13 | ✓✗✗ | fail |
| instance_dropout p=0.15 | 0.7797 | 0.9556 | 77.6 | 10 | ✗✗✗ | fail |
| feature_noise s=0.05 | 0.7754 | 0.9628 | 77.6 | 6 | ✗✓✗ | fail |
| mixup_within_grade s=1.0 | 0.7806 | 0.9645 | 73.5 | 10 | ✗✓✗ | fail |
| bag_bootstrap s=1.5 | 0.7785 | 0.9606 | 73.5 | 6 | ✗✓✗ | fail |
| feature_dropout p=0.1 | 0.7840 | 0.9552 | 65.3 | 11 | ✗✗✗ | fail |
| multimix α=0.5 | 0.7978 | 0.9332 | 69.4 | 25 | ✓✗✗ | fail |
| compose_drop_noise s=0.2 | 0.7837 | 0.9596 | 73.5 | 19 | ✗✗✗ | fail |
| **fibrosis_axis_shift@0.2** | **0.8113** | **0.9629** | 71.4 | 17 | **✓✓✗** | near (G1) |
| fibrosis_axis_shift@0.4 | 0.7924 | 0.9499 | 85.7 | 2 | ✗✗✓ | fail (near-init) |
| feature_scale s=0.05 | 0.7924 | 0.9521 | 65.3 | 15 | ✗✗✗ | fail |
| feature_scale s=0.10 | 0.7789 | 0.9629 | 75.5 | 6 | ✗✓✗ | fail |
| intrabag_spread s=0.2 | 0.8054 | 0.9596 | 65.3 | 23 | ✓✗✗ | fail |
| intrabag_spread s=0.4 | 0.7883 | 0.9558 | 83.7 | 2 | ✗✗✓ | fail (near-init) |
| patch_mixup_self s=0.3 | 0.7884 | 0.9598 | 63.3 | 20 | ✗✗✗ | fail |
| patch_mixup_self s=0.5 | 0.7822 | 0.9593 | 69.4 | 22 | ✗✗✗ | fail |
| **prototype_interp@0.2** | 0.7911 | **0.9627** | **81.6** | 6 | **✗✓✓** | near (val) |
| **prototype_interp@0.4** | **0.7974** | 0.9289 | **81.6** | 19 | **✓✗✓** | near (test) |

## Family-by-family analysis

1. **mixup_instance_pool** (2-bag ordinal instance mix) — α-sweep {0.1..2.0}. Pure val↔test trap:
   raises val (α=1.0 → val MAX 0.8072), tanks test (0.9366). G1 ≤73.5. Closed.
2. **instance_dropout / feature_noise / bag_bootstrap / feature_dropout / compose** — generic
   regularisers. Several reach test>0.9600 (fnoise0.05 0.9628, mixwg1.0 0.9645, boot1.5 0.9606) but
   ALL with val<baseline; none lifts G1. Generic noise can buy test_qwk (mostly via G3 swing) but
   never val AND G1.
3. **mixup_within_grade** — same-grade recombination, test up (0.9645) but val down, G1 down. Trap.
4. **multimix** (3-bag) / **patch_mixup_self** (intra-bag manifold) — both val↑/test↓ or flat; G1 ≤69.
5. **fibrosis_axis_shift** (translate whole bag along learned fibrosis axis v=mean(G2,G3)−mean(G0,G1),
   target shifted by calibrated slope) — **first to pass BOTH QWK gates** (@0.2: val 0.8113/test 0.9629,
   well-trained ep17). BUT G1=71.4 <78 → fails the G1 gate (its test_qwk gain rides on G3 95→99, not
   G1). At strength 0.4 G1 jumps to 85.7 but QWK collapses and ep=2 (near-init). Mechanism: a single
   global axis pushes toward the high-density end → helps G3/extremes, not the G1 boundary.
6. **feature_scale / intrabag_spread** — multiplicative gain / within-bag spread jitter. spread@0.2
   passes val only; high strength lifts G1 (spread@0.4 → 83.7) but near-init + QWK fail. Trap.
7. **prototype_interp** (translate a bag a t-fraction from its own grade centroid toward a RANDOM
   other grade centroid; target interpolated) — **the strongest lead for the 3-gate (G1) goal.**
   - @0.2: test 0.9627 ✓ + **G1 81.6 ✓** + acc 89.2 (>88.4) + mae 0.112 (<0.120), well-trained ep6 —
     test-side near-dominates baseline; **fails ONLY val (0.7911, −0.0057)**, dragged by val G0 38→29.
   - @0.4: val 0.7974 ✓ + G1 81.6 ✓ but **test 0.9289 ✗** — test G0 collapses 89→65 (a G0 bag pulled
     toward a far grade centroid, e.g. G3, is mis-graded). val G0 also 14.
   - **Mechanism**: interpolating between grade centroids manufactures intermediate-density bags →
     stably enriches G1 (81.6 at both strengths). strength↑ raises val (more middle-grade signal) but
     crashes test G0 (extreme bags dragged too far). Sweet spot, if any, ∈ (0.2, 0.4).

## Key insight (the bar is a 3-way Pareto frontier)

The three best candidates each pass a DIFFERENT 2 of the 3 gates:

| candidate | val | test | G1 | fails |
|---|---|---|---|---|
| fibrosis_axis_shift@0.2 | ✓ | ✓ | ✗ | G1 (rides G3, not G1) |
| prototype_interp@0.2 | ✗ | ✓ | ✓ | val (G0 on val cohort) |
| prototype_interp@0.4 | ✓ | ✗ | ✓ | test (G0 collapse) |

So the 3-gate bar is the **val↔test −0.95 trap × the G1↔(G0/G3) recall trade-off** simultaneously.
"Improve G1" pushes mass toward the middle grades → helps G1/val but hurts the extreme grades (G0
recall on the 2-patient test/val cohorts) → drops test_qwk. The binding constraint is **protecting
G0 (and G3) recall while enriching G1**.

## Analysis-driven next candidates (queue)

Derived from the failure modes above, not random:

- **`prototype_interp_adj`** (written, smoke-tested 2026-06-16) — proto_interp but interpolation
  target restricted to an ADJACENT grade (|g_b−y|=1): G0 moves ONLY toward G1, G3 only toward G2.
  Directly fixes proto@0.4's G0 collapse (G0 never dragged toward G3) while keeping the G1-enrichment
  that makes proto the lead. Queue: strengths {0.3, 0.5, 0.7} (`/tmp/aug_jobs_adj.txt`).
- **`g1_synth_mixup`** (written, smoke-tested) — synthesises explicit G1 bags from G0+G2 instance
  mixes (λ=0.5 → target 1.0); only fires on G0/G2 bags. Queue: {0.3, 0.5, 0.8}.
- **In progress** (`bwkd6jjo9`): prototype_interp fine-sweep {0.22, 0.25, 0.28, 0.30, 0.35} — looking
  for the val-crossing strength that keeps test>0.9600 and G1>78.
- **Backlog ideas if the above stall**: (a) class-balanced sampling weight (oversample G1 train bags);
  (b) proto_interp with the shift magnitude capped per-grade so extremes move less; (c) target-region
  reweighting (augment more around the under-represented G1 target band).

## prototype_interp fine-sweep (2026-06-16) — DISJOINT-STRENGTH finding (family closed for 3-gate)

Fine-swept strength {0.20, 0.22, 0.25, 0.28, 0.30, 0.35, 0.40} + g1_synth_mixup {0.3,0.5,0.8}:

| strength | val | test | G1 | ep | v/t/G1 |
|---|---|---|---|---|---|
| 0.20 | 0.7911 | **0.9627** | 81.6 | 6 | ✗✓✓ |
| 0.22 | 0.7907 | 0.9511 | 81.6 | 28 | ✗✗✓ |
| 0.25 | 0.7930 | 0.9491 | 79.6 | 33 | ✗✗✓ |
| 0.28 | 0.7976 | 0.9372 | 79.6 | 43 | ✓✗✓ |
| 0.30 | 0.7981 | 0.9447 | 79.6 | 25 | ✓✗✓ |
| 0.35 | 0.8084 | 0.9403 | 71.4 | 33 | ✓✗✗ |
| 0.40 | 0.7974 | 0.9289 | 81.6 | 19 | ✓✗✓ |

**Finding**: val rises monotonically with strength (crosses 0.7968 at ~0.28) while test falls
monotonically (0.9627 → 0.9289). The val-passing region (strength ≥0.28) and the test-passing
region (strength ≤0.20) are **DISJOINT** — no strength threads both QWK gates. Higher strengths
train longer (ep28-43, not underfit) yet test still drops → genuine val↔test trap, not tuning.
G1 stays ≥79.6 across 0.20-0.30 (the family reliably fixes G1). g1_synth_mixup all fail (forcing
G0/G2→G1 destroys signal; @0.8 gets G1 91.8 but test collapses to 0.808, near-init ep3).
**Conclusion: prototype_interp (random-target) cannot 3-PASS** — blocked by G0 collapse at the
strength val needs. → motivates `prototype_interp_adj` (protect extremes). Best of family stays
proto@0.20 (✗✓✓, val −0.0057).

## prototype_interp_adj (2026-06-17) — G0-protection works but val stays flat → no 3-PASS

Adjacency-only variant {0.3, 0.4, 0.5, 0.7}: val 0.7839-0.7905 (FLAT, never crosses 0.7968),
test 0.9443-0.9573, G1 69-80, **G0 protected (91/91/80/90 vs proto@0.4's 65)**. Hypothesis half-
confirmed: restricting shifts to adjacent grades DOES protect G0 (and test doesn't crash as hard:
adj@0.4 test 0.9573 vs proto@0.4 0.9289). BUT the local shift no longer pushes enough middle-grade
signal to lift val — val is stuck ~0.790 at every strength. Several near-init (ep2-4). So adj trades
the val-boost away for G0-safety → still no strength threads all 3. Best adj@0.4 (✗✗✓, G0 safe).

## Two cleanest near-misses remain (the 3-way frontier)

- **fibrosis_axis_shift@0.2**: ✓✓✗ — passes BOTH QWK (val 0.8113 +0.0145, test 0.9629 +0.0029),
  G1=71 (needs +7). Comfortable QWK margin → room to trade for G1?
- **prototype_interp@0.2**: ✗✓✓ — test 0.9627 + G1 81.6, val 0.7911 (needs +0.0057).

→ Next probe (running `bybum7k2d`): fibrosis_axis_shift fine-sweep {0.22, 0.25, 0.28, 0.30} —
axis@0.4 reaches G1 85.7, so a mid strength may give G1≥78 while QWK stays above gate (axis@0.2
already clears both QWK). If that window doesn't exist, build an axis×G1 hybrid (axis drives val+
test; an added G1-ward nudge for G0/G2 lifts G1) — axis and proto pass complementary gates.

## axis_g1_hybrid (2026-06-17) — best test+G1 ever, but VAL is the structural wall

Composed augmentation: every bag gets fibrosis_axis_shift@0.2 (val+test driver) + a G0/G2 bag is,
with prob q, nudged toward the G1 centroid (G1 booster). Sweep q {0.3,0.5,0.7,1.0}:

| q | val | test | G1 | G0 | G3 | ep | v/t/G1 |
|---|---|---|---|---|---|---|---|
| 0.3 | — | 0.9432 | 59.2 | 90 | 94 | — | ✗✗✗ |
| 0.5 | — | 0.9449 | 65.3 | 85 | 98 | — | ✗✗✗ |
| **0.7** | 0.7846 | **0.9658** | **85.7** | 86 | 98 | 12 | **✗✓✓** |
| 1.0 | 0.7849 | 0.9574 | 79.6 | 85 | 98 | 15 | ✗✗✓ |

**hyb q=0.7 is the strongest TEST result of the whole search**: test_qwk 0.9658 dominates baseline on
EVERY test metric (mae 0.108<0.120, acc 89.2>88.4, f1 0.869>0.861, mRec 88.1>87.05) AND test G1 85.7
(>78), well-trained ep12. But **val 0.7846 < 0.7968** — dragged by val G3 96.7→91.7 (a 2-patient
val grade). Joins proto@0.2 in ✗✓✓.

## CONVERGENT FINDING — the binding gate for "improve G1" is always VAL (val↔G1 tension)

Three independent near-misses, each from a different mechanism, confirm the 3-gate bar is a
cohort-driven Pareto frontier: you can get **(val+test)** OR **(test+G1)** but never all three.

| candidate | val | test | G1 | passes | fails |
|---|---|---|---|---|---|
| fibrosis_axis_shift@0.2 | ✓ 0.8113 | ✓ 0.9629 | ✗ 71.4 | val+test | G1 |
| prototype_interp@0.2 | ✗ 0.7911 | ✓ 0.9627 | ✓ 81.6 | test+G1 | val |
| axis_g1_hybrid q=0.7 | ✗ 0.7846 | ✓ 0.9658 | ✓ 85.7 | test+G1 | val |

Mechanism: improving G1 recall pushes prediction mass toward the middle grades, which (on the
10-patient val cohort, 2 patients each for G0/G3) drops val_qwk via a 2-patient extreme grade
(val G3 here, val G0 elsewhere) — the same −0.95 val↔test/cohort trap that blocks MIL GATE2. The
only candidate with val>0.7968 (axis@0.2) is precisely the one that does NOT improve G1. So under
the **2-gate** bar (val+test) axis@0.2 is a clean seed-2 win; under the **3-gate** bar (+G1) the
val gate and the G1 gate are in direct tension and no augmentation has cleared all three.

→ Next probes (if continuing): hybrid q∈{0.55,0.6,0.65} with a GENTLER G1 nudge (lower _G1_MAX) to
lift G1 without dropping val G3; but the convergent evidence says val is structurally the wall.

## axis_g1_gentle (2026-06-17) — minimal-nudge probe FAILS → val↔G1 tension confirmed decisive

Tunable-magnitude G1 nudge {0.08, 0.12, 0.16, 0.20} on top of axis@0.2: val 0.7768-0.7882 (ALL
below 0.7968), G1 65-76 (all below 78), test mixed. **Even the gentlest nudge (0.08) drops val
from axis@0.2's 0.8113 to ~0.78 AND still fails to lift G1 to 78.** The linear-extrapolation hope
(small nudge → small val loss → threadable) is false: val is hypersensitive to ANY G1 nudge while
G1 needs a large nudge to cross 78 → no threadable point. CONCLUSIVE.

## CONCLUSION (2026-06-17): the 3-gate bar (val + test + G1) is STRUCTURALLY BLOCKED

After ~63 augmentation runs across 15 families + 4 fine-sweeps, **0 clear all three gates**. The
evidence is convergent and mechanistic, not "not-yet-found":

- **axis@0.2** is the ONLY model with val>0.7968 — and it does NOT improve G1 (71). The instant any
  G1-enrichment is added (proto / hybrid / gentle, at ANY strength), val falls below 0.7968.
- The strongest TEST model found — **axis_g1_hybrid q=0.7** — beats baseline on EVERY test metric
  (qwk 0.9658, acc 89.2, mae 0.108, f1 0.869, mRec 88.1) **and** improves test G1 to 85.7, but
  val-selection (val 0.7846) would NOT pick it. This is the val↔test −0.95 trap made concrete: the
  locked val-selection rule is **anti-informative** for the G1 goal on this cohort.
- Root cause = the same one that blocks MIL GATE2: only 2 patients per extreme grade (G0/G3) on the
  10-patient val cohort, so improving G1 (pushing mass to the middle) drops a 2-patient val grade →
  val_qwk falls. The only fixes (val-selection rule, patient split, loss) are thesis-LOCKED.

**Two honest readings (user's call):**
1. STRICT protocol (select on val): no 3-PASS — the val gate blocks every G1-improving model.
2. ACTUAL test quality: `axis_g1_hybrid q=0.7` is a genuinely better TEST model (dominates baseline +
   G1 85.7); it just isn't val-selectable. Defensible thesis point: augmentation CAN improve test G1,
   but val-selection won't choose it → reinforces the val↔test anti-correlation finding.
3. Under the ORIGINAL 2-gate bar (val+test, before G1 was added): **axis@0.2 is a clean seed-2 win**
   (val 0.8113 / test 0.9629, well-trained ep17) — the first augmentation to beat both QWK gates.

## NEW LEVER (2026-06-17): --sampler_weights (patient/class-balanced WeightedRandomSampler) FLIPS the trade-off toward val

User picked "try a new lever". `--sampler_weights` (built-in, weight ∝ 1/(patients_per_class ×
bags_per_patient); never used in this search) oversamples under-represented classes/patients. It is
a TRAINING-DISTRIBUTION lever, orthogonal to feature augmentation. Sweep (titan seed-2):

| config | val | test | G1 | G0 | G3 | ep | v/t/G1 |
|---|---|---|---|---|---|---|---|
| sampler only | **0.8206** | 0.9495 | 71.4 | 84 | 98 | 10 | ✓✗✗ |
| sampler + axis@0.2 | 0.7901 | 0.9410 | 75.5 | 78 | 98 | 15 | ✗✗✗ |
| sampler + proto@0.2 | 0.8064 | 0.9338 | 77.6 | 73 | 89 | 21 | ✓✗✗ |
| **sampler + hyb@0.5** | **0.8049** | 0.9559 | **85.7** | 79 | 100 | 8 | **✓✗✓** |

**Finding**: the balanced sampler is a strong VAL booster (sampler-only val 0.8206, +0.024) because
the val cohort is class-balanced — but it costs test. This FLIPS the failure mode: instead of
val-fail (the augmentation regime), we now get **test-fail**. `sampler+hyb@0.5` is a new **✓✗✓**
near-miss: val 0.8049 (margin +0.008) + G1 85.7 (margin +8), **test 0.9559 short by only 0.0041**,
well-trained ep8. Now all 3 corners of the 2-of-3 Pareto are reached by different configs
(axis@0.2 val+test / hyb q=0.7 test+G1 / sampler+hyb@0.5 val+G1). Since sampler+hyb@0.5 has spare
val+G1 margin, a milder G1 push (lower hyb q) may recover the +0.004 test → probing
sampler×{hyb 0.2/0.3/0.4, gentle 0.12} (`b7p1k53sx`).

## tempered sampler (--sampler_temp beta, 2026-06-17) — bridge FAILS, frontier confirmed

Added `--sampler_temp` (w**beta, beta=1 full balance / 0 uniform). Swept hyb@0.5 × beta {0.3,0.5,0.7}
+ sampler-only beta=0.5: test stays ≤0.9573 at EVERY beta (best temp-only-0.5 = 0.9573), and when the
sampler is weak enough that test could recover, val drops (temp-only-0.5 val 0.7850). No beta threads.

## FINAL CONCLUSION (2026-06-17): 3-gate bar blocked across THREE lever families — baseline is ON the Pareto frontier

| lever | binding wall | best result |
|---|---|---|
| feature augmentation (no sampler) | **val** fails when G1 improves | hyb q=0.7: test 0.9658 + G1 85.7, val 0.7846 |
| full balanced sampler (beta=1) | **test** caps ~0.956 | sampler+hyb@0.5: val 0.8049 + G1 85.7, test 0.9559 |
| tempered sampler (beta 0.3-0.7) | test does not recover | none |

You can buy any **2 of {val>0.7968, test>0.9600, G1>78}** but never all three at seed-2; WHICH gate
fails depends on the lever (augmentation→val, sampler→test). Root cause = the same cohort limit that
blocks MIL GATE2 (2 patients per extreme grade on the 10-patient val/test cohorts). Ceiling-lifting
levers (val-selection rule, patient split, loss) are thesis-LOCKED. Closest single config:
**sampler+hyb@0.5** (val+G1 pass, test −0.0041).

THESIS-USABLE OUTCOMES (no 3-PASS, but real): (1) **axis_shift@0.2** = first augmentation to beat both
QWK gates (val 0.8113/test 0.9629) — a 2-gate seed-2 win + interpretability tie-in (augment along the
fibrosis density axis). (2) **axis_g1_hybrid q=0.7** = test model that beats baseline on every test
metric + G1 78→85.7, but val-selection won't pick it = concrete val↔test anti-correlation evidence.
(3) The 3-lever Pareto-frontier characterization itself = the methodological contribution.

## boundary_contrast (2026-06-17) — FAILS (worst yet); fractional-target boundary mixing confuses the head

bc {0.3,0.5,0.7} ± sampler: val 0.78-0.79, test 0.94-0.95, G1 67-75 — fails ALL gates, worse than
mass-shifting approaches. Sub-integer (0.5/1.5) boundary targets blur rather than sharpen on this
tiny cohort. Last distinct mechanism → exhausted.

## EXHAUSTION DECLARED (2026-06-17): 3-gate bar (val+test+G1) unreachable at seed-2 across all reachable levers

~75 runs / 16 augmentation families + sampler + tempered-sampler, 7 sweeps, **0 full 3-PASS**. The
val↔test↔G1 Pareto frontier is strict and passes through (not above) the baseline:
- augmentation (mass-shift toward G1) → val fails
- balanced sampler → test caps ~0.956
- tempered sampler (bridge) → test never recovers above 0.96 while val/G1 hold
- boundary_contrast (sharpen, not shift) → fails all three
The only ceiling-lifting levers left (val-selection rule, patient split, loss, formulation) are
thesis-LOCKED. Further drop-in augmentation/sampling = churn (mirrors the MIL-aggregator GATE2
exhaustion). Best artifacts to carry forward: axis@0.2 (2-gate win), hyb q=0.7 (test dominates +
G1 85.7, not val-selectable), sampler+hyb@0.5 (val+G1, test −0.004), and the Pareto characterization.

## Post-exhaustion probes (user chose "keep firing", 2026-06-17) — 2 more NEW mechanism classes, both land on the frontier

- **coverage_resample** (resample patches by fibrosis-axis projection → perturb coverage; NEW class:
  distribution-resampling not translation): {1.0,2.0,3.0}±sampler. val ✓ all (up to 0.8134) but G1
  DROPS (63-76) and test <0.96 → a val-booster that HURTS G1. ✓✗✗.
- **feature_smote** (same-grade nearest-neighbour interpolation; NEW class: structured within-grade):
  {0.3,0.5,0.7}±sampler. G1 stays 55-76 (interpolating among existing G1 doesn't sharpen the
  boundary); smote@0.5 test 0.9607✓ but val 0.78/G1 71. ✗✓✗ / ✗✗✗.

Both confirm the frontier from new angles: val-boosters don't lift G1; G1 needs boundary-sharpening
that no augmentation provides (augmentation can't add information not in the frozen features).

## Status

- ~83 aug/sampler runs / 18 augmentation families + 2 sampler levers, 9 sweeps, **0 full 3-PASS**.
- Mechanism CLASSES covered (exhaustive): patch-mix, patch-drop, feature-noise, dim-drop, bag-resample,
  spread-jitter, directional-translation, coverage-resample, within-grade-interp, balanced/tempered
  sampling. Every class lands on the val↔test↔G1 Pareto frontier through the baseline.
- HONEST STOP: further drop-in augmentation = theater (outcome predetermined by the frontier; no
  augmentation adds information absent from the frozen features). Real artifacts: axis@0.2 (2-gate win),
  hyb q=0.7 (test dominates + G1 85.7, not val-selectable), the 3-lever Pareto characterization.

## cov_hyb (2026-06-17) — stack coverage(val-boost) × hyb(test+G1), user "keep firing": confirms frontier again

Hypothesis: hyb q=0.7 clears test+G1 WITHOUT the sampler (test 0.9658 uncapped), missing val by 0.012;
coverage_resample is a non-sampler val-booster → stack to lift val while test+G1 hold. RESULT {0.3-1.0}:
val ✓ ALL (0.806-0.816 — coverage's val-boost DID transfer), G1 crosses at q=1.0 (85.7✓), but **test
drops to 0.949-0.958 (✗)** — coverage trades test for val exactly like the sampler. covhyb@1.0 = ✓✗✓
(val+G1, test −0.011). So a non-sampler val-booster ALSO costs test → the test↔(val,G1) trade is
mechanism-independent, not a sampler artifact. Frontier reconfirmed from the combination angle.

## EXPANDED action space (2026-06-17, 2nd workflow wf_eb7f67a8, 16 agents, 0 kept) — FGSM adversarial margin also lands on the frontier

The 2nd workflow ideated the EXPANDED space the 1st excluded (manifold-mixup / patient-level /
adversarial / curriculum); 0 kept, but endorsed FGSM margin perturbation as the only idea with a
non-circular argument (gradient-sign, per-bag ~zero-mean → adds decision margin, not mass-shift).
Implemented `--adv_eps` (FGSM inner step in train_one_epoch, additive, default 0 = baseline). Sweep
eps {0.005,0.01,0.02,0.05} aug-off + 0.01×axis@0.2:

| run | val | test | G1 | G0 | G3 | v/t/G1 |
|---|---|---|---|---|---|---|
| adv@0.005 | 0.8003 | 0.9546 | 75.5 | 90 | 95 | ✓✗✗ |
| adv@0.01 | 0.7922 | 0.9442 | 75.5 | 89 | 91 | ✗✗✗ |
| adv@0.02 | 0.7987 | **0.8953** | **91.8** | 65 | 70 | ✓✗✓ |
| adv@0.05 | 0.3130 | 0.5889 | 22.4 | — | 0 | collapse |
| adv@0.01+axis | 0.8012 | 0.9188 | 87.8 | 85 | 65 | ✓✗✓ |

**RESULT**: FGSM lifts G1 the MOST of any mechanism (91.8 at eps=0.02) with val passing — but test
CRASHES (0.8953) because the extreme grades collapse (G0 89→65, G3 95→70). So even targeted
adversarial margin perturbation hits the SAME frontier: lifting G1 (any mechanism) costs the
extreme-grade test recall. Confirms the workflow's prediction verbatim — "targeted FGSM = same
frontier as random noise → barrier is structural (cohort + mean-readout), not perturbation type."
The one un-run survivor is patient_roi_consistency (rated likely-null; needs patient-id plumbing).

## MEAN+LABEL-LOCKED class (2026-06-17, via 15-agent adversarial ideation workflow wf_568cf15f) — the ONE untried mechanism class FAILS, and CAUSALLY pins the frontier

The Ultracode workflow ideated 8 mechanisms, adversarially screened → **0 kept**, but flagged one
genuinely-untried CLASS: centroid-locked + label-locked within-bag redistribution (every prior family
moved the bag-mean OR the target; this locks both, changing only the within-bag patch mixture). Ran 3
modules × strengths (centroid_locked_intrabag_resample {0.4,0.6,1.0}, axis_quantile_warp {0.5,1.0},
g1_tail_densify_meanlock {0.4,0.8}):

| run | val | test | G1 | v/t/G1 |
|---|---|---|---|---|
| clock@0.4 | 0.7973 | 0.9540 | 59.2 | ✓✗✗ |
| clock@0.6/1.0 | 0.789/0.779 | 0.949/0.954 | 67/63 | ✗✗✗ |
| warp@0.5/1.0 | 0.794/0.793 | 0.939/0.953 | 69/63 | ✗✗✗ |
| g1tail@0.4/0.8 | 0.778/0.789 | 0.952/0.926 | 67/55 | ✗✗✗ |

**DECISIVE CAUSAL RESULT**: locking the bag-mean did not just fail to help — it **HURT G1 in every run
(55-69, all << baseline 78)**, including g1_tail_densify which targets G1 specifically. This proves G1
recall is read from the **bag-mean position** (the entmax-gated head pools near the mean); locking the
mean removes the only lever that lifts G1. Therefore **lifting G1 REQUIRES moving the bag-mean
(mass-shift), which is exactly what collapses a 2-patient extreme on the val cohort** → the val↔G1
tension is CAUSAL/STRUCTURAL, not coincidental. This is the strongest closing evidence: the frontier
is a property of cohort geometry + the mean-pooling readout, unbreakable by any train-time feature
augmentation. Augmentation lever now CONCLUSIVELY exhausted (adversarial workflow 0-kept + the one
untried class fails + the causal mechanism pinned). NOT TTA — all train-time feature aug.

> NOTE (2026-06-17): inference-side TTA was briefly tested but REMOVED ENTIRELY per user ("เอา tta
> ออกไปเลย") — the `--tta_*` flags, the 4 TTA experiment dirs, and the TTA rows in the CSV are all
> deleted. TTA is excluded from the thesis (it didn't break the frontier and added scope). Not
> re-listed here.

## patient_roi_consistency (--patient_consistency_w, 2026-06-18) — last survivor of workflow-2 FAILS; reconfirms frontier from a NEW information source

The 2nd adversarial workflow (wf_eb7f67a8) left exactly ONE un-run survivor: a **patient-ROI
consistency aux loss** — the only idea that uses an information source no feature-augmentation touches
(the patient grouping: 30 train patients have ≥2 ROIs). Implemented `--patient_consistency_w` (per
step, sample 2 same-patient ROIs, add `w·(ŷ_a−ŷ_b)²` MSE term; default 0 = baseline). This is NOT a
feature augmentation and NOT a sampler — it injects the prior "ROIs of one patient should grade
alike." Swept w {0.05, 0.1, 0.3, 1.0} aug-off + 0.1×axis@0.2 (titan seed-2):

| run | val | test | G1 | ep | acc | f1 | mRec | mae | G0/G1/G2/G3 | v/t/G1 |
|---|---|---|---|---|---|---|---|---|---|---|
| pcons0.05 | 0.7934 | **0.9629** | 73.5 | 9 | 88.0 | 0.842 | 84.7 | 0.120 | 91/73/77/98 | ✗✓✗ |
| pcons0.1 | 0.7925 | 0.9592 | 73.5 | 9 | 86.9 | 0.835 | 83.9 | 0.131 | 89/73/77/96 | ✗✗✗ |
| **pcons0.3** | 0.7959 | **0.9617** | **85.7** | 4 | 88.0 | 0.856 | **87.7** | 0.120 | 86/86/87/92 | ✗✓✓ |
| pcons1.0 | 0.7895 | 0.9478 | 75.5 | 14 | 83.8 | 0.810 | 83.7 | 0.162 | 80/76/87/92 | ✗✗✗ |
| axis0.2+pcons0.1 | **0.7995** | 0.9478 | 71.4 | 12 | 85.7 | 0.822 | 83.8 | 0.151 | 88/71/83/92 | ✓✗✗ |

**RESULT (0/5 3-PASS)**: the patient-consistency prior lands on the SAME Pareto frontier as every
feature aug — and from BOTH faces at once: **pcons0.3** hits the test+G1 corner (G1 77.6→**85.7**,
test 0.9617, val short by only **0.0009** — and best_epoch=4, a near-init flag), while **axis+pcons0.1**
hits the val corner (val 0.7995✓, but test+G1 lost). So injecting genuinely-new information (patient
structure) does NOT break the frontier — it only slides you along it, exactly like input-feature aug
and the sampler. This is the strongest "new information doesn't help" evidence: the ceiling is the
cohort geometry + mean-readout, not the absence of a patient prior. Workflow-2's "likely-null"
verdict confirmed; this null is thesis-valuable (patient-level consistency adds no signal beyond the
per-ROI features on this cohort). **No genuinely-distinct lever remains** — augmentation/training
search is now closed across feature-aug, sampling, mean-lock, adversarial-margin, AND patient-prior.

## mean_teacher (--mean_teacher_w, 2026-06-18) — the 6th genuinely-new training-side mechanism; cleanest NULL yet (doesn't even reach a 2-of-3 corner)

User pushed "หา augmentation ต่อไป" after the lever was declared closed. To honor it WITHOUT firing
a disk/behaviour duplicate, implemented the one canonical training-side mechanism NOT yet tried:
**EMA mean-teacher consistency** (Tarvainen & Valpola 2017, `--mean_teacher_w`). A temporally-EMA-
averaged copy of the model (the "teacher", no grad) supplies a consistency target: penalize
`w·(student(view1) − teacher(view2))²` where view1/view2 are two ~zero-mean-noise views of the same
clean bag. Genuinely distinct from all 5 prior families (NOT feature-aug / sampler / mean-lock / FGSM
/ patient-group) — the regularization signal is the teacher's own averaged predictions. Expected-null
prior DISCLOSED before running (it's a variance-reducer; the frontier is causally pinned). Swept
w {0.05,0.1,0.3,1.0} aug-off + 0.1×axis@0.2, titan seed-2:

| run | val | test | G1 | ep | acc | f1 | mRec | mae | G0/G1/G2/G3 | v/t/G1 |
|---|---|---|---|---|---|---|---|---|---|---|
| mt0.05 | 0.7872 | 0.9553 | 69.4 | 22 | 86.9 | 0.833 | 84.5 | 0.135 | 89/69/83/96 | ✗✗✗ |
| mt0.1 | 0.7801 | 0.9556 | 71.4 | 15 | 86.9 | 0.832 | 83.7 | 0.135 | 89/71/77/98 | ✗✗✗ |
| mt0.3 | 0.7920 | 0.9576 | 75.5 | 22 | 86.5 | 0.830 | 83.7 | 0.135 | 89/76/77/94 | ✗✗✗ (all just-under) |
| mt1.0 | 0.7732 | 0.9464 | 71.4 | 10 | 84.2 | 0.807 | 81.5 | 0.162 | 89/71/77/89 | ✗✗✗ (over-reg) |
| axis0.2+mt0.1 | 0.7917 | 0.9521 | **83.7** | **2** | 86.1 | 0.826 | 84.2 | 0.143 | 89/84/77/88 | ✗✗✓ near-init |

**RESULT (0/5 3-PASS, expected-null confirmed)**: mean-teacher is the cleanest NULL of the whole
search — it doesn't even reach a 2-of-3 corner. Unlike pcons (which hit the test+G1 corner) it just
**over-smooths**: test caps ~0.955-0.958, G1 drops to 69-76, val ≤0.7920 — every metric slightly
BELOW baseline, monotonically worse as w↑ (mt1.0 collapses). The one G1-lift (axis+mt0.1 → 83.7) is
best_epoch=2 = near-init artifact (val+test both fail). So adding an EMA-self-distillation
consistency signal — a genuinely-new mechanism — provides NO lift; it only averages, which on a
saturated cohort costs a little everywhere. This is the strongest "more regularization ≠ progress"
evidence: the ceiling is information (cohort geometry + mean-readout), not regularization strength.
6th mechanism family closed. 0/5 → augmentation/training lever now closed across 6 families, 106
runs, 0 full 3-PASS.

> CODE-CLEANUP NOTE (2026-06-18): per user ("เยอะเกิน รก"), the `--patient_consistency_w` and
> `--mean_teacher_w/_decay/_noise` flags + their train-loop blocks were REMOVED from
> `src/train_grading_reti.py`. These two NULL results are preserved here + in the CSV; the flags no
> longer exist. Trainer flags still live: `--augmentation/--aug_strength`, `--sampler_weights/
> --sampler_temp`, `--adv_eps`, `--max_roi_per_patient`. Baseline behaviour verified unchanged
> (2-epoch smoke post-removal: clean exit, val/test sane).

## cutmix (2026-06-18, REGISTRY-ONLY per user "no new trainer flags") — new BEST-TEST model (cmwg@0.7, well-trained) + G1-boundary hypothesis FALSIFIED

User: "หา augmentation ที่ชนะ a215 และ improve G1 ต่อ — แต่ห้ามเพิ่ม flag ใน trainer." Honored via two NEW
registry modules (no trainer edits — `--augmentation <name>` only), both = discrete cross-bag patch
TRANSPLANT (distinct from the 24 prior families: no blend, no centroid math, real on-manifold
patches):

| run | val | test | G1 | ep | acc | f1 | mRec | mae | G0/G1/G2/G3 | v/t/G1 |
|---|---|---|---|---|---|---|---|---|---|---|
| cmwg0.3 | 0.7742 | 0.9557 | 73.5 | 6 | 86.9 | 0.840 | 84.3 | 0.135 | 90/73/80/94 | ✗✗✗ |
| cmwg0.5 | 0.7826 | 0.9540 | 73.5 | 14 | 85.3 | 0.824 | 82.6 | 0.147 | 89/73/77/91 | ✗✗✗ |
| **cmwg0.7** | 0.7834 | **0.9663** | **79.6** | **35** | **89.2** | **0.872** | **87.4** | **0.108** | 88/80/83/99 | **✗✓✓** |
| g1bc0.1 | 0.7783 | 0.9492 | 67.3 | 15 | 85.7 | 0.811 | 81.3 | 0.151 | 89/67/70/99 | ✗✗✗ |
| g1bc0.2 | 0.7796 | 0.9524 | 63.3 | 19 | 85.7 | 0.816 | 82.4 | 0.147 | 89/63/80/98 | ✗✗✗ |
| g1bc0.3 | 0.7944 | 0.9510 | 61.2 | 20 | 85.3 | 0.807 | 81.9 | 0.151 | 90/61/80/96 | ✗✗✗ |
| g1bc0.5 | 0.7911 | 0.9497 | 63.3 | 24 | 84.9 | 0.802 | 81.3 | 0.154 | 89/63/77/96 | ✗✗✗ |

**RESULT (0/7 3-PASS) + two findings:**
1. **cutmix_within_grade@0.7 = the NEW BEST-TEST model of the entire search**: test_qwk **0.9663**
   (edges out hyb q=0.7's 0.9658) dominating baseline on EVERY test metric (acc 89.2 / f1 0.872 /
   mRec 87.4 / mae 0.108) AND test G1 79.6 (>78) — and crucially **well-trained at ep35** (NOT
   near-init, unlike several prior corner-models). But val 0.7834 → ✗✓✓, joins the test+G1 corner.
   It is the STRONGEST val↔test anti-correlation artifact yet: a well-trained, all-test-metric-
   dominating model that val-selection demonstrably will not pick. (Discrete same-grade patch
   recombination = a clean test-side regularizer; doesn't move the grade-mean → no val rescue.)
2. **g1_boundary_cutmix hypothesis FALSIFIED (informative negative)**: injecting a few G0+G2 patches
   into G1 bags (to "tighten" G1 while protecting extremes) did the OPPOSITE — G1 recall CRATERS
   67→63→61→63 as strength rises. Contaminating a G1 bag with adjacent-grade patches shifts its
   bag-mean OFF 1.0 → the bag is mis-graded. This independently re-confirms the causal mean-readout
   mechanism (G1 recall is read from the bag-mean position; any mass added off-1.0 hurts it) — the
   same reason mean-locked redistribution and FGSM failed. Registry-only, no flags; lever still
   closed (113 runs, 0 full 3-PASS). cmwg@0.7 added to the test-dominating deliverables.

## cutmix_mid_protect (2026-06-19) — "protect extremes to recover val" hypothesis FALSIFIED (registry-only, worse than cmwg on every gate)

Follow-up to cmwg@0.7's val-failure: since cmwg@0.7 (✗✓✓, val 0.7834) augments ALL grades incl. the
2-patient extremes (G0/G3) that drive the tiny val cohort, the hypothesis was that restricting the
within-grade cutmix to the MIDDLE grades (G1/G2 only, leaving G0/G3 pristine) would protect val while
keeping the test+G1 strength. Registry module cutmix_mid_protect.py (NO trainer flags), sweep {0.5,0.7,0.9}:

| run | val | test | G1 | ep | acc | f1 | mRec | mae | G0/G1/G2/G3 | v/t/G1 |
|---|---|---|---|---|---|---|---|---|---|---|
| cmmp0.5 | 0.7391 | 0.9584 | 67.3 | 34 | 87.6 | 0.842 | 84.3 | 0.127 | 90/67/80/100 | ✗✗✗ |
| cmmp0.7 | 0.7575 | 0.9449 | 57.1 | 43 | 82.2 | 0.777 | 78.5 | 0.178 | 90/57/77/90 | ✗✗✗ |
| cmmp0.9 | 0.7293 | 0.9408 | 59.2 | 19 | 83.8 | 0.792 | 78.3 | 0.174 | 90/59/67/98 | ✗✗✗ |

**RESULT (0/3, hypothesis FALSIFIED, worse than cmwg on every gate)**: protecting the extremes did
NOT recover val — val DROPPED to 0.73-0.76 (below cmwg-all-grades' 0.7834) — AND G1 recall CRATERED
(67→57→59 vs cmwg's 79.6). Mechanism: (1) heavy within-grade cutmix (70-90%) on G1/G2 homogenizes
the middle bags toward their population mean, destroying the per-bag G1 signal → G1 collapses; (2)
augmenting middle grades heavily while leaving extremes pristine creates a train/eval distribution
mismatch → val drops, not recovers. So cmwg@0.7's test+G1 strength specifically REQUIRED recombining
ALL grades; the val gate is NOT an "extreme-grade-augmentation" artifact that grade-selectivity can
fix. Registry-only, no flags. 3rd cutmix variant, all confirm the frontier. Lever now 116 runs, 0
full 3-PASS. The val gate has no registry-expressible path (1→1 per-bag map can't rebalance; grade-
selective augmentation makes it worse) — the missing-val conclusion is now empirically hardened.

## sampler × cutmix_within_grade (2026-06-19) — seed-2 3-PASS found but MULTI-SEED EXPOSED it as a LUCKY FOLD (1/5 seeds; a215 stands)

> **VERDICT (multi-seed audit complete): NOT a win — seed-fragile lucky fold.** samp+cmwg0.7 across
> seeds {2,0,1,3,4}: only seed-2 3-passes (1/5). Per-seed test_qwk = 0.9653 / 0.6527 / 0.7435 / 0.7387 /
> 0.9075 → **mean 0.8015 ± 0.116** (catastrophic variance; seed-0 collapsed at ep1, test 0.65) vs
> baseline 0.96. val is high every seed (0.81-0.94, mean 0.889 — the sampler reliably overfits the
> balanced val cohort) but test is wildly unstable. So the sampler+cutmix combo trades robust test for
> val on most seeds; seed-2 happened to also keep test. This is the **a40 failure mode** verbatim (one
> lucky seed up, siblings crash) and vindicates the multi-seed protocol: samp+cmwg0.7 passed 4/5
> scrutiny checks (well-trained ep35 / val-selectable / disclosed / not-test-only) yet the 5th
> (lucky-seed) killed it. **a215 stands; no augmentation/sampler config beats it robustly.** The
> seed-2 3-PASS is retained ONLY as a thesis artifact for "single-seed selection is unreliable on this
> 10-patient cohort." Total now 124 runs.

(Original seed-2 sweep that produced the candidate — all disclosed:)

Combined the EXISTING `--sampler_weights` (val-booster, not a new flag) with the NEW cmwg mechanism
(highest test of the search = most headroom to survive the sampler's test-cost). Rationale: samp+hyb@0.5
previously got val+G1 but test −0.004; cmwg's higher test might thread. Sweep (titan seed-2):

| run | val | test | G1 | ep | acc | f1 | mRec | mae | G0/G1/G2/G3 | gate |
|---|---|---|---|---|---|---|---|---|---|---|
| **samp+cmwg0.7** | **0.8099** | **0.9653** | **81.6** | **35** | 88.8 | 0.866 | 86.3 | 0.112 | 88/82/77/99 | **PASS3 ✓✓✓** |
| samp+cmwg0.5 | 0.7796 | 0.9628 | 85.7 | 26 | 88.0 | 0.854 | 85.0 | 0.120 | 88/86/70/96 | 2/3 (test+G1) |
| tmp0.5+cmwg0.7 | 0.7883 | 0.9615 | 83.7 | 22 | 87.6 | 0.854 | 85.1 | 0.124 | 86/84/73/98 | 2/3 (test+G1) |
| tmp0.3+cmwg0.7 | 0.7886 | 0.9601 | 83.7 | 46 | 87.3 | 0.840 | 84.2 | 0.127 | 88/84/70/95 | 2/3 (test+G1) |

**samp+cmwg0.7 = the FIRST configuration in 117+ runs to clear ALL THREE gates at seed-2**: val 0.8099
(+0.0131), test 0.9653 (+0.0053), test G1 81.6 (+4.1), best_epoch 35 (WELL-TRAINED, not near-init), and
it also beats baseline on acc/f1/mae and is val-selectable (highest val of the 4). Mechanism: the
balanced sampler supplies the val lift that no registry augmentation could (it rebalances class
frequency), while cmwg@0.7's large test headroom (0.9663 solo) survives the sampler's usual ~0.01 test
cost (lands 0.9653, still >0.96) AND keeps G1 up. So the val+test+G1 corner IS reachable — but ONLY by
the sampler (a training-distribution lever), NOT by augmentation alone, consistent with the frontier
analysis (registry aug can't reach val; the sampler can).

> HONESTY — NOT yet a confirmed win. This is a SINGLE seed-2 result, and the entire search established
> that seed-2 is a215's favourable (+2σ) fold. It has cleared 4 of 5 scrutiny checks (well-trained /
> val-selectable / full-sweep-disclosed / not-test-only) — the LAST check is lucky-seed. A multi-seed
> audit (seeds 0,1,3,4) is RUNNING; only if it holds on average / across seeds is this a defensible
> thesis win. Caveat to carry regardless: G2 test recall drops 87→77 (mass shifts into G1) so macro-
> recall dips slightly (86.3 vs 87.05) even as QWK/acc/F1/MAE improve. Do NOT claim a win until the
> multi-seed numbers are in. (Uses only the live --sampler_weights/--sampler_temp flags + the cmwg
> registry module — NO new trainer flags.)

## mixstyle_bag (2026-06-19) — MixStyle (scale-heterogeneity-motivated, the most credible val-path idea) ALSO lands on the frontier

NEW registry mechanism (Zhou et al. MixStyle 2021): cross-bag feature-statistics (per-dim mean/std)
mixing, content/label preserved — the one idea explicitly motivated by this cohort's real SCALE
HETEROGENEITY (20/50/100/200µm), and the most plausible val-path of the remaining mechanisms (it
reduces a real style/scale domain gap rather than just perturbing). NO trainer flags. Sweep {0.1,0.3,0.5}:

| run | val | test | G1 | ep | acc | f1 | mRec | mae | G0/G1/G2/G3 | v/t/G1 |
|---|---|---|---|---|---|---|---|---|---|---|
| mixstyle0.1 | 0.7836 | 0.9319 | 75.5 | 2 | 84.2 | 0.793 | 79.5 | 0.178 | 93/76/63/86 | ✗✗✗ near-init |
| mixstyle0.3 | **0.8092** | 0.9450 | **79.6** | **3** | 85.7 | 0.806 | 80.4 | 0.154 | 89/80/57/96 | ✗✓✗→ val+G1, near-init |
| mixstyle0.5 | 0.7876 | 0.8878 | 65.3 | 22 | 80.7 | 0.731 | 72.0 | 0.243 | 94/65/40/89 | ✗✗✗ collapse |

**RESULT (0/3)**: MixStyle lands on the frontier like every other mechanism. mixstyle0.3 reaches the
**val+G1 corner** (val 0.8092 ✓ + G1 79.6 ✓) but **test 0.9450 ✗** (G2 test recall craters 87→57) AND
best_epoch=3 (near-init red flag). So style/scale mixing behaves as a VAL-BOOSTER that trades away
test (same failure face as the sampler and coverage_resample) — the scale-heterogeneity motivation did
NOT convert to a 3-gate pass. Even the most credible val-path idea confirms: any mechanism that lifts
val on this cohort costs test (G2/extreme collapse). 7th feature-space mechanism family on the
frontier; a215 stands (127 runs, 0 robust 3-gate win). Registry-only, no flags.

## temp-threading sampler×cmwg (2026-06-19) — NO stable threading point; only full-sampler crosses val = the lucky fold

After samp+cmwg0.7 (full sampler) seed-2-3-PASSed-then-lucky-folded, swept sampler_temp to find a
milder balance where val crosses 0.7968 while test stays >0.96 (existing flags + cmwg, NO new flags):

| sampler_temp × cmwg0.7 | val | test | G1 | ep | G2-recall | gate |
|---|---|---|---|---|---|---|
| 0.3 | 0.7886 | 0.9601 | 83.7 | 46 | 70 | 2/3 (test+G1) |
| 0.5 | 0.7883 | 0.9615 | 83.7 | 22 | 73 | 2/3 (test+G1) |
| 0.7 | 0.7701 | 0.9611 | 81.6 | 35 | 77 | 2/3 (test+G1) |
| 0.8 | 0.7574 | 0.9360 | 55.1 | 2 | 63 | 0/3 (near-init) |
| 0.9 | 0.7786 | 0.9475 | 77.6 | 1 | 77 | 1/3 (near-init) |
| 1.0 (full) | **0.8099** | 0.9653 | 81.6 | 35 | 77 | PASS3 = LUCKY FOLD (multi-seed 1/5) |

**RESULT — threading hypothesis FALSIFIED**: val does NOT rise smoothly with sampler strength.
temp 0.3-0.7 all sit at val ~0.77-0.79 (below baseline, stuck at the 2/3 test+G1 corner); temp 0.8-0.9
are near-init noise (ep1-2). ONLY full sampler (temp 1.0) pushes val above 0.7968 — and that is exactly
the seed-fragile lucky fold (multi-seed 1/5, test std 0.116). So the val crossing is DISCONTINUOUS and
bound to the full-sampler overfit; there is no stable intermediate temperature that threads all 3 gates.
Confirms the frontier yet again: val crossing on this cohort is unstable, not a tunable. 130 runs, 0
robust 3-gate win, a215 stands. No new flags (existing --sampler_temp + cmwg registry module).

## cmwg@0.7 SOLO multi-seed (2026-06-19) — the "best-test" artifact is ALSO a lucky seed-2 fold; DOWNGRADED

Tested whether cmwg@0.7 solo (the search's best seed-2 test, 0.9663, no sampler) improves test ROBUSTLY
across seeds (to answer "augmentation that improves" without the sampler's variance). Seeds {2,0,1,3,4}:

| seed | val | test | G1 | ep |
|---|---|---|---|---|
| 2 | 0.7834 | **0.9663** | 79.6 | 35 |
| 0 | 0.8965 | 0.7353 | 56.4 | 3 |
| 1 | 0.9289 | 0.7329 | 56.4 | 6 |
| 3 | 0.8833 | 0.7313 | 76.9 | 18 |
| 4 | 0.9161 | 0.8904 | 59.0 | 16 |
| **mean±std** | 0.882±0.05 | **0.811±0.099** | 65.7±10 | — |

**RESULT — cmwg@0.7 is ALSO seed-fragile (1/5 seeds beat test 0.9600); the "best-test" label was a
LUCKY SEED-2 FOLD.** Multi-seed test mean 0.811±0.099 (far below baseline 0.96). Notably val is HIGH on
every seed (0.88-0.93) but test crashes (0.73-0.89): heavy cutmix@0.7 (70% patch swap) intrinsically
OVERFITS val and destroys test on most seeds; seed-2 alone kept test. **DOWNGRADE: cmwg@0.7 is NOT a
robust test-improver — it is a seed-2-favorable result, same lucky-fold class as samp+cmwg0.7.** So even
the single best-test augmentation of the whole 134-run search does not survive multi-seed. This is the
decisive answer to "find an augmentation that improves": NONE improves a215 robustly; the apparent
winners (cmwg@0.7 best-test, samp+cmwg0.7 3-gate) are both lucky seed-2 folds. Caveat: axis@0.2 (the
"2-gate win") was never multi-seeded and is, by this same evidence, very likely seed-fragile too —
report it as a single-seed result only. a215 STANDS (134 runs, 0 robust improvement on any gate).

## FINAL EXHAUSTION (2026-06-17): augmentation lever closed across the training-side axes (feature aug + sampling)

84 augmentation/sampler runs. Mechanism axes covered: training-side feature aug (10 classes:
patch-mix, patch-drop, feature-noise, dim-drop, bag-resample, spread-jitter, directional-translate,
coverage-resample, within-grade-interp, boundary) + training-distribution sampling (balanced +
tempered). **0 full 3-PASS**; every config lands on the val↔test↔G1 Pareto frontier through the
baseline. No genuinely-distinct augmentation mechanism remains (further runs would be
disk/behaviour duplicates = theater). The 3-gate bar is structurally unreachable at seed-2 via
augmentation; root cause = 2 patients/extreme grade (same as MIL GATE2); ceiling-lifting levers
(val-selection / split / loss) are thesis-LOCKED. THESIS DELIVERABLES: axis_shift@0.2 = first
augmentation to beat both QWK gates (val 0.8113 / test 0.9629) = 2-gate seed-2 win; axis_g1_hybrid
q=0.7 = test-dominating + G1 78→85.7 (not val-selectable); the Pareto-frontier characterization.
- 2-gate win: axis@0.2 (val 0.8113 / test 0.9629). Best test model: hyb q=0.7 (test dominates + G1 85.7,
  not val-selectable). 3-gate bar = structurally blocked by val↔G1 tension on the tiny cohort.
- Lead: **prototype_interp** (G1-stable at 81.6; near-misses on val@0.2 and test@0.4). Fine-sweep +
  adjacency variant running/queued to thread all 3 gates.
- The search is documented + analysed candidate-by-candidate here; cumulative ceiling evidence lives in
  the `mpn-search-exhaustion-proof` memory.

## ROUND 2026-07-25 (54 runs, FULL 6-cell grid) — two NEW axes opened and closed: acquisition METADATA donors + BAG-COHERENT nuisance; 0 beat the champion

Champion for this round = `cutmix_then_noise` (uniform within-grade CutMix @0.8 -> i.i.d. feature
noise sigma .05); per-cell references in the `composed-cutmix-noise-win` / `prototype-cutmix-robust-win`
/ `subspace-noise-titan-win` memories. Protocol: seed 2, MPS, regression, all 3 backbones x
{ASGAP a215, ABMIL simple}, registry-only (no trainer flags). Runs: `experiments/20260725/{scl,xsc,sap,bsh,bns,scls*}_*`.

### Part A — 5 new mechanisms x 6 cells @ fixed strength (30 runs)

| module | v2/ASGAP | v2/ABMIL | u2/ASGAP | u2/ABMIL | ti/ASGAP | ti/ABMIL |
|---|---|---|---|---|---|---|
| cutmix_scale_noise @0.8 (same-magnification donors) | **0.8511/0.9607** ep22 | 0.8372/0.9648 ep22 | 0.8305/0.9183 | 0.8113/0.9557 | 0.7887/0.9604 | 0.7732/0.9490 |
| cutmix_crossscale_noise @0.8 (forced other-magnification) | 0.8529/0.9458 ep20 | 0.8422/0.9544 ep38 | 0.8351/0.9460 | 0.8198/0.9597 | 0.7543/0.9494 | 0.7823/0.9455 |
| cutmix_samepatient_noise @0.8 (donors = same biopsy) | 0.8084/0.9436 | 0.7876/0.9394 | 0.7667/0.9077 | 0.7538/0.9073 | 0.7882/0.9616 | 0.7651/0.9483 |
| cutmix_then_bagshift @0.8 (CutMix + ONE shared nuisance offset) | 0.8234/0.9619 ep23 | 0.8191/0.9627 ep36 | 0.8339/0.9484 | 0.8214/0.9561 | 0.7691/0.9183 | 0.7496/0.9307 |
| bag_nuisance_shift @0.3 (shared offset ALONE) | 0.8002/0.9512 | 0.8043/0.9399 | 0.7677/0.9293 | 0.7712/0.9244 | 0.7819/0.9461 | 0.7913/0.9595 |

### Part B — cutmix_scale_noise strength sweep, full 6-cell grid x {0.70,0.75,0.80,0.85,0.90} (24 runs)

| cell | 0.70 | 0.75 | 0.80 | 0.85 | 0.90 |
|---|---|---|---|---|---|
| v2/ASGAP | 0.8411/0.9580 | 0.8520/0.9552 | 0.8511/0.9607 | 0.8497/0.9606 | 0.8494/0.9594 |
| v2/ABMIL | 0.8384/0.9637 | 0.8281/0.9519 | 0.8372/0.9648 | **0.8449/0.9659** ep30 | 0.8357/0.9606 |
| u2/ASGAP | 0.8216/0.9306 | 0.8277/0.9502 | 0.8305/0.9183 | 0.8270/0.9330 | 0.8280/0.9266 |
| u2/ABMIL | 0.8271/0.9457 | 0.8303/0.9427 | 0.8113/0.9557 | 0.8190/0.9598 | 0.8111/0.9426 |
| ti/ASGAP | 0.7806/0.9537 | 0.7957/0.9565 | 0.7887/0.9604 | 0.7971/0.9574 | 0.7789/0.9481 |
| ti/ABMIL | 0.7743/0.9494 | 0.7797/0.9505 | 0.7732/0.9490 | 0.7871/0.9405 | 0.7628/0.9470 |

### RESULT — 0/54 beat the champion in ANY cell. Four findings:

1. **ACQUISITION-METADATA donor axis (NEW) opened and closed.** This cohort has real magnification
   heterogeneity (scalebar 20/50/100/200um + a no-scalebar group, `results/scalebar_results.csv`,
   1329/1330 ROIs covered, every grade contains several groups), so uniform CutMix builds physically
   impossible mixed-magnification bags. BOTH corrections lose to uniform on test: same-magnification
   donors -0.009 (v2/ASGAP), forced cross-magnification -0.024. Restricting donors by acquisition
   metadata behaves exactly like the feature-geometry restrictions before it (prototype / margin / NN /
   FPS / patient-balanced): val up, test down. Uniform-over-patches remains the unique optimum, now
   also against a non-geometric axis. Note the asymmetry: the *physically coherent* restriction costs
   ~3x less test than the *forced-mismatch* one - the donor diversity ASGAP needs is not "acquisition
   diversity". Also a 3rd independent confirmation of `asgap-wants-diverse-donors`: with the
   diversity-reducing same-scale restriction, ABMIL test (0.9659 @0.85) OVERTAKES ASGAP test (0.9606),
   the same flip prototype donors produce.
2. **cutmix_samepatient_noise = the round's strongest negative, and it kills the "realism" hypothesis.**
   Donors from other ROIs of the SAME biopsy (identical tissue/stain/scanner/magnification = the most
   realistic pseudo-ROI possible) collapse BOTH aggregators on virchow2/uni2 (u2/ABMIL 0.7538/0.9073,
   test G1 recall 36.7). Physical realism does NOT compensate for lost donor diversity; the donor axis
   is now closed at BOTH extremes (patient-balanced = max diversity, lost; same-patient = max
   concentration/realism, lost worse).
3. **The "pooling-cancellation" explanation for the inert noise family is FALSIFIED (most useful result).**
   Hypothesis: i.i.d. per-patch noise fails because it shrinks as 1/sqrt(N) under attention pooling, so
   it barely perturbs the pooled representation the grade is read from. Test: `bag_nuisance_shift` adds
   ONE shared nuisance-subspace offset per bag (calibrated to the between-ROI std, orthogonal to the
   grade-signal subspace) - it does NOT cancel under pooling. It is just as inert (v2 -0.008/-0.008 vs
   no-aug; best cell ti/ABMIL 0.7913/0.9595 = baseline-level, below subspace_noise 0.8050/0.9639). And
   swapping the champion's stage 2 for it (`cutmix_then_bagshift`) lands at champion-minus-a-little on
   every cell. => stage 2 is NOT where the gain comes from and the pooled-representation story is wrong:
   **the discrete same-grade patch transplant carries the entire augmentation effect**; any small
   label-preserving perturbation on top is interchangeable.
4. **titan unchanged and titan's cutmix-hostility is NOT a scale artifact.** Best ti/ASGAP over the whole
   sweep = 0.7971/0.9574 @0.85 (val +0.0003, test -0.0026) - no strength clears both gates; ti/ABMIL
   fails everywhere. Physically-coherent donors DO let titan train longer (ep 18-24 at most strengths vs
   the ep1-2 collapse of plain cutmix, the same on-manifold effect `cutmix_nn_noise` showed) but longer
   training does not convert to test. titan stays at its data-driven ceiling.

**REFINEMENT WORTH KEEPING (defensibility, not a win): cutmix_scale_noise turns the virchow2/ASGAP val
claim from a SPIKE into a PLATEAU.** The champion's val 0.8526 exists only at strength 0.81 (neighbours
0.8254/0.8277). Under scale-matched donors, virchow2/ASGAP val is 0.8411/0.8520/0.8511/0.8497/0.8494
across 0.70-0.90 - a flat ~0.85 plateau, well-trained (ep 7-22), dual-gate over no-aug at 0.80/0.85/0.90.
Cost: test 0.955-0.961 vs the champion's 0.967-0.970. So if the thesis ever needs a val statement that
does not rest on a single strength point, this is the module to cite; the headline stays
`cutmix_then_noise` on TEST+mRec.

**Registry now 126 modules.** 4th independent exhaustion confirmation (`mpn-search-exhaustion-proof`).

## ROUND 2026-07-25 part 2 (34 more runs) — destination rule + bag CARDINALITY + on-manifold patient shift; 0 beat the champion, but ONE new aggregator-level asymmetry

Motivated by two facts established earlier the same day: (a) stage-2 perturbations are interchangeable,
(b) at cutmix strength 1.00 (bag fully replaced by grade-bank draws) uni2/ASGAP still scores 0.8281/0.9571,
i.e. **bag identity carries no information**. What is left of the training distribution is therefore only
WHICH patches are replaced and HOW MANY patches a bag has. Both were tested, plus a repaired version of the
failed bag-shift.

| module | v2/ASGAP | v2/ABMIL | u2/ASGAP | u2/ABMIL | ti/ASGAP | ti/ABMIL |
|---|---|---|---|---|---|---|
| cutmix_adjclean_noise @0.8 (dst = most adjacent-grade-looking patches) | 0.8249/0.9649 | **0.8372/0.9674** ep36 | 0.8339/0.9452 | 0.8261/0.9536 | 0.7847/0.9532 | 0.7677/0.9463 |
| cutmix_sizejitter_noise @0.3 (bag size ~ U(0.7N,1.3N)) | 0.8333/0.9608 | 0.8112/0.9385 ep2 | 0.8347/0.9532 ep34 | 0.7976/0.9380 | 0.7683/0.9580 | 0.7448/0.9326 ep2 |
| patient_pc_shift @1.0 (shift along empirical patient PCs) | 0.8031/0.9447 | 0.8025/0.9423 | 0.7616/0.9356 | 0.7799/0.9372 | 0.7968/0.9518 | 0.7969/0.9478 |
| cutmix_then_pcshift @0.8 | 0.8441/0.9495 | 0.8295/0.9616 | 0.8463/0.9532 | 0.8170/0.9517 | 0.7677/0.9486 | 0.7641/0.9530 |

Follow-up sweeps (10 runs): sizejitter {0.15, 0.50} on the 3 ASGAP cells -> v2 0.8394/0.9583, 0.8289/0.9316;
u2 0.8307/0.9489, 0.8171/0.9535; ti 0.7731/0.9530, 0.7484/0.9532. adjclean {0.70, 0.90} on virchow2 ->
ASGAP 0.8339/0.9652 and 0.8465/0.9632 (mRec 89.1, G1 77.6, ep27); ABMIL 0.8217/0.9652 and 0.8250/0.9612.

### Findings

1. **NEW — bag-CARDINALITY robustness is an ASGAP/ABMIL asymmetry (the round's real result).** Jittering
   bag size +/-30% leaves ASGAP intact and well-trained (v2 0.9608, u2 0.9532 ep34 = champion-level test,
   ti 0.9580) but **collapses ABMIL at epoch 2** (v2 0.9385, ti 0.9326, u2 0.9380). Mechanism: entmax
   support size adapts to how many patches actually carry evidence, whereas gated softmax attention
   spreads mass over all N, so changing N changes its pooled representation. This is a property of the
   PROPOSED aggregator that is not an accuracy claim - and variable patch/ROI count per case is a real
   deployment condition. Worth converting into a proper test-time evaluation (evaluate the trained
   champions on subsampled bags: 100%/75%/50%/25% of patches) rather than another augmentation.
2. **Destination axis closed from the cross-grade side too.** Replacing the patches that look most like the
   ADJACENT grade (score = cos(x, c_adj) - cos(x, c_own)), i.e. cleaning the bag toward its own grade
   mean, gets within 0.001 of the champion on v2/ABMIL test (0.9674 vs 0.9684) and beats the ABMIL
   cutmix_then_noise reference (0.8315/0.9662) on both gates - but never beats the per-cell champion, at
   0.7 / 0.8 / 0.9. Random destinations remain optimal, now against both a within-grade (typicality) and a
   cross-grade (adjacency) rule. **The CutMix stage now has no free parameter left that improves it.**
3. **Nuisance-perturbation family CLOSED for good.** The 2026-07-25 negative bag_nuisance_shift used RANDOM
   nuisance directions, which in ~1500-D are nearly orthogonal to the directions patients actually differ
   along - so it was repaired here: patient_pc_shift shifts along the empirical top-8 patient-effect PCs
   (grade-signal projected out), bag-coherent so it survives pooling. It is negative on ALL 6 cells (best:
   ti/ASGAP 0.7968/0.9518 = val exactly at baseline, test below), and composing it with CutMix again lands
   at champion-minus-a-little. On-manifold, pooling-surviving, signal-orthogonal domain shift does not help.
4. **Same frontier shape as always:** every mechanism here clusters at val +0.01..+0.02 / test -0.004..-0.02
   vs the champion (e.g. adjclean@0.9 v2/ASGAP 0.8465/0.9632; cutmix_then_pcshift u2/ASGAP 0.8463/0.9532).

**Day total: 88 runs (30 + 24 + 24 + 10), 0 champion-beating cells.** Registry = 130 modules.

## ROUND 2026-07-25 part 3 (48 runs) — the GLOBAL-VIEW information injection: the first mechanism that ADDS information, and it lands on the frontier too

All ~130 modules so far re-arrange the same patch features. `data/features_<backbone>_reti_no_patch/`
(1330 ROIs x 3 backbones, already on disk) holds the whole-ROI-resized embedding from the SAME encoder:
same feature space, same dimension (uni2 1536 / virchow2 1280 / titan 768), comparable norm (15.8 vs 14.0),
cos with the patch mean ~0.65 = related but NOT redundant - a genuinely different magnification carrying
the global fibre ARCHITECTURE, which is the WHO criterion for G2/G3 and which no single patch can express.
Donors are taken only from TRAIN ROIs (resolved from each pool path), so there is no leakage.

| module | v2/ASGAP | v2/ABMIL | u2/ASGAP | u2/ABMIL | ti/ASGAP | ti/ABMIL |
|---|---|---|---|---|---|---|
| cutmix_globalview_noise @0.2 (global tokens ONLY, no cutmix) | 0.7675/0.9284 | 0.7067/0.9000 | 0.7803/0.9358 | 0.7277/0.8768 | 0.7708/0.9292 | 0.8132/0.9392 **G1 89.8** |
| cutmix_then_globalview @0.8 (+15% global) | 0.8290/0.9633 | 0.8262/**0.9688** | 0.8261/0.9511 | 0.7992/0.9521 | 0.7693/0.9582 | 0.7578/0.9525 |
| context_infuse @0.2 (blend each patch toward its OWN ROI global) | 0.7808/0.9439 | 0.7904/0.9424 | 0.7420/0.9028 | 0.7423/0.8577 | 0.7765/0.9384 | 0.7870/0.9498 |

**Dose sweep (`cutmix_gvfrac`, CutMix fixed 0.8, strength = global-token fraction):**

| cell | 0.05 | 0.10 | 0.15 | 0.25 |
|---|---|---|---|---|
| v2/ASGAP | 0.8423/0.9636 | 0.8125/0.9498 | 0.8290/0.9633 | 0.8225/0.9595 |
| v2/ABMIL | 0.8273/**0.9685** | 0.8016/**0.9687** | 0.8262/**0.9688** | 0.8192/0.9659 |
| u2/ASGAP | 0.8205/0.9489 | 0.8289/0.9552 | 0.8261/0.9511 | 0.8091/0.9563 |
| u2/ABMIL | 0.8022/0.9551 | 0.8066/0.9446 | 0.7992/0.9521 | 0.7967/0.9422 |
| ti/ASGAP | 0.7556/0.9231 ep1 | 0.7793/0.9619 ep41 | 0.7693/0.9582 | 0.7540/0.9247 ep1 |
| ti/ABMIL | 0.7467/0.9406 | 0.7496/0.9260 | 0.7578/0.9525 | 0.7186/0.9157 |

**Composition `cutmix_scale_gv` (scale-matched donors = the val-plateau mechanism + 10% global tokens = the
test-plateau mechanism; the same "compose the two gate-winners" move that produced the champion):**

| cell | 0.80 | 0.85 |
|---|---|---|
| v2/ASGAP | **0.8591**/0.9601 (highest-but-one val in the study) | 0.8490/0.9542 |
| v2/ABMIL | 0.8446/0.9675 (both gaps < 0.01) | 0.8443/0.9624 |
| u2/ASGAP | 0.8305/0.9505 | 0.8303/0.9525 |
| u2/ABMIL | 0.8237/0.9468 | 0.8207/0.9497 |
| ti/ASGAP | 0.7794/0.9546 | 0.7843/0.9588 |
| ti/ABMIL | 0.7620/0.9398 | 0.7639/0.9534 |

### Findings

1. **The ceiling survives an INFORMATION-ADDING mechanism - the strongest form of the exhaustion argument.**
   Not just "no re-arrangement of the patch features helps" but "adding a second magnification the patch
   view cannot represent does not help either". Everything again lands on the val<->test frontier.
2. **Dose, not idea, was what the first global round measured.** 20% global tokens WITHOUT CutMix is toxic
   (v2/ABMIL 0.7067/0.9000; u2/ABMIL 0.9000-0.8768) - a bag cannot be mostly global view. On top of CutMix
   at 5-15% it is a flat TEST PLATEAU at champion level on v2/ABMIL (0.9685/0.9687/0.9688 vs champion
   0.9684, mRec 89.3-89.6, ep25-30) - a genuine TIE on test, but val 0.80-0.83 << champion 0.8516, so it is
   never val-selectable.
3. **The global view specifically helps G1 - the cohort's weak class.** titan/ABMIL with 20% global tokens
   reaches test **G1 recall 89.8**, the highest G1 anywhere in this study (champion configs sit at 65-82),
   and ti/ASGAP @0.10 trains to ep41 with test 0.9619 (> the 0.9600 no-aug titan baseline). Mechanistically
   coherent: G1 is defined by the CONTINUITY of the fibre network across the field, a global property.
   Overall QWK still drops, so this is a per-class signal, not a win - but it is the one place where the
   extra magnification demonstrably adds something.
4. **The gain arrives as EXTRA TOKENS, not as context in the existing patches.** context_infuse (blend each
   patch toward its own ROI's global embedding) is negative on all 6 cells (u2/ABMIL 0.7423/0.8577) - the
   same variance-shrinkage failure as prototype_shrink.
5. **Composing the two gate-winners no longer works.** cutmix_scale_gv reaches val **0.8591** on v2/ASGAP
   (2nd-highest val in the whole study, test 0.9601) and gets both v2/ABMIL gaps under 0.01
   (0.8446/0.9675) - closest simultaneous approach yet - but never clears both gates. The composition trick
   that created the champion does not repeat.

**Day total: 136 runs across 4 rounds, 0 champion-beating cells. Registry = 135 modules.**

## ROUND 2026-07-26 (36 runs) — the last two structural blind spots: SPATIAL (patch grid) and SCHEDULE (time)

Runs in `experiments/20260726/{spa,cos,pmt,ann,rmp,bbt}_*`. Both axes were completely untouched by the
~147 modules that existed before: nothing had ever used the patch COORDINATES (every .pt stores `rc` =
the (row, col) of each patch in its ROI grid: 5x8, 7x6, 7x16, 4x10 ...), and nothing had ever varied with
TIME (every module applies a fixed i.i.d.-per-bag transform for all 50 epochs).

### Spatial axis

| module | v2/ASGAP | v2/ABMIL | u2/ASGAP | u2/ABMIL | ti/ASGAP | ti/ABMIL |
|---|---|---|---|---|---|---|
| cutmix_spatial_noise @0.5 (contiguous BLOCK from one donor ROI = true CutMix) | 0.8232/0.9568 | 0.8294/0.9576 | 0.7889/0.9542 | 0.7886/0.9608 | 0.7993/0.9554 | 0.7884/0.9392 |
| cutout_spatial @0.3 (drop a contiguous block) | 0.8035/0.9538 | 0.8009/0.9488 | 0.7762/0.9528 | 0.7718/0.9511 | 0.7926/**0.9615** | 0.7874/0.9558 |
| cutmix_posmatch_noise @0.8 (scattered, but donor from the SAME (r,c)) | 0.8190/0.9567 | 0.8175/0.9636 | 0.8166/0.9510 | 0.8101/0.9573 | 0.7671/0.9350 | 0.7483/0.9429 |

1. **Contiguity is worthless here.** The faithful 2-D CutMix (a real, spatially coherent piece of donor
   tissue) LOSES to the champion's scattered random replacement on all 6 cells (test -0.001..-0.025),
   even though preserving the local fibre network inside the transplanted block was the motivation.
2. **Patch position carries no usable signal.** Position-matched donors (centre stays centre, border stays
   border) also lose on all 6 cells. The "operator framing makes positions non-exchangeable" hypothesis
   fails.
3. **The grade is readable from a partial field.** Deleting a contiguous 30% block still gives ti/ASGAP
   test **0.9615 > the 0.9600 no-aug titan baseline** (ep17) and u2/ASGAP 0.9528 (+0.027 vs no-aug), both
   well-trained. Independent support for `asgap-bagsize-robustness` from a structural direction (block
   removal, not random thinning).
   => **the patch grid carries no exploitable information for this task** - a direct empirical
   justification of the permutation-invariant MIL formulation, and the reason scattered random
   replacement has been the optimum all along.

### Schedule axis

| module | v2/ASGAP | v2/ABMIL | u2/ASGAP | u2/ABMIL | ti/ASGAP | ti/ABMIL |
|---|---|---|---|---|---|---|
| cutmix_anneal_noise (0.8 -> 0 by ep30) | **0.8385/0.9631** | 0.7977/0.9524 | 0.8246/0.9348 | 0.8074/0.9377 | 0.7665/**0.9622** ep33 | 0.7616/0.9417 |
| cutmix_ramp_noise (0.2 -> 0.8 over ep20) | 0.8364/**0.9246** ep2 | 0.8238/0.9380 | 0.8238/0.9509 | 0.8000/0.9539 | 0.7692/0.9456 | 0.7752/0.9571 |
| cutmix_bankboot_noise (donor bank re-drawn 50% each epoch) | 0.8272/0.9544 | 0.8275/0.9636 | 0.8225/**0.9562** | 0.8131/0.9502 | 0.7642/0.9588 | 0.7568/0.9365 |

4. **Direction matters and it is ANNEAL, not RAMP.** Starting synthetic and finishing on real bags beats
   the reverse consistently (v2/ASGAP test 0.9631 vs 0.9246; ti/ASGAP 0.9622 vs 0.9456; u2 the only
   cell where ramp is ahead). So the train/test distribution mismatch created by 80%-synthetic bags DOES
   cost something at the END of training - the champion's constant schedule is slightly suboptimal in
   that specific sense - but removing it is not enough to clear both gates (anneal v2/ASGAP = val +0.013
   / test -0.006 vs champion). Notably anneal also lifts the stubborn titan/ASGAP test above its no-aug
   baseline (0.9622 > 0.9600, ep33) while failing val.
5. **Per-epoch donor-bank bootstrapping is neutral.** Restricting each epoch to a random 50% of the bank
   (less within-epoch diversity, more across-epoch diversity) lands at champion +-0.005 everywhere
   (u2/ASGAP test 0.9562 = +0.0026 over champion, val -0.007). The two kinds of donor diversity are
   interchangeable - a further consequence of "bag identity carries no information".

**0/36 champion-beating cells. Session total: 172 runs, 24 new mechanisms, 0 champion-beating cells.
Registry = 150 modules.** With the spatial and schedule axes closed, the augmentation lever has no
structural dimension left that the registry contract can express.

## ROUND 7 (2026-07-26, 28 runs) — multi-mechanism STACKS + second-order transfer: mechanisms do NOT compose

Runs: `experiments/20260726/{sga,sa4,crl,sa4s*}_*`. After every single-mechanism axis closed, the
remaining move was to STACK the near-misses, each of which closed a different part of the gap while
failing a different gate (this is exactly how the champion cutmix_then_noise was found). All compositions
so far had two stages; these are the first three- and four-way stacks.

| module | v2/ASGAP | v2/ABMIL | u2/ASGAP | u2/ABMIL | ti/ASGAP | ti/ABMIL |
|---|---|---|---|---|---|---|
| stack_gv_anneal @0.8 (anneal + global tokens + noise) | 0.7813/0.9508 | 0.7959/0.9604 | 0.8334/0.9381 | 0.8031/0.9456 | 0.7496/0.9321 ep1 | 0.7679/0.9494 |
| stack_all4 @0.8 (scale donors + adjclean dst + global tokens + anneal) | **0.8520**/0.9559 | 0.8400/0.9640 | 0.8281/0.9251 | 0.8123/0.9506 | **0.8040**/0.9538 | 0.7732/0.9631 |
| coral_subspace @0.5 (K=16 covariance re-colouring) | 0.7729/0.9385 | 0.8005/0.9491 | 0.7854/0.9333 | 0.7940/0.9292 | 0.7785/0.9603 **G1 85.7** | 0.7892/0.9226 |

**stack_all4 strength sweep (the titan lead):**

| cell | 0.50 | 0.60 | 0.70 | 0.80 | 0.90 |
|---|---|---|---|---|---|
| ti/ASGAP | 0.7836/**0.9622** | 0.7792/0.9596 | 0.7959/**0.9611** | **0.8040**/0.9538 | 0.7875/0.9581 |
| ti/ABMIL | 0.7756/0.9582 | **0.7949**/0.9494 | 0.7843/0.9404 | 0.7732/**0.9631** | 0.7875/**0.9701** |
| v2/ASGAP | - | - | **0.8469**/0.9586 | **0.8520**/0.9559 | **0.8457**/0.9407 |

### Findings

1. **MECHANISMS DO NOT COMPOSE - the frontier is a constraint, not a set of independent leaks.** The
   three-way stack (stack_gv_anneal) is WORSE than either of its components alone on nearly every cell;
   the four-way stack inherits the val gains of its parts and pays the matching test cost, landing on the
   same val<->test line as everything else. Every mechanism that lifts val gives back test at the SAME
   rate, so stacking only slides further along the identical frontier. This is the cleanest available
   explanation for 200 negative runs, and it also explains why the champion's own composition worked:
   feature_noise was genuinely orthogonal to CutMix - because (as this session proved) it does almost
   nothing at all.
2. **FIRST configuration ever to clear the titan/ASGAP val gate.** stack_all4 @0.8 gives titan/ASGAP
   val **0.8040 > 0.7968** - the hardest cell in the study (proposed aggregator x most augmentation-hostile
   backbone), where every previous augmentation was stuck at val 0.756-0.788. The failure mode simply
   INVERTS: it now passes val and fails test (0.9538 < 0.9600). The strength sweep shows the two gates are
   cleanly anti-correlated across strengths (0.50 test✓ val✗ -> 0.80 val✓ test✗), never simultaneous.
3. **Highest titan test of the whole study**: stack_all4 @0.90 on ti/ABMIL = test **0.9701** (mRec 88.8,
   G1 79.6, ep21) vs the 0.9584 titan baseline and the 0.9639 subspace_noise champion - but val 0.7875
   misses the 0.7902 gate by 0.0027.
4. **Second-order (covariance) transfer is a new mechanism class and it also lands on the frontier**, with
   one signal: coral_subspace gives titan/ASGAP test **G1 recall 85.7** (2nd-highest in the study) and
   mRec 87.2. Together with the global-view round (G1 89.8), TWO unrelated mechanisms that inject
   FIELD-LEVEL structure both lift G1 specifically - consistent with G1 being defined by network
   continuity across the field rather than by single-patch appearance.

**SESSION TOTAL (2026-07-25/26): 200 runs, 27 new mechanisms, 153 registry modules, 0 champion-beating
cells.** Per-cell champions unchanged: virchow2/ASGAP cutmix_then_noise 0.8254/0.9695; virchow2/ABMIL
prototype-cutmix 0.8516/0.9684; uni2/ABMIL cutmix_then_noise 0.8113/0.9619; titan/ABMIL subspace_noise
0.8050/0.9639; titan/ASGAP no augmentation (0.7968/0.9600).
