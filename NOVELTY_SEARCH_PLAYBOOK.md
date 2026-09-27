# Novelty-Search Playbook — ROI-level Reticulin Fibrosis Grading

> **Purpose.** This document is the single source of truth a (human or AI)
> agent should read to **reproduce, extend, and run a new novelty search**
> against the locked baseline
> `experiments/20260523/04_reti_simple_virchow2_regression_20260523_004342`
> (`simple` + `virchow2` + scalar regression, full-data G1_3_3 split, seed=2).
> Following this file end-to-end is sufficient to (a) re-create the current
> leaderboard, (b) add a brand-new pooling/regularisation idea, (c) run it
> reproducibly at the locked config, and (d) decide whether it is a
> defensible thesis novelty.
>
> **Audience.** An AI coding agent that has access to the repo, a shell,
> and the existing `train_grading_reti.py` pipeline. **Do not change the
> data split, the loss, or the model-selection metric.**
>
> **Companion file (read it FIRST).** `NOVELTY_NOTES.md` at the repo root
> is structured agent memory: dead-end families, empirical patterns,
> open hypotheses, batch log. It exists so that prior negative results
> are not re-discovered. Any agent running this workflow must read
> `NOVELTY_NOTES.md` before proposing or coding anything, and must
> append a §9 entry after every batch.

---

## 0. Hard rules (non-negotiable)

1. **Patient split is locked.** Do not edit `patient_split()` in
   `src/train_grading_reti.py` or the hardcoded per-grade targets
   `(2,2)/(3,3)/(3,3)/(2,2)`. Full-data G1_3_3 split:
   Train 30 / Val 10 / Test 10 patients (857 / 214 / 259 bags).
2. **Test set is locked.** Never tune anything on test metrics. Selection
   is on **validation QWK**. Test numbers are reported only once per
   final candidate.
3. **Loss = `SmoothL1Loss`**, **formulation = `regression`**,
   **main metric = `qwk`**, **early-stop patience = 15**, **epochs ≤ 50**.
4. **Same baseline config everywhere** (only `--model_type` and
   `--novelty_id` change between runs):
   ```
   --backbone virchow2 --data_root data
   --epochs 50 --lr 1e-4 --batch_size 1
   --seed 2 --num_workers 4 --topk 0
   --early_stop_patience 15
   --formulation regression --main_metric qwk
   ```
5. **No new dependencies** without asking the user. Use only what is in
   `requirements.txt`.
6. **Destructive ops (file/dir deletion, `git reset`, retraining over an
   existing experiment dir) require explicit user confirmation.**
7. **Search-philosophy diversity rule (added 2026-05-25, after lessons
   from rounds a01–a44).** A "philosophy bucket" is the *aggregator
   philosophy* (e.g. `attention_augment` = "add one more prior on top of
   gated-attention", `attention_replace_parameter_free` = "no learned
   attention", `ordinal_head`, `multi_scale_fusion`, `backbone_fusion`,
   `patch_aux_loss`, `projection`, `ensemble`, …). No more than **5
   consecutive batches in the same philosophy bucket without a win**.
   After 5 in-bucket losses the next batch **must** propose a module
   from a bucket that is not yet exhausted, NOT a 6th refinement of the
   same prior. Every new `aNN_*.py` docstring must declare its
   `Philosophy bucket:` (see §3 template). The §11 autonomous loop
   terminates if this rule is violated.
8. **Legacy-win port mandate (added 2026-05-25).** Before generating
   diagnostic-driven hints for a new search line, every prior winner
   (from any baseline / backbone) listed in
   `NOVELTY_NOTES.md` → `legacy_winners_to_port` MUST be re-implemented
   against the current locked baseline first. A re-tested legacy winner
   is worth more than 5 diagnostic-derived hints because it carries one
   real positive signal across baselines. After porting, mark the entry
   `status: family_explored` (or `status: family_dead`) in the frontmatter.
9. **Thesis-novelty criteria (added 2026-05-25, refined 2026-05-25).**
   This repository exists to support a Master's thesis whose advisor
   (Dr. Pittipol Kantavat) requires **Type-2 (methodological) novelty
   AND an engineering win**. The §7 win gates
   (`val_qwk > 0.8182 AND test_qwk > 0.9476`) are the **primary
   success criterion** — beating the baseline at seed=2 is non-optional
   for the thesis to claim its proposed method works. The five
   `thesis_contribution` criteria below are **secondary** and exist
   only as bookkeeping so that runs which do NOT yet beat the baseline
   can still be written up cleanly as systematic-study evidence (see
   §13.1, Ch. 4.3). They are **NOT a substitute** for the §7 gates and
   **do NOT authorise stopping the search early**.

   `thesis_contribution` bookkeeping criteria (all five must hold for a
   run to feed the Ch. 4.3 / Ch. 4.4 prose):

   - **Methodological identity**: the method is a *replacement* or a
     *principled augmentation* of MIL aggregation, not a hyperparameter
     sweep. It belongs to a named philosophy bucket from §0 rule 7.
   - **Pathology-motivated rationale**: the design choice is justified
     by reticulin/MPN-specific reasoning that can be stated in 2–3
     sentences without invoking the empirical result.
   - **Clean ablation**: a paired ablation companion exists (or is
     planned in the same batch) that isolates the active ingredient.
     The ablation result is reported even when it disconfirms the main.
   - **Reproducible diagnostics**: a `method_note.md`, `confusion_*.csv`,
     and `per_roi.csv` exist for the run; the per-class behaviour is
     describable in the thesis text.
   - **Non-redundant with prior contributions**: the method is not
     covered by a published baseline (e.g. ABMIL, DeepSet, Set
     Transformer) unless the *application* to MPN reticulin grading
     itself is the contribution.

   **Search continues until an `engineering_win` (§7 Tier A) is
   produced or a hard stop condition in §11 fires.** Logging a run as
   a `thesis_contribution` does NOT decrement the `consec_no_wins`
   counter and does NOT stop the autonomous loop. The purpose of the
   tier-B tag is only to mark which runs are write-up-ready for the
   systematic-study chapter, so that on defense day the agent can show
   *both* the engineering winner *and* the surrounding evidence
   without re-deriving it from raw `.out` files.

---

## 1. Repository map (only the files this workflow touches)

```
src/
  train_grading_reti.py          # the ONLY trainer entrypoint
  core/config.py                 # EXPERIMENTS_DIR, SEED
  data/bag_dataset.py            # loads .pt feature bags; defines patient lists
  models/
    mean_pool_mil.py             # legacy baseline (MeanPoolMIL)
    simple_mil.py                # ABMIL — current locked baseline head
    novelty_attempts/            # currently EMPTY (only __init__.py).
      __init__.py                #   a-series restarted from scratch.
      # a<NN>_<short_name>.py    #   Next id: a01. (Prior a2..a35 were
      # _template.py             #    deleted 2026-05-16; new modules may
                                 #    re-use any id.) See §3 template.
scripts/
  loop_runner.sh                 # SINGLE parameterized batch runner — takes attempts as args
  batches/                       # optional: one .txt per historical/planned batch
  append_to_global_leaderboard.py# idempotent: scans experiments/ -> leaderboard.csv
  summarize_loop.py              # prints sorted leaderboard table
  leader_diagnostics.py          # MANDATORY pre-step (§3.5): turns the current
                                 # leader into rule-based "next novelty hints"
                                 # (supports --json for agent consumption)
  novelty_status.py              # AGENT ENTRYPOINT: single JSON dump of repo
                                 # state (constraints, baselines, leader,
                                 # leaderboard top, dead-ends, hypotheses,
                                 # next-action recommendation). Run FIRST.
results/
  leaderboard.csv                # one row per completed run
  diagnostics/<leader_name>/     # leader_diagnostics.py output (per leader)
experiments/
  20260413/                                          # date-bucketed (YYYYMMDD)
    reti_mean_pool_uni2_20260413_145001/             #   LEGACY baseline (no longer active)
  20260523/
    04_reti_simple_virchow2_regression_20260523_004342/  # CURRENT locked baseline
    reti_novelty_attempt_virchow2_<id>_s<seed>_<ts>/     # new-baseline novelty runs
  reti_*_<old_ts>/                                   # legacy flat dirs — still discovered
runs/<novelty_id>.out            # stdout per run
NOVELTY_SEARCH_PLAYBOOK.md       # THIS FILE
```

### Per-run artifact contract (already implemented by the trainer)

Every run writes to `experiments/<YYYYMMDD>/reti_novelty_attempt_virchow2_<postfix>_<ts>/`:

| File | Purpose |
|---|---|
| `config.json` | full argparse + git/env metadata |
| `training_log.csv` | per-epoch train/val loss, qwk, mae, acc, f1, recall |
| `val_metrics.json` | best-epoch val_qwk / val_mae / val_acc / val_f1 / val_macro_recall |
| `val_predictions.csv` | per-ROI y_true, y_pred, y_score |
| `val_confusion_matrix.{csv,png}` | confusion matrix at best epoch |
| `test_metrics.json` | test_qwk / test_mae / test_acc / test_f1 / test_macro_recall |
| `test_predictions.csv` | per-ROI test predictions |
| `test_confusion_matrix.{csv,png}` | test confusion matrix |
| `best_*.pth` | model checkpoint at best val_qwk epoch |
| `method_note.md` | short rationale (auto-written from the novelty docstring) |

A row is appended to `results/leaderboard.csv` on completion **only when the
run is invoked with `--leaderboard`** (the trainer's default is OFF). The
batch / novelty-search scripts (`loop_runner.sh`, `rerun_proposal_ablations*`)
pass it automatically. For ad-hoc / sanity runs you start by hand, omit the
flag — per-run artifacts (config, logs, checkpoint, predictions, CMs,
`method_note`) are still saved. If you forget and want to backfill rows
afterwards, run `scripts/append_to_global_leaderboard.py` (idempotent
re-sync that scans `experiments/`).

> **Layout note (May 2026 onwards).** Every new run is written to
> `experiments/<YYYYMMDD>/<run_name>_<HHMMSS>/` instead of the flat
> `experiments/<run_name>_<ts>/`. This keeps `experiments/` tidy and
> makes "what did I run today" trivial: `ls experiments/$(date +%Y%m%d)/`.
> The leaderboard scripts and `loop_runner.sh` already glob
> both layouts, so legacy runs remain visible.

---

## 2. The locked baseline

```
experiments/20260523/04_reti_simple_virchow2_regression_20260523_004342/
    # current locked baseline: simple + virchow2 + scalar regression, seed=2
```

**Numbers a new novelty must beat to count as a win:**

| Metric | Baseline (simple + virchow2 + regression, seed=2) |
|---|---:|
| val_qwk  | 0.8182 |
| test_qwk | 0.9476 |
| test_acc | 85.33 |

> A candidate **wins** only if `val_qwk > 0.8182 AND test_qwk > 0.9476`
> *at seed=2*. (Selection is on val_qwk; test_qwk is the gate-keeper for
> "no test-set collapse".) Constants `TARGET_VAL_QWK` / `TARGET_TEST_QWK`
> in `scripts/loop_runner.sh` enforce this automatically.

**Legacy baseline (no longer active, do not gate against):**

```
experiments/20260413/reti_mean_pool_uni2_20260413_145001/    # UNI2-h + mean_pool, 42-patient split
experiments/20260506/reti_mean_pool_uni2_s2_20260506_232602/ # seed=2 repro
```

| Metric | Legacy original | Legacy seed=2 repro |
|---|---:|---:|
| val_qwk  | 0.8130 | 0.8090 |
| test_qwk | 0.9080 | 0.9129 |
| test_acc | 85.91  | 86.58  |

---

## 3. Add a new novelty (template)

A "novelty module" is a single self-contained Python file in
`src/models/novelty_attempts/`. It must expose two symbols:

- **`Model`** — an `nn.Module` whose `forward(features)` returns
  `(bag_score, patch_attention_or_weights, None)`.
  - Input `features` is `[N, D]` for a single bag (or `[B, N, D]` batched).
  - Output `bag_score` is `[1]` (or `[B, 1]`), already clamped to `[0, 3]`
    if `clamp_output=True`. **Required for regression.**
- **`KWARGS`** — a `dict` of constructor kwargs. The trainer overrides
  `input_dim` and `num_classes` to match the chosen backbone, so just
  pin them to Virchow2 defaults: `input_dim=1280, num_classes=1`.

### Canonical template (copy any prior `aNN_*.py` or use the structure below)

```python
"""<aXX> — <one-line idea>.

Philosophy bucket: <one of: attention_augment | attention_replace_learned |
    attention_replace_parameter_free | projection | ensemble |
    ordinal_head | multi_scale_fusion | backbone_fusion |
    patch_aux_loss | norm_based_salience | other>     # REQUIRED, see §0 rule 7

Why it might beat the ABMIL baseline: <2-3 sentences of clinical /
statistical rationale>. Note whether this is an *augment* (adds on top of
the existing attention head) or a *replace* (swaps the aggregator philosophy).
After 5 consecutive in-bucket failures, switching philosophy bucket is
mandatory per §0 rule 7.
"""
from typing import Optional, Tuple
import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, input_dim: int, num_classes: int = 1,
                 dropout: float = 0.5, clamp_output: bool = True,
                 # ... your hyperparameters ...
                 ) -> None:
        super().__init__()
        assert num_classes == 1, "regression only"
        self.classifier = nn.Sequential(
            nn.Dropout(dropout), nn.Linear(input_dim, 1)
        )
        self.clamp_output = clamp_output
        # ... your aggregator state ...

    def forward(self, features: torch.Tensor):
        squeeze = features.dim() == 2
        if squeeze:
            features = features.unsqueeze(0)          # [1, N, D]

        # === YOUR AGGREGATOR ===
        # bag = aggregator(features)  -> [B, D]
        bag = features.mean(dim=1)                    # placeholder

        # weights for interpretability slot (optional, can be None)
        weights = torch.full(features.shape[:2], 1.0 / features.shape[1],
                             device=features.device)

        y = self.classifier(bag)                      # [B, 1]
        if self.clamp_output:
            y = y.clamp(0.0, 3.0)

        if squeeze:
            y = y.squeeze(0)                          # [1]
            weights = weights.squeeze(0)              # [N]
        return y, weights, None


KWARGS = dict(input_dim=1280, num_classes=1, dropout=0.5)  # Virchow2 default
```

### Naming convention

`a<NN>_<short_snake_name>.py` — strictly increment `NN`. The a-series is
**restarted from scratch** under the new baseline: next available id is
**`a01`** (see `NOVELTY_NOTES.md` frontmatter `search_config.next_module_id`).
Prior a2..a35 files were deleted on 2026-05-16, so any id in 1..35 is
free to re-use for *new* content. Update the docstring header with the
exact rationale that will appear in `method_note.md` of every run.

### Defensibility checklist (do not skip)

- [ ] Identical or fewer trainable parameters than the `simple` baseline (~197K).
- [ ] Permutation-invariant in the patch dimension (or note why not).
- [ ] Bag-size-invariant (so it works on ROIs of different N).
- [ ] Deterministic at inference (`model.eval()`).
- [ ] Output strictly in `[0, 3]`.
- [ ] Has a clean ablation companion already in the queue (e.g. a
      "raw" version vs a "rank-normalised" version) so you can argue
      the *active ingredient*.

---

## 3.5. Mandatory pre-step: diagnose the current leader first

> **Rule.** Before opening a new `aNN_*.py`, run the diagnostics script
> against the current leader. The output `next_novelty_hints.md` is the
> *only* legitimate starting point for the next batch. This replaces
> "I have an intuition" with "I'm targeting failure mode X".

### Run it

```bash
# Default: diagnoses the current leader (or the locked baseline directly when
# no leader exists) against the locked baseline. ~10 seconds.
PYTHONPATH=src python scripts/leader_diagnostics.py

# Diagnose a specific candidate instead:
PYTHONPATH=src python scripts/leader_diagnostics.py \
    --leader-dir experiments/<YYYYMMDD>/<run_dir>

# Or let it pick the top val_qwk row that ALSO passes test gate
# (skips val-overfit anti-patterns like a4_deepset):
PYTHONPATH=src python scripts/leader_diagnostics.py --auto-leader
```

### What it produces

`results/diagnostics/<leader_name>/`:

| File | Use |
|---|---|
| `summary.md` | val confusion matrices (leader vs baseline), per-grade error stats, bag-norm stats by grade |
| `next_novelty_hints.md` | **rule-based suggested aggregator families + ablation companions** |
| `per_roi.csv` | joined per-ROI predictions (both runs) + true label |
| `disagreement.csv` | only ROIs where leader / baseline rounded preds differ, with winner column |
| `bag_stats.csv` | per-bag norm statistics (mean, std, p10/50/90, top-10% share) |
| `confusion_{leader,baseline}.csv` | raw 4×4 matrices |

### The dispatch rules (so you know what triggers what)

| Rule fires when… | Suggested family |
|---|---|
| G2↔G3 ≥ 30% of leader errors | density / coverage prior |
| G1↔G2 ≥ 30% of leader errors | focal / peak prior |
| G0↔G1 ≥ 25% of leader errors | absence / noise-floor prior |
| \|corr(\|err\|, bag_size)\| ≥ 0.20 | bag-size-adaptive blend |
| Cross-grade mean(p90 norm) span < 5% of mean | switch family → projection-onto-prototype |
| Leader loses to baseline on ≥40% of disagreement ROIs | hedged late-fuse ensemble |
| ≥3 consecutive batches in `attention_augment` bucket failed | switch to **`attention_replace_parameter_free`** (rank-norm, statistical pooling, top-K with no learned scorer) |
| Disagreement-CSV shows leader wins ROIs where baseline attention is peaked | **`ordinal_head`** family (CORAL / CORN / cumulative-link head) |
| Bag-norm CV across grades > 50% | **`norm_based_salience`** (a43 / a44 family — rank-by-norm, norm-weighted pool) |
| Scale metadata (20/50/100/200µm) available but unused by the leader | **`multi_scale_fusion`** family (scale-conditioned aggregator) |
| Single-backbone leader plateaus for ≥3 batches | **`backbone_fusion`** family (concat Virchow2 + UNI2 features) |

Each hint comes with concrete module ideas **and a paired ablation companion** — so the *active ingredient* is isolable.

### Pre-registration stub (required in every new `aNN`)

The first lines of every new `src/models/novelty_attempts/aNN_<name>.py`
docstring must reference the diagnostic hint it targets and a kill
criterion. Copy this block verbatim and fill in:

```python
"""<aNN> — <one-line idea>.

Philosophy bucket: <see §3 template — one of the enumerated values>
Hint targeted (from results/diagnostics/<leader>/next_novelty_hints.md,
or NOVELTY_NOTES.md §6 when no leader exists):
    Hint #<N> — <family name, e.g. "density / coverage prior">
Hypothesis: <1 sentence — what change in val behaviour you expect>.
Ablation companion: <aNN-pair filename>.
Kill criterion: abandon if val_qwk < <threshold> at seed=2.
"""
```

This is enforced by social convention, not code. Reviewers (including
your future self at the defense) will check it.

### When to re-run diagnostics

Re-run **only when the leader changes** (i.e. a new candidate passes
the §7 gates). The hints are leader-specific — they go stale the moment
a different prior is on top. Running diagnostics on every failed `aNN`
is noise; running it on every new *winner* keeps ideation grounded.

---

## 4. Run a single novelty (the canonical command)

```bash
PYTHONPATH=src python src/train_grading_reti.py \
  --backbone virchow2 --data_root data \
  --epochs 50 --lr 1e-4 --batch_size 1 \
  --seed 2 --num_workers 4 --topk 0 \
  --early_stop_patience 15 \
  --formulation regression --main_metric qwk \
  --device "${DEVICE:-mps}" \
  --model_type novelty_attempt --novelty_id a01_<your_name> \
  --postfix a01_<your_name>_s2 \
  --leaderboard \
  > runs/a01_<your_name>.out 2>&1
```

- `--device`: use `cuda` on the GPU box, `mps` on this Mac, `cpu` as
  last resort. The script auto-detects with `--device auto`.
- `--postfix` is appended to the experiment dir name; **always include
  `_s<seed>` suffix** so multi-seed runs do not clash.
- `--leaderboard` appends a row to `results/leaderboard.csv` on completion.
  **Omit it for ad-hoc / sanity runs you don't want tracked** — the trainer
  defaults to NOT writing to the leaderboard, so manual experiments stay
  off the global table unless you opt in. `loop_runner.sh` and the proposal
  rerun scripts pass `--leaderboard` for you.
- Re-running the same `--postfix` creates a new timestamped dir; it
  does **not** overwrite. Old runs are safe.

---

## 5. Run a batch (loop runner)

The runner is a **single parameterized script**, `scripts/loop_runner.sh`.
You do **not** create a new file per batch — pick one of the three input
modes below:

```bash
# (A) Pass attempts inline (simplest):
DEVICE=mps NUM_WORKERS=4 EPOCHS=50 \
  ./scripts/loop_runner.sh a01_my_idea a02_ablation_companion

# (B) Read from a batch file (one aNN_name per line, # comments allowed).
#     Recommended for >3 attempts — the file becomes the historical record.
cat > scripts/batches/a01.txt <<'EOF'
# Batch a01 — first batch of the new-baseline search (simple + virchow2)
a01_my_idea
a02_ablation_companion
EOF
DEVICE=mps NUM_WORKERS=4 EPOCHS=50 \
  ./scripts/loop_runner.sh --batch scripts/batches/a01.txt

# (C) Auto-discover every aNN_*.py module that has no completed _s${SEED}_
#     experiment yet. Useful after you drop in several new files.
DEVICE=mps NUM_WORKERS=4 EPOCHS=50 \
  ./scripts/loop_runner.sh --auto
```

All knobs are env vars: `DEVICE`, `NUM_WORKERS`, `EPOCHS`, `SEED`,
`TARGET_VAL_QWK`, `TARGET_TEST_QWK`, `STOP_ON_WIN` (default 1; set to 0
to run every attempt to completion regardless of wins), `TAG` (free-form
label written to the log).

Key behaviour of `loop_runner.sh`:

- Runs each novelty in order with the **exact baseline config** (only
  `--model_type novelty_attempt --novelty_id <ATT>` and `--postfix`
  vary).
- Skips any novelty whose experiment dir already has `test_metrics.json`
  (idempotent — safe to re-launch).
- Calls `baseline_check` after every run; **`exit 0` on the first win**
  unless `STOP_ON_WIN=0`.
- Logs to `runs/loop_runner.log` and `runs/<ATT>.out`.

### Manual queue (no shell loop)

If you only want to run 1–3 ideas, just call the canonical command in
section 4 once per idea. Then refresh the leaderboard (section 6).

---

## 6. Update / inspect the leaderboard

```bash
# Append any new experiment dirs that aren't in the leaderboard yet.
python scripts/append_to_global_leaderboard.py

# Pretty table sorted by val_qwk (top 20).
python scripts/summarize_loop.py | head -25

# Or read the CSV directly.
column -ts, results/leaderboard.csv | less -S
```

Schema (`results/leaderboard.csv`):

```
timestamp, run_name, experiment_dir, model_type, backbone, seed, split_seed,
formulation, main_metric, best_epoch,
val_qwk, val_mae, val_accuracy, val_f1_macro, val_macro_recall,
val_loss, checkpoint
```

Test metrics live in `experiments/<dir>/test_metrics.json` — load them
when you need test_qwk for the "did it beat the baseline" decision.

---

## 7. Decide if a candidate is the new winner

> **Primary criterion = Tier A (engineering win).** A `thesis_contribution`
> (Tier B) is bookkeeping that helps write Ch. 4.3 / 4.4; it is **not** a
> winner and it does **not** end the search. The search continues until
> a Tier-A pass appears or a §11 hard stop condition fires.

### Tier A — Engineering win (PRIMARY — selection on val_qwk only)

Apply in order; **stop at the first failure**:

1. **Selection**: `val_qwk` (at seed=2) > **0.8182**.
2. **No-test-collapse gate**: `test_qwk` (at seed=2) > **0.9476**.
3. **Per-class sanity**: no class has `test_recall == 0` (check the
   test confusion matrix).
4. **Ablation companion in leaderboard**: at least one neighbouring
   novelty in the same family is present for contrast (one variant with
   the active ingredient on, one with it off / replaced by a control).

(Param count is no longer a gate — any capacity is allowed as long as the
two QWK gates above hold at seed=2. Still record it in `method_note.md`.)

Only candidates that pass all 4 update `current_leader` in
`NOVELTY_NOTES.md` frontmatter and are eligible to be reported as the
thesis's *proposed method* (Ch. 3). **Until such a candidate exists, the
search MUST keep running** (per §11).

### Tier B — Thesis-contribution bookkeeping (SECONDARY — never stops the search)

A run is logged as a **`thesis_contribution`** when **all five** of the
§0 rule 9 criteria are satisfied (methodological identity, pathology-
motivated rationale, clean ablation, reproducible diagnostics, non-
redundant with prior published methods). Crucially:

- **Tier B does NOT replace Tier A.** A Tier-B-only run is *not* a
  winner, *not* `current_leader`, and *cannot* be presented as the
  thesis's proposed method. It is allowed to appear in the
  systematic-study chapter (Ch. 4.3) as supporting evidence only.
- **Tier B does NOT stop the autonomous loop.** Logging a run as
  `thesis_contribution` does NOT decrement `consec_no_wins` and does
  NOT trigger any §11 stop condition. The agent keeps generating
  batches until a Tier-A win or a hard stop.
- **Tier B exists so write-up evidence is not lost.** When defense day
  arrives and the agent must produce Ch. 4.3 prose ("we evaluated N
  aggregator families…"), the tier-B tag tells it which runs are
  publication-ready and which are noisy/inconclusive.

> **Note on multi-seed robustness.** A multi-seed sweep
> (e.g. seeds 43 / 44 / 45) is the strongest possible robustness check,
> but it is intentionally **out of scope for the current workflow** —
> we focus on a single locked seed (`--seed 2`) so the search loop stays
> fast. If you later want to add it back, re-run the canonical command
> in §4 with `--seed 43/44/45 --postfix <id>_s<seed>`, then take
> mean ± std of `test_qwk` across seeds.

---

## 8. End-to-end "I want a new novelty" recipe (TL;DR for next time)

> Copy/paste this block into the next AI session along with the link to
> this playbook. The agent should be able to execute it autonomously
> without asking clarifying questions.

```text
You are running the novelty-search workflow defined in
NOVELTY_SEARCH_PLAYBOOK.md and NOVELTY_NOTES.md. Do this:

 0. Run `python scripts/novelty_status.py --pretty`. This returns the
    entire decision-relevant state as JSON in a single call:
        hard_constraints, baseline (with gates), current_leader,
        leaderboard_top, novelty_modules_on_disk, next_free_novelty_id,
        dead_end_families (DE<NN> ids), open_hypotheses (Hn ids),
        bootstrap_hints (BHk ids), diagnostics_runs, warnings,
        next_action_recommendation.
    If `warnings` is non-empty, surface them and stop.

 1. Read NOVELTY_NOTES.md §4 (dead-ends), §5 (empirical patterns),
    §6 (bootstrap hints if no leader), §7 (open hypotheses), §8
    (decision-helper checklist). Then read playbook §0 and §3.5.

 2. If `current_leader` is non-null, run
        PYTHONPATH=src python scripts/leader_diagnostics.py
    and read results/diagnostics/<leader>/next_novelty_hints.md and
    summary.md. Pick exactly ONE hint family.
    If `current_leader` is null, pick exactly ONE BH<k> hint from
    §6 of NOVELTY_NOTES.md (or one H<n> from §7 if you have a
    mechanistic reason to prefer it).

 3. Cross-check the chosen family against:
       - `dead_end_families` from step 0 (do not retry without a
         fundamentally new mechanism — state it explicitly),
       - prior §9 batch-log entries in NOVELTY_NOTES.md.

 4. Propose 2–4 new novelty modules at indices starting from
    `next_free_novelty_id` that:
       - target the chosen hint id,
       - pass every check in NOVELTY_NOTES.md §8 (the 6-item list),
       - keep parameter count ≤ baseline mean_pool,
       - include at least one ablation companion that removes the
         active ingredient.
    Each file's docstring MUST include the pre-registration stub
    from playbook §3.5 (referenced hint id, hypothesis, ablation
    companion, kill criterion).

 5. Implement each module as a separate file in
    src/models/novelty_attempts/aXX_<name>.py using the §3 template.
    Do not modify any other source file.

 6. Optionally write the attempt list to scripts/batches/<tag>.txt.
    scripts/loop_runner.sh is parameterized; no new shell script
    is needed.

 7. Ask the user once before launching: confirm device
    (mps/cuda/cpu). Then execute one of:
       ./scripts/loop_runner.sh aXX_... aYY_... ...
       ./scripts/loop_runner.sh --batch scripts/batches/<tag>.txt
       ./scripts/loop_runner.sh --auto

 8. After the loop finishes:
       python scripts/append_to_global_leaderboard.py
       python scripts/summarize_loop.py | head -30
       python scripts/novelty_status.py --pretty
    Report:
       - targeted hint id + 1-paragraph justification,
       - leaderboard delta (new rows),
       - candidates that passed both §2 gates (val_qwk > 0.8182 AND
         test_qwk > 0.9476 at seed=2),
       - if a winner emerged: re-run leader_diagnostics.py on it
         and report the new hints (do NOT spawn another batch
         unless the user confirms),
       - if no winner: explicitly choose (a) refine the SAME hint
         family next batch, or (b) move to a different hint id.
       (Multi-seed robustness is out of scope — seed=2 only.)

 9. APPEND a new entry to NOVELTY_NOTES.md §9 using the template
    at the top of that section. Required even for failed batches.
    Update the YAML frontmatter when §3/§4/§7 change:
       - if a winner emerged: set frontmatter `current_leader`,
       - if a family was cleanly invalidated: add a DE<NN> id to
         `dead_end_family_ids` and a row to §4,
       - if a new testable direction surfaced: add an H<n> id to
         `open_hypothesis_ids` and a row to §7.
    Never edit prior §9 entries — append only.

10. Never edit the patient split, the loss, the early-stop patience,
    or the baseline config. Never look at test metrics during ideation.

Deliverables in your final reply:
   * the `next_action_recommendation` you executed,
   * the targeted hint id + justification linking to specific numbers,
   * list of files created (each with its kill criterion),
   * exact commands run,
   * leaderboard table sorted by val_qwk (top 15),
   * winner candidate (or "no winner this batch") with rationale,
   * if winner: fresh diagnostics output on the new leader,
   * the new NOVELTY_NOTES.md §9 entry you appended,
   * honest noise/significance discussion,
   * recommended next batch + hint id.
```

---

## 9. Current state snapshot (May 2026)

- Baseline switched on **2026-05-23** to `simple + virchow2 + scalar regression`
  (full-data G1_3_3 split, seed=2):
  `experiments/20260523/04_reti_simple_virchow2_regression_20260523_004342`
  → val_qwk **0.8182** (epoch 5), test_qwk **0.9476**, test_acc **85.33%**.
- **Clean slate on disk.** `src/models/novelty_attempts/` contains only
  `__init__.py`. All prior `a`-series modules (a2..a35) were deleted on
  2026-05-16 and the legacy results are not comparable to the new baseline.
  **The a-series is restarted from scratch at `a01`** — any id in 1..35
  is free to re-use for new content. See `NOVELTY_NOTES.md` frontmatter
  `search_config.next_module_id`.
- **Failure-mode diagnostics on the new baseline have not been re-derived yet.**
  Before the first batch, run `scripts/leader_diagnostics.py` (or equivalent
  inspection of `val_predictions.csv` / `val_confusion_matrix.csv` in the
  baseline dir) to obtain new BH1/BH2/BH3 magnitudes. Pick **one** failure-mode
  family for the first batch — see §3.5 and `NOVELTY_NOTES.md` §6.
- **Search-loop policy** (mirrors `NOVELTY_NOTES.md` frontmatter `search_config`):
  full auto on MPS, 2 modules per batch (1 main + 1 ablation), stop on
  first BEATS_BASELINE OR by **2026-06-05**, whichever first.
  **The agent does NOT run `git add` / `git commit`** — version control
  is managed manually by the user.

---

## 10. Anti-patterns — common mistakes that wasted time

- ❌ Tuning anything on the test set, even "just to look".
- ❌ Re-running with a different `--seed` but forgetting the `_s<seed>`
  postfix → leaderboard rows look duplicated.
- ❌ Comparing against the legacy `mean_pool + uni2` baseline (or any
  other reproduction) instead of the locked
  `experiments/20260523/04_reti_simple_virchow2_regression_20260523_004342`.
- ❌ Forgetting `--formulation regression` — the `novelty_attempt` path
  refuses anything else; the run will fail loudly. Good.
- ❌ Editing `train_grading_reti.py` to "support" a novelty. The
  novelty module API is fixed. If you need a new behaviour, refactor
  the novelty, not the trainer.
- ❌ **Anchoring bias** (added 2026-05-25, after a01–a40 post-mortem):
  adding +1 mechanism on top of ABMIL for 5+ consecutive
  batches. If the val_qwk gain has been < 0.005 for 5 batches in a row
  in the same `attention_augment` bucket, **change philosophy bucket
  per §0 rule 7**, not the hint family within the same bucket.
- ❌ **Diagnostics tunnel vision**: `leader_diagnostics.py` hints
  default to "add a new prior to the leader". Once a leader
  has accumulated 3+ priors and still loses, the next batch must
  propose a **replacement aggregator** (a `*_replace_*` bucket), not
  a 4th additive prior.
- ❌ **Ignoring legacy wins** (§0 rule 8): filing a legacy positive
  result as "different setup" without porting it to the current
  baseline. A re-tested legacy winner carries 5×–10× the signal of a
  diagnostic-derived hint and is therefore mandatory to attempt first.


---

## 11. Autonomous outer-loop policy

When `NOVELTY_NOTES.md` frontmatter has `search_config.autonomous_outer_loop: true`,
the agent has been pre-authorised to iterate batches without per-batch user
approval. Apply the §8 recipe in a loop; stop only when **(a)** a candidate
passes both §7 gates at seed=2 (`val_qwk > 0.8182 AND test_qwk > 0.9476`),
**(b)** `consec_no_wins >= max_consecutive_no_wins` (default 8),
**(c)** today ≥ `hard_deadline` (2026-06-05),
**(d)** all BH-/H-hints in `NOVELTY_NOTES.md` are already attempted in §9
or dead-ended in §4, or
**(e)** §0 rule 7 is violated — i.e. the last 5 batches were all in the
same philosophy bucket without a win. In case (e), do NOT auto-spawn
another batch; instead surface the situation to the user with a one-line
recommendation of the next *unexplored* philosophy bucket.

Operational tips: launch `loop_runner.sh` with `isBackground=true`; poll
every 200–290 s (the shell tool times out at ~300 s); never read
`test_*.json` during hint/family selection; never skip the ablation
companion.


---

## 12. Post-mortem — what rounds a01–a44 taught us (added 2026-05-25)

> A short retrospective written after 42 novelty modules failed to beat
> the locked `simple + virchow2` baseline. The lessons below are now
> encoded as §0 rules 7–8, the §3 template tag, the §3.5 dispatch-rule
> extensions, the §10 anti-patterns, and the §11 termination condition
> (e). Future agents: read this once, then trust the rules.

### 12.1 What happened

- 40 novelty modules (`a01`–`a40`) were generated, all built on top of
  the `ABMIL` architecture. Every single one stayed inside the
  `attention_augment` philosophy bucket (add a coverage / length-norm /
  top-K / multi-query / projection / hedged-blend modifier on top of
  gated attention). None passed both `§7` gates.
- 2 modules (`a43`, `a44`, port of legacy `a25_rank_norm_softmean` from
  the previous UNI2 + mean_pool baseline) were the first to leave the
  `attention_augment` bucket. `a44` immediately landed a top-3 val_qwk
  (0.798) with `test_qwk` exceeding the baseline gate (+0.011), proving
  the value of bucket diversification.

### 12.2 Failure modes diagnosed

1. **Architecture anchoring.** The diagnostic-driven search (§3.5)
   produces hints that *add a prior to the leader*. After 40 priors
   stacked on the same attention head, marginal returns flattened to
   below noise. Hint diversity ≠ philosophy diversity.
2. **Legacy-win amnesia.** `legacy_a25_rank_norm_softmean` was the
   single novelty that beat the legacy `mean_pool + uni2` baseline on
   both val and test. When the baseline was switched on 2026-05-23 to
   `simple + virchow2`, the legacy results were filed as
   "not comparable" and the rank-norm idea was never ported. It took
   42 in-bucket failures before it was tried — and it became the
   best non-`attention_augment` candidate on the first try.
3. **Single-seed gate misleads near the noise floor.** Top 5 novelty
   modules cluster in `val_qwk ∈ [0.797, 0.809]` while the baseline at
   seed=2 sits at 0.8182 — a gap of ≤0.02 which is well within typical
   small-cohort val noise (10 val patients, 2 each of G0/G3). This
   playbook intentionally does **not** mandate multi-seed validation
   (it stays out of scope per §7 note), but agents must report when a
   leader lands inside this tie zone instead of claiming a definitive
   loss.

### 12.3 Rules added as a result

| New rule | Where | Purpose |
|---|---|---|
| Philosophy-bucket diversity (max 5 in a row) | §0 rule 7 | Force exit from a saturated bucket |
| Legacy-win port mandate | §0 rule 8 | Re-test prior winners before novel ideation |
| `Philosophy bucket:` docstring tag | §3 template + §3.5 stub | Make bucket trackable per module |
| Extended dispatch rules (5 new families) | §3.5 table | Give diagnostics a path *out* of `attention_augment` |
| Anti-patterns: anchoring, tunnel vision, legacy amnesia | §10 | Make the failure modes nameable |
| Termination condition (e) | §11 | Stop autonomous loop before another 5 wasted batches |

### 12.4 Buckets still un-explored as of 2026-05-25

These are first-class candidates for the next 1–2 batches per §0 rule 7
(do not return to `attention_augment` until at least one win or one
clean dead-end appears in each):

- `ordinal_head` — CORAL / CORN / cumulative-link regression head
  replacing the `Linear(128, 1) + round` of the current baseline.
- `multi_scale_fusion` — scale (20/50/100/200µm) as a conditioning
  signal for the aggregator (currently unused).
- `backbone_fusion` — concat Virchow2 (1280-d) + UNI2 (1536-d) frozen
  features before pooling.
- `patch_aux_loss` — auxiliary patch-level consistency / pseudo-label
  loss to regularise the aggregator without adding parameters at
  inference.
- `norm_based_salience` — extended a44 family (rank-norm + length-norm
  temperature, rank-norm + multi-query, density-aware pooling without
  learned attention).



---

## 13. Thesis-story tracking (added 2026-05-25)

> The novelty search exists to support a thesis whose **defensible
> contribution** must be statable in 1–2 sentences at the proposal
> exam (15 มิ.ย. 2026) and the final defense (ปลายปี 2026). This
> section is the bridge between *runs on disk* and *paragraphs in the
> thesis*. Update it whenever §9 of `NOVELTY_NOTES.md` is updated.

### 13.1 Mapping runs → thesis chapters

| Thesis chapter / paper section | Which runs feed it | What it claims |
|---|---|---|
| Ch. 3 — Methods (proposed aggregator) | The single `engineering_win` (if any), or the strongest `thesis_contribution` (if none) | The custom MIL aggregator design and its rationale. |
| Ch. 4.1 — Formulation comparison | Legacy `mean_pool + uni2` multi-class vs scalar regression runs | Scalar regression > multi-class (+0.148 QWK; G0 recall 0% → 75–88%). **Strongest piece of defense evidence.** |
| Ch. 4.2 — Backbone × aggregator ablation | Cross-tabulated mean_pool / simple × uni2 / virchow2 / titan runs (proposal-era 12 cells) | MIL × backbone interaction is non-uniform. |
| Ch. 4.3 — Systematic aggregator study | The full `philosophy_buckets_tried` map (all aNN runs, both passes and fails) | Reports which philosophy families are competitive on small-cohort reticulin MIL and which are dead. **42+ aNN runs are evidence, not waste.** |
| Ch. 4.4 — Ablation isolation | Every paired `(main, ablation)` from §9 batch log | Active-ingredient attribution per family. |
| Ch. 5 — Limitations | Tie-zone discussion (top 5 candidates within 0.02 val_qwk of baseline) + G0/G3 small-cohort variance | Honest scope statement; protects against over-claim. |

### 13.2 The "story decision tree" the agent must apply at the end of every batch

After appending §9 and updating `philosophy_buckets_tried`, the agent
must classify the run into ONE of:

| Outcome class | What it gives the thesis | Counts toward `consec_no_wins`? | What to do next |
|---|---|---|---|
| **A. Engineering win** (Tier A passes) | Ch. 3 candidate | Resets to 0 | Stop autonomous loop; surface to user for multi-seed / writing decisions. |
| **B. Thesis contribution, not engineering win** (Tier B passes, Tier A fails) | Ch. 4.3 / 4.4 supporting evidence (NOT a winner) | **Yes — increments** | **Continue search per §11**; a Tier-A win is still required. |
| **C. Clean dead-end** (paired ablation invalidates a hypothesis family) | Closes a hypothesis (HNN → DENN in §4 of NOVELTY_NOTES) | **Yes — increments** | Continue; the dead-end is also a contribution to Ch. 4.3 but is not a winner. |
| **D. Inconclusive** (neither A nor B; ablation noisy / family not isolated) | Nothing thesis-grade | **Yes — increments** | Discard from thesis, optionally re-run with cleaner companion. |

Only outcome class A resets `consec_no_wins` to 0. **B, C, and D all
advance the counter** — beating the baseline (Tier A) is the only thing
that constitutes a win for the autonomous loop. Tier B/C runs are
*saved for the thesis text* but the search must continue until A or a
§11 hard stop.

### 13.3 Story-readiness checklist (run after every batch)

Append to every §9 entry:

```
Story slot filled:
  [ ] Ch. 3 (candidate aggregator)
  [ ] Ch. 4.3 (systematic study entry — bucket: <name>)
  [ ] Ch. 4.4 (ablation isolation — paired with <aNN>)
  [ ] Ch. 5 (limitation / tie-zone evidence)
Outcome class: <A | B | C | D>
1-sentence thesis claim this batch supports:
  "<one sentence; if class D, write 'no thesis-grade claim this batch'>"
```

This forces the agent to think in *thesis paragraphs* not just
*leaderboard rows*. The user reviewing the §9 log before defense should
be able to read straight from these claims into Ch. 4.3 prose.

---

## 14. Search-mode switch (added 2026-05-25)

> Different stages of the thesis require different **search shapes**
> (where to explore, how deep). The mode does NOT change the win
> criterion — beating the §7 Tier-A gates is always the goal — it only
> changes *which philosophy bucket / depth* the agent prioritises this
> batch. Set the mode in `NOVELTY_NOTES.md` frontmatter
> `search_config.search_mode`.

### 14.1 Modes

| Mode | When to use | What the agent does (search shape only — Tier-A is still the win) |
|---|---|---|
| `engineering_chase` | A Tier-B candidate is within 0.02 val_qwk of baseline | Run ablation refinements + compound variants of the closest candidate; bucket-diversity rule still applies but compounds *inside* the closest bucket are allowed. Goal: convert the close-to-gate candidate into a Tier-A win. |
| `thesis_exploration` | No close-to-gate candidate exists | Cycle through unexplored philosophy buckets at one main+ablation per bucket; prioritise breadth so the next Tier-A attempt has more *families* to draw from. **Search still does not stop until Tier A passes**; tier-B runs along the way are bookkeeping for Ch. 4.3, not a success state. |
| `writing_freeze` | ≤2 weeks to a deadline AND user explicitly enables it | No new aNN files; only re-runs of existing top-3 candidates for confirmation; agent generates Ch. 3 / Ch. 4 prose drafts from `method_note.md`s. This is the only mode in which the search pauses without a Tier-A win, and it requires explicit user activation. |

### 14.2 Defaults

- Until 2026-06-01: **`thesis_exploration`**. The post-mortem (§12)
  showed 42 runs in 2 attention-family buckets and no Tier-A win. The
  fastest path to a Tier-A win is to widen the philosophy bucket
  search before re-deepening — every unexplored bucket is a fresh
  chance at the gate that 42 in-family runs did not produce. Buckets
  to prioritise: `ordinal_head`, `multi_scale_fusion`,
  `backbone_fusion`, `patch_aux_loss`, `norm_based_salience` compounds
  built on a44.
- 2026-06-01 → 2026-06-05: **`engineering_chase`** if any
  Tier-B candidate is within 0.015 val_qwk of baseline; otherwise stay
  in `thesis_exploration`. The Tier-A win bar does NOT relax as the
  deadline approaches.
- 2026-06-05 → defense: **`writing_freeze` only on explicit user
  approval**. Without user approval, the search keeps running until
  Tier A or the §11 hard deadline.

### 14.3 Override rule

The user may set `search_mode` explicitly in the frontmatter; the
agent must honour it without arguing. If the mode and the schedule
disagree, surface a one-line warning at the top of the next batch
report but proceed with the user's setting.
