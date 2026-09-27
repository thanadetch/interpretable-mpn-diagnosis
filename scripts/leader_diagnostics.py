"""leader_diagnostics.py — turn the current leader into a list of *next-novelty hints*.

Purpose
-------
Stop guessing what aggregator prior to try next. This script ingests the
val_predictions.csv of the current leader and the locked baseline
(`mean_pool` at seed=2), joins per-ROI, computes failure-mode statistics,
loads patch-level feature stats from the underlying .pt bags, and writes a
rule-based ``next_novelty_hints.md`` that tells you which *family* of
aggregator to design next (density / focal / absence / bag-size-adaptive /
projection / etc.).

It is strictly **validation-only**. It does not touch test predictions.

Usage
-----
    # Auto-pick the best val_qwk row from results/leaderboard.csv
    PYTHONPATH=src python scripts/leader_diagnostics.py

    # Or pin a specific leader / baseline experiment dir
    PYTHONPATH=src python scripts/leader_diagnostics.py \\
        --leader-dir experiments/20260508/reti_novelty_attempt_uni2_a25_rank_norm_softmean_s2_20260508_115535 \\
        --baseline-dir experiments/20260506/reti_mean_pool_uni2_s2_20260506_232602

Outputs (all under ``results/diagnostics/<leader_name>/``)
---------------------------------------------------------
    per_roi.csv               # joined per-ROI: true, pred_leader, pred_baseline, bag stats
    confusion_leader.csv      # leader val confusion matrix
    confusion_baseline.csv    # baseline val confusion matrix
    disagreement.csv          # ROIs where leader/baseline disagree, who wins
    bag_stats.csv             # per-bag patch-norm statistics
    summary.md                # human-readable diagnostic narrative
    next_novelty_hints.md     # >>> the actionable output <<<
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BASELINE = (
    REPO_ROOT
    / "experiments/20260506/reti_mean_pool_uni2_s2_20260506_232602"
)
# The leader is resolved at runtime from NOVELTY_NOTES.md frontmatter (via
# scripts/novelty_status.py). If no leader is set, this script exits cleanly
# with a clear message — there is nothing to diagnose against.
DEFAULT_FEATURES_DIR = REPO_ROOT / "data/features_uni2_reti"
DEFAULT_LEADERBOARD = REPO_ROOT / "results/leaderboard.csv"
DEFAULT_OUT_ROOT = REPO_ROOT / "results/diagnostics"

# Baseline gates from NOVELTY_SEARCH_PLAYBOOK.md §2 — a candidate must beat BOTH.
BASELINE_VAL_QWK = 0.8130
BASELINE_TEST_QWK = 0.9080


# ---------------------------------------------------------------------------
# Resolving leader / baseline
# ---------------------------------------------------------------------------

def resolve_leader_from_notes() -> Optional[Path]:
    """Read NOVELTY_NOTES.md YAML frontmatter to get current_leader.path.
    Returns None if no leader is set or notes file is missing.
    """
    notes = REPO_ROOT / "NOVELTY_NOTES.md"
    if not notes.exists():
        return None
    text = notes.read_text()
    if not text.startswith("---\n"):
        return None
    end = text.find("\n---\n", 4)
    if end < 0:
        return None
    front = text[4:end]
    in_block = False
    for raw in front.splitlines():
        line = raw.rstrip()
        if line.startswith("current_leader:"):
            tail = line.split(":", 1)[1].strip()
            if tail in ("null", "~", "None", ""):
                return None
            in_block = True
            continue
        if in_block:
            if not line.startswith(" "):
                break
            stripped = line.strip()
            if stripped.startswith("path:"):
                p = stripped.split(":", 1)[1].strip().strip('"').strip("'")
                return (REPO_ROOT / p).resolve()
    return None


def pick_leader_from_leaderboard(csv_path: Path, seed: int = 2) -> Path:
    """Return the eligible run with the highest val_qwk that ALSO passes the
    test-collapse gate (test_qwk > BASELINE_TEST_QWK). This avoids picking
    val-overfit anti-patterns (e.g. DeepSet / Set Transformer) as the 'leader'.
    Mean-pool rows are excluded — they are baselines, not leaders.
    """
    df = pd.read_csv(csv_path)
    df = df[(df["seed"] == seed) & (df["backbone"] == "uni2")
            & (df["formulation"] == "regression")
            & (df["model_type"] != "mean_pool")].copy()
    df = df.sort_values("val_qwk", ascending=False)
    for _, row in df.iterrows():
        exp_dir = Path(row["experiment_dir"])
        tmj = exp_dir / "test_metrics.json"
        if not tmj.exists():
            continue
        try:
            tqwk = float(json.loads(tmj.read_text()).get("test_qwk", 0.0))
        except Exception:
            continue
        if tqwk > BASELINE_TEST_QWK:
            return exp_dir
    raise RuntimeError(
        f"No row in {csv_path} passes both gates (val_qwk > {BASELINE_VAL_QWK} "
        f"AND test_qwk > {BASELINE_TEST_QWK})."
    )


# ---------------------------------------------------------------------------
# Bag (patch-feature) stats
# ---------------------------------------------------------------------------

def reconstruct_val_paths(features_dir: Path, seed: int = 2) -> List[Path]:
    """Use the trainer's own dataset + patient_split to recover the ordered
    list of .pt paths that make up the val set. This is the canonical join
    key: position i in val_predictions.csv == position i in this list.
    """
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from data.bag_dataset import GradingBagDatasetFull  # noqa: WPS433
    from train_grading_reti import patient_split  # noqa: WPS433
    ds = GradingBagDatasetFull(features_dir)
    _, val_idx, _ = patient_split(ds, seed=seed)
    return [ds.samples[i][0] for i in val_idx]


def bag_stats_for_path(pt_path: Path) -> Dict[str, float]:
    """Compute patch-norm statistics for a single bag .pt file."""
    data = torch.load(pt_path, map_location="cpu", weights_only=False)
    feats = data["feats"] if isinstance(data, dict) else data  # [N, D]
    feats = feats.float()
    norms = feats.norm(dim=-1).numpy()  # [N]
    N = int(feats.shape[0])
    q = np.quantile(norms, [0.10, 0.50, 0.90])
    k = max(1, int(round(0.10 * N)))
    top_sum = float(np.sort(norms)[-k:].sum())
    tot = float(norms.sum() + 1e-12)
    return {
        "bag_size": N,
        "norm_mean": float(norms.mean()),
        "norm_std": float(norms.std()),
        "norm_p10": float(q[0]),
        "norm_p50": float(q[1]),
        "norm_p90": float(q[2]),
        "norm_iqr": float(q[2] - q[0]),
        "norm_cv": float(norms.std() / (norms.mean() + 1e-12)),
        "top10pct_norm_share": top_sum / tot,
    }


# ---------------------------------------------------------------------------
# Joined per-ROI table
# ---------------------------------------------------------------------------

def load_predictions(run_dir: Path, suffix: str) -> pd.DataFrame:
    """Load val_predictions.csv, attach a positional ``row_id``, rename cols."""
    df = pd.read_csv(run_dir / "val_predictions.csv").reset_index(drop=True)
    df["row_id"] = df.index
    df = df.rename(columns={
        "pred_idx": f"pred_{suffix}",
        "pred_name": f"pred_name_{suffix}",
        "raw_output": f"score_{suffix}",
    })
    return df[["row_id", "slide_id", "label_idx", "label_name",
               f"pred_{suffix}", f"pred_name_{suffix}", f"score_{suffix}"]]


def confusion_matrix(df: pd.DataFrame, pred_col: str) -> pd.DataFrame:
    """4x4 confusion matrix (rows = true, cols = pred)."""
    cm = pd.crosstab(df["label_idx"], df[pred_col],
                     rownames=["true"], colnames=["pred"], dropna=False)
    for g in range(4):
        if g not in cm.index:
            cm.loc[g] = 0
        if g not in cm.columns:
            cm[g] = 0
    return cm.sort_index().reindex(sorted(cm.columns), axis=1).astype(int)


# ---------------------------------------------------------------------------
# Diagnostic rules → next-novelty hints
# ---------------------------------------------------------------------------

@dataclass
class Hint:
    family: str
    rationale: str
    suggested_ideas: List[str]
    ablation_companion: str


def _adjacent_error_pcts(cm: pd.DataFrame) -> Dict[Tuple[int, int], float]:
    """Fraction of all errors attributable to each (true, pred) off-diagonal pair."""
    total_err = int(cm.values.sum() - np.trace(cm.values))
    if total_err == 0:
        return {}
    out: Dict[Tuple[int, int], float] = {}
    for t in range(4):
        for p in range(4):
            if t == p:
                continue
            out[(t, p)] = int(cm.loc[t, p]) / total_err
    return out


def derive_hints(
    per_roi: pd.DataFrame,
    bag_stats: pd.DataFrame,
    cm_leader: pd.DataFrame,
    cm_baseline: pd.DataFrame,
) -> List[Hint]:
    """Apply the rule-based dispatcher → list of Hint suggestions."""
    hints: List[Hint] = []

    # ---- (1) Confusion-pair targeted families ----
    err_pcts = _adjacent_error_pcts(cm_leader)

    def pair_pct(a: int, b: int) -> float:
        return err_pcts.get((a, b), 0.0) + err_pcts.get((b, a), 0.0)

    g0g1, g1g2, g2g3 = pair_pct(0, 1), pair_pct(1, 2), pair_pct(2, 3)

    if g2g3 >= 0.30:
        hints.append(Hint(
            family="density / coverage prior",
            rationale=(
                f"G2↔G3 confusions account for {g2g3:.0%} of all leader errors. "
                "G3 pathologically means diffuse dense fibres *everywhere* — the leader "
                "is under-weighting the *fraction* of high-norm patches."
            ),
            suggested_ideas=[
                "weighted-mean where weight = soft-indicator(norm > learned-threshold), "
                "so the bag score scales with COVERAGE not just average intensity",
                "two-stat pool: (mean of norms) and (fraction of patches in top quantile), "
                "linearly combined with a single learned scalar",
            ],
            ablation_companion=(
                "raw coverage (hard threshold, no temperature) vs soft coverage "
                "(temperature-scaled indicator) — isolates the smoothness prior"
            ),
        ))

    if g1g2 >= 0.30:
        hints.append(Hint(
            family="focal / peak prior",
            rationale=(
                f"G1↔G2 confusions account for {g1g2:.0%} of errors. The G1/G2 boundary "
                "is driven by a few strong-evidence patches rather than overall density, "
                "so a peakier aggregator should help."
            ),
            suggested_ideas=[
                "log-sum-exp pool with a learned (small) temperature — peakier than mean",
                "top-q soft-mean over rank-of-norm (q ≈ 0.2), parameter-free",
            ],
            ablation_companion=(
                "soft-mean over top-q vs over top-1: isolates 'a few patches' vs "
                "'single strongest patch'"
            ),
        ))

    if g0g1 >= 0.25:
        hints.append(Hint(
            family="absence / noise-floor prior",
            rationale=(
                f"G0↔G1 confusions account for {g0g1:.0%} of errors. The G0/G1 boundary "
                "is about *absence* of fibrosis evidence — the model needs a robust "
                "estimate of the noise floor, not the peak."
            ),
            suggested_ideas=[
                "robust mean of the BOTTOM-q patches (q ≈ 0.3): a 'background' score "
                "that is added to mean as a soft offset",
                "two-branch late-fuse: presence head (G0 vs ≥G1) + severity head "
                "(G1..G3), combined deterministically",
            ],
            ablation_companion=(
                "bottom-q mean offset vs no offset — isolates the noise-floor signal"
            ),
        ))

    # ---- (2) Bag-size sensitivity ----
    merged = per_roi.merge(bag_stats, on="row_id", how="left") if "row_id" in bag_stats.columns else per_roi.assign(bag_size=np.nan)
    merged["abs_err_leader"] = (merged["label_idx"] - merged["score_leader"]).abs()
    if merged["bag_size"].notna().sum() >= 5:
        corr = float(merged["abs_err_leader"].corr(merged["bag_size"]))
        if abs(corr) >= 0.20:
            direction = "small" if corr < 0 else "large"
            hints.append(Hint(
                family="bag-size-adaptive aggregator",
                rationale=(
                    f"|corr(|err|, bag_size)| = {abs(corr):.2f} (sign: errors grow on "
                    f"{direction} bags). The leader does not adapt to N."
                ),
                suggested_ideas=[
                    "convex blend α(N)·leader + (1-α(N))·mean, with α monotone in N "
                    "(e.g. α = sigmoid((N - N0)/τ), N0 and τ as 2 scalar params)",
                    "length-normalised soft-mean with temperature ∝ 1/√N",
                ],
                ablation_companion=(
                    "fixed-α blend (α=0.5) vs learned α(N): isolates the *adaptivity*"
                ),
            ))

    # ---- (3) Family-saturation check: rank vs raw-norm distribution overlap ----
    if "row_id" in bag_stats.columns and "norm_p90" in bag_stats.columns:
        by_grade_p90 = bag_stats.merge(
            per_roi[["row_id", "label_idx"]], on="row_id"
        ).groupby("label_idx")["norm_p90"].mean()
    else:
        by_grade_p90 = pd.Series(dtype=float)
    if len(by_grade_p90) >= 3:
        span = float(by_grade_p90.max() - by_grade_p90.min())
        scale = float(by_grade_p90.mean() + 1e-9)
        rel = span / scale
        if rel < 0.05:
            hints.append(Hint(
                family="switch family: projection-onto-prototype",
                rationale=(
                    f"Across grades, mean p90(norm) varies by only {rel:.1%}. "
                    "The norm-rank family is saturated — no separability signal left in "
                    "patch norms. Switch to *direction-based* aggregation."
                ),
                suggested_ideas=[
                    "compute grade prototypes c_g = mean of train bag-means per grade; "
                    "score = mean of cosine(patch, c_3 - c_0); ordinal-aware, parameter-free",
                    "PCA-1 of train patches → use signed projection as the 'fibrosis "
                    "score' per patch, then soft-mean over rank-of-projection",
                ],
                ablation_companion=(
                    "cosine-to-prototype mean vs rank-of-cosine soft-mean — replicates the "
                    "a12/a25 dichotomy in a *new* feature space"
                ),
            ))

    # ---- (4) Leader-vs-baseline regression check ----
    merged["err_baseline"] = (merged["label_idx"] - merged["score_baseline"]).abs()
    merged["err_leader"] = merged["abs_err_leader"]
    leader_loses = int((merged["err_leader"] > merged["err_baseline"] + 0.1).sum())
    leader_wins = int((merged["err_baseline"] > merged["err_leader"] + 0.1).sum())
    if leader_loses > 0 and leader_loses >= 0.4 * (leader_wins + leader_loses):
        hints.append(Hint(
            family="hedged ensemble (defensive)",
            rationale=(
                f"Leader loses to baseline on {leader_loses} ROIs vs wins on "
                f"{leader_wins}. The leader's prior has a *cost* on some bags — a "
                "deterministic late blend would Pareto-dominate."
            ),
            suggested_ideas=[
                "score = 0.5·leader + 0.5·mean (zero new params, parameter-free)",
                "score = α·leader + (1-α)·mean with single learned α (clamp to [0,1])",
            ],
            ablation_companion=(
                "fixed 50/50 blend vs learned α — separates 'ensemble effect' from 'tuning'"
            ),
        ))

    return hints


# ---------------------------------------------------------------------------
# Renderers
# ---------------------------------------------------------------------------

def _df_to_md(df: pd.DataFrame, float_fmt: str = "{:.3f}") -> str:
    """Minimal Markdown table renderer (no 'tabulate' dep)."""
    df = df.copy()
    # Promote index to a column if it's named or non-default
    if df.index.name is not None or not isinstance(df.index, pd.RangeIndex):
        df = df.reset_index()
    cols = [str(c) for c in df.columns]
    rows: List[List[str]] = []
    for _, r in df.iterrows():
        out: List[str] = []
        for v in r.tolist():
            if isinstance(v, float):
                out.append(float_fmt.format(v))
            else:
                out.append(str(v))
        rows.append(out)
    header = "| " + " | ".join(cols) + " |"
    sep = "|" + "|".join(["---"] * len(cols)) + "|"
    body = "\n".join("| " + " | ".join(r) + " |" for r in rows)
    return "\n".join([header, sep, body])

def render_summary(
    leader_dir: Path,
    baseline_dir: Path,
    per_roi: pd.DataFrame,
    cm_leader: pd.DataFrame,
    cm_baseline: pd.DataFrame,
    bag_stats: pd.DataFrame,
) -> str:
    leader_metrics = json.loads((leader_dir / "val_metrics.json").read_text())
    baseline_metrics = json.loads((baseline_dir / "val_metrics.json").read_text())

    n = len(per_roi)
    abs_err_l = (per_roi["label_idx"] - per_roi["score_leader"]).abs()
    abs_err_b = (per_roi["label_idx"] - per_roi["score_baseline"]).abs()

    lines: List[str] = []
    lines.append(f"# Leader diagnostics\n")
    lines.append(f"- **Leader**: `{leader_dir.name}`")
    lines.append(f"- **Baseline**: `{baseline_dir.name}`")
    lines.append(f"- Val ROIs: {n}\n")

    lines.append("## Headline val metrics\n")
    lines.append("| metric | leader | baseline |")
    lines.append("|---|---:|---:|")
    for k in ("val_qwk", "val_mae", "val_accuracy", "val_f1_macro", "val_macro_recall"):
        lv = leader_metrics.get(k, float("nan"))
        bv = baseline_metrics.get(k, float("nan"))
        lines.append(f"| {k} | {lv:.4f} | {bv:.4f} |")
    lines.append("")
    lines.append(f"Mean |continuous err|: leader {abs_err_l.mean():.3f} | "
                 f"baseline {abs_err_b.mean():.3f}\n")

    lines.append("## Leader confusion matrix (rows=true)\n")
    lines.append(_df_to_md(cm_leader, float_fmt="{:.0f}"))
    lines.append("\n\n## Baseline confusion matrix (rows=true)\n")
    lines.append(_df_to_md(cm_baseline, float_fmt="{:.0f}"))
    lines.append("")

    lines.append("\n## Per-true-grade error stats (leader, continuous score)\n")
    g = per_roi.assign(err=(per_roi["label_idx"] - per_roi["score_leader"]).abs()) \
                .groupby("label_idx")["err"].agg(["count", "mean", "std", "max"])
    lines.append(_df_to_md(g))
    lines.append("")

    if not bag_stats.empty and "row_id" in bag_stats.columns:
        lines.append("\n## Bag-norm stats by true grade (means)\n")
        merged = bag_stats.merge(per_roi[["row_id", "label_idx"]], on="row_id")
        bg = merged.groupby("label_idx")[
            ["bag_size", "norm_mean", "norm_std", "norm_p90",
             "norm_cv", "top10pct_norm_share"]
        ].mean()
        lines.append(_df_to_md(bg))
        lines.append("")

    return "\n".join(lines) + "\n"


def render_hints(hints: List[Hint]) -> str:
    if not hints:
        return ("# Next-novelty hints\n\n"
                "No rule was triggered with high confidence. Manual ideation needed.\n"
                "Suggest reviewing `summary.md` and `disagreement.csv` directly.\n")

    out: List[str] = ["# Next-novelty hints\n",
                      "Rule-based suggestions derived from val-set diagnostics. "
                      "Pick **one** family for the next batch, implement two modules "
                      "(idea + ablation companion), then re-run this script after the "
                      "new leader is found.\n"]
    for i, h in enumerate(hints, 1):
        out.append(f"## Hint {i}: {h.family}\n")
        out.append(f"**Rationale.** {h.rationale}\n")
        out.append("**Suggested modules:**")
        for s in h.suggested_ideas:
            out.append(f"- {s}")
        out.append(f"\n**Ablation companion:** {h.ablation_companion}\n")
    out.append(
        "\n---\n"
        "**Discipline rule.** Before coding any `aNN`, add a 4-line stub in "
        "`src/models/novelty_attempts/aNN_<name>.py` docstring referencing the "
        "*specific hint* above and the *kill criterion* (e.g. abandon if "
        "val_qwk < 0.81 at seed=2). See `NOVELTY_SEARCH_PLAYBOOK.md` §3.5.\n"
    )
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--leader-dir", type=Path, default=None,
                    help="Path to leader experiment dir. Default: resolve from "
                         "NOVELTY_NOTES.md frontmatter `current_leader.path`. "
                         "Use --auto-leader to pick the top val_qwk row that "
                         "ALSO passes the test gate from leaderboard.csv.")
    ap.add_argument("--auto-leader", action="store_true",
                    help="Ignore notes; resolve from leaderboard.csv "
                         "(filtered by test-collapse gate).")
    ap.add_argument("--baseline-dir", type=Path, default=DEFAULT_BASELINE,
                    help=f"Default: {DEFAULT_BASELINE.relative_to(REPO_ROOT)}")
    ap.add_argument("--features-dir", type=Path, default=DEFAULT_FEATURES_DIR,
                    help="UNI2 patch-feature root (for bag-norm stats).")
    ap.add_argument("--leaderboard", type=Path, default=DEFAULT_LEADERBOARD)
    ap.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT)
    ap.add_argument("--no-bag-stats", action="store_true",
                    help="Skip the slow per-bag .pt loading step.")
    ap.add_argument("--json", action="store_true",
                    help="After writing files, also print a JSON summary "
                         "of leader/baseline metrics and the rule-based "
                         "hints to stdout (for agent consumption).")
    args = ap.parse_args()

    if args.leader_dir is not None:
        leader_dir = args.leader_dir
    elif args.auto_leader:
        try:
            leader_dir = pick_leader_from_leaderboard(args.leaderboard)
        except RuntimeError as e:
            print(f"[diagnostics] --auto-leader: {e}", file=sys.stderr)
            return 2
    else:
        notes_leader = resolve_leader_from_notes()
        if notes_leader is None:
            print(
                "[diagnostics] No current_leader set in NOVELTY_NOTES.md "
                "frontmatter. Nothing to diagnose.\n"
                "  - Bootstrap a first batch from NOVELTY_NOTES.md §6.\n"
                "  - When a winner emerges, set frontmatter `current_leader.path` "
                "and re-run this script.\n"
                "  - Or pass --leader-dir <path> / --auto-leader explicitly.",
                file=sys.stderr,
            )
            return 2
        leader_dir = notes_leader
    leader_dir = leader_dir.resolve()
    baseline_dir = args.baseline_dir.resolve()
    if leader_dir == baseline_dir:
        print(
            f"[diagnostics] Leader == baseline ({leader_dir.name}). Nothing to "
            "diagnose (the diff would be all zeros). Set a real leader first.",
            file=sys.stderr,
        )
        return 2
    out_dir = args.out_root / leader_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[diagnostics] leader   = {leader_dir}")
    print(f"[diagnostics] baseline = {baseline_dir}")
    print(f"[diagnostics] out      = {out_dir}")

    # --- Load & join predictions (by row position; slide_id is NOT unique) ---
    df_l = load_predictions(leader_dir, "leader")
    df_b = load_predictions(baseline_dir, "baseline")
    if len(df_l) != len(df_b):
        raise RuntimeError(
            f"val_predictions row counts differ: leader={len(df_l)} "
            f"vs baseline={len(df_b)}. Cannot join."
        )
    # Sanity: same true labels in same positions (both runs share dataset+seed)
    if not (df_l["label_idx"].values == df_b["label_idx"].values).all():
        raise RuntimeError("True labels disagree across runs — different splits?")

    per_roi = df_l.merge(
        df_b[["row_id", "pred_baseline", "pred_name_baseline", "score_baseline"]],
        on="row_id", how="inner",
    )
    per_roi.to_csv(out_dir / "per_roi.csv", index=False)
    print(f"[diagnostics] joined per_roi rows = {len(per_roi)}")

    # --- Confusion matrices ---
    cm_leader = confusion_matrix(per_roi, "pred_leader")
    cm_baseline = confusion_matrix(per_roi, "pred_baseline")
    cm_leader.to_csv(out_dir / "confusion_leader.csv")
    cm_baseline.to_csv(out_dir / "confusion_baseline.csv")

    # --- Disagreement set ---
    disagree = per_roi[per_roi["pred_leader"] != per_roi["pred_baseline"]].copy()
    disagree["leader_abs_err"] = (disagree["label_idx"] - disagree["score_leader"]).abs()
    disagree["baseline_abs_err"] = (disagree["label_idx"] - disagree["score_baseline"]).abs()
    disagree["winner"] = np.where(
        disagree["leader_abs_err"] < disagree["baseline_abs_err"], "leader",
        np.where(disagree["leader_abs_err"] > disagree["baseline_abs_err"],
                 "baseline", "tie"))
    disagree.to_csv(out_dir / "disagreement.csv", index=False)

    # --- Bag stats (positional join with reconstructed val .pt paths) ---
    bag_stats = pd.DataFrame(columns=["row_id"])
    if not args.no_bag_stats:
        val_paths = reconstruct_val_paths(args.features_dir, seed=2)
        if len(val_paths) != len(per_roi):
            print(f"[diagnostics] WARNING: val_paths={len(val_paths)} vs "
                  f"per_roi={len(per_roi)} — skipping bag stats.",
                  file=sys.stderr)
        else:
            records = []
            for i, p in enumerate(val_paths):
                rec = bag_stats_for_path(p)
                rec["row_id"] = i
                rec["pt_path"] = str(p)
                records.append(rec)
            bag_stats = pd.DataFrame.from_records(records)
            bag_stats.to_csv(out_dir / "bag_stats.csv", index=False)
            print(f"[diagnostics] bag_stats rows = {len(bag_stats)}")

    # --- Render outputs ---
    (out_dir / "summary.md").write_text(
        render_summary(leader_dir, baseline_dir, per_roi,
                       cm_leader, cm_baseline, bag_stats)
    )
    hints = derive_hints(per_roi, bag_stats, cm_leader, cm_baseline)
    (out_dir / "next_novelty_hints.md").write_text(render_hints(hints))

    print(f"[diagnostics] wrote {out_dir}/summary.md")
    print(f"[diagnostics] wrote {out_dir}/next_novelty_hints.md ({len(hints)} hints)")
    print("[diagnostics] done.")

    if args.json:
        leader_metrics = json.loads((leader_dir / "val_metrics.json").read_text())
        baseline_metrics = json.loads((baseline_dir / "val_metrics.json").read_text())
        payload = {
            "leader_dir": str(leader_dir.relative_to(REPO_ROOT)),
            "baseline_dir": str(baseline_dir.relative_to(REPO_ROOT)),
            "out_dir": str(out_dir.relative_to(REPO_ROOT)),
            "n_val_rois": int(len(per_roi)),
            "leader_val_metrics": {k: v for k, v in leader_metrics.items()
                                   if isinstance(v, (int, float))},
            "baseline_val_metrics": {k: v for k, v in baseline_metrics.items()
                                     if isinstance(v, (int, float))},
            "n_disagreement": int(len(disagree)),
            "n_leader_wins": int((disagree["winner"] == "leader").sum()) if len(disagree) else 0,
            "n_baseline_wins": int((disagree["winner"] == "baseline").sum()) if len(disagree) else 0,
            "hints": [
                {
                    "family": h.family,
                    "rationale": h.rationale,
                    "suggested_ideas": h.suggested_ideas,
                    "ablation_companion": h.ablation_companion,
                }
                for h in hints
            ],
        }
        print("---JSON---")
        print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())














