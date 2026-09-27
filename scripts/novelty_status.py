"""novelty_status.py — single agent entrypoint that dumps repo state as JSON.

Why this exists
---------------
The novelty-search workflow has many state files (NOVELTY_NOTES.md,
NOVELTY_SEARCH_PLAYBOOK.md, results/leaderboard.csv, experiments/**/,
results/diagnostics/**/). An AI agent that has to read all of them
spends most of its context just locating facts. This script collapses
the entire decision-relevant state into ONE JSON payload that an agent
can consume in a single tool call.

Output is JSON to stdout. Status is via exit code:
    0 = OK
    2 = state inconsistent (e.g., leader path missing on disk)

Usage
-----
    python scripts/novelty_status.py            # JSON to stdout
    python scripts/novelty_status.py --pretty   # human-readable
    python scripts/novelty_status.py --field current_leader.val_qwk
        # extract one nested field (dot path)

The payload schema is documented in the SCHEMA constant below and is
the contract the agent should rely on. Do not break it without bumping
SCHEMA_VERSION.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
NOTES_PATH = REPO_ROOT / "NOVELTY_NOTES.md"
PLAYBOOK_PATH = REPO_ROOT / "NOVELTY_SEARCH_PLAYBOOK.md"
LEADERBOARD = REPO_ROOT / "results/leaderboard.csv"
EXPERIMENTS = REPO_ROOT / "experiments"
NOVELTY_DIR = REPO_ROOT / "src/models/novelty_attempts"
DIAGNOSTICS_DIR = REPO_ROOT / "results/diagnostics"

SCHEMA_VERSION = "1.0.0"

SCHEMA = {
    "schema_version": "str",
    "repo_root": "str",
    "hard_constraints": {
        "patient_split_locked": "bool",
        "test_set_locked": "bool",
        "loss": "str",
        "formulation": "str",
        "main_metric": "str",
        "early_stop_patience": "int",
        "max_epochs": "int",
        "seed": "int",
        "backbone": "str",
        "max_trainable_params_rule": "str",
    },
    "baseline": {
        "original": {"path": "str", "val_qwk": "float", "test_qwk": "float"},
        "seed2_repro": {"path": "str", "val_qwk": "float", "test_qwk": "float"},
        "gate_val_qwk": "float",
        "gate_test_qwk": "float",
    },
    "current_leader": "object | null",  # {name, path, val_qwk, test_qwk, exists}
    "leaderboard_top": "list[object]",   # top-N rows sorted by val_qwk
    "novelty_modules_on_disk": "list[str]",  # aNN_*.py basenames
    "next_free_novelty_id": "str",        # e.g. "a2"
    "dead_end_families": "list[object]",  # parsed from NOVELTY_NOTES.md §4
    "open_hypotheses": "list[object]",    # parsed from NOVELTY_NOTES.md §7
    "bootstrap_hints": "list[object]",    # parsed from NOVELTY_NOTES.md §6
    "diagnostics_runs": "list[str]",      # subdirs of results/diagnostics/
    "bucket_diagnostics": "object",       # philosophy-bucket saturation (§0 rule 7, added 2026-05-25)
    "search_mode": "str",                 # thesis_exploration | engineering_chase | writing_freeze (§14)
    "legacy_winners_port_status": "list[object]",  # frontmatter legacy_winners_to_port
    "next_action_recommendation": "str",  # rule-based hint to the agent
    "warnings": "list[str]",
}


# ---------------------------------------------------------------------------
# Parsers — read the markdown frontmatter / tables in a brittleness-tolerant way
# ---------------------------------------------------------------------------

def parse_yaml_frontmatter(md_path: Path) -> Dict[str, Any]:
    """Parse the YAML frontmatter of a markdown file (first ``---``-delimited
    block). Prefers PyYAML for nested structures (e.g. dict-of-list-of-scalar
    like ``philosophy_buckets_tried`` added 2026-05-25); falls back to the
    minimal in-house parser if PyYAML is not importable.
    """
    if not md_path.exists():
        return {}
    text = md_path.read_text()
    if not text.startswith("---\n"):
        return {}
    end = text.find("\n---\n", 4)
    if end < 0:
        return {}
    block = text[4:end]
    try:
        import yaml  # PyYAML; present in the project venv since 2026-05-25
        loaded = yaml.safe_load(block)
        return loaded if isinstance(loaded, dict) else {}
    except Exception:
        # Fall back to the minimal parser (legacy behaviour).
        return _yaml_min_parse(block)


def _yaml_min_parse(block: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    stack: List[tuple] = [(0, out)]  # (indent, dict)
    lines = block.splitlines()
    i = 0
    while i < len(lines):
        raw = lines[i]
        if not raw.strip() or raw.lstrip().startswith("#"):
            i += 1
            continue
        indent = len(raw) - len(raw.lstrip(" "))
        line = raw.strip()
        # pop stack to current indent level
        while stack and stack[-1][0] > indent:
            stack.pop()
        parent = stack[-1][1]

        if line.startswith("- "):
            # list item — parent must be a list, but our parent is a dict;
            # peek the previous key on stack.
            # Simplified: treat as scalar list entry attached to the last-set
            # key in parent. Find that key.
            if isinstance(parent, list):
                parent.append(_scalar(line[2:].strip()))
            i += 1
            continue

        if ":" in line:
            key, _, val = line.partition(":")
            key = key.strip()
            val = val.strip()
            if val == "":
                # could be either dict-of or list-of
                # peek next non-empty line
                j = i + 1
                while j < len(lines) and not lines[j].strip():
                    j += 1
                if j < len(lines) and lines[j].lstrip().startswith("- "):
                    new_list: List[Any] = []
                    parent[key] = new_list
                    stack.append((indent + 2, new_list))
                else:
                    new_map: Dict[str, Any] = {}
                    parent[key] = new_map
                    stack.append((indent + 2, new_map))
            else:
                parent[key] = _scalar(val)
        i += 1
    return out


def _scalar(s: str) -> Any:
    s = s.strip()
    if s.startswith('"') and s.endswith('"'):
        return s[1:-1]
    if s.startswith("'") and s.endswith("'"):
        return s[1:-1]
    if s.lower() in ("true", "false"):
        return s.lower() == "true"
    if s.lower() in ("null", "~", "none"):
        return None
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    return s


_MD_TABLE_ROW = re.compile(r"^\|(.+)\|\s*$")
# Split on `|` that is NOT preceded by `\` (so escaped pipes inside cells survive).
_MD_CELL_SPLIT = re.compile(r"(?<!\\)\|")


def _split_row(inner: str) -> List[str]:
    parts = _MD_CELL_SPLIT.split(inner)
    return [p.strip().replace(r"\|", "|") for p in parts]


def parse_md_table(md_text: str, header_signature: List[str]) -> List[Dict[str, str]]:
    """Find the first markdown table whose header cells contain ALL strings
    in `header_signature` (case-insensitive substring match), return rows as
    list of dicts keyed by header cell text.
    """
    lines = md_text.splitlines()
    for i in range(len(lines) - 1):
        m = _MD_TABLE_ROW.match(lines[i])
        sep = _MD_TABLE_ROW.match(lines[i + 1]) if i + 1 < len(lines) else None
        if not m or not sep:
            continue
        cells = _split_row(m.group(1))
        if not all(any(sig.lower() in c.lower() for c in cells)
                   for sig in header_signature):
            continue
        # Confirm separator row
        sep_cells = _split_row(sep.group(1))
        if not all(set(c) <= set("-: ") and "-" in c for c in sep_cells if c):
            continue
        rows: List[Dict[str, str]] = []
        for j in range(i + 2, len(lines)):
            mj = _MD_TABLE_ROW.match(lines[j])
            if not mj:
                break
            vals = _split_row(mj.group(1))
            if len(vals) != len(cells):
                continue
            rows.append({cells[k]: vals[k] for k in range(len(cells))})
        return rows
    return []


# ---------------------------------------------------------------------------
# Repo introspection
# ---------------------------------------------------------------------------

def load_metrics(exp_dir: Path) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for fname, prefix in (("val_metrics.json", "val_"), ("test_metrics.json", "test_")):
        p = exp_dir / fname
        if p.exists():
            try:
                data = json.loads(p.read_text())
                for k, v in data.items():
                    if isinstance(v, (int, float)) and k.startswith(prefix):
                        out[k] = float(v)
            except Exception:
                pass
    return out


def list_novelty_modules() -> List[str]:
    if not NOVELTY_DIR.exists():
        return []
    return sorted(
        p.name for p in NOVELTY_DIR.glob("a*.py")
        if p.is_file() and p.name != "__init__.py"
    )


# Buckets enumerated in NOVELTY_SEARCH_PLAYBOOK.md §0 rule 7 / §3 template
# (added 2026-05-25). Keep this list in sync with the playbook.
KNOWN_PHILOSOPHY_BUCKETS: List[str] = [
    "attention_augment",
    "attention_replace_learned",
    "attention_replace_parameter_free",
    "projection",
    "ensemble",
    "ordinal_head",
    "multi_scale_fusion",
    "backbone_fusion",
    "patch_aux_loss",
    "norm_based_salience",
    "other",
]


def _extract_bucket_from_docstring(py_path: Path) -> Optional[str]:
    """Read ``Philosophy bucket: <name>`` from the module docstring (first
    50 lines). Returns the bucket name (lower-stripped) or None if missing.
    """
    try:
        with py_path.open() as f:
            head = "".join([next(f) for _ in range(50)])
    except (StopIteration, OSError):
        try:
            head = py_path.read_text()
        except OSError:
            return None
    m = re.search(r"Philosophy bucket:\s*([a-z_]+)", head, flags=re.IGNORECASE)
    if not m:
        return None
    return m.group(1).strip().lower()


def collect_module_buckets() -> Dict[str, List[str]]:
    """Return ``{module_basename: [bucket_or_'untagged']}`` for every
    aNN_*.py file currently on disk. Used to compute live bucket counts
    that may differ from frontmatter (e.g. if the agent forgot to update
    ``philosophy_buckets_tried`` after creating a new module).
    """
    out: Dict[str, str] = {}
    for name in list_novelty_modules():
        bucket = _extract_bucket_from_docstring(NOVELTY_DIR / name)
        out[name] = bucket or "untagged"
    return out


def bucket_diagnostics(
    front: Dict[str, Any],
    module_buckets: Dict[str, str],
) -> Dict[str, Any]:
    """Compute the live bucket-saturation picture used by §0 rule 7.

    Combines two sources of truth:
        1. ``search_config.philosophy_buckets_tried`` in the frontmatter
           (curated by the user / agent after each batch).
        2. ``Philosophy bucket:`` docstring tags scanned from disk.

    The frontmatter is treated as the canonical record; the disk scan is
    surfaced as a cross-check so the agent can spot tag-drift early.

    Returns a dict with:
        - counts (frontmatter + disk),
        - unexplored buckets (count == 0 in BOTH sources),
        - consecutive-in-bucket window (last K modules in id order),
        - saturated_bucket (the bucket name if the window ≥ max cap),
        - max_consecutive_in_bucket (the configured cap from frontmatter),
        - rule7_violated (boolean),
        - recommendation (one-line string for the agent).
    """
    sc = front.get("search_config") or {}
    cap = int(sc.get("max_consecutive_in_bucket", 5) or 5)
    frontmatter_buckets: Dict[str, List[str]] = (
        sc.get("philosophy_buckets_tried") or {}
    )

    # Counts per bucket (frontmatter source of truth)
    fm_counts = {b: len(frontmatter_buckets.get(b, []) or [])
                 for b in KNOWN_PHILOSOPHY_BUCKETS}
    # Counts per bucket from on-disk docstring tags
    disk_counts: Dict[str, int] = {b: 0 for b in KNOWN_PHILOSOPHY_BUCKETS}
    disk_counts["untagged"] = 0
    for bucket in module_buckets.values():
        if bucket in disk_counts:
            disk_counts[bucket] += 1
        else:
            disk_counts.setdefault(bucket, 0)
            disk_counts[bucket] += 1

    unexplored = [
        b for b in KNOWN_PHILOSOPHY_BUCKETS
        if fm_counts.get(b, 0) == 0 and disk_counts.get(b, 0) == 0
        and b != "other"
    ]

    # Consecutive-in-bucket window: last ``cap`` modules by aNN id (disk order).
    def _id(name: str) -> int:
        m = re.match(r"a(\d+)_", name)
        return int(m.group(1)) if m else -1
    sorted_modules = sorted(module_buckets.keys(), key=_id)
    window = sorted_modules[-cap:] if len(sorted_modules) >= cap else sorted_modules
    window_buckets = [module_buckets[m] for m in window]
    saturated_bucket: Optional[str] = None
    rule7_violated = False
    if len(window_buckets) >= cap:
        # Only declare violation when EVERY slot is a known bucket tag AND
        # they all agree. Untagged slots are unknown and cannot establish a
        # violation on their own.
        all_known = all(b not in (None, "untagged") for b in window_buckets)
        if all_known and len(set(window_buckets)) == 1:
            saturated_bucket = window_buckets[0]
            rule7_violated = True

    # Recommendation text
    if rule7_violated:
        suggest = unexplored[0] if unexplored else "any non-attention bucket"
        rec = (
            f"§0 rule 7 violated: last {cap} modules all in '{saturated_bucket}'. "
            f"Next module MUST belong to bucket '{suggest}' (currently 0 runs)."
        )
    elif unexplored:
        rec = (
            f"§14 mode='thesis_exploration': unexplored buckets = "
            f"{unexplored}. Prefer one of these for the next batch (§12.4 priority)."
        )
    else:
        rec = "All known buckets have ≥1 run. §0 rule 7 not triggered."

    return {
        "max_consecutive_in_bucket": cap,
        "frontmatter_counts": fm_counts,
        "disk_counts": disk_counts,
        "unexplored_buckets": unexplored,
        "recent_window_modules": window,
        "recent_window_buckets": window_buckets,
        "saturated_bucket": saturated_bucket,
        "rule7_violated": rule7_violated,
        "recommendation": rec,
    }


def next_free_id(modules: List[str]) -> str:
    used: List[int] = []
    for name in modules:
        m = re.match(r"a(\d+)_", name)
        if m:
            used.append(int(m.group(1)))
    nxt = (max(used) + 1) if used else 2  # a2 is the first novelty slot
    return f"a{nxt}"


def read_leaderboard_top(n: int = 10) -> List[Dict[str, Any]]:
    if not LEADERBOARD.exists():
        return []
    with LEADERBOARD.open() as f:
        rows = list(csv.DictReader(f))
    # Coerce val_qwk, then sort desc.
    for r in rows:
        try:
            r["_val_qwk_num"] = float(r.get("val_qwk", "nan"))
        except ValueError:
            r["_val_qwk_num"] = float("-inf")
    rows.sort(key=lambda r: r["_val_qwk_num"], reverse=True)
    out: List[Dict[str, Any]] = []
    for r in rows[:n]:
        out.append({
            "run_name": r.get("run_name"),
            "experiment_dir": r.get("experiment_dir"),
            "model_type": r.get("model_type"),
            "backbone": r.get("backbone"),
            "seed": r.get("seed"),
            "formulation": r.get("formulation"),
            "val_qwk": r["_val_qwk_num"],
        })
    return out


def collect_dead_end_families() -> List[Dict[str, str]]:
    if not NOTES_PATH.exists():
        return []
    text = NOTES_PATH.read_text()
    rows = parse_md_table(text, header_signature=["family", "reason"])
    # Keep only the parsed columns we care about.
    return [{k.lower().replace(" ", "_"): v for k, v in r.items()} for r in rows]


def collect_open_hypotheses() -> List[Dict[str, str]]:
    """Open hypotheses live in a markdown table in §7 with columns
    `id | hypothesis | mechanism summary | required ablation companion`.
    """
    if not NOTES_PATH.exists():
        return []
    text = NOTES_PATH.read_text()
    rows = parse_md_table(text, header_signature=["id", "hypothesis", "mechanism"])
    return [{k.lower().replace(" ", "_"): v for k, v in r.items()} for r in rows]


def collect_bootstrap_hints() -> List[Dict[str, str]]:
    """Bootstrap hints (used when no leader exists) live in §6 as a table
    with columns `id | failure mode | magnitude | suggested family`.
    """
    if not NOTES_PATH.exists():
        return []
    text = NOTES_PATH.read_text()
    rows = parse_md_table(text, header_signature=["id", "failure mode", "magnitude"])
    return [{k.lower().replace(" ", "_"): v for k, v in r.items()} for r in rows]


def list_diagnostics_runs() -> List[str]:
    if not DIAGNOSTICS_DIR.exists():
        return []
    return sorted(p.name for p in DIAGNOSTICS_DIR.iterdir() if p.is_dir())


# ---------------------------------------------------------------------------
# Assemble the status payload
# ---------------------------------------------------------------------------

def build_status() -> Dict[str, Any]:
    warnings: List[str] = []
    front = parse_yaml_frontmatter(NOTES_PATH)

    # Hard constraints — read from NOVELTY_NOTES.md frontmatter if present,
    # else fall back to playbook-defined defaults.
    constraints = front.get("hard_constraints", {}) or {
        "patient_split_locked": True,
        "test_set_locked": True,
        "loss": "SmoothL1Loss",
        "formulation": "regression",
        "main_metric": "qwk",
        "early_stop_patience": 15,
        "max_epochs": 50,
        "seed": 2,
        "backbone": "uni2",
        "max_trainable_params_rule": "<= mean_pool baseline",
    }

    # Baselines
    baseline_block = front.get("baseline", {})
    baseline_orig = (REPO_ROOT / baseline_block.get(
        "original_path",
        "experiments/20260413/reti_mean_pool_uni2_20260413_145001"))
    baseline_seed2 = (REPO_ROOT / baseline_block.get(
        "seed2_repro_path",
        "experiments/20260506/reti_mean_pool_uni2_s2_20260506_232602"))
    baseline = {
        "original": {
            "path": str(baseline_orig.relative_to(REPO_ROOT)),
            "exists": baseline_orig.exists(),
            **load_metrics(baseline_orig),
        },
        "seed2_repro": {
            "path": str(baseline_seed2.relative_to(REPO_ROOT)),
            "exists": baseline_seed2.exists(),
            **load_metrics(baseline_seed2),
        },
        "gate_val_qwk": float(baseline_block.get("gate_val_qwk", 0.8130)),
        "gate_test_qwk": float(baseline_block.get("gate_test_qwk", 0.9080)),
    }
    if not baseline_orig.exists():
        warnings.append(f"baseline.original.path missing: {baseline_orig}")
    if not baseline_seed2.exists():
        warnings.append(f"baseline.seed2_repro.path missing: {baseline_seed2}")

    # Current leader
    leader_block = front.get("current_leader")
    leader: Optional[Dict[str, Any]] = None
    if leader_block and leader_block.get("path"):
        lp = REPO_ROOT / leader_block["path"]
        leader = {
            "name": leader_block.get("name"),
            "path": leader_block["path"],
            "exists": lp.exists(),
            "active_ingredient": leader_block.get("active_ingredient"),
            **load_metrics(lp),
        }
        if not lp.exists():
            warnings.append(f"current_leader.path missing on disk: {lp}")

    modules = list_novelty_modules()
    next_id = next_free_id(modules)
    module_buckets = collect_module_buckets()
    bucket_info = bucket_diagnostics(front, module_buckets)
    sc = front.get("search_config") or {}
    search_mode = sc.get("search_mode", "thesis_exploration")
    legacy_port = front.get("legacy_winners_to_port") or []

    # Surface bucket-saturation as a warning so callers that exit on warnings
    # halt the loop, matching playbook §11 termination condition (e).
    if bucket_info.get("rule7_violated"):
        warnings.append(
            "RULE7_BUCKET_SATURATION: " + bucket_info["recommendation"]
        )
    # Warn on untagged modules (added 2026-05-25 — every aNN must declare
    # a Philosophy bucket: tag per §3 template).
    untagged = [m for m, b in module_buckets.items() if b == "untagged"]
    if untagged:
        warnings.append(
            f"UNTAGGED_MODULES: {len(untagged)} module(s) missing "
            f"'Philosophy bucket:' docstring tag: {untagged[:5]}"
            + ("…" if len(untagged) > 5 else "")
        )

    payload: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "repo_root": str(REPO_ROOT),
        "hard_constraints": constraints,
        "baseline": baseline,
        "current_leader": leader,
        "leaderboard_top": read_leaderboard_top(),
        "novelty_modules_on_disk": modules,
        "next_free_novelty_id": next_id,
        "dead_end_families": collect_dead_end_families(),
        "open_hypotheses": collect_open_hypotheses(),
        "bootstrap_hints": collect_bootstrap_hints(),
        "diagnostics_runs": list_diagnostics_runs(),
        "bucket_diagnostics": bucket_info,
        "search_mode": search_mode,
        "legacy_winners_port_status": legacy_port,
        "warnings": warnings,
    }

    # Next-action recommendation (rule-based, bucket-aware as of 2026-05-25).
    # Order of precedence: rule-7 saturation > legacy-port pending > standard.
    pending_legacy_ports = [
        e for e in legacy_port
        if (e.get("status") or "").lower() not in {"family_explored", "family_dead"}
    ]

    if bucket_info.get("rule7_violated"):
        unexp = bucket_info.get("unexplored_buckets") or []
        target = unexp[0] if unexp else "any non-attention bucket"
        rec = (
            f"§0 RULE 7 VIOLATED — last {bucket_info['max_consecutive_in_bucket']} "
            f"modules all in '{bucket_info['saturated_bucket']}'. "
            f"Next module ({next_id}_<name>.py) MUST belong to bucket "
            f"'{target}'. Do NOT propose another '{bucket_info['saturated_bucket']}' "
            "variant. See playbook §12.4 for the priority queue."
        )
    elif pending_legacy_ports:
        ids = [e.get("id") for e in pending_legacy_ports]
        rec = (
            f"§0 RULE 8 — legacy winners NOT yet ported to current baseline: {ids}. "
            "Port these as the next batch before generating diagnostic-driven hints."
        )
    elif leader is None:
        unexp = bucket_info.get("unexplored_buckets") or []
        prio = (
            f"Prefer unexplored buckets {unexp[:3]} per §12.4 priority. "
            if unexp else ""
        )
        rec = (
            "No current leader. Bootstrap from NOVELTY_NOTES.md §6. "
            f"{prio}Pick ONE family. Implement main idea + ablation companion "
            f"starting at module {next_id}_<name>.py. "
            "Pre-register both per playbook §3.5 with a Philosophy bucket: tag."
        )
    else:
        # Has the agent run diagnostics for this leader?
        diag_done = leader["name"] in (
            d.replace("reti_novelty_attempt_uni2_", "").replace(
                f"_s{constraints['seed']}", "_s2"
            ) for d in list_diagnostics_runs()
        )
        if not diag_done:
            rec = (
                f"Leader exists ({leader['name']}) but diagnostics not run. "
                "Execute: PYTHONPATH=src python scripts/leader_diagnostics.py "
                "Then read results/diagnostics/<leader>/next_novelty_hints.md."
            )
        else:
            rec = (
                "Leader and diagnostics both present. Read "
                "results/diagnostics/<leader>/next_novelty_hints.md, pick ONE "
                "hint family, propose a batch starting at module "
                f"{next_id}_<name>.py with an ablation companion "
                "(remember: declare Philosophy bucket: tag in the docstring)."
            )
    payload["next_action_recommendation"] = rec

    return payload


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _get_field(obj: Any, dotted: str) -> Any:
    cur = obj
    for part in dotted.split("."):
        if isinstance(cur, dict):
            cur = cur.get(part)
        elif isinstance(cur, list):
            try:
                cur = cur[int(part)]
            except (ValueError, IndexError):
                return None
        else:
            return None
    return cur


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pretty", action="store_true",
                    help="Indent JSON for human reading.")
    ap.add_argument("--field", default=None,
                    help="Print one nested field via dot-path (e.g. "
                         "current_leader.val_qwk).")
    ap.add_argument("--schema", action="store_true",
                    help="Print the payload schema and exit.")
    args = ap.parse_args()

    if args.schema:
        print(json.dumps({"schema_version": SCHEMA_VERSION, "fields": SCHEMA},
                         indent=2))
        return 0

    payload = build_status()

    if args.field:
        v = _get_field(payload, args.field)
        if isinstance(v, (dict, list)):
            print(json.dumps(v, indent=2 if args.pretty else None))
        else:
            print(v)
    else:
        print(json.dumps(payload, indent=2 if args.pretty else None))

    # Exit code: 2 if any path warning, else 0.
    return 2 if payload["warnings"] else 0


if __name__ == "__main__":
    sys.exit(main())






