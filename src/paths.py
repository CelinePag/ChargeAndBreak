"""
paths.py — the single source of truth for every directory the project reads or
writes.

Why this module exists
----------------------
Before the src/ restructure every module carried bare relative literals
("solutions", "logs", "figures", ...).  Those resolve against the *current
working directory*, so a script only worked when launched from the repository
root; running it from anywhere else silently created a second, empty tree of
output folders.  Anchoring on __file__ instead makes every path independent of
where the process was started.

Layout
------
    <ROOT>/
      src/                  all Python sources (this package)
      instances/            generated base instances        (input)
      instances_sens/       variant instances, one dir/axis (input)
      data/                 external datasets (R15-PGLT, ...)(input)
      solutions/            run results + oracle_<inst>.json caches
      logs/                 per-run .txt, gurobi .log, *_scenarios.json
      figures/              .pdf/.png plots
      tex/tables/           .tex — GENERATED tables (safe to overwrite)
      tex/sections/         .tex — hand-written manuscript prose (never written
                            by any script; kept apart so a table regeneration
                            can never be confused with the manuscript)
      data_output/          .csv/.xlsx — tabular exports
      archive/              files retired by audit_runs.py (never deleted)

Usage
-----
    from src import paths
    p = paths.SOLUTIONS / f"{run_id}.json"          # pathlib
    p = paths.solutions(f"{run_id}.json")           # str, for os.path-style code

Every accessor returns a plain ``str`` so it drops into the existing
``os.path.join`` call sites without further change; the module-level constants
are ``Path`` objects for new code.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

# ── anchor ───────────────────────────────────────────────────────────────────
# paths.py lives at <ROOT>/src/paths.py, so the root is one level up.
ROOT: Path = Path(__file__).resolve().parent.parent

# ── input trees ──────────────────────────────────────────────────────────────
INSTANCES      : Path = ROOT / "instances"
INSTANCES_SENS : Path = ROOT / "instances_sens"
DATA           : Path = ROOT / "data"

# ── output trees ─────────────────────────────────────────────────────────────
SOLUTIONS   : Path = ROOT / "solutions"
LOGS        : Path = ROOT / "logs"
FIGURES     : Path = ROOT / "figures"
TEX         : Path = ROOT / "tex"
TEX_TABLES  : Path = TEX / "tables"      # generated — scripts write here
TEX_SECTIONS: Path = TEX / "sections"    # hand-written — scripts NEVER write here
DATA_OUTPUT : Path = ROOT / "data_output"
ARCHIVE     : Path = ROOT / "archive"
RESULTS_VSS : Path = ROOT / "results_vss"

# Directories a run may write into.  Inputs are deliberately absent: a missing
# instances/ is a real error and must not be papered over by mkdir.  TEX_SECTIONS
# is absent on purpose — nothing generated may land next to the manuscript.
_WRITABLE = (SOLUTIONS, LOGS, FIGURES, TEX_TABLES, DATA_OUTPUT)


def ensure_dirs() -> None:
    """Create the output directories if they do not exist (idempotent)."""
    for d in _WRITABLE:
        d.mkdir(parents=True, exist_ok=True)


# ── str accessors ────────────────────────────────────────────────────────────
# `paths.solutions()` -> "<ROOT>/solutions"
# `paths.solutions("x.json")` -> "<ROOT>/solutions/x.json"

def _joiner(base: Path):
    def join(*parts: str) -> str:
        return os.path.join(str(base), *parts) if parts else str(base)
    return join


instances      = _joiner(INSTANCES)
instances_sens = _joiner(INSTANCES_SENS)
data           = _joiner(DATA)
solutions      = _joiner(SOLUTIONS)
logs           = _joiner(LOGS)
figures        = _joiner(FIGURES)
tex            = _joiner(TEX)
tex_tables     = _joiner(TEX_TABLES)
tex_sections   = _joiner(TEX_SECTIONS)
data_output    = _joiner(DATA_OUTPUT)
archive        = _joiner(ARCHIVE)
results_vss    = _joiner(RESULTS_VSS)


# ── run-id naming convention ─────────────────────────────────────────────────
# A run_id is the stem of solutions/<run_id>.json and logs/<run_id>.txt:
#
#     <instance>_<ALGO>[_<VARIANT>]_<YYYYmmdd>_<HHMMSS>[_<idx>]
#
# VARIANT is the METHOD-CONFIGURATION label (e.g. "S25H12" = 25 scenarios,
# 12 h look-ahead).  It exists because the reporting dedup is keyed on
# (instance, method, supervised, variant): without it, a sweep over a method's
# own parameters would have to be smuggled in by duplicating the INSTANCE under
# a "__tag" stem — which is what the earlier guard/gamma/diesel sweeps did.
# That is the wrong home for the label (the instance is unchanged) and it also
# orphans the run from the instance's already-solved oracle cache.
#
# Parsing is shared by the runner (which stamps `variant` into every solution
# JSON) and by compile_solutions (which ranks runs by recency), so the two can
# never drift apart.  The variant must START WITH A LETTER: that is what keeps
# it unambiguous against the 8-digit date that follows the algorithm when no
# variant is present.
RUN_ID_RE = re.compile(
    r"^(?P<instance>.+)_(?P<algo>LA|ROBU|RO|GREEDY|2SP|ORACLE)"
    r"(?:_(?P<variant>[A-Za-z][A-Za-z0-9_.-]*))?"
    r"_(?P<ts>\d{8}_\d{6})(?:_(?P<idx>\d+))?$"
)


def parse_run_id(run_id: str) -> dict | None:
    """Split a run_id into instance / algo / variant / ts / idx, or None."""
    m = RUN_ID_RE.match(run_id or "")
    if not m:
        return None
    d = m.groupdict()
    d["idx"] = int(d["idx"]) if d["idx"] else 0
    return d


def make_run_id(instance: str, algo: str, ts: str,
                idx: int | None = None, variant: str | None = None) -> str:
    """Build a run_id that ``parse_run_id`` round-trips."""
    parts = [instance, algo.upper()]
    if variant:
        parts.append(str(variant))
    parts.append(ts)
    if idx is not None:
        parts.append(f"{int(idx):03d}")
    return "_".join(parts)
