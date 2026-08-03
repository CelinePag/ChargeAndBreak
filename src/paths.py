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
