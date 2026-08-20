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


def redirect_outputs(root: str | os.PathLike) -> None:
    """Send every OUTPUT tree under ``root`` instead of the repo root.

    Inputs (instances/, data/) are deliberately untouched — a side experiment
    reads the same instances but must never scatter its runs among the
    manuscript's.  Used by ML/code/rollout.py so that learned-policy runs,
    logs and figures stay inside ML/ and can never contaminate the reporting
    pipeline, which globs solutions/ by method name.

    Rebinds the module-level constants AND the str accessors, so callers that
    did ``from src import paths`` and call ``paths.solutions(...)`` at run
    time pick the change up; call it before the first write.
    """
    global SOLUTIONS, LOGS, FIGURES, DATA_OUTPUT, _WRITABLE
    global solutions, logs, figures, data_output
    base = Path(root).resolve()
    SOLUTIONS   = base / "solutions"
    LOGS        = base / "logs"
    FIGURES     = base / "figures"
    DATA_OUTPUT = base / "data_output"
    _WRITABLE   = (SOLUTIONS, LOGS, FIGURES, DATA_OUTPUT)
    solutions   = _joiner(SOLUTIONS)
    logs        = _joiner(LOGS)
    figures     = _joiner(FIGURES)
    data_output = _joiner(DATA_OUTPUT)
    ensure_dirs()


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


# ══════════════════════════════════════════════════════════════════════════════
# THE STANDARD LA CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════
# Since 2026-08-18 the standard look-ahead policy solves its tail subproblem as
# a MILP.  The LP relaxation it used to run is now a VARIANT of it, the exact
# inverse of the earlier arrangement, and the corpus on disk still carries the
# old spelling: the MILP runs are tagged "MIPTAIL" in their run_id and the LP
# runs carry no tag at all.
#
# Rather than rewrite ~13 000 file names, the reporting layer maps a stored
# (variant, solve_mode) pair onto the EFFECTIVE variant every consumer keys on,
# so "no variant" means "the standard configuration" throughout, whichever
# spelling produced the run.  Everything downstream — the dedup key, the
# base-case figures and tables, the coverage and audit inventories, the LA
# sweep cells — then needs no knowledge of the swap.
#
# It lives here, next to the run-id vocabulary it belongs to, because the
# consumers span reporting modules that do not otherwise import one another.
LA_STD_VARIANT    = "MIPTAIL"   # historic tag of the runs that are now standard
LA_LEGACY_VARIANT = "LPTAIL"    # synthetic tag for the superseded LP tail


def effective_variant(method: str | None, variant: str | None,
                      solve_mode: str | None = None,
                      energy_q: float | None = None) -> str | None:
    """Stored (variant, solve_mode) -> the variant reporting should key on.

    For every method but LA this is the stored variant unchanged.  For LA:

      untagged or MIPTAIL, solved as a MIP   -> None    (the standard)
      untagged or MIPTAIL, solved as an LP   -> LPTAIL  (the superseded default)
      any other tag (S25H12, TB0, LOCAL, …)  -> unchanged

    ``solve_mode`` is what the run actually did and therefore outranks the tag,
    so a run launched after the default flipped needs no tag to be recognised as
    standard, and one launched with the wrong pair of flags is filed by what it
    did rather than by what it was called.  Where the mode is unknown — the rows
    compile_solutions reconstructs from the log of a run that never finished —
    the tag decides, and its absence reads as the old LP default, which is what
    the whole stored corpus is; such rows are unfinished and never reach an
    aggregate either way.
    """
    if (method or "").upper() != "LA":
        return variant or None
    tagged_std = variant in (None, "", LA_STD_VARIANT)
    mode = (solve_mode or "").lower()
    if mode == "mip":
        eff = None if tagged_std else variant
    elif mode == "lp":
        # A TAGGED cell keeps its tag and gains the solver: S25H12 solved as an
        # LP and S25H12 solved as a MIP are different configurations and must
        # not share a cell.  Without this the two pool together and the newer
        # one silently wins the dedup, which is how a completed LP ladder rung
        # turned into a half-MIP one.
        eff = (LA_LEGACY_VARIANT if tagged_std
               else f"{variant}+{LA_LEGACY_VARIANT}")
    elif variant == LA_STD_VARIANT:      # mode unknown: the tag decides
        eff = None
    else:
        eff = variant or LA_LEGACY_VARIANT
    # The committed-charge energy guard splits the cell only on the SUPERSEDED
    # LP side.  On the MILP side it is being adopted as part of the standard
    # configuration, so a guarded and an unguarded MILP run are two samples of
    # one cell rather than two cells — and where an instance has both, the
    # report keeps the guarded one (see _prefer_energy_guard).  Leaving the tag
    # off here is what merges them; the preference rule is what stops the pair
    # being averaged.
    if energy_q and (mode == "lp" or (eff or "").endswith(LA_LEGACY_VARIANT)):
        eff = f"{eff}+EQ{int(round(float(energy_q) * 100))}" if eff               else f"EQ{int(round(float(energy_q) * 100))}"
    return eff
