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
        └─ basecase/ LAconfig/ usecase/ sensitivity/ VSS/
                            those three trees are split one level deep by
                            EXPERIMENT; see "EXPERIMENT BUCKETS" below.  Writers
                            call solution_out()/log_out()/figure_out() and
                            readers call find_*()/glob_*()/scan_*(), so neither
                            side has to know the bucket of a given file.
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

# ══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT BUCKETS
# ══════════════════════════════════════════════════════════════════════════════
# solutions/, logs/ and figures/ are each split one level deep, by EXPERIMENT
# rather than by method or by date:
#
#     basecase/     the 3x3x4x25 base grid, standard method configurations
#     LAconfig/     the look-ahead configuration sweep (S..H.. ladders, LOCAL)
#     usecase/      the real corridor instances (usecase_*)
#     sensitivity/  the one-at-a-time axes (instances_sens, "<inst>__<tag>")
#     VSS/          the value-of-stochastic-solution experiment: every DET run
#                   (deterministic plan built at nominal travel times, executed
#                   as is) that measures what planning on averages costs
#
# Why one level and not more: the flat trees had grown to ~18 000 solutions and
# ~25 000 logs, which is slow to list and impossible to read.  Anything deeper
# would need the reader to know the bucket to find a file, and the reader
# usually does not — a run_id names the run, not the experiment.  With exactly
# one level, "search the buckets" is a two-glob operation (see _multi_glob), so
# every consumer can keep addressing artefacts by NAME alone.
#
# Files that classify into no bucket (ad-hoc runs, concept figures, the
# framework diagram) stay at the tree root.  That is deliberate: a file we
# cannot place is better left visible than filed under a guess.

BASECASE    = "basecase"
LACONFIG    = "LAconfig"
USECASE     = "usecase"
SENSITIVITY = "sensitivity"
VSS         = "VSS"
BUCKETS: tuple[str, ...] = (BASECASE, LACONFIG, USECASE, SENSITIVITY, VSS)

# Trees that are bucketed.  data_output/ and tex/tables/ are NOT: they hold a
# few dozen named exports whose names already say which experiment they belong
# to, and the LaTeX \input paths in the manuscript would all have to change.
_BUCKETED = ("SOLUTIONS", "LOGS", "FIGURES")

# Directories a run may write into.  Inputs are deliberately absent: a missing
# instances/ is a real error and must not be papered over by mkdir.  TEX_SECTIONS
# is absent on purpose — nothing generated may land next to the manuscript.
_WRITABLE = (SOLUTIONS, LOGS, FIGURES, TEX_TABLES, DATA_OUTPUT)


def ensure_dirs() -> None:
    """Create the output directories if they do not exist (idempotent)."""
    for d in _WRITABLE:
        d.mkdir(parents=True, exist_ok=True)
    for base in (SOLUTIONS, LOGS, FIGURES):
        for b in BUCKETS:
            (base / b).mkdir(parents=True, exist_ok=True)


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
    r"^(?P<instance>.+)_(?P<algo>LA|ROBU|RO|GREEDY|2SP|DET|ORACLE)"
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
    # The committed-charge energy guard USED to split the cell on the LP side:
    # the MILP arm had adopted the guard while the stored LP corpus had not, so
    # an "+EQ50" tag kept the two populations apart.  That split is retired
    # (2026-08-23).  The whole LP corpus was deleted and the sweep is being
    # re-run with the guard on in BOTH arms, so the tag now only ever appends a
    # suffix that every consumer would have to be taught about — and would not
    # be: the ladder figures look up "<TAG>+LPTAIL" exactly, so an LP cell
    # arriving as "S25H12+LPTAIL+EQ50" would silently vanish from every figure
    # rather than error.  `energy_q` is kept in the signature because callers
    # pass it positionally and because a future split may need it again.
    return eff


# ══════════════════════════════════════════════════════════════════════════════
# BUCKET ROUTING
# ══════════════════════════════════════════════════════════════════════════════
# Which of the four experiment buckets an artefact belongs to is derived from
# its NAME, never stored anywhere.  That is what makes the split reversible and
# self-healing: re-deriving the bucket of every file on disk reproduces the
# layout exactly, so a file dropped in the wrong place (or left in the root by
# an older script) is still found by the readers below and is re-filed the next
# time the migration runs.
#
# Precedence is instance first, configuration second.  A look-ahead sweep run
# on a sensitivity instance is a SENSITIVITY run: the axis is the experiment,
# the LA tag only says how that cell was solved.  Only on the base grid does an
# LA configuration tag mean "this run exists to compare configurations".

# Base-grid instance stems: R<route>C<customers>T<window>_<seed>, optionally
# carrying a "__<axis>" sensitivity suffix.
_BASE_INSTANCE_RE = re.compile(
    r"^R(?:short|medium|long)C(?:few|medium|many)T(?:tight|medium|large|none)_\d+$"
)

# Suffixes an artefact name may carry on top of its run_id / instance stem.
_ARTEFACT_EXTS     = (".json", ".txt", ".log", ".png", ".pdf", ".pptx", ".csv")
_ARTEFACT_SUFFIXES = ("_gurobi", "_scenarios")


def instance_bucket(instance: str | None) -> str | None:
    """Bucket implied by an INSTANCE stem alone, or None if unrecognised."""
    stem = os.path.basename(str(instance or "")).removesuffix(".json")
    if not stem:
        return None
    if stem.startswith("usecase"):
        return USECASE
    core, sep, _axis = stem.partition("__")
    if not _BASE_INSTANCE_RE.match(core):
        return None                       # ad-hoc / benchmark name: leave at root
    return SENSITIVITY if sep else BASECASE


def bucket_for(instance: str | None, algo: str | None = None,
               variant: str | None = None) -> str | None:
    """Bucket for a run, from its instance and its method configuration.

    ``variant`` is the raw stored tag, not ``effective_variant``'s output: the
    routing has to work at WRITE time, before a run has a solve_mode to key on.
    Both spellings of the standard look-ahead (untagged and MIPTAIL) are base
    case; every other LA tag is a configuration sweep.
    """
    b = instance_bucket(instance)
    if b is None:
        return None
    # DET exists only to serve the VSS experiment — it is not a configuration
    # of another method but a method of its own, so it takes the bucket
    # regardless of which instance it ran on.  This is the one place where
    # "instance first, configuration second" does not apply: the DET runs on a
    # sensitivity instance are still the VSS experiment, and splitting them
    # across two buckets would mean no single directory holds the experiment.
    if (algo or "").upper() == "DET":
        return VSS
    if (b == BASECASE and (algo or "").upper() == "LA"
            and variant not in (None, "", LA_STD_VARIANT)):
        return LACONFIG
    return b


def bucket_of_run_id(run_id: str | None) -> str | None:
    """Bucket for a run_id (``<inst>_<ALGO>[_<VAR>]_<ts>[_<idx>]``)."""
    d = parse_run_id(run_id or "")
    return bucket_for(d["instance"], d["algo"], d["variant"]) if d else None


def bucket_of_artefact(name: str) -> str | None:
    """Bucket for any solutions/ or logs/ FILE NAME.

    Handles every shape that lives in those trees::

        <run_id>.json / .txt          a run
        <run_id>_gurobi.log           its solver log
        <run_id>_scenarios.json       its realised scenarios
        oracle_<inst>.json            the shared per-instance oracle cache
        oracle_<inst>_gurobi.log      and its bound log
        oracle_trace_<inst>.log       a bound-trace re-solve

    The oracle cache is keyed by instance only — one oracle serves every method
    on that instance — so it is filed by the INSTANCE's bucket.  An LAconfig run
    therefore finds its oracle in basecase/, which is why every reader searches
    all buckets rather than assuming its own.
    """
    stem = os.path.basename(str(name or ""))
    for ext in _ARTEFACT_EXTS:
        if stem.endswith(ext):
            stem = stem[: -len(ext)]
            break
    if not stem:
        return None
    if stem.startswith("oracle_trace_"):
        return instance_bucket(stem[len("oracle_trace_"):])
    for sfx in _ARTEFACT_SUFFIXES:
        if stem.endswith(sfx):
            stem = stem[: -len(sfx)]
    if stem.startswith("oracle_"):
        return instance_bucket(stem[len("oracle_"):])
    return bucket_of_run_id(stem)


# Figure names are not run_ids — they are named after the SECTION they serve —
# so they need their own table.  Longest matching prefix wins, and a name that
# matches nothing falls through to bucket_of_artefact, which catches the per-run
# diagnostic PNGs (``<run_id>.png``).  Concept and framework figures match
# neither and stay at the figures/ root: they belong to no experiment.
_FIGURE_PREFIXES: tuple[tuple[str, str], ...] = (
    ("paper_",                 BASECASE),
    ("additional_la_",         LACONFIG),
    ("additional_sens_",       SENSITIVITY),
    ("additional_diesel_",     SENSITIVITY),
    ("additional_grid_",       SENSITIVITY),
    ("diesel_vs_ev_",          SENSITIVITY),
    ("check_diesel_timeline_", SENSITIVITY),
    ("check_power_timeline_",  SENSITIVITY),
    ("real_route_",            USECASE),
)


def bucket_of_figure(name: str) -> str | None:
    """Bucket for a figure FILE NAME (or bare stem)."""
    stem = os.path.basename(str(name or ""))
    hit = max((p for p, _b in _FIGURE_PREFIXES if stem.startswith(p)),
              key=len, default=None)
    if hit is not None:
        return dict(_FIGURE_PREFIXES)[hit]
    return bucket_of_artefact(stem)


# ── write-side resolvers ─────────────────────────────────────────────────────
# Every writer goes through these instead of paths.solutions(name), so that the
# bucket is decided in exactly one place.  They create the directory, so no
# caller has to.

def _out(base: Path, name: str, bucket: str | None) -> str:
    d = base / bucket if bucket in BUCKETS else base
    d.mkdir(parents=True, exist_ok=True)
    return str(d / os.path.basename(name))


def solution_out(name: str, bucket: str | None = None) -> str:
    """Path to WRITE solutions/<bucket>/<name>."""
    return _out(SOLUTIONS, name, bucket or bucket_of_artefact(name))


def log_out(name: str, bucket: str | None = None) -> str:
    """Path to WRITE logs/<bucket>/<name>."""
    return _out(LOGS, name, bucket or bucket_of_artefact(name))


def figure_out(name: str, bucket: str | None = None) -> str:
    """Path to WRITE figures/<bucket>/<name>."""
    return _out(FIGURES, name, bucket or bucket_of_figure(name))


# ── read-side resolvers ──────────────────────────────────────────────────────
# Readers address artefacts by NAME and must not care where they sit.  The tree
# root is searched too, so an unbucketed leftover is still found.

def _search_dirs(base: Path) -> list[str]:
    return [str(base)] + [str(base / b) for b in BUCKETS]


def _find(base: Path, name: str) -> str | None:
    name = os.path.basename(str(name))
    # Try the derived bucket first, so the common case is a single stat call.
    guess = bucket_of_figure(name) if base == FIGURES else bucket_of_artefact(name)
    order = ([str(base / guess)] if guess in BUCKETS else []) + _search_dirs(base)
    seen: set[str] = set()
    for d in order:
        if d in seen:
            continue
        seen.add(d)
        p = os.path.join(d, name)
        if os.path.isfile(p):
            return p
    return None


def _multi_glob(base: Path, pattern: str) -> list[str]:
    """Glob `pattern` at the tree root AND inside every bucket."""
    import glob as _glob
    out: list[str] = []
    for d in _search_dirs(base):
        out.extend(_glob.glob(os.path.join(d, pattern)))
    return sorted(out)


def _scan_tree(base: Path) -> list[tuple[str, str]]:
    """-> [(basename, full path)] over the tree root and every bucket.

    Where a name exists in more than one place the BUCKETED copy wins; the root
    is only a fallback for artefacts that were never filed.
    """
    found: dict[str, str] = {}
    for d in _search_dirs(base):          # root first, buckets overwrite it
        try:
            with os.scandir(d) as it:
                for e in it:
                    if e.is_file():
                        found[e.name] = e.path
        except OSError:
            continue
    return sorted(found.items())


def find_solution(name: str) -> str | None:
    """Existing solutions/**/<name>, or None."""
    return _find(SOLUTIONS, name)


def find_log(name: str) -> str | None:
    """Existing logs/**/<name>, or None."""
    return _find(LOGS, name)


def solution_path(name: str) -> str:
    """Where <name> IS, or — if it does not exist yet — where it would go."""
    return find_solution(name) or solution_out(name)


def log_path(name: str) -> str:
    """Where <name> IS, or — if it does not exist yet — where it would go."""
    return find_log(name) or log_out(name)


def glob_solutions(pattern: str) -> list[str]:
    """Glob a pattern across solutions/ and all of its buckets."""
    return _multi_glob(SOLUTIONS, pattern)


def glob_logs(pattern: str) -> list[str]:
    """Glob a pattern across logs/ and all of its buckets."""
    return _multi_glob(LOGS, pattern)


def expand_logs(spec: str) -> list[str]:
    """Resolve a comma-separated --glob spec against the bucketed logs/ tree.

    A pattern with no directory part ("oracle_*_gurobi.log") is a NAME pattern
    and is searched across logs/ and all of its buckets.  One that carries a
    directory ("logs/basecase/oracle_R*.log", an absolute path) is a location
    the caller chose and is used verbatim.  That split is what lets the CLI
    defaults stay readable while a user can still point the tool anywhere.
    """
    import glob as _glob
    out: list[str] = []
    for part in str(spec or "").split(","):
        pat = part.strip()
        if not pat:
            continue
        out.extend(sorted(_glob.glob(pat)) if os.path.dirname(pat)
                   else glob_logs(pat))
    return out


def glob_figures(pattern: str) -> list[str]:
    """Glob a pattern across figures/ and all of its buckets."""
    return _multi_glob(FIGURES, pattern)


def scan_solutions() -> list[tuple[str, str]]:
    """[(basename, path)] for every file under solutions/ and its buckets."""
    return _scan_tree(SOLUTIONS)


def scan_logs() -> list[tuple[str, str]]:
    """[(basename, path)] for every file under logs/ and its buckets."""
    return _scan_tree(LOGS)


def find_in(directory: str | os.PathLike, name: str) -> str | None:
    """find_solution for an arbitrary bucketed tree — see in_tree."""
    return _find(Path(directory), name)


def out_in(directory: str | os.PathLike, name: str,
           bucket: str | None = None) -> str:
    """solution_out for an arbitrary bucketed tree — see in_tree."""
    return _out(Path(directory), name, bucket or bucket_of_artefact(name))


def path_in(directory: str | os.PathLike, name: str) -> str:
    """Where <name> IS in that tree, or where it would go — see in_tree."""
    return find_in(directory, name) or out_in(directory, name)


def in_tree(directory: str | os.PathLike, pattern: str) -> list[str]:
    """Glob inside an arbitrary solutions-shaped directory, buckets included.

    Several tools take a --dir so they can be pointed at ML/solutions or at an
    archived copy.  Those trees are bucketed the same way, so the search has to
    be too; this is the directory-argument form of glob_solutions.
    """
    return _multi_glob(Path(directory), pattern)


def scan_tree(directory: str | os.PathLike) -> list[tuple[str, str]]:
    """[(basename, path)] over an arbitrary bucketed tree — see in_tree."""
    return _scan_tree(Path(directory))
