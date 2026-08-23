"""
additional_analysis.py — single entry point for the paper's additional analyses.

Every experiment block of Section 8 beyond the base-case comparison (8.2) is
launched from here.  Each subcommand maps to one paper section:

  Subcommand     Paper section                       What it produces
  ─────────────  ──────────────────────────────────  ─────────────────────────────
  sensitivity    8.3 Sensitivity analysis            one-at-a-time sweeps off the
                                                     base case (CS spacing, charger
                                                     power, beta, travel CV, AR(1)
                                                     rho, no-split break, battery
                                                     capacity)
  grid           8.3 Sensitivity analysis            two axes crossed instead of
                                                     one-at-a-time; default is
                                                     battery x charger power,
                                                     which one-at-a-time cannot
                                                     separate (the pack moves
                                                     both range and the taper)
  diesel         8.4 VS diesel trucks                same instances re-run with
                                                     --diesel (HoS only, no
                                                     charging) for the EV-vs-diesel
                                                     makespan penalty
  vss            8.5 Effect of uncertainty           VSS / EVPI decomposition via
                                                     experiments/vss_evpi.py
                                                     (EEV / RP / WS, common random
                                                     numbers)
  la             8.3 LA configuration sensitivity    look-ahead horizon and
                                                     scenario-count sweep, run on
                                                     the BASE instances under a
                                                     --variant label
  la-report      8.3 LA configuration sensitivity    data_output/additional_la_
                                                     stats.csv from those runs
  compile        8.3-8.5 tables                      re-run compile_solutions on
                                                     solutions/ (variant runs are
                                                     tagged by instance-stem suffix)

Design decisions baked in (agreed 2026-07-20 / 2026-07-29):
  * Long routes are EXCLUDED by default: with 50 seeds the full protocol is
    intractable there and only Greedy (+ partially LA) survives.  The default
    combo set is the short/medium representative grid; pass --combos to widen.
  * Sensitivity sweeps are one-at-a-time off the base case, on a reduced seed
    set (default 10) and a reduced method set (greedy,LA,2SP,ORACLE) — the
    sweeps chart the problem's response surface, the base case does the
    method ranking.
  * Variant instances are materialised as SEPARATE json files whose stem
    carries a double-underscore tag (e.g. RshortCfewTlarge_1__cs30.json) so
    that solution files, oracle caches, and the latest-run-per-(instance,
    method) dedup in compile_solutions never collide with base runs.
    This applies to axes that change the INSTANCE (cs_spacing, charger_power,
    cv, ar1_rho, beta, no_split) — a different instance deserves a different
    file, and a different oracle.
  * An axis that changes only a METHOD's own configuration (the `la` block)
    does NOT copy instances.  It runs on the base instances under a
    --variant label carried in the run_id, which is what the compile dedup
    keys on.  Copying would be doubly wrong there: the instance is unchanged,
    and a copy would orphan the run from the instance's already-solved oracle
    cache, forcing hours of needless re-solving to recover the gap.

Everything is launched through runner_dispatch.py subprocesses, so all solver
flags/logs behave exactly as in the base experiments.  Use --dry-run to print
the commands without running them (handy while the base runs still occupy the
machine).

Examples
--------
  # print what the diesel block would run (nothing executed)
  python -m src.output_analysis.additional_analysis diesel --dry-run

  # 8.4: diesel counterpart, greedy + hindsight oracle, 10 seeds
  python -m src.output_analysis.additional_analysis diesel --seeds 1-10 --jobs 4

  # 8.3: CS-spacing sweep at 30 and 90 km (60 km = base case, already run)
  python -m src.output_analysis.additional_analysis sensitivity --axis cs_spacing --values 30,90

  # 8.3: charger-power sweep incl. MCS
  python -m src.output_analysis.additional_analysis sensitivity --axis charger_power --values 150,350,1000

  # 8.3: no-split-break regime (Art. 7 15'+30' split forbidden; no --values)
  python -m src.output_analysis.additional_analysis sensitivity --axis no_split

  # 8.3: pack x charge point.  ALWAYS --dry-run first — a grid multiplies out
  # fast (3 x 3 cells x 160 instances = 1440 runs per algorithm).
  python -m src.output_analysis.additional_analysis grid --dry-run
  python -m src.output_analysis.additional_analysis grid \
      --x-values 500,600,800 --y-values 350,700 --seeds 1-5 --jobs 4

  # 8.5: VSS / EVPI on the short/medium grid, 20 scenarios
  python -m src.output_analysis.additional_analysis vss --n-scenarios 20

  # 8.3: LA look-ahead / scenario-count sweep (base instances, labelled runs)
  python -m src.output_analysis.additional_analysis la --configs S25H12,S25H14,S10H24,S50H24
  python -m src.output_analysis.additional_analysis la-report

  # refresh tables (Excel + LaTeX) after any block
  python -m src.output_analysis.additional_analysis compile --tex-dir tex/tables
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import shutil
import subprocess
import sys
from src import paths as _paths
from src.settings import BATTERY_CAPACITY, CHARGER_POWER_BASE_KW

# ══════════════════════════════════════════════════════════════════════════════
# DEFAULTS
# ══════════════════════════════════════════════════════════════════════════════

# Representative combos for the additional analyses.  Long routes are excluded
# by default (see module docstring); the base-case table still covers them.
DEFAULT_COMBOS = ["RshortCfew", "RshortCmany", "RmediumCfew", "RmediumCmany"]

DEFAULT_TW      = ["tight", "medium", "large", "none"]
DEFAULT_SEEDS   = "1-10"          # sensitivity does not need the full 50
DEFAULT_ALGOS   = "greedy,LA,2SP,ORACLE"   # reduced method set for sweeps
# Greedy baseline guard (2026-07-29): the base case runs greedy with a 0.95
# departure quantile ("a driver keeps a few minutes in hand"), so every greedy
# run launched here must match or the sweep deltas mix a guard change with the
# axis change.
DEFAULT_GUARD   = 0.95
DIESEL_ALGOS    = "greedy,ORACLE"          # 8.4 needs the hindsight optimum
                                           # + the practice baseline

SENS_DIR    = _paths.instances_sens()    # variant instance files live here
VSS_DIR     = _paths.results_vss()       # per-instance VSS/EVPI json results

# Sensitivity axes: how each one is materialised.
#   regen  — re-generate the instance via instance_io.generate_instance_file
#            (same (route,cust,tw,seed) => same geometry seed, so corridor
#            length / customer draws stay paired with the base instance)
#   patch  — copy the base json and override a data field (identical geometry
#            AND identical realisation; the cleanest paired comparison)
#   stub   — needs model plumbing first; refuses with a pointer
_AXES = {
    "cs_spacing":    dict(kind="regen", kw="cs_spacing_km",
                          default_values=[30, 90], tag="cs"),
    # `base` = the value the UNLABELLED base-case runs already carry.  Only the
    # axes the grid can cross need it, so that a cell landing on the base case
    # is skipped rather than regenerated under a variant tag (which would
    # duplicate the base case under a second name and split its dedup group).
    "charger_power": dict(kind="regen", kw="charger_power_kw",
                          default_values=[150, 350, 1000], tag="kw",
                          base=CHARGER_POWER_BASE_KW),
    "cv":            dict(kind="regen", kw="cv",
                          default_values=[0.10, 0.25], tag="cv"),
    "ar1_rho":       dict(kind="regen", kw="ar1_rho",
                          default_values=[0.4], tag="rho"),
    "beta":          dict(kind="patch", kw="beta",
                          default_values=[2.0, 5.0], tag="beta"),
    # I3.  Regen (not patch): the pack needs five coupled fields (E0, Ecap,
    # Emin, Ebar, Tbar) and make_data already derives all of them from Bcap.
    # Capacity touches neither the geometry nor the realisation — Q is drawn
    # before Bcap is applied — so a regen at the same seed stays exactly paired
    # with the base instance, as on the charger_power axis.
    #
    # READ THE RESULTS WITH CARE: two things move with the pack, not one.
    # Emin = SOC_MIN_FRAC·Ecap, so +100 kWh of pack is only +80 kWh of usable
    # energy; and the tail acceptance is TAIL_C_RATE·Ecap, so a bigger pack
    # ALSO pushes back where the charge curve tapers (at 350 kW the taper only
    # binds below 875 kWh).  A one-at-a-time sweep at fixed charger power
    # therefore measures range and taper avoidance together; cross it with
    # charger_power if the two need separating.
    "battery":       dict(kind="regen", kw="battery_kwh",
                          default_values=[400, 600, 800], tag="kwh",
                          base=BATTERY_CAPACITY),
    # Art. 7 second subparagraph only PERMITS the 15'+30' split, so a carrier
    # may forbid it.  allow_split=False drops x_b15/x_b30 from MILP/2SP and
    # takes b15/b30 out of the greedy, LA and supervisor rules.  Patch (not
    # regen): geometry AND realisation stay identical to the base instance,
    # and generation itself is unaffected because the nominal greedy pass that
    # centres the time windows never takes a split break anyway.
    "no_split":      dict(kind="patch", kw="allow_split", cast=bool,
                          default_values=[0], tag="nosplit"),
}

_ROUTE_FROM_TAG = {"Rshort": "short", "Rmedium": "medium", "Rlong": "long"}
_CUST_FROM_TAG  = {"Cfew": "few", "Cmedium": "medium", "Cmany": "many"}


# ══════════════════════════════════════════════════════════════════════════════
# SMALL HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _expand_seeds(spec: str) -> list[int]:
    """Parse '1-10' or '1,2,7' (or a mix: '1-3,7') into a sorted int list."""
    out: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(part))
    return sorted(out)


def _split_combo(combo: str) -> tuple[str, str]:
    """'RshortCfew' -> ('short', 'few')."""
    for rtag, route in _ROUTE_FROM_TAG.items():
        if combo.startswith(rtag):
            ctag = combo[len(rtag):]
            if ctag in _CUST_FROM_TAG:
                return route, _CUST_FROM_TAG[ctag]
    raise SystemExit(f"unrecognised combo '{combo}' "
                     f"(expected e.g. RshortCfew, RmediumCmany)")


def _base_instance_path(route: str, cust: str, tw: str, seed: int) -> str:
    from src.instance_gen.instance_io import instance_filename
    return _paths.instances(instance_filename(route, cust, tw, seed))


def _tagged(path_stem: str, tag: str) -> str:
    """RshortCfewTlarge_1 + 'cs30' -> RshortCfewTlarge_1__cs30 (double
    underscore = variant marker, keeps base and variant runs apart in
    solutions/, oracle caches, and the compile dedup)."""
    return f"{path_stem}__{tag}"


def _run(cmd: list[str], dry: bool) -> None:
    print(("DRY-RUN  " if dry else "RUN      ") + " ".join(cmd))
    if not dry:
        subprocess.run(cmd, check=False)


def _dispatch(pattern: str, algos: str, jobs: int, dry: bool,
              extra: list[str] | None = None,
              guard: float | None = None,
              n_scenarios: int | None = None,
              horizon: float | None = None,
              la_energy_quantile: float | None = None,
              resume: bool = True) -> None:
    """Launch runner_dispatch for one instance pattern.

    ``resume`` is ON by default.  An LA cell is the most expensive thing these
    sweeps launch, and without a checkpoint a crash at stop 140 of 150 throws
    away the whole instance; with one it resumes from the last completed stop.
    It is safe to leave on: the checkpoint key covers every LA parameter, a
    corrupt or unreadable checkpoint is caught and the run starts fresh, and a
    clean finish deletes it — so a normal rerun never resumes stale state.
    It composes with --skip-existing rather than fighting it: a FINISHED run is
    skipped outright (its checkpoint is already gone), while a CRASHED one has
    no finished solution to skip, so it is relaunched and picks the checkpoint
    up.

    ``guard`` is the greedy departure quantile (--prune_quantile).  Because
    that flag is SHARED — it also drives LA's action pruning and the opt-in
    supervisor — it must not leak onto the other algorithms, whose base runs
    were made without it.  So when greedy is requested alongside others, the
    batch is split: greedy runs guarded, the rest run untouched.

    ``n_scenarios`` / ``horizon`` must match the BASE case whenever LA or 2SP
    is in the sweep.  Left unset they fall back to runner_dispatch's own
    defaults (10 scenarios, 12 h), which silently differ from a base case run
    with other values — the sweep delta would then blend the axis change with
    a scenario-count/horizon change.  Both are forwarded unconditionally:
    greedy, RO and ORACLE accept and ignore them.

    ``la_energy_quantile`` is the same kind of trap and is left unset by
    default, matching settings.LA_ENERGY_QUANTILE: it sizes LA's committed
    charge to cover the legs to the next CS at a quantile of CONSUMPTION rather
    than at nominal, so a sweep run without it is a different policy from a
    base run made with it.  Forwarded only to a batch containing LA, where it
    has an effect — the flag is LA-only downstream.
    """
    def _go(alg_spec: str, guarded: bool) -> None:
        cmd = [sys.executable, "-m", "src.simulation.runner_dispatch", pattern, alg_spec,
               "--jobs", str(jobs), "--skip-existing"]
        # --resume is LA-only downstream (it reaches run_simulation_precomputed
        # and nothing else), so it is forwarded only to a batch that contains
        # LA rather than sprayed onto greedy/2SP/ORACLE where it would be dead
        # weight in the command line and in the logs.
        if resume and any(a.strip().upper() == "LA"
                          for a in alg_spec.split(",")):
            cmd += ["--resume"]
        if guarded and guard is not None:
            cmd += ["--prune_quantile", str(guard)]
        if n_scenarios is not None:
            cmd += ["--n_scenarios", str(int(n_scenarios))]
        if horizon is not None:
            cmd += ["--horizon", str(float(horizon))]
        if la_energy_quantile is not None and any(
                a.strip().upper() == "LA" for a in alg_spec.split(",")):
            cmd += ["--la_energy_quantile", str(float(la_energy_quantile))]
        if extra:
            cmd += extra
        _run(cmd, dry)

    algo_list = [a.strip() for a in algos.split(",") if a.strip()]
    greedy    = [a for a in algo_list if a.lower() == "greedy"]
    others    = [a for a in algo_list if a.lower() != "greedy"]

    if greedy and guard is not None:
        _go(",".join(greedy), True)
        if others:
            _go(",".join(others), False)
    else:
        _go(algos, False)


# ══════════════════════════════════════════════════════════════════════════════
# VARIANT INSTANCE MATERIALISATION
# ══════════════════════════════════════════════════════════════════════════════


def _retitle(path: str, tagged_stem: str) -> None:
    """Stamp the tagged stem into the instance's own ``title``.

    CRITICAL: every run records ``instance = full_data["title"]``, and BOTH the
    compile dedup (keyed on instance+method) and the oracle cache file name
    (oracle_<title>.json) derive from it.  A variant that keeps the BASE title
    therefore (a) displaces the base run in every table and figure and (b)
    would overwrite the base oracle cache with a variant result.  Diesel is
    exempt: _apply_diesel_mode appends its own "_diesel" suffix at run time.
    """
    if tagged_stem.endswith("__diesel"):
        return
    with open(path, "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    inst = payload.get("instance", {})
    old  = inst.get("title", "")
    inst["title"] = tagged_stem
    if isinstance(inst.get("label"), str) and old:
        inst["label"] = inst["label"].replace(old, tagged_stem, 1)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


def _materialise_regen(axis: str, value, combos, tws, seeds,
                       out_dir: str, dry: bool) -> str:
    """Re-generate variant instances with one axis changed; returns glob."""
    from src.instance_gen.instance_io import generate_instance_file
    spec = _AXES[axis]
    tag  = f"{spec['tag']}{value:g}" if isinstance(value, float) \
           else f"{spec['tag']}{value}"
    # Not under --dry-run: a dry run must not touch the filesystem, and
    # creating the dir up front left empty instances_sens/<axis>_<v>/ trees
    # behind that read as "this sweep was started" when it never was.
    if not dry:
        os.makedirs(out_dir, exist_ok=True)
    for combo in combos:
        route, cust = _split_combo(combo)
        for tw in tws:
            for seed in seeds:
                stem   = os.path.splitext(os.path.basename(
                             _base_instance_path(route, cust, tw, seed)))[0]
                target = os.path.join(out_dir, _tagged(stem, tag) + ".json")
                if os.path.isfile(target):
                    continue
                if dry:
                    print(f"DRY-RUN  generate {target}")
                    continue
                path = generate_instance_file(
                    route, cust, tw, seed,
                    output_dir=out_dir, verbose=False,
                    **{spec["kw"]: value})
                os.replace(path, target)
                _retitle(target, _tagged(stem, tag))
                print(f"generated {target}")
    return os.path.join(out_dir, f"*__{tag}.json")


def _axis_tag(spec: dict, value) -> str:
    """'kwh' + 600 -> 'kwh600'.  ``:g`` keeps 600.0 as 600 and 0.1 as 0.1, so a
    value that arrives as int or float tags identically — the tag is part of the
    instance stem, and two spellings of one cell would split its dedup group."""
    return f"{spec['tag']}{value:g}"


def _materialise_grid(active: list, combos, tws, seeds,
                      out_dir: str, dry: bool) -> str:
    """Re-generate variant instances with N axes changed at once.

    ``active`` is a list of (axis_spec, value) pairs — the axes that actually
    DIFFER from the base case in this cell.  A cell sitting on a base row or
    column therefore arrives here with a single pair and produces exactly the
    stem and kwargs the one-at-a-time sweep would have produced, which is what
    lets the grid reuse those runs instead of recomputing them under a second
    name (see cmd_grid).

    Same contract as _materialise_regen — one geometry seed per
    (route,cust,tw,seed), so every cell stays paired with the base instance AND
    with every other cell.  Only regen axes qualify: a patch axis overrides a
    stored field, and two patches would have to be composed by hand, which is a
    different (and so far unneeded) code path.
    """
    from src.instance_gen.instance_io import generate_instance_file
    tag    = "_".join(_axis_tag(s, v) for s, v in active)
    kwargs = {s["kw"]: v for s, v in active}
    # Not under --dry-run: a dry run must not touch the filesystem, and
    # creating the dir up front left empty instances_sens/<axis>_<v>/ trees
    # behind that read as "this sweep was started" when it never was.
    if not dry:
        os.makedirs(out_dir, exist_ok=True)
    for combo in combos:
        route, cust = _split_combo(combo)
        for tw in tws:
            for seed in seeds:
                stem   = os.path.splitext(os.path.basename(
                             _base_instance_path(route, cust, tw, seed)))[0]
                target = os.path.join(out_dir, _tagged(stem, tag) + ".json")
                if os.path.isfile(target):
                    continue
                if dry:
                    print(f"DRY-RUN  generate {target}")
                    continue
                path = generate_instance_file(
                    route, cust, tw, seed,
                    output_dir=out_dir, verbose=False, **kwargs)
                os.replace(path, target)
                _retitle(target, _tagged(stem, tag))
                print(f"generated {target}")
    return os.path.join(out_dir, f"*__{tag}.json")


def _patch_tag(spec: dict, value) -> str:
    """Variant tag for one patched value.  Boolean axes are a single regime
    rather than a sweep, so they carry the bare tag (…__nosplit, not
    …__nosplit0)."""
    if spec.get("cast") is bool:
        return spec["tag"]
    return f"{spec['tag']}{value:g}"


def _materialise_patch(axis: str, value, combos, tws, seeds,
                       out_dir: str, dry: bool) -> str:
    """Copy base instances and override one data field (identical geometry
    and realisation).  Used for beta (out-of-window penalty) and no_split
    (Art. 7 split break available or not)."""
    spec = _AXES[axis]
    cast = spec.get("cast", float)
    val  = cast(value)
    tag  = _patch_tag(spec, value)
    # Not under --dry-run: a dry run must not touch the filesystem, and
    # creating the dir up front left empty instances_sens/<axis>_<v>/ trees
    # behind that read as "this sweep was started" when it never was.
    if not dry:
        os.makedirs(out_dir, exist_ok=True)
    for combo in combos:
        route, cust = _split_combo(combo)
        for tw in tws:
            for seed in seeds:
                src  = _base_instance_path(route, cust, tw, seed)
                stem = os.path.splitext(os.path.basename(src))[0]
                target = os.path.join(out_dir, _tagged(stem, tag) + ".json")
                if os.path.isfile(target):
                    continue
                if dry:
                    print(f"DRY-RUN  patch {src} [{spec['kw']}={val}] "
                          f"-> {target}")
                    continue
                with open(src, "r", encoding="utf-8") as fh:
                    payload = json.load(fh)
                payload["instance"][spec["kw"]] = val
                payload["meta"][f"variant_{axis}"] = val
                with open(target, "w", encoding="utf-8") as fh:
                    json.dump(payload, fh, indent=2)
                _retitle(target, _tagged(stem, tag))
                print(f"patched   {target}")
    return os.path.join(out_dir, f"*__{tag}.json")


def _materialise_copy(tag: str, combos, tws, seeds,
                      out_dir: str, dry: bool) -> str:
    """Verbatim copies of base instances under a tagged stem — used when the
    variant differs only by SOLVER flags (diesel mode, ROBU gamma), so the
    tagged stem is what keeps runs/caches separate."""
    # Not under --dry-run: a dry run must not touch the filesystem, and
    # creating the dir up front left empty instances_sens/<axis>_<v>/ trees
    # behind that read as "this sweep was started" when it never was.
    if not dry:
        os.makedirs(out_dir, exist_ok=True)
    for combo in combos:
        route, cust = _split_combo(combo)
        for tw in tws:
            for seed in seeds:
                src  = _base_instance_path(route, cust, tw, seed)
                stem = os.path.splitext(os.path.basename(src))[0]
                target = os.path.join(out_dir, _tagged(stem, tag) + ".json")
                if os.path.isfile(target):
                    continue
                if dry:
                    print(f"DRY-RUN  copy {src} -> {target}")
                    continue
                shutil.copyfile(src, target)
                _retitle(target, _tagged(stem, tag))
    return os.path.join(out_dir, f"*__{tag}.json")


# ══════════════════════════════════════════════════════════════════════════════
# SUBCOMMANDS
# ══════════════════════════════════════════════════════════════════════════════

def cmd_sensitivity(args) -> None:
    """Section 8.3 — one-at-a-time sensitivity sweeps off the base case."""
    spec = _AXES.get(args.axis)
    if spec is None:
        raise SystemExit(f"unknown axis '{args.axis}' "
                         f"(choose from {', '.join(_AXES)})")
    if spec["kind"] == "stub":
        raise SystemExit(f"axis '{args.axis}' not runnable yet: {spec['msg']}")

    values = ([float(v) for v in args.values.split(",")] if args.values
              else spec["default_values"])
    # A boolean axis has one variant (the flag OFF) and a tag with no value
    # suffix, so a non-zero --values would write __nosplit instances that are
    # really the base case.  Refuse rather than mislabel the batch.
    if spec.get("cast") is bool and any(float(v) != 0 for v in values):
        raise SystemExit(f"axis '{args.axis}' is a single regime, not a sweep "
                         f"— drop --values (the variant is the flag turned "
                         f"OFF, i.e. 0)")
    seeds  = _expand_seeds(args.seeds)
    combos = args.combos.split(",")
    tws    = args.tw.split(",")

    for value in values:
        v = int(value) if float(value).is_integer() and args.axis != "cv" \
            else value
        # A boolean axis is one alternative regime, not a sweep — no suffix.
        sub = (args.axis if spec.get("cast") is bool else f"{args.axis}_{v}")
        out_dir = os.path.join(SENS_DIR, sub)
        if spec["kind"] == "regen":
            pattern = _materialise_regen(args.axis, v, combos, tws, seeds,
                                         out_dir, args.dry_run)
        else:
            pattern = _materialise_patch(args.axis, v, combos, tws, seeds,
                                         out_dir, args.dry_run)
        _dispatch(pattern, args.algorithms, args.jobs, args.dry_run,
                  guard=args.prune_quantile, resume=args.resume,
                  n_scenarios=args.n_scenarios, horizon=args.horizon,
                  la_energy_quantile=args.la_energy_quantile)


def _grid_values(spec: dict, raw: str | None) -> list:
    """Parse --x-values / --y-values, keeping integral values as ints so the
    tag reads kwh600, not kwh600.0."""
    vals = ([float(v) for v in raw.split(",")] if raw else
            [float(v) for v in spec["default_values"]])
    return [int(v) if float(v).is_integer() else v for v in vals]


def cmd_grid(args) -> None:
    """Section 8.3 — a 2-D grid over two regen axes (default: battery capacity
    x charger power).

    Why this exists separately from `sensitivity`: one-at-a-time cannot
    separate range from charge speed, because the two are coupled in the model.
    Emin = SOC_MIN_FRAC.Ecap and the tail acceptance is TAIL_C_RATE.Ecap, so
    resizing the pack ALSO moves where the charge curve tapers; sweeping the
    pack at a fixed charge point therefore measures range and taper avoidance
    together.  Only the grid separates them.
    """
    axes = tuple(a.strip() for a in args.axes.split(","))
    if len(axes) != 2:
        raise SystemExit(f"--axes needs exactly two axis names, got '{args.axes}'")
    if axes[0] == axes[1]:
        raise SystemExit(f"--axes needs two DIFFERENT axes, got '{axes[0]}' twice")
    for a in axes:
        spec = _AXES.get(a)
        if spec is None:
            raise SystemExit(f"unknown axis '{a}' (choose from {', '.join(_AXES)})")
        if spec["kind"] != "regen":
            raise SystemExit(
                f"axis '{a}' is kind='{spec['kind']}'; the grid only crosses "
                f"regen axes (a patch axis overrides a stored field and would "
                f"have to be composed by hand)")

    sx, sy = (_AXES[a] for a in axes)
    xs = _grid_values(sx, args.x_values)
    ys = _grid_values(sy, args.y_values)
    seeds  = _expand_seeds(args.seeds)
    combos = args.combos.split(",")
    tws    = args.tw.split(",")

    # A grid CONTAINS the one-at-a-time sweeps: the row at the base charger
    # power is the battery sweep, the column at the base pack is the charger
    # sweep, and the crossing cell is the base case itself.  Those cells must
    # therefore carry the SAME identity as the single-axis runs — same stem,
    # same directory — or the grid would recompute already-finished work under
    # a second name and split its dedup group.  Giving them a one-element
    # `active` list does exactly that: the tag and the kwargs collapse to what
    # `sensitivity --axis <that axis>` produces, so the existing instance files
    # are found on disk and runner_dispatch's --skip-existing finds the runs.
    def _active(vx, vy) -> list:
        return [(s, v) for s, v in ((sx, vx), (sy, vy))
                if s.get("base") is None or float(v) != float(s["base"])]

    per_cell = len(combos) * len(tws) * len(seeds)
    cells    = [(vx, vy) for vx in xs for vy in ys]
    n_base   = sum(1 for c in cells if not _active(*c))
    n_edge   = sum(1 for c in cells if len(_active(*c)) == 1)
    print(f"GRID  {axes[0]} x {axes[1]}  =  {len(xs)} x {len(ys)} = {len(cells)} cells"
          f"  x {per_cell} instances/cell  ({args.algorithms} each)")
    if n_base:
        print(f"      {n_base} base cell skipped (its unlabelled runs already exist)")
    if n_edge:
        print(f"      {n_edge} cells lie on a base row/column = the one-at-a-time "
              f"sweeps; they reuse those stems, so anything already run is skipped")
    print(f"      {len(cells) - n_base - n_edge} interaction cells are new to "
          f"this grid ({(len(cells) - n_base - n_edge) * per_cell} instances)")

    for vx, vy in cells:
        active = _active(vx, vy)
        if not active:                      # the base case itself
            continue
        if len(active) == 1:                # on a base row/column: one-at-a-time
            spec, val = active[0]
            axis = axes[0] if spec is sx else axes[1]
            out_dir = os.path.join(SENS_DIR, f"{axis}_{val}")
        else:
            out_dir = os.path.join(
                SENS_DIR, "grid_" + "_".join(_axis_tag(s, v) for s, v in active))
        pattern = _materialise_grid(active, combos, tws, seeds,
                                    out_dir, args.dry_run)
        _dispatch(pattern, args.algorithms, args.jobs, args.dry_run,
                  guard=args.prune_quantile, resume=args.resume,
                  n_scenarios=args.n_scenarios, horizon=args.horizon,
                  la_energy_quantile=args.la_energy_quantile)


def cmd_diesel(args) -> None:
    """Section 8.4 — diesel counterpart (HoS only, no charging) on the same
    instances; tagged copies keep diesel runs/caches apart from EV runs."""
    seeds   = _expand_seeds(args.seeds)
    combos  = args.combos.split(",")
    tws     = args.tw.split(",")
    out_dir = os.path.join(SENS_DIR, "diesel")
    pattern = _materialise_copy("diesel", combos, tws, seeds,
                                out_dir, args.dry_run)
    _dispatch(pattern, args.algorithms, args.jobs, args.dry_run,
              extra=["--diesel"], guard=args.prune_quantile,
              resume=args.resume,
              n_scenarios=args.n_scenarios, horizon=args.horizon,
              la_energy_quantile=args.la_energy_quantile)


# The ROBU price-of-robustness (Gamma) frontier was removed in August 2026 —
# it is not part of the paper (see the header comment in
# tex/sections/results_section.tex).  ROBU itself remains available as a method
# through runner_dispatch; only the budget sweep and its figure are gone.


# ══════════════════════════════════════════════════════════════════════════════
# 8.3 — LA CONFIGURATION SENSITIVITY (look-ahead horizon x scenario count)
# ══════════════════════════════════════════════════════════════════════════════

# The base case actually run for the paper.  NOT the runner_dispatch CLI
# defaults (10 / 12.0) — every stored LA run is S25 H24, and a sweep quoted
# against the wrong centre point is meaningless.
LA_BASE_SCEN    = 25
LA_BASE_HORIZON = 24.0

# One-at-a-time off (S25, H24): two horizons at the base scenario count, two
# scenario counts at the base horizon.  The base cell itself is deliberately
# absent — it is the existing unlabelled runs on these very instances.
#
# The horizon ladder is 12 / 24 / 48 rather than a fine grid, because the axis
# has natural breakpoints rather than a smooth trend: 12 h falls short of the
# 13 h regular spread T_SPR1, 24 h is exactly one duty cycle (T_SPR1 + Tr1 =
# 13 + 11), and 48 h is two.  A doubling ladder brackets the cycle length from
# both sides, which is what the threshold claim in the paper needs.
LA_DEFAULT_CONFIGS = "S25H12,S25H48,S10H24,S50H24"

# Seeds the REPORT is quoted over (2026-08-22).  The sweep itself was launched
# over 25 seeds, but every cell must stand on the SAME instances or the level
# column moves for a reason that has nothing to do with the configuration, and
# the cells that ran short (S10H24, S50H24) would otherwise be compared on a
# different population than the ones that ran full.  Ten is the window every
# cell of the design was meant to cover; whatever is still missing inside it is
# printed by the coverage block in cmd_la_report rather than being averaged
# over silently.  --seeds still overrides.
LA_REPORT_SEEDS = tuple(range(1, 11))

# Combos for the LA sweep.  Unlike the instance-level axes this one DOES span
# long routes: the whole question is how the look-ahead behaves as the route
# grows, and the base LA runs exist for all three.
LA_COMBOS = ["RshortCmedium", "RmediumCmedium", "RlongCmedium"]
# Both window regimes: the value of look-ahead should rise with temporal
# coupling, and time windows are the second coupling channel after HOS, so the
# none/tight contrast is part of the finding rather than a robustness check.
LA_TW     = ["none", "tight"]


def _parse_la_config(cfg: str) -> tuple[int, float]:
    """'S25H12' -> (25, 12.0).  The label is also the run's variant tag.

    'L' (look-ahead) is accepted as a synonym of 'H' (horizon) so that tags
    written either way refer to the same cell; see _norm_la_tag.
    """
    import re
    m = re.fullmatch(r"S(?P<s>\d+)[HL](?P<h>\d+(?:\.\d+)?)", cfg.strip(),
                     re.IGNORECASE)
    if not m:
        raise SystemExit(
            f"bad LA config '{cfg}': expected S<scenarios>H<horizon>, "
            f"e.g. S25H12 or S50H24 ('L' accepted for 'H')")
    return int(m.group("s")), float(m.group("h"))


def _instance_seed(instance: str | None) -> int | None:
    """'RshortCmediumTnone_7' -> 7; anything unparseable -> None."""
    if not instance:
        return None
    tail = instance.rsplit("_", 1)[-1]
    return int(tail) if tail.isdigit() else None


def _norm_la_tag(cfg: str) -> str:
    """Canonical cell label: 'S25L48' / 's25h48' -> 'S25H48', 'base' -> 'base'.

    Both spellings occur in stored runs — the batch was launched with L-tags
    while this module's defaults are written with H — and a report that keyed
    on the raw string would split one cell into two half-empty ones.
    """
    if not cfg or cfg == "base":
        return "base"
    try:
        n_scen, horizon = _parse_la_config(cfg)
    except SystemExit:
        return cfg
    return f"S{n_scen}H{horizon:g}"


_N_STOPS_CACHE: dict = {}


def _n_stops(instance: str | None) -> int | None:
    """Stops on the route, from the instance file (run_cache keeps no lists).

    Decisions happen once per stop, so this is the divisor that turns effort
    per RUN into effort per DECISION — the only form in which two route
    classes of very different length can be compared at all.
    """
    if not instance:
        return None
    if instance not in _N_STOPS_CACHE:
        _N_STOPS_CACHE[instance] = _read_n(instance)
    return _N_STOPS_CACHE[instance]


def _read_n(instance: str) -> int | None:
    """N off the instance file, wherever that file lives.

    A regime copy (…__nosplit) is written under instances_sens/<axis>/, not
    instances/, so a lookup that only tried instances/ returned None for every
    such run and dropped the whole NOSPLIT column out of the per-stop cost.
    The last resort is the STRIPPED stem: every axis reported here is a patch
    axis, so the copy has the same geometry — same stops, same order — as the
    instance it was patched from, and its N is the same number.
    """
    cands = [_paths.instances(instance + ".json")]
    cands += sorted(glob.glob(os.path.join(str(_paths.INSTANCES_SENS), "**",
                                           instance + ".json"),
                              recursive=True))
    stem, tag = _split_instance_tag(instance)
    if tag:
        cands.append(_paths.instances(stem + ".json"))
    for path in cands:
        try:
            with open(path, encoding="utf-8") as fh:
                raw = json.load(fh)
        except (OSError, ValueError):
            continue
        n = (raw.get("instance", raw) or {}).get("N")
        if n:
            return int(n)
    return None


def _t_per_stop_s(r) -> float | None:
    """Average decision time per stop: run wall clock / number of stops.

    Deliberately the crudest available measure (agreed 2026-08-22).  It is the
    total time the run took divided by the decisions it made, with NO
    correction for --jobs, for n_workers, or for how the |Xi| sub-problems of
    one action pack into a parallel wave.  Two earlier cost columns did try to
    correct for those — a stored per-stop mean that timed only the decision
    call, and a CPU-effort back-fill that multiplied the clock by the worker
    count — and both put a modelled number on the axis, one that no
    configuration was ever observed to produce and that moved whenever a batch
    was launched with different parallelism.

    What this measures is what an operator waits: the run took `wall_clock_s`
    and made `n_stops` decisions in it.  It carries the batch's own contention
    with it, so cells launched under different load are not perfectly
    comparable — that is a property of the measurement, and it is stated
    rather than modelled away.
    """
    wall = r.get("wall_clock_s")
    # Stops actually EXECUTED, falling back to the route length for runs made
    # before the halt semantics landed.  A halted run decided fewer stops than
    # the route has, so dividing by the route length would understate its cost
    # — but such a run is infeasible and is filtered out of this column
    # upstream, so this only ever matters for diagnosis.
    n = (r.get("metrics") or {}).get("n_stops_executed")         or _n_stops(r.get("instance"))
    if not wall or not n:
        return None
    return float(wall) / n


def cmd_la(args) -> None:
    """Section 8.3 — LA look-ahead horizon / scenario-count sensitivity.

    Runs on the BASE instances, with each cell labelled by --variant.  No
    instance copies: the instance is identical, so a copy would only orphan the
    run from the oracle cache that is already solved for it.  The compile dedup
    is keyed on (instance, method, supervised, variant), so these runs sit
    alongside the base LA runs instead of displacing them, and the base cell of
    the sweep IS the standard-configuration runs of those instances.

    Each cell inherits the runner's default tail solver, which since the
    2026-08-18 swap is the MIP: a ladder launched before that ran the LP tail
    and its rungs are no longer the same configuration as the cell they are
    read against, so it has to be re-run to be comparable.
    """
    seeds  = _expand_seeds(args.seeds)
    combos = args.combos.split(",")
    tws    = args.tw.split(",")
    tails  = [t.strip().lower() for t in args.solve_mode.split(",") if t.strip()]
    for t in tails:
        if t not in ("mip", "lp"):
            raise SystemExit(f"bad --solve-mode '{t}': expected mip, lp, or "
                             f"mip,lp")

    files = []
    for combo in combos:
        route, cust = _split_combo(combo)
        for tw in tws:
            for seed in seeds:
                p = _base_instance_path(route, cust, tw, seed)
                if not os.path.isfile(p):
                    print(f"missing  {p} (skipped)")
                    continue
                files.append(p)
    if not files:
        raise SystemExit("no instances matched --combos/--tw/--seeds")

    # One unit of work per (config, tail).  Both arms of a cell carry the SAME
    # --variant tag: paths.effective_variant files the LP-solved one as
    # "<TAG>+LPTAIL" at read time, so they never collide in the dedup and the
    # figures find both without the launcher having to invent a second tag.
    units = []
    for cfg in [c.strip() for c in args.configs.split(",") if c.strip()]:
        n_scen, horizon = _parse_la_config(cfg)
        for tail in tails:
            units.append((cfg, n_scen, horizon, tail))

    print(f"\n=== LA sweep: {len(units)} cell(s) x {len(files)} instance(s) "
          f"= {len(units) * len(files)} run(s) ===")
    for cfg, n_scen, horizon, tail in units:
        base = " (base cell, run under its own tag)" if (
            (n_scen, horizon) == (LA_BASE_SCEN, LA_BASE_HORIZON)) else ""
        print(f"     {cfg:<8} |Xi|={n_scen:<3} L={horizon:g}h  {tail}-tail{base}")

    def _launch(unit):
        cfg, n_scen, horizon, tail = unit
        # prune_quantile is NOT passed: the flag drives LA's action pruning and
        # every base LA run was made without it, so passing it would confound
        # the configuration axis with a guard change.
        _dispatch(",".join(files), "LA", args.jobs, args.dry_run,
                  extra=["--variant", cfg,
                         "--n_scenarios", str(n_scen),
                         "--horizon", f"{horizon:g}",
                         "--solve_mode", tail],
                  guard=None, la_energy_quantile=args.la_energy_quantile,
                  resume=args.resume)

    # Two nested levels of concurrency, and they multiply.  runner_dispatch
    # parallelises INSTANCES within one cell (--jobs); this parallelises the
    # CELLS on top (--config-jobs).  One level alone leaves cores idle: a cell
    # of 8 instances at --jobs 8 finishes its slowest run with seven cores
    # already free, and with the cells serialised that tail is paid once per
    # cell.  Since n_workers is 1 per run, the product is the number of
    # concurrent LA processes — keep --jobs x --config-jobs at or below the
    # core count.
    if args.config_jobs <= 1 or len(units) <= 1:
        for u in units:
            _launch(u)
        return
    print(f"     [{args.config_jobs} cell(s) in flight x --jobs {args.jobs} "
          f"= up to {args.config_jobs * args.jobs} concurrent run(s)]")
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=args.config_jobs) as ex:
        list(ex.map(_launch, units))


def _fmt_seeds(seeds: list) -> str:
    """[1,2,3,7] -> '1-3,7' — the same spec _expand_seeds accepts back."""
    if not seeds:
        return "-"
    parts, run_start, prev = [], seeds[0], seeds[0]
    for s in seeds[1:] + [None]:
        if s == prev + 1:
            prev = s
            continue
        parts.append(f"{run_start}-{prev}" if prev > run_start else f"{run_start}")
        run_start = prev = s
    return ",".join(parts)


# Instance tags the LA report folds in as CELLS of its own sweep.  Deliberately
# an allowlist, not "any tag": every one-at-a-time sensitivity axis also runs LA
# on tagged copies (…__cs30, …__kw700, …__kwh300), and those are infrastructure
# perturbations reported by section_sensitivity — pulling them onto the
# configuration sweep would answer a different question with the same picture.
# What belongs here is a change to the RULES the policy plans under, at
# unchanged infrastructure: nosplit forbids the Art. 7 15'+30' split, so the
# instance is physically the base case and only the legal action set moves.
_LA_REGIMES = {"nosplit"}

# Variant tags that are NOT cells of the sweep.  A --variant label is also the
# handiest way to keep an ad-hoc run from colliding with a stored one, so tags
# turn up that were never meant as configurations: LOCAL marks a single-machine
# timing probe, run on a handful of instances to measure what the policy costs
# on cab-grade hardware.
#
# Excluding them is not cosmetic.  Scope is DISCOVERED from the sweep rows, so
# an off-footprint variant does not merely add a spurious cell — it widens the
# combos, window classes and seeds every OTHER cell is then filtered on.  Three
# LOCAL runs, one of them on a window class the sweep does not use, pulled the
# whole Tmedium population into the pooled rows and moved the base cell from 150
# runs at 5.24 % to 224 at 5.47 %.  A cell must be evidence of its own scope
# only.
_LA_NON_CELL_VARIANTS = {"TB0"}
# Matched by PREFIX, not by exact string.  An exact-match list was tried and
# failed twice: a LOCAL batch varies the solver while keeping one label, so the
# tags arrive as LOCAL, LOCAL_MIPTAIL, and whatever comes next — and each new
# spelling silently re-entered the sweep, widening the discovered combos to the
# off-grid instances these probes run on and moving every pooled cell with it.
_LA_NON_CELL_PREFIXES = ("LOCAL",)


def _is_non_cell_variant(variant: str | None) -> bool:
    """True for a --variant that labels an ad-hoc run rather than a sweep cell."""
    v = (variant or "").upper()
    return v in _LA_NON_CELL_VARIANTS or v.startswith(_LA_NON_CELL_PREFIXES)


def _split_instance_tag(instance: str | None) -> tuple[str | None, str | None]:
    """'RmediumCfewTnone_10__nosplit' -> ('RmediumCfewTnone_10', 'nosplit').

    Sensitivity regimes run on TAGGED instance copies, so their route/window/
    seed do not parse off the raw id and compile_solutions annotates them as
    None.  Everything the LA report keys on has to come off the stripped stem.
    """
    if not instance or "__" not in instance:
        return instance, None
    stem, _, tag = instance.rpartition("__")
    return stem, (tag or None)


def _la_coords(rec, cs) -> tuple:
    """(route, cust, window, seed, regime) for one LA run, tag-aware."""
    stem, regime = _split_instance_tag(rec.get("instance"))
    tags = cs._parse_instance_tags(stem)
    return (tags["route_class"], tags["customers_class"],
            tags["window_class"], _instance_seed(stem), regime)


def _la_cell_tag(variant: str | None, regime: str | None,
                 method: str = "LA") -> str:
    """Config label for one cell: the tail solver crossed with the regime.

    Two orthogonal things pick out an LA cell.  The VARIANT says what the
    policy does (absent = the standard configuration, which since the 2026-08-18
    swap is the MIP tail; LPTAIL = the superseded LP tail), and the instance TAG
    says which regime it runs under (absent = the base instances, nosplit =
    Art. 7 split break forbidden).  They are crossed rather than merged so that,
    e.g., LPTAIL+NOSPLIT can never be silently pooled into LPTAIL — the
    collision that cost us a batch once already.  A cell run with an explicit
    S25H24 tag is the base cell written the long way, so it normalises onto it.

    The variant it reads has already been through paths.effective_variant, so
    "absent" means the standard configuration whichever tag the run was launched
    under — the MIP-tail runs are stored under the historic "MIPTAIL" label.
    """
    # Greedy is carried alongside the LA cells as the do-nothing reference —
    # the policy the look-ahead has to beat to justify its compute — so it takes
    # a cell of its own rather than being folded into the base cell, which is
    # LA's own standard configuration.
    cfg = (_norm_la_tag(variant or "base") if method == "LA"
           else method.upper())
    if cfg == f"S{LA_BASE_SCEN}H{LA_BASE_HORIZON:g}":
        cfg = "base"
    if not regime:
        return cfg
    reg = regime.upper()
    return reg if cfg == "base" else f"{cfg}+{reg}"


def _regap_regimes_to_base_oracle(rows, cs) -> int:
    """Re-measure regime runs against the UNRESTRICTED (base) hindsight optimum.

    By default every run is scored against the oracle for its own instance id,
    which for a …__nosplit copy is the oracle that was itself denied the split
    break.  That answers "how well did the policy do given the rule", and it is
    the wrong question here: scored that way a regime can look BETTER simply
    because its optimum got worse, and the cells stop being comparable across
    the plane.

    Scoring against the base oracle instead makes every cell in the figure a
    distance to one common reference — the best achievable with every option
    available — so forbidding the split shows up as the loss it is.  This is
    only legitimate because no_split is a PATCH axis: the copy has identical
    geometry and identical realisations, so the base oracle is the optimum of
    the very same instance under the unrestricted rules, not of a different one.
    Do not extend this to a regen axis, where the instances genuinely differ.
    """
    tagged = []
    for r in rows:
        if r.get("method") != "LA":
            continue
        stem, regime = _split_instance_tag(r.get("instance"))
        if regime in _LA_REGIMES:
            tagged.append((r, r["instance"], stem))
    if not tagged:
        return 0
    # Reuse the real annotator rather than restating its arithmetic: point each
    # record at the base stem, let it resolve that oracle, then put the id back
    # so the pairing keys downstream still identify the run's own instance.
    for r, _orig, stem in tagged:
        r["instance"] = stem
    cs._annotate_gap_to_oracle([r for r, _o, _s in tagged], _paths.solutions())
    for r, orig, _s in tagged:
        r["instance"] = orig
    return len(tagged)


def _la_local_tag(rec, regime: str | None) -> str:
    """Cell label inside the LOCAL family: LP or MILP, split break or not.

    The sweep separates its arms by --variant, but a LOCAL batch varies the
    SOLVER instead and keeps one variant label, so the tag is read off
    solve_mode.  Matching on the LOCAL prefix rather than the exact string
    keeps LOCAL_MIP and friends in the family instead of stranding them as
    cells of their own.
    """
    tag = "LOCAL" + ("+MIP" if rec.get("solve_mode") == "mip" else "")
    return tag + ("+NOSPLIT" if regime == "nosplit" else "")


def _prefer_energy_guard(rows, cs):
    """Where one instance has both a guarded and an unguarded LA MILP run of the
    same cell, keep the guarded one.

    effective_variant merges the two into a single MILP cell, because the
    committed-charge guard is being adopted as part of the standard rather than
    studied as a variant.  Merging alone would AVERAGE them, which is the one
    thing a cell must not do: it would report a number no configuration
    produces.  Preferring the higher quantile makes the merged cell mean "the
    configuration we are going forward with, on every instance that has it, and
    the older run only where it does not."

    Runs at the same quantile are left alone for _dedup_latest to resolve on
    timestamp, which is the existing rule for genuine repeats.
    """
    best = {}
    for r in rows:
        if r.get("method") != "LA" or (r.get("solve_mode") or "").lower() != "mip":
            continue
        key = (r.get("instance"), r.get("variant"), bool(r.get("supervised")))
        q = float(r.get("la_energy_quantile") or 0.0)
        if key not in best or q > best[key][0]:
            best[key] = (q, id(r))
    out, dropped = [], 0
    for r in rows:
        if r.get("method") == "LA" and (r.get("solve_mode") or "").lower() == "mip":
            key = (r.get("instance"), r.get("variant"), bool(r.get("supervised")))
            hit = best.get(key)
            if hit and hit[1] != id(r) and                     float(r.get("la_energy_quantile") or 0.0) < hit[0]:
                dropped += 1
                continue
        out.append(r)
    return out, dropped


def _la_discover(rows, cs) -> dict:
    """Footprint of the LA sweep as ACTUALLY RUN: configs, combos, TW classes
    and seeds read off the stored variant runs.

    Derived from the SWEEP rows only, never the base ones.  The base cell is
    the pre-existing unlabelled runs and covers every seed and combo ever run,
    so discovering from it would widen the scope to instances no variant cell
    has, and the level column would then compare a wide base average against
    narrow variant averages — exactly the distortion the --seeds filter exists
    to prevent.

    A sweep row is one carrying a --variant OR one running under a regime tag
    (…__nosplit): the regime cell has no variant of its own — its standard arm
    is an ordinary unlabelled run — but it is still a sweep cell rather than the
    base, and leaving it out of discovery would filter its own runs away.
    """
    cfgs, combos, tws, seeds = set(), set(), set(), set()
    for r in rows:
        if r.get("method") != "LA" or r.get("status") != "OK":
            continue
        route, cust, tw, seed, regime = _la_coords(r, cs)
        if _is_non_cell_variant(r.get("variant")):
            continue                              # ad-hoc run, not a cell
        if regime and regime not in _LA_REGIMES:  # another axis's sweep
            continue
        # Neither the base cell nor the LP-tail one is evidence of the sweep's
        # footprint: LPTAIL is the FORMER base (every seed and combo ever run,
        # under the old default), so discovering from it would widen the scope
        # to instances no sweep cell has and set the level column comparing a
        # full-design average against ten-seed ones.  Both are still reported —
        # cmd_la_report adds them to wanted_cfg by name.
        if r.get("variant") in (None, _paths.LA_LEGACY_VARIANT) and not regime:
            continue
        if not route or not cust or seed is None:
            continue
        cfgs.add(_la_cell_tag(r.get("variant"), regime))
        combos.add(f"R{route}C{cust}")
        tws.add(tw)
        seeds.add(seed)
    return dict(configs=cfgs, combos=combos, tws=tws - {None}, seeds=seeds)


# Per-stop decision times are not persisted: runner.py keeps only the mean and
# the max in metrics.  They ARE in the run log, though — each stop opens with a
# header carrying its type and closes with a '-> CHOSEN ... <t>s' line — so the
# CS-only mean is recoverable from logs/ without re-running anything.
#
# Why CS stops specifically: they are the ones where a decision actually has
# branching structure (charge / how long / and the break-rest interaction on top
# of it), so their cost is what an operator would feel at the stop.  Averaging
# over every stop dilutes that with laybys and customers, which enumerate far
# fewer actions and run about half as long.
_LA_LOG_HDR    = re.compile(r"^\[LA\] stop (\d+) \((\w+)\)")
_LA_LOG_CHOSEN = re.compile(r"-> CHOSEN .*?([\d.]+)s\s*$")
_LA_CS_CACHE: dict = {}


def _cs_decision_mean_s(run_id: str | None):
    """Mean decision time over CS stops only, parsed from logs/<run_id>.txt.

    Returns None when the log is missing or carries no CS stop, which _agg then
    drops — the same treatment any other absent measurement gets.
    """
    if not run_id:
        return None
    if run_id in _LA_CS_CACHE:                  # pooled rows re-read the same runs
        return _LA_CS_CACHE[run_id]
    path = _paths.log_path(f"{run_id}.txt")
    out, cur = None, None
    try:
        vals = []
        with open(path, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                m = _LA_LOG_HDR.match(line)
                if m:
                    cur = m.group(2)
                    continue
                if cur == "CS":
                    m = _LA_LOG_CHOSEN.search(line)
                    if m:
                        vals.append(float(m.group(1)))
        # A forced rest short-circuits decide_stop and prints no CHOSEN line, so
        # a stop can be missing here; the mean is over the CS stops that were
        # actually decided, which is the quantity being reported.
        out = (sum(vals) / len(vals)) if vals else None
    except OSError:
        out = None
    _LA_CS_CACHE[run_id] = out
    return out


_FORCED_REST_CACHE: dict = {}


def _tripped_forced_rest(run_id: str | None) -> bool:
    """True when this run hit the old forced-rest safety net.

    Runs stored before 2026-08-22 were simulated under the OLD semantics: when
    every scored action came back infeasible over the look-ahead horizon, the
    policy inserted a minimum corrective rest and carried on, so the run
    finished the route and was recorded FEASIBLE.  Under the halt semantics the
    same situation ends the run and it is infeasible.

    Re-running the whole corpus to reclassify is unnecessary, because the net
    printed a marker every time it fired.  Reading it back off the log recovers
    the new classification exactly: the net fired iff the run had no feasible
    action at some stop, which is precisely the new halt condition.

    A missing log reads as False — the run then keeps whatever classification
    it was stored with, which is the conservative direction (it stays feasible
    and keeps contributing, rather than being silently dropped on the strength
    of a file that is not there).
    """
    if not run_id:
        return False
    if run_id in _FORCED_REST_CACHE:
        return _FORCED_REST_CACHE[run_id]
    hit = False
    try:
        with open(_paths.log_path(f"{run_id}.txt"), encoding="utf-8",
                  errors="replace") as fh:
            for line in fh:
                if "FORCED REST" in line or "NO FEASIBLE ACTION" in line:
                    hit = True
                    break
    except OSError:
        hit = False
    _FORCED_REST_CACHE[run_id] = hit
    return hit


def _reclassify_forced_rest(rows) -> int:
    """Mark every run that tripped the net as infeasible, in place.

    Runs after cs._annotate_outcome, because it overrides the outcome that
    function assigned.  Everything downstream keys on `outcome` — _gap_usable,
    _is_truly_infeasible, the n_infeasible counts — so this one override is
    enough to make the whole report speak the halt semantics.
    """
    n = 0
    for r in rows:
        if r.get("method") != "LA" or r.get("outcome") != "feasible":
            continue
        if _tripped_forced_rest(r.get("run_id")):
            r["outcome"] = "infeasible"
            r["_reclassified_no_feasible_action"] = True
            n += 1
    return n


def _dec_cs(rec, cs):
    """Decision time per CS stop for one run, whatever produced it.

    LA prints a per-stop line and gets the real CS-only mean.  Greedy prints no
    timing at all and does not need one: its decision is a rule evaluation that
    measures 0.0 s to four decimals at every stop, CS or not, so the stored
    all-stop mean is the same number to any precision this report can show.
    """
    if rec.get("method") == "LA":
        return _cs_decision_mean_s(rec.get("run_id"))
    return cs._get(rec, ("metrics", "decision_time_mean_s"))


def _run_date(run_id: str | None) -> str | None:
    """'..._LA_S50H24_20260821_160214_000' -> '20260821'."""
    m = re.search(r"_(\d{8})_\d{6}", run_id or "")
    return m.group(1) if m else None


def _batch_spread(recs) -> tuple[int, float | None]:
    """(distinct batch dates, max/min of the per-date median t/stop).

    The cost column is run wall clock over stops, uncorrected for parallelism,
    so it is only comparable between runs that met the same machine.  It is
    not a small effect: the base cell's own runs span seven batches over ten
    days and their per-batch median ranges from 17 to 98 s/stop — 5.7x on ONE
    configuration.  Paired against the 2026-08-18 base batch, S50 comes out
    2.9x slower and L48 7.9x slower (the direction the work implies); paired
    against 2026-08-17 both come out faster.  Whichever batch dominates a cell
    therefore decides which way it reads.

    Reporting the spread next to the number is what keeps that visible.  A
    cell drawn from ONE batch gets (1, None) and needs no caveat; a cell with
    a large ratio is a cell whose cost cannot be compared with another's.
    Dates carrying fewer than three runs are skipped — a single straggler
    re-run months later would otherwise set the ratio on its own.
    """
    import statistics as stat
    by: dict = {}
    for r in recs:
        t = _t_per_stop_s(r)
        d = _run_date(r.get("run_id"))
        if t is not None and d:
            by.setdefault(d, []).append(t)
    meds = [stat.median(v) for v in by.values() if len(v) >= 3]
    if len(meds) < 2 or not min(meds):
        return len(by), None
    return len(by), max(meds) / min(meds)


def _la_panel(by_cell: dict, exclude: set) -> tuple[set, list]:
    """The BALANCED PANEL: instances carrying an OK run in every cell.

    Reporting each cell on whatever instances it happens to have is what
    produced the two anomalies chased on 2026-08-22.  A cell's median is a
    median over ITS OWN route mix, and the mixes are not equal: S25H48/MIP is
    53% short-route runs and 5% long (four long runs exist in total), so its
    pooled cost lands inside the short group at 27 s/stop, while the base cell
    is a balanced 34/34/32 and lands at 72.  Side by side that reads as "a
    longer horizon is cheaper" — the opposite of the truth, since on long
    routes alone the same cell is 455 s/stop.

    Intersecting first removes the confound at its root rather than patching
    the symptom: every cell then reports the same instances, so a difference
    between cells cannot be a difference in population, and the paired deltas
    are paired on the whole panel instead of on whatever overlapped.

    Keyed on the STRIPPED stem, so a regime copy (…__nosplit) counts as the
    instance it was patched from and the NOSPLIT arms can join the panel
    instead of emptying it.

    ``exclude`` names cells that are reported but may not CONSTRAIN the panel.
    One sparse cell otherwise decides the scope for all the others: including
    S25H48/MIP costs 40 instances and 31 of the 34 long routes.
    """
    cells = [c for c in by_cell if c not in exclude and by_cell[c]]
    if not cells:
        return set(), []
    return set.intersection(*[by_cell[c] for c in cells]), sorted(cells)


def _panel_breakdown(stems, cs) -> str:
    """'111 instances (short 39, medium 38, long 34)' for the console."""
    n = {}
    for stem in stems:
        rt = cs._parse_instance_tags(stem).get("route_class")
        n[rt] = n.get(rt, 0) + 1
    parts = ", ".join(f"{r} {n[r]}" for r in ("short", "medium", "long")
                      if r in n)
    return f"{len(stems)} instance(s)" + (f" ({parts})" if parts else "")


def _la_coverage(present, cfgs, combos, tws, seeds) -> None:
    """What the fixed seed window actually has, and what it is missing.

    The report is quoted over LA_REPORT_SEEDS rather than over whatever each
    batch reached, which is the only way the cells compare — but a fixed window
    turns "this cell ran short" from something that silently widens the scope
    into something that has to be SEEN.  So the full design grid (config x
    combo x window x seed) is walked and every hole in it is named, both on the
    console and in data_output/additional_la_coverage.csv.

    A hole means "no run with status OK on that instance": a run that exists but
    stranded the truck counts as missing here, because it contributes nothing to
    a median either.  The n_infeasible column of the stats CSV is where those
    are counted; this is where their absence from the medians is explained.
    """
    import csv
    rows, short = [], []
    for cfg in sorted(cfgs):
        miss_cfg = 0
        for combo in sorted(combos):
            for tw in sorted(tws):
                got  = present.get((cfg, combo, tw), set())
                miss = sorted(seeds - got)
                miss_cfg += len(miss)
                rows.append([cfg, combo, tw, len(got), len(miss),
                             _fmt_seeds(miss) if miss else ""])
        if miss_cfg:
            short.append((cfg, miss_cfg))
    full = len(cfgs) * len(combos) * len(tws) * len(seeds)
    have = sum(r[3] for r in rows)
    print(f"  coverage: {have}/{full} cells of the "
          f"{len(cfgs)}x{len(combos)}x{len(tws)}x{len(seeds)} design "
          f"(config x combo x window x seed) have an OK run")
    if not short:
        print("            nothing missing")
    for cfg, n in sorted(short, key=lambda t: -t[1]):
        det = [f"{c}/{t} {m}" for c, t, m in
               ((r[1], r[2], r[5]) for r in rows if r[0] == cfg and r[5])]
        print(f"            {cfg:<14} {n:>4} missing : " + "; ".join(det))
    out = _paths.data_output("additional_la_coverage.csv")
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["config", "combo", "window_class", "n_present",
                    "n_missing", "missing_seeds"])
        w.writerows(rows)
    print(f"  coverage CSV: {out}")


def cmd_la_report(args) -> None:
    """Write data_output/additional_la_stats.csv: one row per (config, route
    class, window class), with the unlabelled base runs as the reference row.

    With no --configs/--combos/--tw/--seeds, the scope is DISCOVERED from the
    runs on disk rather than assumed from the launcher's defaults: a sweep is
    routinely launched over a different footprint than LA_DEFAULT_*, and a
    report that filtered on the defaults would silently write an empty CSV and
    an empty figure.  Any flag that IS passed overrides the discovered value.

    Two effect measures are reported side by side, and they answer different
    questions:

      gap_pen_median_pct  — level: distance to the hindsight optimum.  Bounded
                            below by the oracle's own residual MIP gap, which
                            on long routes is structural (~9%), so it is a
                            calibrated level, not a certified one.
      delta_vs_base_pct   — effect: the paired per-instance change in route
                            duration against the base cell.  The oracle cancels
                            out of a difference of two policy runs on the SAME
                            instance, so this is the number the sweep is really
                            about and it is unaffected by the oracle's gap.

    Cost is reported as decision_mean_s (per-stop decision time) as well as
    wall clock, because wall clock is only comparable across configs when every
    batch was launched with the same --jobs.
    """
    import csv
    import statistics as stat
    from src.output_analysis import compile_solutions as cs

    rows = cs.load_solutions(_paths.solutions())
    cs._annotate_instance_tags(rows)
    cs._annotate_gap_to_oracle(rows, _paths.solutions())
    rows, n_pref = _prefer_energy_guard(rows, cs)
    if n_pref:
        print(f"  MILP runs superseded by a guarded run on the same instance: "
              f"{n_pref} dropped")
    n_regap = _regap_regimes_to_base_oracle(rows, cs)
    if n_regap:
        print(f"  regime runs re-scored against the base-case oracle: {n_regap}")
    cs._annotate_outcome(rows)
    n_rc = _reclassify_forced_rest(rows)
    if n_rc:
        print(f"  runs reclassified infeasible (no feasible action, halt "
              f"semantics): {n_rc}")
    rows, _ = cs._dedup_latest(rows)

    found = _la_discover(rows, cs)
    if not any((args.configs, args.combos, args.tw, args.seeds)) and not found["configs"]:
        print("  no LA variant runs found in solutions/ — nothing to report")
        return

    wanted_cfg = ({_la_cell_tag(c.strip(), None) for c in args.configs.split(",")
                   if c.strip()}
                  if args.configs else set(found["configs"]))
    wanted_cfg.add("base")
    wanted_cfg.add("GREEDY")      # the do-nothing reference, never swept
    # The superseded LP tail: reported like any other cell, but excluded from
    # scope discovery for the reason given in _la_discover, so it has to be
    # named here or it would never be asked for.  Asked for only when runs of
    # it still EXIST: the LP-tail corpus was deleted on 2026-08-22, and a cell
    # requested by name with nothing behind it reports its whole footprint as
    # missing, which reads as a data gap rather than as a deliberate removal.
    if any(r.get("method") == "LA"
           and (r.get("variant") or "").upper().endswith(_paths.LA_LEGACY_VARIANT)
           for r in rows):
        wanted_cfg.add(_paths.LA_LEGACY_VARIANT)
    # A regime cell needs its own standard arm as the reference the variant arm
    # is read against, and that arm carries no --variant, so asking for
    # LPTAIL+NOSPLIT has to pull NOSPLIT in with it.
    wanted_cfg |= {c.split("+", 1)[1] for c in list(wanted_cfg) if "+" in c}
    combos = set(args.combos.split(",")) if args.combos else found["combos"]
    tws    = set(args.tw.split(","))     if args.tw     else found["tws"]
    # Seeds are NOT discovered.  The base cell is the pre-existing unlabelled
    # runs and covers every seed ever run, while each sweep cell covers only the
    # seeds its own batch reached — so discovering the union would compare a
    # 25-instance average against 10- and 24-instance ones, and the LEVEL column
    # would move between cells for a reason that has nothing to do with the
    # configuration.  Fixed at LA_REPORT_SEEDS (1-10) so every cell stands on
    # one population; what is missing inside that window is printed by
    # _la_coverage rather than averaged over.  --seeds still overrides.
    seeds = set(_expand_seeds(args.seeds)) if args.seeds else set(LA_REPORT_SEEDS)

    src = lambda given: "given" if given else "found"
    print(f"  scope   : configs {','.join(sorted(wanted_cfg - {'base'})) or '-'} "
          f"({src(args.configs)})")
    print(f"            combos  {','.join(sorted(combos))} ({src(args.combos)})")
    print(f"            tw      {','.join(sorted(tws))} ({src(args.tw)})")
    print(f"            seeds   {_fmt_seeds(sorted(seeds))} "
          f"({'given' if args.seeds else 'fixed'})")

    groups: dict = {}
    present: dict = {}
    by_cell: dict = {}          # cell -> the instance stems it has an OK run on
    for r in rows:
        if r.get("method") not in ("LA", "greedy") or r.get("status") != "OK":
            continue
        route, cust, tw, seed, regime = _la_coords(r, cs)
        if _is_non_cell_variant(r.get("variant")):
            continue                              # ad-hoc run, not a cell
        if regime and regime not in _LA_REGIMES:  # another axis's sweep
            continue
        if not route or not cust:
            continue
        if f"R{route}C{cust}" not in combos:
            continue
        if tw not in tws:
            continue
        if seed not in seeds:
            continue
        cfg = _la_cell_tag(r.get("variant"), regime, r.get("method"))
        if cfg not in wanted_cfg:
            continue
        # The annotation upstream reads these off the raw instance id, which is
        # None for a tagged copy; downstream pairing keys on window_class, so
        # write the stripped-stem value back before it is used.
        r["window_class"] = tw
        # The stem is the panel key: a regime copy counts as the instance it
        # was patched from, so a NOSPLIT arm joins the panel rather than
        # emptying it.
        r["_stem"] = _split_instance_tag(r.get("instance"))[0]
        groups.setdefault((cfg, route, tw), []).append(r)
        present.setdefault((cfg, f"R{route}C{cust}", tw), set()).add(seed)
        by_cell.setdefault(cfg, set()).add(r["_stem"])

    _la_coverage(present, wanted_cfg, combos, tws, seeds)

    # ── the balanced panel ───────────────────────────────────────────────────
    if args.panel == "common" and by_cell:
        excl = {c.strip() for c in (args.panel_exclude or "").split(",")
                if c.strip()}
        keep, held = _la_panel(by_cell, excl)
        print(f"  panel   : {_panel_breakdown(keep, cs)} with an OK run in "
              f"every one of {len(held)} cell(s)")
        if excl:
            print(f"            not constraining the panel: "
                  f"{','.join(sorted(excl))} (still reported, still filtered)")
        # Named individually rather than as a count: the cell that costs the
        # most is the one worth re-running or excluding, and that is only
        # visible per cell.
        for c in held:
            lost = len(by_cell[c] - keep)
            if lost:
                print(f"            {c:<16} drops {lost:>3} of its "
                      f"{len(by_cell[c])} instance(s) to the panel")
        short = sorted(by_cell, key=lambda c: len(by_cell[c]))[:1]
        if short and len(by_cell[short[0]]) < len(keep) * 1.15:
            print(f"            binding cell: {short[0]} "
                  f"({len(by_cell[short[0]])} instances) — "
                  f"--panel-exclude {short[0]} to free the rest")
        groups = {k: [r for r in v if r["_stem"] in keep]
                  for k, v in groups.items()}
        groups = {k: v for k, v in groups.items() if v}
    elif args.panel == "all":
        print("  panel   : none (--panel all) — cells may differ in route mix, "
              "so a difference between them can be a difference in population")

    # ── the LOCAL family ─────────────────────────────────────────────────────
    # Same measurements, separate corpus and separate CSV.  These runs answer a
    # different question — what the policy costs on cab-grade hardware, not how
    # it should be configured — and they are launched on whatever handful of
    # instances is convenient, off the sweep's footprint.  Pooling them into the
    # sweep is what pulled the Tmedium population into every cell once already
    # (see _LA_NON_CELL_VARIANTS), so they are deliberately NOT scope-filtered
    # and NOT written to the same file; nothing here can move a sweep number.
    local: dict = {}
    for r in rows:
        if r.get("method") != "LA" or r.get("status") != "OK":
            continue
        if not (r.get("variant") or "").upper().startswith("LOCAL"):
            continue
        route, cust, tw, seed, regime = _la_coords(r, cs)
        if not route or tw is None:
            continue
        r["window_class"] = tw
        local.setdefault((_la_local_tag(r, regime), route, tw), []).append(r)

    if not groups:
        print("  no LA runs matched — nothing written")
        return

    def _agg(vals):
        vals = [v for v in vals if v is not None]
        return (stat.median(vals) if vals else None), len(vals)

    # Paired reference: the standard arm's duration per instance, keyed the same
    # way the variant rows are, so a missing reference run drops the pair
    # instead of silently comparing against a different instance.
    #
    # Selected on "carries no --variant" rather than on cfg == 'base', because
    # a regime cell has its own standard arm and must be read against THAT, not
    # against the split-break base: the key already carries the instance id, and
    # a nosplit copy has a different one, so keying this way pairs each variant
    # arm with the standard arm on its own instance and can never cross the two
    # regimes.
    # Restricted to LA: greedy also carries no variant, and since the key is
    # (instance, window) a greedy run would otherwise overwrite the LA arm on
    # the same instance and every delta in the table would silently become a
    # comparison against greedy.
    _ref = lambda r: (r.get("method") == "LA" and not r.get("variant")
                      and not cs._is_truly_infeasible(r))
    base_dur = {(r.get("instance"), r.get("window_class")): r.get("duration_h")
                for recs in groups.values() for r in recs if _ref(r)}
    # Same pairing on the PENALISED duration (arrival + beta * window misses),
    # which is the model's actual objective.  A configuration can shorten the
    # route by arriving late at customers, and the duration-only delta scores
    # that as an improvement; the penalised delta does not.
    base_pen = {(r.get("instance"), r.get("window_class")):
                r.get("duration_pen_h")
                for recs in groups.values() for r in recs if _ref(r)}

    _paths.ensure_dirs()
    def _write(out_name, grp):
        """One CSV from one group dict.  The sweep and the LOCAL family share
        the column set and the aggregation exactly; only the corpus differs, so
        they share the writer rather than drifting apart in two copies."""
        out = _paths.data_output(out_name)
        with open(out, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["config", "n_scenarios", "horizon_h", "route_class",
                        "window_class", "n_runs", "n_infeasible",
                        # what every median in this row is computed over
                        "n_feasible",
                        "gap_pen_median_pct", "gap_nopen_median_pct",
                        "duration_median_h", "delta_vs_base_pct", "n_paired",
                        "duration_pen_median_h", "delta_pen_vs_base_pct",
                        "n_paired_pen",
                        # The reported cost of a decision: run wall clock over
                        # the number of stops, uncorrected for parallelism (see
                        # _t_per_stop_s).  Every figure and table reads THIS
                        # column; the three that follow are the finer-grained
                        # instruments kept for diagnosis.
                        "t_per_stop_s_median",
                        # How far the cost column can be trusted ACROSS cells:
                        # how many batches the cell was built from, and how far
                        # apart their per-batch medians are.  See _batch_spread.
                        "n_batches", "t_per_stop_batch_ratio",
                        "decision_mean_s_median", "decision_cs_mean_s_median",
                        "decision_max_s_median",
                        "wall_clock_s_median", "n_stops_median",
                        # Solver EFFORT, measured only — a run made before the
                        # 2026-08-21 instrumentation simply has none, and is
                        # left out rather than back-filled from the clock.
                        # cap_rate says whether that work actually converged.
                        "solve_cpu_h_median", "n_subproblems_median",
                        "cap_rate_median"])
            # Window classes are also POOLED into a synthetic 'all' row per
            # (config, route).  A figure that wants one row per configuration must
            # not average the two per-window medians: the classes carry different
            # run counts (and the base cell carries more than the sweep cells), so
            # the average of medians is weighted by nothing meaningful.  Pooling the
            # raw runs and re-taking the median is the honest version, and the
            # per-instance pairing behind delta_vs_base_pct is unaffected because it
            # keys on (instance, window_class) either way.
            pooled = {}
            for (cfg, route, _tw), recs in grp.items():
                pooled.setdefault((cfg, route, "all"), []).extend(recs)
                # Route classes pool the same way and for the same reason, so a
                # figure asking what a configuration costs OVERALL reads one
                # pooled median rather than an unweighted average of three
                # per-corridor medians carrying very different run counts.
                pooled.setdefault((cfg, "all", _tw), []).extend(recs)
                pooled.setdefault((cfg, "all", "all"), []).extend(recs)
            for (cfg, route, tw), recs in sorted(list(grp.items())
                                                 + list(pooled.items())):
                # EVERY reported value is read from the FEASIBLE runs only
                # (2026-08-22).  Since a violation now halts the run, an
                # infeasible run has no completed route and no full decision
                # sequence: its duration is None and its wall clock covers only
                # the stops it reached before dying.  Averaging those in would
                # reward failing early — the sooner a cell strands, the cheaper
                # and shorter it would look.
                #
                # `recs` stays the whole cell, because n_runs and n_infeasible
                # are what the infeasibility rate is built from, and that rate
                # is the correction the reader applies to every column below.
                ok = [r for r in recs if cs._gap_usable(r)]
                n_ok = len(ok)
                gp, n_gp = _agg([100.0 * r["gap_pen"] for r in ok
                                 if r.get("gap_pen") is not None])
                gn, _    = _agg([100.0 * r["gap_nopen"] for r in ok
                                 if r.get("gap_nopen") is not None])
                dur, _   = _agg([r.get("duration_h") for r in ok])
                dec, _   = _agg([cs._get(r, ("metrics", "decision_time_mean_s"))
                                 for r in ok])
                dcs, _   = _agg([_dec_cs(r, cs) for r in ok])
                decx, _  = _agg([cs._get(r, ("metrics", "decision_time_max_s"))
                                 for r in ok])
                wall, _  = _agg([r.get("wall_clock_s") for r in ok])
                cpu, _   = _agg([cs._get(r, ("metrics", "solve_cpu_s_total"))
                                 for r in ok])
                nsub, _  = _agg([cs._get(r, ("metrics", "n_subproblems"))
                                 for r in ok])
                caps, _  = _agg([cs._get(r, ("metrics", "subproblem_cap_rate"))
                                 for r in ok])
                tps, _   = _agg([_t_per_stop_s(r) for r in ok])
                nbat, brat = _batch_spread(ok)
                nstp, _  = _agg([_n_stops(r.get("instance")) for r in ok])
                n_inf = sum(1 for r in recs if cs._is_truly_infeasible(r))

                durp, _  = _agg([r.get("duration_pen_h") for r in ok])

                deltas, deltas_pen = [], []
                for r in recs:
                    if cs._is_truly_infeasible(r):
                        continue
                    key = (r.get("instance"), r.get("window_class"))
                    b, d = base_dur.get(key), r.get("duration_h")
                    if b and d:
                        deltas.append(100.0 * (d / b - 1.0))
                    bp, dp = base_pen.get(key), r.get("duration_pen_h")
                    if bp and dp:
                        deltas_pen.append(100.0 * (dp / bp - 1.0))
                dl, n_pair = _agg(deltas)
                dlp, n_pair_pen = _agg(deltas_pen)

                ns = recs[0].get("n_scenarios")
                hh = recs[0].get("horizon_hours")
                w.writerow([cfg, ns, hh, route, tw, len(recs), n_inf, n_ok,
                            None if gp is None else round(gp, 3),
                            None if gn is None else round(gn, 3),
                            None if dur is None else round(dur, 3),
                            None if dl is None else round(dl, 3), n_pair,
                            None if durp is None else round(durp, 3),
                            None if dlp is None else round(dlp, 3), n_pair_pen,
                            None if tps is None else round(tps, 2),
                            nbat, None if brat is None else round(brat, 2),
                            None if dec is None else round(dec, 4),
                            None if dcs is None else round(dcs, 4),
                            None if decx is None else round(decx, 4),
                            None if wall is None else round(wall, 1),
                            None if nstp is None else round(nstp, 0),
                            None if cpu is None else round(cpu / 3600.0, 3),
                            None if nsub is None else round(nsub, 0),
                            None if caps is None else round(caps, 4)])
                print(f"  {cfg:<8} {route:<7} {tw:<6} "
                      f"n={len(recs):<3} ok={n_ok:<3} "
                      f"gap_pen={'—' if gp is None else f'{gp:.2f}%':<7} "
                      f"(from {n_gp} run(s) with an oracle bound)  "
                      f"delta={'—' if dl is None else f'{dl:+.2f}%':<7} "
                      f"(n={n_pair})  "
                      f"t/CS={'—' if dcs is None else f'{dcs:.1f}s'}"
                      f" (wall/stop {'—' if tps is None else f'{tps:.1f}s'})"
                      + (f" [{nbat} batches, x{brat:.1f} apart]"
                         if brat and brat > 1.5 else ""))
        print(f"  CSV saved   : {out}")

    _write("additional_la_stats.csv", groups)
    if local:
        print(f"  LOCAL family: {sum(len(v) for v in local.values())} run(s) in "
              f"{len(local)} cell(s) — separate corpus, separate file")
        _write("additional_la_local_stats.csv", local)


def _latest_2sp_solution(stem: str) -> str | None:
    """Newest solutions/<stem>_2SP_*.json (run ids end in a timestamp, so
    lexicographic max = latest), for the RP leg of the VSS harness."""
    hits = sorted(_paths.glob_solutions(f"{stem}_2SP_*.json"),
                  key=os.path.basename)
    return hits[-1] if hits else None


def cmd_guard(args) -> None:
    """Section 8.2/8.3 — guarded-greedy sweep (departure guard at the xi
    q-quantile).  Runs on TAGGED instance copies (__q95 etc.) because the
    compile dedup is keyed on (instance, method, supervised) only: running
    guarded greedy on the base instances would silently displace the
    unguarded base runs in every table at the next compile."""
    seeds  = _expand_seeds(args.seeds)
    combos = args.combos.split(",")
    tws    = args.tw.split(",")
    for q in [float(x) for x in args.quantiles.split(",")]:
        tag     = f"q{int(round(q * 100))}"
        out_dir = os.path.join(SENS_DIR, f"guard_{tag}")
        pattern = _materialise_copy(tag, combos, tws, seeds,
                                    out_dir, args.dry_run)
        _dispatch(pattern, args.algorithms, args.jobs, args.dry_run,
                  guard=q, resume=args.resume,
                  n_scenarios=args.n_scenarios, horizon=args.horizon,
                  la_energy_quantile=args.la_energy_quantile)


def cmd_vss(args) -> None:
    """Section 8.5 — VSS / EVPI decomposition, one call of the harness per
    instance (see experiments/vss_evpi.py for the EEV/RP/WS definitions).
    The RP leg reuses the latest base-case 2SP solution when one exists."""
    seeds  = _expand_seeds(args.seeds)
    combos = args.combos.split(",")
    tws    = args.tw.split(",")
    os.makedirs(VSS_DIR, exist_ok=True)
    for combo in combos:
        route, cust = _split_combo(combo)
        for tw in tws:
            for seed in seeds:
                inst = _base_instance_path(route, cust, tw, seed)
                stem = os.path.splitext(os.path.basename(inst))[0]
                out  = os.path.join(VSS_DIR, f"{stem}_vss.json")
                if os.path.isfile(out) and not args.overwrite:
                    print(f"skip     {out} (exists)")
                    continue
                cmd = [sys.executable,
                       "-m", "src.output_analysis.vss_evpi",
                       inst, "--out", out,
                       "--n-scenarios", str(args.n_scenarios),
                       "--seed", str(args.crn_seed),
                       "--time-limit", str(args.time_limit)]
                plan = _latest_2sp_solution(stem)
                if plan:
                    cmd += ["--plan-from", plan]
                else:
                    print(f"note     {stem}: no 2SP solution found — "
                          f"RP leg will be null (VSS/EVPI incomplete)")
                _run(cmd, args.dry_run)


def cmd_compile(args) -> None:
    """Refresh the Excel summary + LaTeX tables over solutions/ (variant runs
    are distinguished by their '__tag' instance stems)."""
    cmd = [sys.executable, "-m", "src.output_analysis.compile_solutions"]
    if args.tex_dir:
        cmd += ["--tex-dir", args.tex_dir]
    _run(cmd, args.dry_run)


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def _add_common(p: argparse.ArgumentParser, algos_default: str) -> None:
    p.add_argument("--combos", default=",".join(DEFAULT_COMBOS),
                   help="Comma-separated route+customer combos "
                        f"(default: {','.join(DEFAULT_COMBOS)}; long routes "
                        "deliberately excluded — see module docstring)")
    p.add_argument("--tw", default=",".join(DEFAULT_TW),
                   help="Comma-separated TW classes (default: all four)")
    p.add_argument("--seeds", default=DEFAULT_SEEDS,
                   help="Seed spec, e.g. '1-10' or '1,2,7' "
                        f"(default: {DEFAULT_SEEDS})")
    p.add_argument("--algorithms", default=algos_default,
                   help=f"Algorithm spec for runner_dispatch "
                        f"(default: {algos_default})")
    p.add_argument("--jobs", type=int, default=2,
                   help="Concurrent (instance, algorithm) runs (default: 2)")
    p.add_argument("--prune_quantile", type=float, default=DEFAULT_GUARD,
                   help="Greedy departure guard: depart only if the leg still "
                        f"fits at this xi quantile (default: {DEFAULT_GUARD}, "
                        "matching the base case). Applied to GREEDY only — "
                        "the flag is shared with LA pruning, so other "
                        "algorithms are dispatched without it. Pass an empty "
                        "value via --prune_quantile nan to disable.")
    p.add_argument("--n_scenarios", type=int, default=None,
                   help="Scenario count forwarded to LA / 2SP.  MUST match the "
                        "base case, otherwise the sweep delta mixes the axis "
                        "change with a scenario-count change.  Unset = "
                        "runner_dispatch's default (10).")
    p.add_argument("--horizon", type=float, default=None,
                   help="LA look-ahead horizon (h) forwarded to runner_dispatch. "
                        "Match the base case for the same reason as "
                        "--n_scenarios.  Unset = runner_dispatch's default (12).")
    p.add_argument("--la_energy_quantile", type=float, default=None,
                   help="LA energy guard forwarded to runner_dispatch: size "
                        "the committed charge to cover the legs to the next CS "
                        "at this quantile of consumption instead of at "
                        "nominal.  Match the base case for the same reason as "
                        "--n_scenarios.  Unset = off "
                        "(settings.LA_ENERGY_QUANTILE), which is NOT what the "
                        "current base LA batch uses.")
    p.add_argument("--no-resume", dest="resume", action="store_false",
                   default=True,
                   help="Disable LA checkpoint/resume (on by default): every "
                        "LA batch launched here checkpoints after each stop "
                        "and continues a crashed run from where it stopped. "
                        "Pass this only if the per-stop checkpoint write is "
                        "unwanted — it is the flag that saves a half-finished "
                        "long-route LA run.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print commands / files without executing anything")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Head block for the paper's additional analyses "
                    "(Sections 8.3-8.5). See module docstring for the "
                    "subcommand -> section map.")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("sensitivity",
                       help="8.3 one-at-a-time sensitivity sweeps")
    p.add_argument("--axis", required=True, choices=sorted(_AXES),
                   help="Which parameter to sweep")
    p.add_argument("--values", default=None,
                   help="Comma-separated values (default: per-axis classes "
                        "from settings.py)")
    _add_common(p, DEFAULT_ALGOS)
    p.set_defaults(func=cmd_sensitivity)

    p = sub.add_parser("grid",
                       help="8.3 two-axis grid (default: battery x charger power)")
    p.add_argument("--axes", default="battery,charger_power",
                   help="Two comma-separated regen axes to cross "
                        "(default: battery,charger_power)")
    p.add_argument("--x-values", dest="x_values", default=None,
                   help="Comma-separated values for the FIRST axis "
                        "(default: that axis's default_values)")
    p.add_argument("--y-values", dest="y_values", default=None,
                   help="Comma-separated values for the SECOND axis")
    _add_common(p, DEFAULT_ALGOS)
    p.set_defaults(func=cmd_grid)

    p = sub.add_parser("diesel", help="8.4 diesel counterpart runs")
    _add_common(p, DIESEL_ALGOS)
    p.set_defaults(func=cmd_diesel)


    p = sub.add_parser("guard", help="8.2 guarded-greedy quantile sweep")
    p.add_argument("--quantiles", default="0.9,0.95,1.0",
                   help="Comma-separated prune quantiles (default: "
                        "0.9,0.95,1.0; 1.0 = worst-case corner)")
    _add_common(p, "greedy")
    p.set_defaults(func=cmd_guard)

    # LA configuration sweep — its own arg block rather than _add_common:
    # the algorithm is fixed (LA) and --prune_quantile must NOT be offered,
    # since that flag also drives LA's action pruning and would confound the
    # axis with a guard change.
    def _add_la_common(pp) -> None:
        pp.add_argument("--configs", default=LA_DEFAULT_CONFIGS,
                        help="Comma-separated S<scenarios>H<horizon> cells "
                             f"(default: {LA_DEFAULT_CONFIGS}).  The base cell "
                             f"S{LA_BASE_SCEN}H{LA_BASE_HORIZON:g} may be named "
                             f"like any other: it then runs under its own tag "
                             f"and is filed in the LAconfig bucket, so it "
                             f"cannot displace the unlabelled base-case runs.")
        pp.add_argument("--combos", default=",".join(LA_COMBOS),
                        help=f"Route+customer combos (default: "
                             f"{','.join(LA_COMBOS)})")
        pp.add_argument("--tw", default=",".join(LA_TW),
                        help=f"TW classes (default: {','.join(LA_TW)})")
        pp.add_argument("--seeds", default=DEFAULT_SEEDS,
                        help=f"Seed spec (default: {DEFAULT_SEEDS})")

    p = sub.add_parser("la", help="8.3 LA horizon / scenario-count sweep")
    _add_la_common(p)
    # Carried for the same reason as --no-resume below: this subcommand
    # launches LA without taking _add_common's block, and a cell run at a
    # different energy guard from the base cell is a different policy, not a
    # point on the horizon/scenario ladder.
    p.add_argument("--la_energy_quantile", type=float, default=None,
                   help="LA energy guard forwarded to runner_dispatch (size "
                        "the committed charge to cover the legs to the next CS "
                        "at this quantile of consumption instead of at "
                        "nominal).  Match the base cell, or the ladder mixes "
                        "the guard with the configuration.  Unset = off "
                        "(settings.LA_ENERGY_QUANTILE).")
    p.add_argument("--solve-mode", dest="solve_mode", default="mip",
                   help="Look-ahead TAIL solver: 'mip' (the standard), 'lp', "
                        "or 'mip,lp' to launch both arms of every cell.  Both "
                        "arms carry the same --variant tag; the LP one is "
                        "filed as '<TAG>+LPTAIL' at read time, so they never "
                        "collide and both reach the figures.")
    p.add_argument("--config-jobs", dest="config_jobs", type=int, default=1,
                   help="How many CELLS to have in flight at once (default: "
                        "1, i.e. cells run one after another).  Multiplies "
                        "with --jobs, which parallelises instances WITHIN a "
                        "cell: keep --jobs x --config-jobs at or below the "
                        "core count, since n_workers is 1 per run.")
    p.add_argument("--jobs", type=int, default=8,
                   help="Concurrent runs (default: 8, matching the base LA "
                        "batch).  Keep it identical across every cell or the "
                        "wall-clock column is contention, not compute.")
    # The LA sweep is exactly the batch resume exists for, so it carries the
    # flag even though it does not take _add_common's block.
    p.add_argument("--no-resume", dest="resume", action="store_false",
                   default=True,
                   help="Disable LA checkpoint/resume (on by default): each "
                        "cell checkpoints after every stop and a crashed run "
                        "continues from where it stopped instead of "
                        "restarting.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print commands without executing anything")
    p.set_defaults(func=cmd_la)

    # la-report deliberately does NOT take _add_la_common's defaults.  The
    # launcher must state what to run; the reporter should describe what WAS
    # run, and a sweep is routinely launched over a different footprint than
    # LA_DEFAULT_*.  Unset here means "discover from solutions/".
    p = sub.add_parser("la-report",
                       help="8.3 LA sweep -> data_output/additional_la_stats.csv "
                            "(scope auto-discovered from the runs on disk)")
    p.add_argument("--configs", default=None,
                   help="Comma-separated S<scenarios>H<horizon> cells "
                        "(default: every LA variant found in solutions/)")
    p.add_argument("--combos", default=None,
                   help="Route+customer combos (default: those the found "
                        "variant runs actually cover)")
    p.add_argument("--tw", default=None,
                   help="TW classes (default: those the found runs cover)")
    p.add_argument("--seeds", default=None,
                   help=f"Seed spec, e.g. '1-25' (default: "
                        f"{_fmt_seeds(list(LA_REPORT_SEEDS))}, fixed so every "
                        f"cell stands on one population; what is missing inside "
                        f"that window is printed as coverage)")
    p.add_argument("--panel", default="common", choices=["common", "all"],
                   help="'common' (default) reports every cell on the BALANCED "
                        "PANEL — only instances with an OK run in every cell — "
                        "so each cell is the same population and every "
                        "comparison is exactly paired.  'all' reports each cell "
                        "on whatever it has, which lets route mix differ "
                        "between cells")
    p.add_argument("--panel-exclude", default=None,
                   help="Cells that may NOT constrain the panel, comma "
                        "separated (e.g. 'S25H48').  They are still reported, "
                        "and still filtered to the panel; they just do not "
                        "shrink it.  Use for a cell whose runs are too sparse "
                        "to hold the rest hostage")
    p.set_defaults(func=cmd_la_report)

    p = sub.add_parser("vss", help="8.5 VSS / EVPI decomposition")
    p.add_argument("--n-scenarios", type=int, default=20,
                   help="Common-random-number scenarios per instance "
                        "(default: 20)")
    p.add_argument("--crn-seed", type=int, default=42,
                   help="Seed for the common scenario draw (default: 42)")
    p.add_argument("--time-limit", type=int, default=600,
                   help="Per-MILP time limit (s) inside the harness "
                        "(default: 600)")
    p.add_argument("--overwrite", action="store_true",
                   help="Recompute instances whose result file exists")
    _add_common(p, "-")   # harness runs its own solves; algo spec unused
    p.set_defaults(func=cmd_vss)

    p = sub.add_parser("compile", help="Refresh Excel/LaTeX result tables")
    p.add_argument("--tex-dir", default=None,
                   help="Also write LaTeX tables into this directory")
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=cmd_compile)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
