"""
additional_analysis.py — single entry point for the paper's additional analyses.

Every experiment block of Section 8 beyond the base-case comparison (8.2) is
launched from here.  Each subcommand maps to one paper section:

  Subcommand     Paper section                       What it produces
  ─────────────  ──────────────────────────────────  ─────────────────────────────
  sensitivity    8.3 Sensitivity analysis            one-at-a-time sweeps off the
                                                     base case (CS spacing, charger
                                                     power, beta, travel CV, AR(1)
                                                     rho, no-split break; battery
                                                     is a guarded stub until plumbed)
  diesel         8.4 VS diesel trucks                same instances re-run with
                                                     --diesel (HoS only, no
                                                     charging) for the EV-vs-diesel
                                                     makespan penalty
  vss            8.5 Effect of uncertainty           VSS / EVPI decomposition via
                                                     experiments/vss_evpi.py
                                                     (EEV / RP / WS, common random
                                                     numbers)
  gamma          8.5 Effect of uncertainty           ROBU price-of-robustness
                                                     frontier (--robu_gamma sweep)
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

  # 8.5: budget frontier on short routes
  python -m src.output_analysis.additional_analysis gamma --gammas 0,1,2,4,8 --combos RshortCfew

  # 8.5: VSS / EVPI on the short/medium grid, 20 scenarios
  python -m src.output_analysis.additional_analysis vss --n-scenarios 20

  # refresh tables (Excel + LaTeX) after any block
  python -m src.output_analysis.additional_analysis compile --tex-dir tex/tables
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from src import paths as _paths

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
    "charger_power": dict(kind="regen", kw="charger_power_kw",
                          default_values=[150, 350, 1000], tag="kw"),
    "cv":            dict(kind="regen", kw="cv",
                          default_values=[0.10, 0.25], tag="cv"),
    "ar1_rho":       dict(kind="regen", kw="ar1_rho",
                          default_values=[0.4], tag="rho"),
    "beta":          dict(kind="patch", kw="beta",
                          default_values=[2.0, 5.0], tag="beta"),
    "battery":       dict(kind="stub",
                          msg="battery axis needs Bcap plumbed through "
                              "generate_instance_file -> instance_realistic "
                              "-> make_data (mirror the charger_power_kw "
                              "path) before it can run"),
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
              guard: float | None = None) -> None:
    """Launch runner_dispatch for one instance pattern.

    ``guard`` is the greedy departure quantile (--prune_quantile).  Because
    that flag is SHARED — it also drives LA's action pruning and the opt-in
    supervisor — it must not leak onto the other algorithms, whose base runs
    were made without it.  So when greedy is requested alongside others, the
    batch is split: greedy runs guarded, the rest run untouched.
    """
    def _go(alg_spec: str, guarded: bool) -> None:
        cmd = [sys.executable, "-m", "src.simulation.runner_dispatch", pattern, alg_spec,
               "--jobs", str(jobs), "--skip-existing"]
        if guarded and guard is not None:
            cmd += ["--prune_quantile", str(guard)]
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
                  guard=args.prune_quantile)


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
              extra=["--diesel"], guard=args.prune_quantile)


def cmd_gamma(args) -> None:
    """Section 8.5 — ROBU price-of-robustness frontier (budget sweep)."""
    seeds  = _expand_seeds(args.seeds)
    combos = args.combos.split(",")
    tws    = args.tw.split(",")
    for g in [int(x) for x in args.gammas.split(",")]:
        out_dir = os.path.join(SENS_DIR, f"gamma_{g}")
        pattern = _materialise_copy(f"g{g}", combos, tws, seeds,
                                    out_dir, args.dry_run)
        _dispatch(pattern, "ROBU", args.jobs, args.dry_run,
                  extra=["--robu_gamma", str(g)])   # ROBU: no greedy guard


def _latest_2sp_solution(stem: str) -> str | None:
    """Newest solutions/<stem>_2SP_*.json (run ids end in a timestamp, so
    lexicographic max = latest), for the RP leg of the VSS harness."""
    import glob as _glob
    hits = sorted(_glob.glob(_paths.solutions(f"{stem}_2SP_*.json")))
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
                  guard=q)


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

    p = sub.add_parser("diesel", help="8.4 diesel counterpart runs")
    _add_common(p, DIESEL_ALGOS)
    p.set_defaults(func=cmd_diesel)

    p = sub.add_parser("gamma", help="8.5 ROBU budget (Gamma) frontier")
    p.add_argument("--gammas", default="0,1,2,4,8",
                   help="Comma-separated integer budgets (default: 0,1,2,4,8; "
                        "base case already covers Gamma = sqrt(N))")
    _add_common(p, "ROBU")
    p.set_defaults(func=cmd_gamma)

    p = sub.add_parser("guard", help="8.2 guarded-greedy quantile sweep")
    p.add_argument("--quantiles", default="0.9,0.95,1.0",
                   help="Comma-separated prune quantiles (default: "
                        "0.9,0.95,1.0; 1.0 = worst-case corner)")
    _add_common(p, "greedy")
    p.set_defaults(func=cmd_guard)

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
