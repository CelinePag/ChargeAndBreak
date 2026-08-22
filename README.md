# ChargeAndBreak

Joint charging and hours-of-service (HoS) break scheduling for battery-electric
heavy-duty vehicles on long-haul routes.

## Layout

```
src/                      all Python sources
  paths.py                ROOT-anchored directory constants — the ONLY place
                          an output/input directory name is written down
  settings.py             shared model + protocol constants

  instance_gen/           building the instances
    instances.py            instance generator (geometry, chargers, windows)
    instance_io.py          read/write instances/*.json, batch generation
    norway_instance.py      real-corridor instance from data/norway/
    pglt.py                 R15-PGLT benchmark loader + validation

  methods/                the solvers / policies
    MILP.py                 core deterministic MILP formulation
    oracle.py               hindsight optimum (shared cache per instance)
    greedy.py               myopic online policy
    RO.py                   robust (conservative box)
    RObudget.py             budgeted robust (ROBU, C&CG)
    twosp.py                two-stage stochastic program (method label "2SP")
    recourse.py             plan execution with recourse
    SP.py                   (empty placeholder)

  simulation/             executing a plan against realised uncertainty
    Simulation.py           look-ahead (LA) policy + rolling simulation
    BEHDV.py                vehicle model (energy, charging curve)
    scenarios.py            scenario generation + ScenarioTracker
    supervisor.py           optional feasibility supervisor (off by default)
    runner.py               finalize_run: metrics, artefacts, logs
    runner_dispatch.py      batch CLI — the main entry point for runs

  output_analysis/        turning runs into numbers
    compile_solutions.py    solutions/ -> Excel + LaTeX tables
    additional_analysis.py  sensitivity / diesel / VSS orchestration
    vss_evpi.py             VSS + EVPI harness
    audit_runs.py           integrity audit of solutions/ + logs/
    coverage_report.py      per-class run coverage matrix
    oracle_*                oracle bound / stall diagnostics

  plot/                   figures
    paper_figures.py        main gap figures
    additional_figures.py   §8.3-8.5 figures + tables
    paper_style.py          shared palette and chrome
    plots.py                per-run diagnostic plots
    concept_solution_*.py   conceptual solution figure (matplotlib / pptx)

  misc/                   one-off maintenance + regression scripts

instances/                generated base instances        (input)
instances_sens/           variant instances, one dir/axis (input)
instances_usecase/        real-corridor instances         (input)
data/                     external datasets               (input)
solutions/                run results + oracle_<inst>.json caches
logs/                     per-run .txt, gurobi .log, *_scenarios.json
figures/                  .pdf / .png
tex/tables/               .tex — GENERATED tables (scripts overwrite these)
tex/sections/             .tex — hand-written manuscript prose (never written
                          by any script)
data_output/              .csv / .xlsx — tabular exports
archive/                  files retired by audit_runs.py (never deleted)
```

### Experiment buckets

`solutions/`, `logs/` and `figures/` are each split one level deep, by
experiment rather than by method or by date:

```
<tree>/basecase/          the 3x3x4x25 base grid, standard method configs
<tree>/LAconfig/          the look-ahead configuration sweep (S..H.., LOCAL)
<tree>/usecase/           the real-corridor instances (usecase_*)
<tree>/sensitivity/       the one-at-a-time axes ("<instance>__<tag>")
```

Which bucket a file belongs to is **derived from its name**, never stored, by
`bucket_of_artefact` / `bucket_of_figure` in `src/paths.py`. Writers call
`paths.solution_out()` / `log_out()` / `figure_out()`; readers call
`paths.find_solution()` / `glob_solutions()` / `scan_solutions()` (and the
`*_log` / `in_tree` / `scan_tree` equivalents), which search the tree root and
all four buckets. Neither side has to know a file's bucket, so a run_id is
still the only handle you need:

```bash
python -m src.plot.plots RlongCfewTlarge_10_LA_MIPTAIL_20260817_150816_000
```

Precedence is instance first, configuration second — an LA sweep run on a
sensitivity instance is a sensitivity run. A name that classifies into no
bucket (an ad-hoc run, the framework diagram) stays at the tree root rather
than being filed under a guess.

To re-file a tree after adding files by hand, or after changing a
classification rule:

```bash
python -m src.misc.bucketize_outputs           # report what would move
python -m src.misc.bucketize_outputs --apply   # move it
```

## Install

```bash
pip install -e .
```

The editable install puts the package on the import path, so the commands below
work from any working directory. Gurobi is driven through Pyomo and must be
installed separately with a valid licence.

## Running

```bash
python -m src.simulation.runner_dispatch "instances/RmediumCfew_1.json" LA,RO
python -m src.output_analysis.compile_solutions   # Excel + tex/tables/*.tex
python -m src.output_analysis.coverage_report
python -m src.output_analysis.audit_runs
python -m src.plot.paper_figures
python -m src.plot.additional_figures
python -m src.plot.plots <run_id>
```

`compile_solutions` refreshes the LaTeX tables by default; pass `--tex-dir ''`
to skip them.

Paths resolve against the repository root via `src/paths.py`, not the current
working directory, so output always lands in the same tree regardless of where
the process was started.
