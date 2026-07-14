# CHANGES_IMPLEMENTED.md — July-2026 model revision

Implementation record for **every item in `code_change_list.md`** (which maps
to the annotated paper rewrite `annotated_draft_rewrite.pdf`).  Equation tags:
`(o.XX)` = old paper numbering, `(RXX)/(VIX)` = rewrite numbering.

Status legend: ✅ implemented · 🟡 implemented with a decision you should
review · 📦 skeleton (needs external data) · 📝 documented-only (P3/optional).

**Everything previously computed must be rerun** — the P0 items change the
feasible set and the objective of every model, and the instance JSONs should
be regenerated (new δ=0.15 base case, MILP-based windows, destination
deadlines, 60 km spacing default).

---

## Decisions taken on the open forks (review these first)

| Fork | Decision | Where to change it |
|---|---|---|
| M6 (10 h extension vs conservative 9 h) | **Full extension mechanism (R16–R19)** implemented — the rewrite includes it in the paper body and the code already tracked `ext_shift_used` | drop by setting `ext_bar=0` in `settings.py` (`EXT_BAR`) |
| M8 (layby nodes) | Implemented **behind a flag, default OFF** (`add_laybys=False` in `instances.instance_realistic` / `instance_io.generate_instance_file`) | turn on per-instance; §3.4(d) of the rewrite assumes ON — enable before final experiments or revert the paper text to "at stops only" |
| M7 β (lateness weight) | `BETA_LATENESS = 10.0` obj-hours per lateness-hour (`settings.py`); `hard_tw=True` flag recovers hard windows | `settings.py` / per-instance `hard_tw` |
| Destination deadline (R6) | `T_dead = t_nom_N + max(2 h, 0.20·(t_nom_N − T_START))`, generated with the windows (I1) | `instance_io._DEADLINE_KAPPA/_DEADLINE_DMIN` |
| Base δ | **0.15** everywhere (S5); 0.25 available as sensitivity | `settings.LOWER_PCT` |
| I1 window widths | Uniform half-width `Δ = Δmin` for every customer (no exposure scaling), per-class Δmin: tight 0.5 h, medium 1 h, large 3 h; small ±10 % centre jitter keeps the per-seed RNG contract | `instance_io._WINDOW_DMIN` |
| CS spacing | Base **60 km** per AFIR (was 40); 30/90 sensitivities | `settings.CS_SPACING_KM` |

---

## A. Deterministic MILP (`MILP.py`, `instances.py`, `settings.py`)

- **M1 (P0) ✅ — valid inequalities.** The `−1` fix was already present in
  `_add_vi3/_add_vi4`; the remaining invalidity — using the 9 h limit — is
  fixed: VI-3 and VI-5 now use `Tdrv_sh2 = 10 h`
  (`MILP.add_valid_inequalities`).  Unit tests
  `test_m1_vi_reset_count_five_hours` (5 h driving → exactly 1 reset admitted)
  and `test_m1_vi_shift_uses_extended_limit` (9.5 h driving → 0 rests
  admitted) pass.
- **M2 (P0) ✅ — break credit `g` (R7–R12).** `p` deleted; `g[i]` (credited
  charging) added with `g ≤ tauc`, `g ≤ T_R·x`, `g ≤ T_R(1−σ)`,
  `g ≥ tauc − T_R(1−x) − T_R·σ` and `taub_hat = g + taub` at CS
  (`= taub` at customers **and laybys**).  The redundant concurrent-coverage
  block (o.42)–(o.44) is deleted.  Tests m2a/m2b/m2c pass.
- **M3 (P0) ✅ — `u` deleted.** Work contribution of charging is the linear
  expression `tauc − g` in the `sw` propagation (R20) and the weekly cap
  (R21).  Test m3 passes (fully counted / zero when credited / fully counted
  sequential + M_seq).
- **M4 (P0) ✅ — σ rules (R13–R14).** `σ ≤ y`, `σ ≥ y + ρ − 1`
  (charge+rest forced sequential) plus the tightening `σ ≤ x + ρ` (no
  sequential flag without an activity to sequence).  `BEHDV.advance`'s
  no-MILP fallback now also sets σ=1 for charge+rest.  Test m4 passes.
- **M5 (P1) ✅ — shift spread (R22–R25).** `sw ≤ 13 h` (o.62) deleted; new
  variables `h`, `l5` with `o_i := td−ta−taur` as an expression, big-M
  `M_h = 15 h`; pre-rest cap `h+o ≤ 13 + 2·ρ2 + 15(1−ρ)`; global `h ≤ 15`.
  `sw` is **retained** (weekly cap R21, reporting, S3 verification).  Waiting
  is inside `td−ta` so it consumes spread but never working time (S4).
  BEHDV tracks `h` in the state/trajectory.  Tests m5 pass
  (14 h pre-rest spread: r1 infeasible, r2 feasible).
- **M6 (P1) ✅ — 10 h extension (R16–R19).** First-stage binaries `z`,
  `q_ext`; `sd ≤ 9 + z`, persistence `z_{i+1} ≥ z_i − ρ_i`,
  `q ≥ z + ρ − 1`, `Σq + z_N ≤ ext_bar (=2)`.  The old shallow-copy hack in
  `Simulation.select_best_action` (raising `Tdrv_sh1`/`M_sd` to 10 h) is
  **removed**; every sub-problem now receives `ext_remaining = ext_bar −
  vehicle.ext_shift_used`.  Tests m6 pass (9.5 h feasible with budget,
  infeasible with `ext_bar=0`).
- **M7 (P1) ✅ — soft windows + waiting (R1–R6, new objective).** Variables
  `w`, `ell` (customers); objective `min ta_N + β·Σ ell`; `ta+w ≥ Wha`,
  `ell ≥ ta+w − Whf` (or hard closing under `hard_tw=True` with `ell=0`);
  destination deadline `ta_N ≤ T_dead` when the instance provides one.
  `w` enters the spread via `o` and never enters `sw` (verified in the S4
  path).  Tests m7 pass (wait, penalised lateness, hard-tw sensitivity).
- **M8 (P2) 🟡 — layby nodes.** `instances.insert_laybys()` splits legs
  > 30 min at ~25 km spacing into `L`-type stops (break/rest only, 2-min
  parking overhead `M_lay`, counted as work when a break/rest is taken).
  Model support in `MILP` (td_L, taub_hat, sw, weekly cap), `2SP`, and the
  sub-problem slicing.  **Default OFF** (`add_laybys=False`) — flag it on and
  report the solve-time impact before adopting §3.4(d).
- **M9 (P1) ✅ — budgets & constants.** `rho_bar=3` (was already 3),
  `ext_bar=2`, weekly working cap R21 (60 h; per scenario in 2SP/RO, skipped
  when it contains no variables), instance-generation assertion
  `ΣD_nom ≤ 56 h` in `make_data` (regenerated with offset seed in
  `instance_io`), all `T_K` big-Ms are `T_R` (`m.TK`), `v ≤ y+x+ρ` present.
  Big-Ms raised: `M_sd = 10 h`, `M_sw = M_h = 15 h`.

## B. Rolling horizon (`Simulation.py`)

- **RH1 (P0) ✅** — VI fix in place; **old Table 7 LA numbers are invalid,
  rerun before interpreting.**
- **RH2 (P1) ✅** — pruning is now literally the supervisor's checks
  (`supervisor.compute_flags` / `action_passes` — single source of truth);
  `prune_quantile` config (1.0 = exact under bounded uniform; <1 for
  lognormal, report α).  The pruning and the realisation draw use the same
  δ by construction of the run loop.
- **RH3 (P1) ✅** — `criterion ∈ {mean, worst, best, cvar_0.8}` (generic
  `cvar_<alpha>`); infeasible penalty is the named constant `T_PEN`.
  Ablation runnable via `experiments/rh_sweep.py --criteria mean cvar_0.8 worst`.
- **RH4 (P1) ✅** — with `--solve_mode both`, per-stop LP-vs-MIP agreement
  and the MIP-score delta of the LP choice are recorded (`events["cmp_log"]`),
  summarised into `metrics["lp_vs_mip"]` (agreement rate + mean delta) —
  the data for Table `lp-vs-milp`.
- **RH5 (P2) ✅** — `experiments/rh_sweep.py`: T_hor × S (× criterion) sweep
  with per-stop decision-time reporting (the real-time argument).
- **RH6 (P1) ✅** — `find_horizon_end_stop` now carries the precise
  step-by-step algorithm docstring (appendix pseudocode).

## C. Two-stage stochastic plan (`2SP.py`, `recourse.py`)

- **SP1 (P0) ✅** — scenario-averaged-duration execution **replaced** by
  `recourse.run_plan_with_recourse`: per stop, fixed-structure re-solve over
  `[i, N]` from the realized state with nominal remaining times (tiny MIP —
  only PWL/mode flags stay integer); on infeasibility an **add-only repair**
  MILP (binaries may only increase, each addition penalised at 1000 in the
  objective, then arrival+lateness); if repair fails → recorded plan
  violation + supervisor takeover.  Repair frequency / additions / plan
  violations are S2 metrics.  Implemented via `fixed_plan` / `plan_mode`
  ("fix" | "repair") in `MILP.build_horizon_model`.
  Smoke run: 3 add-only repairs fired, 0 violations, feasible, 4.4 % gap.
- **SP2 (P1) ✅ / P3 📝** — the non-anticipativity caveat is in the `run_2sp`
  docstring (paper §5.5 sentence).  The optional `SP-resolve` interpolation
  method (re-solve the full 2SP every k stops) is **not** implemented (P3).
- **SP3 (P2) ✅ (doc)** — parity note in the docstring; both LA and 2SP
  default to `n_scenarios=10` in the dispatcher.
- Model updates: the extensive form got all of M2–M7/M9 per scenario
  (g-credit, tauc−g work, σ rules, spread h, extension z/q first-stage, soft
  TW with per-scenario w/ell, weekly cap), plus `objective="mean"|"max"`.

## D. Robust plan (`RO.py`)

- **RO1 (P0) ✅** — full rewrite: budgeted set `U_Γ` (Bertsimas–Sim),
  min–max by **scenario duplication reusing the 2SP builder**
  (`build_2sp_model(objective="max")`, epigraph θ) — binaries shared,
  durations scenario-indexed (adjustable).  Initial set = nominal +
  Γ-longest-legs@+δ + Γ-most-energy-critical@−δ + K random; **cutting-plane
  loop** stress-tests the incumbent plan (fixed-structure evaluation) against
  greedy-flip and random candidates in `U_Γ`, appending violating/θ-exceeding
  realizations (cap `max_cut_iters=10`).  Time/energy ECR coupling is applied
  consistently *within* each scenario.  Execution shares the SP1 recourse
  path.  `--gamma` sweeps the frontier; `--legacy_box` keeps the old
  constraint-wise mixed-extremes counterpart as the conservatism baseline.
  Smoke run (Γ=2): feasible, 8.4 % gap.
- **RO2 (P1) ✅ (code-side)** — Bertsimas & Sim (2004) and Ben-Tal et al.
  (2004) DOIs are in the module docstring; **add them to the paper .bib**
  (fills the three `[? ]` placeholders in §5.4/§6.4).

## E. Simulator (`supervisor.py`, `BEHDV.py`, `runner.py`, `oracle.py`)

- **S1 (P0) ✅** — `supervisor.py`: one-step feasibility guard applied
  identically to **every** policy (LA loop, greedy, 2SP/RO recourse), same
  function as the RH pruning.  Cheapest preventive action in the order
  parallel-break-upgrade < break < charge-to-cover < rest.  Modes:
  supervised (default) / **raw** (`--raw`), interventions logged.
- **S2 (P0) ✅** — violation semantics: `BEHDV` **records instead of
  raising** — stranding (SOC clipped to Emin so the run completes
  observably), retroactive mid-leg `hos_cd`/`hos_sd`/`hos_spread`, TW
  lateness (violation only under `hard_tw`).  `runner.finalize_run` builds a
  `metrics` block (violations by type, stranding count, lateness hours,
  waits, interventions, repairs, plan violations, per-stop decision times,
  offline solve time, S3 compliance, RH4 agreement) saved in the results
  JSON and returned in `results["metrics"]`.
- **S3 (P1) ✅** — `oracle.check_directive_compliance`: ex-post scan for
  >6 h consecutive work without a break and per-shift break totals
  (30/45 min bands); reported per run (`metrics["directive_compliance"]`).
  The explicit consecutive-working counter block was **not** added — add it
  only if the reported compliance is ≪100 % across the final experiments.
- **S4 (P1) ✅** — waiting unified: model variable `w` (offline plans) and
  simulator-inserted wait (online), both consume spread `h` and never `sw`.
- **S5 (P2) ✅** — `settings.sample_multipliers`: uniform (base, δ=0.15),
  lognormal (CV-matched), AR(1) correlation on the normal driver; wired into
  `scenarios.generate_scenarios(dist=, ar1_rho=)` and the instance-file
  realisation draws.  δ inconsistency resolved (0.15 base, 0.25 sensitivity).
- **S6 (P2) ✅ (decision + doc)** — `Q_i` is a **known parameter** (expected
  access delay, visible to every method); documented in `settings.py`.
  No arrival-time revelation, no endogenous queuing.

## F. Instances & experiments (`instance_io.py`, `runner_dispatch.py`)

- **I1 (P1) ✅** — windows from the **deterministic MILP** nominal solve
  (1 % gap, 300 s limit; greedy fallback flagged in the JSON meta via
  `window_half_widths._source`), centred on nominal service starts,
  exposure-scaled half-widths, destination deadline, nominal-infeasible
  geometry regenerated with offset seeds (requested seed kept in the
  filename; `geometry_seed` in meta).
- **I2 (P1) ✅** — charger power classes {150, 200, 350, 1000 kW} rescale the
  PWL breakpoints (`settings.scale_tbar`, `charger_power_kw` through
  `make_data`/`instance_realistic`/`generate_instance_file`); CS spacing
  base 60 km (AFIR) with 30/90 sensitivities (`cs_spacing_km` parameter).
- **I3 (P1) ✅ (pre-existing)** — diesel benchmark via
  `runner_dispatch --diesel` was already present and still works.
- **I4 (P2) 📦** — `norway_instance.py`: complete loader skeleton (NOBIL
  chargers CSV + customer terminals CSV + optional OSRM legs + Statens
  vegvesen dispersion) emitting a standard instance JSON.  Needs the data
  exports (documented in its docstring).
- **I5 (P2) ✅ (partial)** — common random numbers were already guaranteed by
  the per-file `D_real`; the new `metrics` block in every results JSON gives
  `compile_solutions.py` what it needs for distribution plots (box plots) and
  paired comparisons.  Extending `compile_solutions.py` plotting itself was
  left as a reporting task.

## G. Energy model

- **E1 (P1) ✅** — citable calibration comment (Nykvist & Olsson 2021;
  NACFE Run on Less—Electric) + startup assertion
  `1.0 ≤ ECR(80) ≤ 1.5 kWh/km` in `settings.py` (ECR(80) ≈ 1.10).
- **E2 (P3) 📝 not implemented** — payload-dependent `E_i` needs
  per-customer load quantities in the instance format; parameter-only once
  the format carries loads.  Deliberately skipped (optional).

---

## New / removed API surface

| Change | Detail |
|---|---|
| New modules | `supervisor.py`, `recourse.py`, `experiments/rh_sweep.py`, `norway_instance.py`, `tests/test_model_changes.py` |
| `MILP.solve_horizon` | new kwargs `ext_remaining`, `fixed_plan`, `plan_mode` |
| `MILP` model | new vars `g, h, l5, z, q_ext, w, ell`; **removed** `u, p` and `conc_*`; solution dicts gained `h, g, w, ell, z` |
| `BEHDV` | new: `h` property/history, `violations`, `wait_list`, `lateness`; `advance` no longer raises on TW/SOC |
| `run_*` entry points | new kwargs `supervised`, `prune_quantile`; greedy: `delta` (queue avoidance removed); RO: `Gamma`, `legacy_box`, `n_random_scen`, `max_cut_iters` |
| CLI (`runner_dispatch`) | `--gamma`, `--legacy_box`, `--raw`, `--prune_quantile`, `--criterion cvar_0.8` |
| Results JSON | new `metrics` block (S2/S3/RH4) |

## What to rerun

1. **Regenerate all instance JSONs** (`python instance_io.py instances 50 0.15`)
   — new δ, 60 km spacing, MILP windows + deadlines, weekly-cap guard.
   Old JSONs still load (keys back-filled) but embed the old geometry/windows.
2. **Delete oracle caches** (`solutions/oracle_*.json`) — the model changed.
3. Rerun the full base-case table and sensitivities; report supervised + raw
   modes per the paper's §7 reporting plan.

## Verification performed

- `tests/test_model_changes.py`: **14/14 pass** (M1–M7 unit tests from the
  change list).
- End-to-end smoke runs on `RshortCfewTnone_1.json` (old-format JSON,
  back-filled): greedy ✅ feasible (gap 12.3 %), 2SP ✅ feasible (gap 4.4 %,
  3 add-only repairs, 0 violations), RO Γ=2 ✅ feasible (gap 8.4 %), LA ✅
  (see log).  Oracle solves with the new model + warm start.
