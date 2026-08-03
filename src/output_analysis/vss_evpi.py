"""
vss_evpi.py — VSS / EVPI harness for Section 8.5 (Effect of uncertainty).

Launched per instance by ``python -m src.output_analysis.additional_analysis vss``; can also be
run standalone:

    python experiments/vss_evpi.py instances/RshortCfewTlarge_1.json \
        --out results_vss/RshortCfewTlarge_1_vss.json --n-scenarios 20

Definitions (all on the SAME common-random-number scenario set, drawn once
with --seed via scenarios.generate_scenarios):

  WS   (wait-and-see)      : mean over scenarios m of the perfect-hindsight
                             optimum oracle_solve(data, D_m) — the paper's
                             oracle machinery reused verbatim.
  EEV  (expectation of the : the plan of the EXPECTED-VALUE problem (the
        expected-value sol.)  deterministic MILP at nominal travel times,
                             i.e. oracle_solve at D = D_nominal), executed
                             under each scenario with online duration
                             recourse (recourse.run_plan_with_recourse) —
                             the same execution semantics as the 2SP plan.
  RP   (recourse problem)  : the 2SP plan executed the same way.  Supplied
                             via --plan-from (a solution json containing the
                             committed plan); when absent, RP is left null
                             and can be filled from the base-case 2SP runs
                             at compile time (then NOT common-random — flag
                             it in the paper's table notes).

  VSS = EEV - RP   (value of modelling the distribution at plan time)
  EVPI = RP - WS   (value of perfect information — irreducible online loss)

Both a raw-arrival and a window-penalised (obj = arrival + beta * misses)
version of every quantity are reported, matching compile_solutions'
gap_nopen / gap_pen convention.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

from src.instance_gen.instance_io import load_instance_json
from src.methods.oracle      import oracle_solve
from src.methods.recourse    import run_plan_with_recourse
from src.simulation.scenarios   import generate_scenarios


# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _sol_to_plan(sol: list, N: int) -> list[dict]:
    """Full-route MILP solution -> per-stop plan list (index 0..N) with the
    {y, break_type, rest_type} keys recourse.run_plan_with_recourse expects.
    Mirrors recourse._sol_to_plan_updates with start_stop = 0."""
    plan = [dict(y=0, break_type=None, rest_type=None) for _ in range(N + 1)]
    for s in sol:
        i = int(s["i"])
        brk = ("b45" if s.get("b45") else "b15" if s.get("b15") else
               "b30" if s.get("b30") else None)
        rst = ("r1" if s.get("rho1") else "r2" if s.get("rho2") else None)
        plan[i] = dict(y=int(s.get("y", 0)), break_type=brk, rest_type=rst)
    return plan


def _plan_from_solution_file(path: str, N: int) -> list[dict]:
    """Extract a committed plan from a saved solution json (2SP runs).  Looks
    for a 'plan' list first, then falls back to a MILP-style 'sol' list."""
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    for key in ("plan", "committed_plan"):
        if isinstance(data.get(key), list):
            return [dict(y=int(e.get("y", 0) or 0),
                         break_type=e.get("break_type"),
                         rest_type=e.get("rest_type"))
                    for e in data[key]]
    if isinstance(data.get("sol"), list):
        return _sol_to_plan(data["sol"], N)
    raise SystemExit(f"--plan-from {path}: no 'plan'/'committed_plan'/'sol' "
                     f"list found")


def _scen_lists(scen: dict, N: int) -> tuple[list, list]:
    D = [float(scen["D"][i]) for i in range(N)]
    E = [float(scen["E"][i]) for i in range(N)]
    return D, E


def _execute_plan(full_data, plan, D, E, cv, label) -> dict:
    """Run a committed plan under one scenario with duration recourse and
    return arrival / penalised objective / feasibility."""
    vehicle, tracker, events = run_plan_with_recourse(
        full_data, plan, D, E, method_name=label,
        log_fn=lambda *_: None, cv=cv, supervised=False, verbose=False)
    beta     = float(full_data.get("beta", 2.0))
    misses   = len(getattr(vehicle, "tw_misses", {}) or {})
    feasible = (not vehicle.violations) and (not events["plan_violations"])
    return dict(arrival=float(vehicle.t_arr),
                obj=float(vehicle.t_arr) + beta * misses,
                tw_misses=misses,
                feasible=bool(feasible),
                repairs=len(events["repairs"]),
                plan_violations=len(events["plan_violations"]))


def _mean(vals):
    vals = [v for v in vals if v is not None and np.isfinite(v)]
    return float(np.mean(vals)) if vals else None


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    ap = argparse.ArgumentParser(description="VSS/EVPI harness (Section 8.5)")
    ap.add_argument("instance", help="Instance json (instances/*.json)")
    ap.add_argument("--out", required=True, help="Output json path")
    ap.add_argument("--n-scenarios", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42,
                    help="Common-random-number seed for the scenario draw")
    ap.add_argument("--time-limit", type=int, default=600,
                    help="Per-MILP time limit (s) for the WS and EV solves")
    ap.add_argument("--mip-gap", type=float, default=0.005)
    ap.add_argument("--plan-from", default=None,
                    help="Solution json of the base-case 2SP run; enables the "
                         "RP leg under the same scenarios")
    args = ap.parse_args()

    full_data, _D_real, _E_real, cv_file = load_instance_json(args.instance)
    N = full_data["N"]

    scens = generate_scenarios(full_data, 0, N,
                               n_scenarios=args.n_scenarios,
                               cv=cv_file, seed=args.seed)

    # ── EV plan: deterministic MILP at nominal travel times ──────────────────
    D_nom = [float(full_data["D"][i]) for i in range(N)]
    print(f"[vss] EV solve (nominal MILP), N={N} ...")
    ev = oracle_solve(full_data, D_nom, sim_results=None,
                      time_limit=args.time_limit, mip_gap=args.mip_gap,
                      tee=False, verbose=False)
    if not ev["feasible"]:
        raise SystemExit("[vss] nominal MILP infeasible/unsolved — cannot "
                         "form the EV plan")
    ev_plan = _sol_to_plan(ev["sol"], N)

    rp_plan = (_plan_from_solution_file(args.plan_from, N)
               if args.plan_from else None)

    # ── Per-scenario legs ────────────────────────────────────────────────────
    rows = []
    for m, scen in enumerate(scens):
        D, E = _scen_lists(scen, N)

        print(f"[vss] scenario {m + 1}/{len(scens)}: WS solve ...")
        ws = oracle_solve(full_data, D, sim_results=None,
                          time_limit=args.time_limit, mip_gap=args.mip_gap,
                          tee=False, verbose=False)

        row = dict(
            scenario=m,
            ws_obj=(float(ws["obj"]) if ws["feasible"] else None),
            ws_gap=(float(ws["gap"]) if ws["feasible"] else None),
            eev=_execute_plan(full_data, ev_plan, D, E, cv_file, "EEV"),
        )
        if rp_plan is not None:
            row["rp"] = _execute_plan(full_data, rp_plan, D, E, cv_file, "RP")
        rows.append(row)

    # ── Aggregate ────────────────────────────────────────────────────────────
    def _leg(key, field):
        return [r[key][field] if r.get(key) and r[key]["feasible"] else None
                for r in rows]

    ws_mean  = _mean([r["ws_obj"] for r in rows])
    eev_mean = _mean(_leg("eev", "obj"))
    rp_mean  = _mean(_leg("rp", "obj")) if rp_plan is not None else None

    summary = dict(
        instance=os.path.basename(args.instance),
        n_scenarios=len(scens), crn_seed=args.seed,
        ws_mean=ws_mean,
        eev_mean=eev_mean,
        eev_infeasible=sum(1 for r in rows if not r["eev"]["feasible"]),
        rp_mean=rp_mean,
        rp_infeasible=(sum(1 for r in rows
                           if r.get("rp") and not r["rp"]["feasible"])
                       if rp_plan is not None else None),
        vss=(eev_mean - rp_mean
             if (eev_mean is not None and rp_mean is not None) else None),
        evpi=(rp_mean - ws_mean
              if (rp_mean is not None and ws_mean is not None) else None),
        evpi_vs_eev=(eev_mean - ws_mean
                     if (eev_mean is not None and ws_mean is not None)
                     else None),
    )

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(dict(summary=summary, scenarios=rows), fh, indent=2)

    print(f"[vss] {summary['instance']}: "
          f"WS={ws_mean and round(ws_mean, 3)}  "
          f"EEV={eev_mean and round(eev_mean, 3)}  "
          f"RP={rp_mean and round(rp_mean, 3)}  "
          f"VSS={summary['vss'] and round(summary['vss'], 3)}  "
          f"EVPI={summary['evpi'] and round(summary['evpi'], 3)}")
    print(f"[vss] wrote {args.out}")


if __name__ == "__main__":
    main()
