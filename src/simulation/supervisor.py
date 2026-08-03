"""
supervisor.py — S1: shared one-step safety supervisor
======================================================
A single feasibility guard applied identically to EVERY policy (LA, greedy,
RO, 2SP) before the vehicle departs a stop:

    before departing stop i, if the worst-case realization of leg i -> i+1
    (over the known support of the travel-time distribution) would violate a
    driving, spread, or SOC constraint before the next stopping opportunity,
    the supervisor overrides the action with the cheapest preventive action
    (parallel break upgrade < break < charge-to-cover < rest) at stop i.

Runs in two modes (set per run):
    raw (default)  — supervisor off, exposing each architecture's intrinsic
                     feasibility risk (violations recorded by BEHDV, see S2);
                     an infeasible run is recorded as-is, never rescued.
    supervised     — overrides are applied and logged as interventions
                     (opt-in via supervised=True / --supervised).

This module is ALSO the single source of truth for the rolling-horizon
action pruning (RH2): Simulation._prune_actions calls compute_flags() here,
so the pruning rule and the supervisor are literally the same check.

Quantile parameter (RH2)
------------------------
`quantile` is a probability level on the bounded shifted-lognormal
multiplier xi (settings.GUARD_QUANTILE sets the project default):
  None — guard disabled: the checks run at NOMINAL travel time / energy
         (xi = 1 on both sides); no uncertainty margin at all.
  0.95 — guard uses xi_quantile(0.95) (slow/time side) and
         xi_quantile(0.05) (fast/energy side); the residual violation risk
         alpha = 5% per leg must be reported alongside the results.
  1.0  — the full support corners [XI_MIN, XI_MAX]: the guard removes no
         action that is feasible under all realizations (the support is
         hard, so no residual risk).

Import chain
------------
  supervisor.py -> settings (no local imports) — usable by every layer.
"""

from __future__ import annotations

from src.settings import V_NOM, ecr, xi_quantile, GUARD_QUANTILE


# ══════════════════════════════════════════════════════════════════════════════
# WORST-CASE HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def worst_case_energy_to_next_cs(full_data: dict, stop: int,
                                 cv: float,
                                 quantile: float | None = GUARD_QUANTILE
                                 ) -> float:
    """
    Worst-case energy (kWh) to reach the next CS stop (or destination) from
    `stop`, driving non-stop.  The energy worst case is the FASTEST
    realization (ECR is increasing and convex in speed), consistent with the
    robust counterpart: xi at the low quantile, bounded by the speed-limiter
    corner XI_MIN.  quantile=None → nominal energies (no margin).
    """
    N     = full_data["N"]
    K_set = set(full_data["K"])
    xi_lo = 1.0 if quantile is None else xi_quantile(1.0 - quantile, cv)

    e_needed, cur = 0.0, stop
    while cur < N:
        d_nom = full_data["D"].get(cur, 0.0)
        L_km  = full_data.get("km", {}).get(cur, d_nom * V_NOM)
        d_min = max(d_nom * xi_lo, 1e-9)
        v_wc  = L_km / d_min
        e_needed += L_km * ecr(v_wc)
        cur += 1
        if cur in K_set or cur == N:
            break
    return e_needed


def _action_min_dwell(full_data: dict, stop: int, action: dict) -> float:
    """
    Minimum on-duty dwell (h) implied by `action` at `stop` (excludes rests):
    service time, queue, minimum break duration.  Used to estimate the spread
    consumption o(a) of an action; charging duration is not known before the
    solve and is conservatively omitted (it only makes the guard tighter to
    include, never looser to omit for the checks that matter, because a longer
    dwell delays the leg but does not change its worst case).
    """
    C_set = set(full_data["C"])
    K_set = set(full_data["K"])
    brk   = action.get("break_type")
    dwell = 0.0
    if stop in C_set:
        dwell += full_data["S"].get(stop, 0.0)
    if stop in K_set and action.get("y", 0):
        dwell += full_data["Q"].get(stop, 0.0)
    dwell += {"b45": full_data["Tb45"], "b15": full_data["Tb15"],
              "b30": full_data["Tb30"]}.get(brk, 0.0)
    return dwell


# ══════════════════════════════════════════════════════════════════════════════
# FLAGS — the one-step feasibility checks (single source of truth)
# ══════════════════════════════════════════════════════════════════════════════

def compute_flags(full_data: dict, stop: int, state,
                  cv: float, quantile: float | None = GUARD_QUANTILE) -> dict:
    """
    Evaluate the one-step feasibility checks at `stop` for vehicle `state`.

    Checks (S1 pseudocode):
      must_charge   : usable SOC does not cover the worst-case energy demand
                      to the next charging opportunity          (only at CS)
      must_reset_cd : cd + D_wc > 4.5 h
      must_rest     : sd + D_wc > effective shift-driving limit
                      (10 h while the extension budget lasts, else 9 h), OR
                      spread h + D_wc > 15 h (next rest type unknown -> 15 h)

    SIM3: the spread check here uses o(a) = 0 (action-independent lower
    bound).  The action-specific pre-rest spread check h + o(a) + D_wc,
    applied identically at ALL stop types, lives in action_passes() /
    _spread_with_dwell_fails() — the v2 customer-only h_arrival branch is
    gone with the single rest-last convention.

    Returns a dict with the three booleans plus the diagnostic quantities.
    """
    N     = full_data["N"]
    K_set = set(full_data["K"])
    if stop >= N:
        return dict(must_charge=False, must_reset_cd=False, must_rest=False,
                    D_next_wc=0.0, e_needed=0.0, sd_limit=full_data["Tdrv_sh1"])

    xi_hi     = 1.0 if quantile is None else xi_quantile(quantile, cv)
    D_next_wc = full_data["D"].get(stop, 0.0) * xi_hi

    e_needed = worst_case_energy_to_next_cs(full_data, stop, cv, quantile)
    must_charge = (stop in K_set
                   and state.e_arr - e_needed < full_data["Emin"])

    must_reset_cd = state.cd + D_next_wc > full_data["Tdrv_cons"]

    ext_bar  = int(full_data.get("ext_bar", 2))
    ext_used = getattr(state, "ext_shift_used", 0)
    sd_limit = (full_data.get("Tdrv_sh2", 10.0) if ext_used < ext_bar
                else full_data["Tdrv_sh1"])
    Tspr2    = full_data.get("Tspr2", 15.0)
    h_state  = getattr(state, "h", 0.0)

    must_rest = (state.sd + D_next_wc > sd_limit
                 or h_state + D_next_wc > Tspr2)

    return dict(
        must_charge=must_charge, must_reset_cd=must_reset_cd,
        must_rest=must_rest,
        D_next_wc=D_next_wc, e_needed=e_needed, sd_limit=sd_limit,
        h_state=h_state, Tspr2=Tspr2,
    )


def _spread_with_dwell_fails(full_data: dict, stop: int, action: dict,
                             flags: dict) -> bool:
    """
    SIM3 — action-specific pre-rest spread check, identical at ALL stop
    types: with the single rest-last convention, the elapsed time entering
    the next leg is h_arrival + o(a), where o(a) is the minimum on-duty
    dwell implied by the action (service, queue, minimum break).  Fails
    when h + o(a) + D_wc would breach the 15 h spread ceiling and no rest
    is declared to reset it.
    """
    if action.get("rest_type") in ("r1", "r2"):
        return False
    o_a = _action_min_dwell(full_data, stop, action)
    return (flags["h_state"] + o_a + flags["D_next_wc"]
            > flags["Tspr2"] + 1e-9)


def action_passes(full_data: dict, stop: int, state, action: dict,
                  flags: dict) -> bool:
    """True iff `action` clears every raised flag (used for RH pruning)."""
    y   = action.get("y", 0)
    brk = action.get("break_type")
    rst = action.get("rest_type")
    has_reset = brk in ("b45", "b30") or rst in ("r1", "r2")
    has_rest  = rst in ("r1", "r2")
    K_set     = set(full_data["K"])

    if flags["must_charge"] and stop in K_set and not y:
        return False
    if flags["must_reset_cd"] and not has_reset and not has_rest:
        return False
    if flags["must_rest"] and not has_rest:
        return False
    if _spread_with_dwell_fails(full_data, stop, action, flags):
        return False
    return True


# ══════════════════════════════════════════════════════════════════════════════
# SUPERVISOR — override with the cheapest preventive action
# ══════════════════════════════════════════════════════════════════════════════

def supervise_action(full_data: dict, stop: int, state, action: dict,
                     cv: float, quantile: float | None = GUARD_QUANTILE
                     ) -> tuple:
    """
    S1 — one-step feasibility guard.

    Checks the proposed `action` against the worst-case realization of the
    next leg and, if a check fails, overrides it with the CHEAPEST preventive
    addition, in the order:

        parallel break upgrade  <  break  <  charge-to-cover  <  rest

    (i.e. a failing driving check is first fixed by upgrading an
    already-planned charge into a parallel break, then by inserting a break;
    a failing SOC check adds a charge; a failing spread/shift check adds the
    appropriate rest — reduced if the budget allows, regular otherwise.)

    Returns
    -------
    (action, intervention) — `action` possibly modified (a NEW dict when
    modified); `intervention` is None or a dict {stop, checks, before, after}.
    """
    N = full_data["N"]
    if stop >= N or stop == 0:
        return action, None

    flags = compute_flags(full_data, stop, state, cv, quantile)
    if action_passes(full_data, stop, state, action, flags):
        return action, None

    K_set  = set(full_data["K"])
    is_CS  = stop in K_set
    before = dict(action)
    new    = dict(action)
    fixed  = []

    # 1) SOC check: charge-to-cover
    if flags["must_charge"] and is_CS and not new.get("y", 0):
        new["y"] = 1
        fixed.append("charge")

    # 2) Driving check: cheapest qualifying break
    _has_reset = (new.get("break_type") in ("b45", "b30")
                  or new.get("rest_type") in ("r1", "r2"))
    if flags["must_reset_cd"] and not _has_reset:
        # 8.3 no-split axis: with the Art. 7 split unavailable phi stays 0 and
        # the 45' block is the only break that resets consecutive driving.
        _phi = (getattr(state, "phi", 0)
                if full_data.get("allow_split", True) else 0)
        brk = "b30" if _phi == 1 else "b45"
        # parallel break upgrade: at a CS with a charge planned, the break
        # runs inside the charging window at (near) zero marginal time cost.
        new["break_type"] = brk
        fixed.append("break" if not new.get("y", 0) else "parallel-break")

    # 3) Shift-driving / spread check: rest (reduced if budget allows).
    # SIM3: the spread check includes the action's own dwell o(a) —
    # h + o(a) + D_wc against the 15 h ceiling — at every stop type.
    if ((flags["must_rest"]
         or _spread_with_dwell_fails(full_data, stop, new, flags))
            and new.get("rest_type") not in ("r1", "r2")):
        rho_bar = int(full_data.get("rho_bar", 3))
        new["rest_type"]  = ("r2" if getattr(state, "rho2_used", 0) < rho_bar
                             else "r1")
        new["break_type"] = None   # rest supersedes the break (resets cd too)
        fixed.append("rest")

    if not fixed:
        return action, None

    intervention = dict(
        stop=stop,
        checks=[k for k in ("must_charge", "must_reset_cd", "must_rest")
                if flags[k]],
        fixes=fixed, before=before, after=dict(new),
    )
    return new, intervention
