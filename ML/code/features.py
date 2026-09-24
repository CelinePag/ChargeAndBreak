"""
features.py — the ONE state/action description, used by training and by driving
==============================================================================
Every feature here is computable from (full_data, BEHDV state, stop) alone.
Nothing reads D_real/E_real for a leg that has not been driven yet: the
simulator reveals the travel-time multiplier only after departure, so a
feature that peeked would score brilliantly offline and collapse in the loop.

This module is deliberately the single source of the feature vector.  The
extractor rebuilds each stored state and calls these functions; the deployed
policy calls the same functions at the same point of the loop, passing a live
BEHDV.  Both objects are read only through the attribute names StaticState
declares, so the feature code cannot tell them apart -- which is what makes
train/serve skew impossible by construction rather than by discipline.

Design notes
------------
* SLACKS, NOT LEVELS.  The regulation is a set of thresholds (4.5 / 9 / 13 /
  15 h).  Feeding cd_slack = 4.5 - cd instead of cd puts every threshold at
  zero, which a tree splits exactly and which also happens to be where a ReLU
  hinges -- so the same features serve a neural student if one is ever built.
* TYPED LOOKAHEAD, NOT RAW STOPS.  The teacher's window is a median of 41
  stops, but about two thirds of those are identical laybys.  Describing the
  next few charging stations and customers carries the same information in a
  tenth of the columns.
* Hard rules are NOT features to be learned: must_charge / must_reset_cd /
  must_rest come from supervisor.compute_flags, the same call greedy and the
  look-ahead pruner make.
"""
from __future__ import annotations

import numpy as np

from src.simulation.BEHDV import _charging_time_needed, _energy_after_charging
from src.simulation.supervisor import compute_flags
from src.simulation.Simulation import find_horizon_end_stop

N_CS_AHEAD = 3        # how many charging stations the lookahead describes
N_CUST_AHEAD = 2      # how many customers the lookahead describes

# The complete, fixed action vocabulary: y in {0,1} x (nothing | break | rest).
ACTION_VOCAB = [
    "y0_go", "y0_b45", "y0_b15", "y0_b30", "y0_r1", "y0_r2",
    "y1_go", "y1_b45", "y1_b15", "y1_b30", "y1_r1", "y1_r2",
]


def action_key(y, brk, rst) -> str:
    b = "none" if str(brk).lower() in ("0", "none", "-", "") else str(brk).lower()
    r = "none" if str(rst).lower() in ("0", "none", "-", "") else str(rst).lower()
    tail = b if b != "none" else (r if r != "none" else "go")
    return f"y{int(y)}_{tail}"


def _get(d, i, default=0.0):
    """Route dicts come back from JSON with string keys; replays use ints."""
    if i in d:
        return d[i]
    return d.get(str(i), default)


class Precomp:
    """Per-instance route geometry that never changes during a run.

    Computed once per instance and reused at every stop, because the
    alternative -- walking the route inside every feature call -- turns a
    74k-decision extraction into something that takes hours.
    """

    def __init__(self, fd: dict):
        N = int(fd["N"])
        self.N = N
        self.K = set(int(k) for k in fd["K"])
        self.C = set(int(c) for c in fd["C"])
        self.L = set(int(x) for x in fd.get("L", []))
        self.ferry = set(int(k) for k in (fd.get("ferry") or {}))
        D, E = fd["D"], fd["E"]
        self.D = np.array([float(_get(D, i)) for i in range(N + 1)])
        self.E = np.array([float(_get(E, i)) for i in range(N + 1)])
        # cumulative nominal drive time / energy from the origin
        self.cumD = np.concatenate([[0.0], np.cumsum(self.D)])[: N + 1]
        self.cumE = np.concatenate([[0.0], np.cumsum(self.E)])[: N + 1]
        self.totD = float(self.D[:N].sum())
        self.totE = float(self.E[:N].sum())
        # next_cs[i] = smallest CS index > i (N if none); likewise customers
        self.next_cs = np.full(N + 2, N, dtype=int)
        nc = N
        for i in range(N, -1, -1):
            self.next_cs[i] = nc
            if i in self.K:
                nc = i
        self.cs_list = np.array(sorted(self.K), dtype=int)
        self.cust_list = np.array(sorted(self.C), dtype=int)
        idx = np.arange(N + 2)
        self.n_cs_after = np.searchsorted(self.cs_list, idx, "right")
        self.n_cust_after = np.searchsorted(self.cust_list, idx, "right")

    def cs_ahead(self, stop: int, k: int) -> int:
        """Index of the k-th charging station strictly after stop (-1 if none)."""
        j = int(self.n_cs_after[stop]) + k
        return int(self.cs_list[j]) if j < len(self.cs_list) else -1

    def cust_ahead(self, stop: int, k: int) -> int:
        j = int(self.n_cust_after[stop]) + k
        return int(self.cust_list[j]) if j < len(self.cust_list) else -1



class StaticState:
    """A state snapshot that satisfies the same attribute protocol as BEHDV.

    The extractor rebuilds states from stored runs; the policy passes a live
    BEHDV.  Both are read only through these attribute names, so the feature
    code cannot tell them apart -- which is the point: one code path computes
    the training inputs and the serving inputs.
    """

    __slots__ = ("t_arr", "e_arr", "cd", "sd", "sw", "h", "phi",
                 "rho2_used", "ext_shift_used", "stop")

    def __init__(self, t_arr, e_arr, cd, sd, sw, h, phi, rho2_used,
                 ext_shift_used, stop):
        self.t_arr = t_arr
        self.e_arr = e_arr
        self.cd = cd
        self.sd = sd
        self.sw = sw
        self.h = h
        self.phi = phi
        self.rho2_used = rho2_used
        self.ext_shift_used = ext_shift_used
        self.stop = stop

def _walk_forward(fd: dict, pre: Precomp, stop: int, target: int,
                  cd: float, sd: float):
    """Nominal walk from stop to target, applying the HoS reset rules.

    Returns (drive_h, added_dwell_h, n_breaks, n_rests, legs_to_break,
    legs_to_rest).  This is the same accounting the look-ahead's own horizon
    routine does, reused here as a feature generator so the student sees the
    forcing structure the teacher conditioned on.
    """
    Tcd = float(fd["Tdrv_cons"])
    Tsd = float(fd["Tdrv_sh1"])
    Tb45 = float(fd["Tb45"])
    Tr1 = float(fd["Tr1"])
    drive = added = 0.0
    nb = nr = 0
    legs_to_break = legs_to_rest = -1
    n = 0
    for i in range(stop, min(target, pre.N)):
        d = float(pre.D[i])
        if cd + d > Tcd + 1e-9:
            if legs_to_break < 0:
                legs_to_break = n
            added += Tb45
            nb += 1
            cd = 0.0
        if sd + d > Tsd + 1e-9:
            if legs_to_rest < 0:
                legs_to_rest = n
            added += Tr1
            nr += 1
            cd = sd = 0.0
        cd += d
        sd += d
        drive += d
        n += 1
    return drive, added, nb, nr, legs_to_break, legs_to_rest


def state_features(fd: dict, pre: Precomp, stop: int, state,
                   cv: float, guard_q):
    """Everything about the situation that does not depend on the action.

    Returns (features dict, supervisor flags dict).
    """
    N = pre.N
    Ecap = float(fd["Ecap"])
    Emin = float(fd["Emin"])
    e = float(state.e_arr)
    usable = e - Emin
    t = float(state.t_arr)
    t0 = float(fd.get("T_START", 8.0))
    h = float(getattr(state, "h", 0.0))

    flags = compute_flags(fd, stop, state, cv, guard_q)
    D_next = float(pre.D[stop]) if stop < N else 0.0
    E_next = float(pre.E[stop]) if stop < N else 0.0

    f = {}
    # -- A. raw state --------------------------------------------------------
    f["t_elapsed"] = t - t0
    f["tod"] = t % 24.0
    f["soc_kwh"] = e
    f["soc_frac"] = e / Ecap
    f["usable_kwh"] = usable
    f["cd"] = float(state.cd)
    f["sd"] = float(state.sd)
    f["sw"] = float(state.sw)
    f["spread_h"] = h
    f["phi"] = float(state.phi)
    f["rho2_used"] = float(state.rho2_used)
    f["ext_shift_used"] = float(getattr(state, "ext_shift_used", 0))

    # -- B. slack to every regulatory boundary (thresholds moved to zero) -----
    f["cd_slack"] = float(fd["Tdrv_cons"]) - state.cd
    f["sd_slack"] = float(flags["sd_limit"]) - state.sd
    f["sw_slack"] = float(fd["Twrk_cons2"]) - state.sw
    f["spread_slack1"] = float(fd.get("Tspr1", 13.0)) - h
    f["spread_slack2"] = float(fd.get("Tspr2", 15.0)) - h
    f["rho2_left"] = float(fd.get("rho_bar", 3)) - state.rho2_used
    f["ext_left"] = float(fd.get("ext_bar", 2)) - getattr(state, "ext_shift_used", 0)
    f["D_next"] = D_next
    f["E_next"] = E_next
    f["D_next_wc"] = float(flags["D_next_wc"])
    f["cd_slack_after"] = f["cd_slack"] - flags["D_next_wc"]
    f["sd_slack_after"] = f["sd_slack"] - flags["D_next_wc"]
    f["spread_slack2_after"] = f["spread_slack2"] - flags["D_next_wc"]
    f["must_charge"] = float(flags["must_charge"])
    f["must_reset_cd"] = float(flags["must_reset_cd"])
    f["must_rest"] = float(flags["must_rest"])

    # -- C. energy and reachability ------------------------------------------
    e_needed = float(flags["e_needed"])
    f["e_needed_next_cs_wc"] = e_needed
    f["e_margin_next_cs_wc"] = usable - e_needed
    nxt = int(pre.next_cs[stop])
    f["stops_to_next_cs"] = float(nxt - stop)
    f["drive_to_next_cs"] = float(pre.cumD[min(nxt, N)] - pre.cumD[stop])
    f["energy_to_next_cs"] = float(pre.cumE[min(nxt, N)] - pre.cumE[stop])
    f["soc_at_next_cs_frac"] = (e - f["energy_to_next_cs"]) / Ecap
    f["charge_time_to_full"] = _charging_time_needed(e, fd)
    f["charge_rate_now_kw"] = (_energy_after_charging(e, 0.1, fd) - e) / 0.1
    f["e_nom_margin_next_cs"] = usable - f["energy_to_next_cs"]

    # -- D. the node we are standing on --------------------------------------
    is_cs = stop in pre.K
    is_cu = stop in pre.C
    f["is_cs"] = float(is_cs)
    f["is_cust"] = float(is_cu)
    f["is_layby"] = float(stop in pre.L)
    f["is_ferry"] = float(stop in pre.ferry)
    f["queue_here"] = float(_get(fd["Q"], stop)) if is_cs else 0.0
    f["service_here"] = float(_get(fd["S"], stop)) if is_cu else 0.0
    wha = _get(fd.get("Wha", {}), stop, None)
    whf = _get(fd.get("Whf", {}), stop, None)
    f["tw_early_slack"] = float(wha) - t if (is_cu and wha is not None) else 0.0
    f["tw_late_slack"] = float(whf) - t if (is_cu and whf is not None) else 0.0

    # -- E. typed lookahead ---------------------------------------------------
    for k in range(N_CS_AHEAD):
        j = pre.cs_ahead(stop, k)
        p = f"cs{k + 1}_"
        if j < 0:
            f[p + "drive"] = -1.0
            f[p + "soc_frac"] = -1.0
            f[p + "queue"] = -1.0
            f[p + "reach"] = 0.0
        else:
            dd = float(pre.cumD[j] - pre.cumD[stop])
            ee = float(pre.cumE[j] - pre.cumE[stop])
            f[p + "drive"] = dd
            f[p + "soc_frac"] = (e - ee) / Ecap
            f[p + "queue"] = float(_get(fd["Q"], j))
            f[p + "reach"] = float(usable - ee > 0)
    for k in range(N_CUST_AHEAD):
        j = pre.cust_ahead(stop, k)
        p = f"cu{k + 1}_"
        if j < 0:
            f[p + "drive"] = -1.0
            f[p + "open_slack"] = 0.0
            f[p + "close_slack"] = 0.0
            f[p + "rests_between"] = 0.0
        else:
            dd = float(pre.cumD[j] - pre.cumD[stop])
            dr, ad, nb, nr, _, _ = _walk_forward(fd, pre, stop, j, state.cd, state.sd)
            eta = t + dr + ad
            wa = _get(fd.get("Wha", {}), j, None)
            wf = _get(fd.get("Whf", {}), j, None)
            f[p + "drive"] = dd
            f[p + "open_slack"] = (float(wa) - eta) if wa is not None else 0.0
            f[p + "close_slack"] = (float(wf) - eta) if wf is not None else 0.0
            f[p + "rests_between"] = float(nr)

    # -- F. horizon summary (the teacher's own window) ------------------------
    hz_end, hz_rests = find_horizon_end_stop(
        fd, stop, float(fd.get("_horizon_h", 24.0)), state)
    f["hz_stops"] = float(hz_end - stop)
    f["hz_drive"] = float(pre.cumD[min(hz_end, N)] - pre.cumD[stop])
    f["hz_n_rests"] = float(hz_rests)
    _, _, nb_h, _, l_brk, l_rest = _walk_forward(
        fd, pre, stop, min(hz_end, N), state.cd, state.sd)
    f["hz_n_breaks"] = float(nb_h)
    f["legs_to_forced_break"] = float(l_brk if l_brk >= 0 else 99)
    f["legs_to_forced_rest"] = float(l_rest if l_rest >= 0 else 99)

    # -- G. whole-route remainder ---------------------------------------------
    f["stops_left"] = float(N - stop)
    f["drive_left"] = float(pre.totD - pre.cumD[stop])
    f["energy_left"] = float(pre.totE - pre.cumE[stop])
    f["cs_left"] = float(len(pre.cs_list) - pre.n_cs_after[stop])
    f["cust_left"] = float(len(pre.cust_list) - pre.n_cust_after[stop])
    f["route_frac_done"] = float(pre.cumD[stop] / max(pre.totD, 1e-9))

    return f, flags


def action_features(fd: dict, pre: Precomp, stop: int, state, action: dict,
                    sf: dict) -> dict:
    """The action half of the (state, action) row."""
    y = int(action.get("y", 0))
    brk = action.get("break_type")
    rst = action.get("rest_type")
    brk = None if str(brk).lower() in ("0", "none", "-", "") else str(brk).lower()
    rst = None if str(rst).lower() in ("0", "none", "-", "") else str(rst).lower()

    bmin = {"b45": float(fd["Tb45"]), "b15": float(fd["Tb15"]),
            "b30": float(fd["Tb30"])}.get(brk, 0.0)
    rmin = float(fd["Tr1"]) if rst == "r1" else float(fd["Tr2"]) if rst == "r2" else 0.0

    # If we charge, how long would a charge-to-full take, and would the
    # declared break disappear inside it?  That concurrency is the coupling
    # the whole problem turns on, so it is an explicit input rather than an
    # inference the model has to make.
    tauc_full = sf["charge_time_to_full"] if y else 0.0

    # -- the action's effect on the NEXT CUSTOMER'S WINDOW -------------------
    # The teacher's objective already prices a window miss (beta * delta), so
    # the regret labels do too -- yet every student misses ~2x as many windows
    # as the teacher.  The reason is representational: `cu1_close_slack` is a
    # STATE feature and the dwell an ACTION feature, so the model has to form
    # their difference itself, across a split boundary, from two columns whose
    # interaction it is never shown.  The quantity that actually decides a
    # miss is the slack LEFT AFTER this action, so give it directly.
    dwell = (max(tauc_full, bmin) if y else bmin) + rmin
    close_slack = sf.get("cu1_close_slack", 0.0)
    open_slack = sf.get("cu1_open_slack", 0.0)
    has_cu = sf.get("cu1_drive", -1.0) >= 0.0

    return {
        "a_y": float(y),
        "a_cu1_slack_after": (close_slack - dwell) if has_cu else 99.0,
        "a_cu1_late": float(has_cu and (close_slack - dwell) < 0.0),
        "a_cu1_early": float(has_cu and (open_slack - dwell) > 0.0),
        "a_dwell_vs_cu1_drive": (dwell / max(sf.get("cu1_drive", 1.0), 0.05)
                                 if has_cu else 0.0),
        "a_b45": float(brk == "b45"),
        "a_b15": float(brk == "b15"),
        "a_b30": float(brk == "b30"),
        "a_r1": float(rst == "r1"),
        "a_r2": float(rst == "r2"),
        "a_break_min": bmin,
        "a_rest_min": rmin,
        "a_dwell_min": bmin + rmin,
        "a_resets_cd": float(brk in ("b45", "b30") or rst in ("r1", "r2")),
        "a_resets_sd": float(rst in ("r1", "r2")),
        "a_tauc_full": tauc_full,
        "a_break_absorbable": float(y == 1 and bmin > 0 and tauc_full >= bmin),
        "a_residual_break": max(0.0, bmin - tauc_full),
    }


def key_to_action(key: str) -> dict:
    """'y1_b45' -> {'y': 1, 'break_type': 'b45', 'rest_type': None}."""
    ys, tail = key.split("_", 1)
    y = int(ys[1])
    brk = tail if tail.startswith("b") else None
    rst = tail if tail.startswith("r") else None
    return dict(y=y, break_type=brk, rest_type=rst)
