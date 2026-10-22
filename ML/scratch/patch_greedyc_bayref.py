"""One-off patch: add the 'bayref' rule set (variant CORR3, new default) to
greedy_corrected.py.  Each replacement must match exactly once."""
import io

p = "src/methods/greedy_corrected.py"
s = io.open(p, encoding="utf-8").read()

reps = [
# docstring
(r'''      With these rules no decision uses the charging time, so for the same
      sequence of activities every dwell is non-increasing in charger power:
      max(tauc, b) for a charge with a break, tauc + M_seq before a rest.  The
      only remaining channel is the spread rule, which reads the elapsed duty
      time (shorter at higher power) to decide where to rest.
''',
r'''      With these rules no decision uses the charging time, so for the same
      sequence of activities every dwell is non-increasing in charger power:
      max(tauc, b) for a charge with a break, tauc + M_seq before a rest.  The
      only remaining channel is the spread rule, which reads the elapsed duty
      time (shorter at higher power) to decide where to rest.
  rules="bayref" (variant CORR3, the default) - "bay" plus one planning rule:
        B4  The duty-time (spread) rules plan with the charging time of a
            PLAN_POWER_KW = 350 kW point (the base case, AFIR's per-point
            minimum) or with the actual one if the charger is slower.  The
            greedy tracks the duty time it WOULD have accumulated at that
            speed (actual elapsed time + the dwell it saved since the last
            rest) and uses it for every rest decision; the simulator still
            executes the actual, shorter stops.
      Consequence: at any charger power >= 350 kW the greedy takes exactly the
      activities it takes at 350 kW (energy is identical because it always
      charges to full; driving is identical), and every stop lasts at most as
      long.  Route duration therefore cannot increase from 350 to 700 to
      1000 kW, for any battery size.  At 350 kW and below the planning curve
      IS the actual curve, so results equal "bay" there.  Legality is kept:
      the actual duty time never exceeds the planned one.
'''),
('''VARIANTS      = {"milp": "CORR", "bay": "CORR2"}
DEFAULT_RULES = "bay"
VARIANT       = VARIANTS[DEFAULT_RULES]
''',
'''VARIANTS      = {"milp": "CORR", "bay": "CORR2", "bayref": "CORR3"}
DEFAULT_RULES = "bayref"
VARIANT       = VARIANTS[DEFAULT_RULES]
# Rule sets that let the truck stay at the bay to complete a break (B1-B3).
BAY_RULES     = ("bay", "bayref")
# B4: charger power the duty-time rules plan with (bayref only).
from src.settings import CHARGER_POWER_BASE_KW as PLAN_POWER_KW, charging_curve


def _planning_data(full_data: dict) -> dict:
    """full_data whose charging curve is the SLOWER of the actual one and the
    PLAN_POWER_KW one, breakpoint by breakpoint (the curves never cross:
    charging time is non-increasing in power on every segment)."""
    ref = charging_curve(PLAN_POWER_KW, full_data["Ecap"])

    def _slower(curve):
        return {r: max(float(t), float(ref[int(r)])) for r, t in curve.items()}

    pd = dict(full_data)
    pd["Tbar"] = _slower(full_data["Tbar"])
    if full_data.get("TbarK"):
        pd["TbarK"] = {k: _slower(c) for k, c in full_data["TbarK"].items()}
    return pd


class _PlanState:
    """The vehicle as the bayref planner sees it: every attribute of the real
    BEHDV except the duty time h, replaced by the planned duty time."""
    def __init__(self, vehicle, h_plan: float):
        self._v = vehicle
        self.h = h_plan

    def __getattr__(self, name):
        return getattr(self._v, name)
'''),
# durations: bay branch for both bay rule sets
('''            elif rules == "bay":
                # B1: stay at the bay until the break is complete''',
'''            elif rules in BAY_RULES:
                # B1: stay at the bay until the break is complete'''),
# decision signature: plan_data
('''                              verbose: bool = False,
                              rules: str = DEFAULT_RULES) -> tuple[dict, str]:''',
'''                              verbose: bool = False,
                              rules: str = DEFAULT_RULES,
                              plan_data: dict | None = None) -> tuple[dict, str]:'''),
# decision: planning charge time
('''    # C3: a charge that would end before T_min is not possible at all.
    tauc_full  = (_charging_time_needed(state.e_arr, full_data, stop=stop_global)
                  if is_CS else 0.0)''',
'''    # Planning data: the actual instance, or (bayref, B4) its slower-of-actual-
    # and-350-kW charging curve.  Every charging time a DECISION reads comes
    # from pd; execution (run loop) always uses the actual curve.
    pd = plan_data if plan_data is not None else full_data

    # C3: a charge that would end before T_min is not possible at all.
    tauc_full  = (_charging_time_needed(state.e_arr, pd, stop=stop_global)
                  if is_CS else 0.0)'''),
('''        if can_charge and not avoid_queue and rules == "bay":''',
'''        if can_charge and not avoid_queue and rules in BAY_RULES:'''),
('''        if can_charge and rules == "bay":''',
'''        if can_charge and rules in BAY_RULES:'''),
('''        d = _durations_corrected(
            full_data, stop_global,''',
'''        d = _durations_corrected(
            pd, stop_global,'''),
('''                  + _charging_time_needed(soc_next, full_data, stop=_stop)''',
'''                  + _charging_time_needed(soc_next, pd, stop=_stop)'''),
('''    if rules != "bay" and y and brk and rst is None and not must_charge \\''',
'''    if rules not in BAY_RULES and y and brk and rst is None and not must_charge \\'''),
# run loop: planning state
('''    K_set, C_set, L_set = (set(full_data["K"]), set(full_data["C"]),
                           set(full_data.get("L", [])))
''',
'''    K_set, C_set, L_set = (set(full_data["K"]), set(full_data["C"]),
                           set(full_data.get("L", [])))
    # B4 (bayref): planning curve and the duty time saved since the last rest
    plan_data = _planning_data(full_data) if rules == "bayref" else None
    h_saved   = 0.0
'''),
('''        action, reason = greedy_corrected_decision(
            full_data, stop, vehicle,''',
'''        state_for_plan = (_PlanState(vehicle, vehicle.h + h_saved)
                          if plan_data is not None else vehicle)
        action, reason = greedy_corrected_decision(
            full_data, stop, state_for_plan,'''),
('''            safety_buffer_frac = safety_buffer,
            rules              = rules,
        )''',
'''            safety_buffer_frac = safety_buffer,
            rules              = rules,
            plan_data          = plan_data,
        )'''),
('''        dur      = _durations_corrected(full_data, stop, action, vehicle, rules)''',
'''        dur      = _durations_corrected(full_data, stop, action, vehicle, rules)
        if plan_data is not None:
            dur_plan = _durations_corrected(plan_data, stop, action, vehicle,
                                            rules)
            saved = ((dur_plan["tauc"] + dur_plan["taub"])
                     - (dur["tauc"] + dur["taub"]))'''),
('''        tracker.record_realisation(stop, D_act, E_actual=E_act)
''',
'''        tracker.record_realisation(stop, D_act, E_actual=E_act)
        if plan_data is not None:
            # a daily rest resets the duty time, and with it the saving
            h_saved = 0.0 if action.get("rest_type") else h_saved + max(0.0, saved)
'''),
]
for a, b in reps:
    assert s.count(a) == 1, a[:70]
    s = s.replace(a, b)
io.open(p, "w", encoding="utf-8").write(s)
print("patched", p)

d = "src/simulation/runner_dispatch.py"
t = io.open(d, encoding="utf-8").read()
reps = [
('''    # rule set of the corrected greedy: "bay" (variant CORR2, default) lets
    # the truck stay at the bay to complete a break; "milp" (variant CORR)
    # applies the MILP's sequential rule.  Ignored unless greedy_corrected.
    greedy_rules: str                = "bay",''',
'''    # rule set of the corrected greedy: "bayref" (variant CORR3, default)
    # stays at the bay to complete a break and plans its duty time at 350 kW
    # (route duration cannot increase with charger power); "bay" (CORR2)
    # without the planning rule; "milp" (CORR) applies the MILP's sequential
    # rule.  Ignored unless greedy_corrected.
    greedy_rules: str                = "bayref",'''),
('''    parser.add_argument("--greedy_rules", choices=["bay", "milp"],
                        default="bay",
                        help="Corrected greedy rule set: 'bay' (variant CORR2, "
                             "default) may stay at the bay to complete a break "
                             "and takes a break at every charging stop; 'milp' "
                             "(variant CORR) applies the MILP's sequential "
                             "rule.")''',
'''    parser.add_argument("--greedy_rules", choices=["bayref", "bay", "milp"],
                        default="bayref",
                        help="Corrected greedy rule set: 'bayref' (variant "
                             "CORR3, default) = 'bay' + duty-time planning at "
                             "350 kW, so route duration cannot increase with "
                             "charger power; 'bay' (CORR2) may stay at the bay "
                             "to complete a break and takes a break at every "
                             "charging stop; 'milp' (CORR) applies the MILP's "
                             "sequential rule.")'''),
]
for a, b in reps:
    assert t.count(a) == 1, a[:70]
    t = t.replace(a, b)
io.open(d, "w", encoding="utf-8").write(t)
print("patched", d)
