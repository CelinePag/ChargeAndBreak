"""One-off patch: add the 'bay' rule set (variant CORR2) to greedy_corrected.py
and the matching switch to runner_dispatch.py.  Each replacement must match once."""
import io

def patch(path, reps):
    s = io.open(path, encoding="utf-8").read()
    for a, b in reps:
        assert s.count(a) == 1, (path, a[:70])
        s = s.replace(a, b)
    io.open(path, "w", encoding="utf-8").write(s)
    print("patched", path)

MOD = "src/methods/greedy_corrected.py"
patch(MOD, [
('''Run ids carry the variant label:  <instance>_GREEDY_CORR_<ts>[_<idx>].  The
reporting dedup is keyed on the variant, so these runs sit next to the
baseline greedy runs and never displace them; compile_solutions and
paper_figures exclude variants from the published tables and figures.

Usage
-----
  python -m src.simulation.runner_dispatch "instances/*.json" greedy \\
      --greedy_corrected --prune_quantile 0.95 --safety 0.1 --jobs 8
''',
'''Two rule sets
-------------
  rules="milp"  (variant CORR)  - C1-C3 and D1-D2 above, exactly.
  rules="bay"   (variant CORR2, the default since 2026-10-21) - C1, C3 and D2,
      but the truck may STAY AT THE BAY after a charge to complete a break
      (C2 relaxed), and the break decision no longer depends on the charge time:
        B1  Charge + break always runs in parallel: the charge is credited as
            break and the stop lasts max(charge, break).  No repositioning.
        B2  Whenever the greedy stops to charge (must-charge stop) it also takes
            the break there (b45, or b30 to complete a split), staying at the
            bay if the charge ends first.  greedy.py inserted the break only if
            the charge lasted >= 45 min, so a faster charger could REMOVE the
            break from a charging stop and push it to a later standalone stop:
            the cause of its worse results at 1000 kW than at 700 kW.
        B3  At a mandatory break at a charging station the greedy charges (as
            greedy.py does); the stop lasts max(charge, break).
      With these rules no decision uses the charging time, so for the same
      sequence of activities every dwell is non-increasing in charger power:
      max(tauc, b) for a charge with a break, tauc + M_seq before a rest.  The
      only remaining channel is the spread rule, which reads the elapsed duty
      time (shorter at higher power) to decide where to rest.

Run ids carry the variant label:  <instance>_GREEDY_<VARIANT>_<ts>[_<idx>].
The reporting dedup is keyed on the variant, so these runs sit next to the
baseline greedy runs and never displace them; compile_solutions and
paper_figures exclude variants from the published tables and figures.

Usage
-----
  python -m src.simulation.runner_dispatch "instances/*.json" greedy \\
      --greedy_corrected [--greedy_rules bay|milp] --prune_quantile 0.95 \\
      --safety 0.1 --jobs 8
'''),
('''# Variant label stamped into the run_id (and from there into the solution JSON).
VARIANT = "CORR"
''',
'''# Variant label stamped into the run_id (and from there into the solution JSON),
# per rule set.  VARIANT is the default rule set's label.
VARIANTS      = {"milp": "CORR", "bay": "CORR2"}
DEFAULT_RULES = "bay"
VARIANT       = VARIANTS[DEFAULT_RULES]
'''),
('''def _durations_corrected(full_data: dict, stop: int, action: dict,
                         state: BEHDV) -> dict:''',
'''def _durations_corrected(full_data: dict, stop: int, action: dict,
                         state: BEHDV, rules: str = DEFAULT_RULES) -> dict:'''),
('''            if tauc >= break_min - 1e-9:
                sigma, taub = 0, 0.0
            else:
                sigma, taub = 1, break_min''',
'''            if tauc >= break_min - 1e-9:
                sigma, taub = 0, 0.0
            elif rules == "bay":
                # B1: stay at the bay until the break is complete
                sigma, taub = 0, break_min - tauc
            else:
                sigma, taub = 1, break_min'''),
('''                              safety_buffer_frac: float = 0.0,
                              verbose: bool = False) -> tuple[dict, str]:''',
'''                              safety_buffer_frac: float = 0.0,
                              verbose: bool = False,
                              rules: str = DEFAULT_RULES) -> tuple[dict, str]:'''),
('''        if can_charge and not avoid_queue:
            if tauc_full >= b_min - 1e-9:''',
'''        if can_charge and not avoid_queue and rules == "bay":
            y = 1                                        # B3: max(charge, break)
        elif can_charge and not avoid_queue:
            if tauc_full >= b_min - 1e-9:'''),
('''        if can_charge:
            y = 1
            if phi == 1 and tauc_full >= Tb30:''',
'''        if can_charge and rules == "bay":
            # B2: take the break at every charging stop, whatever the charge
            # time; the truck stays at the bay if the charge ends first.
            y = 1
            brk = "b30" if phi == 1 else "b45"
            reason = f"MUST-CHARGE + {brk} (stay at bay if needed)"
        elif can_charge:
            y = 1
            if phi == 1 and tauc_full >= Tb30:'''),
('''             "rest_type": "r1" if rest_here else None}, state)''',
'''             "rest_type": "r1" if rest_here else None}, state, rules)'''),
('''    if y and brk and rst is None and not must_charge \\''',
'''    if rules != "bay" and y and brk and rst is None and not must_charge \\'''),
('''                         persist: bool                = True) -> dict:''',
'''                         persist: bool                = True,
                         rules: str                   = DEFAULT_RULES) -> dict:'''),
('''                                    ts, variant=VARIANT)''',
'''                                    ts, variant=VARIANTS[rules])'''),
('''            safety_buffer_frac = safety_buffer,
        )
        if supervised and stop > 0:''',
'''            safety_buffer_frac = safety_buffer,
            rules              = rules,
        )
        if supervised and stop > 0:'''),
('''        dur      = _durations_corrected(full_data, stop, action, vehicle)''',
'''        dur      = _durations_corrected(full_data, stop, action, vehicle, rules)'''),
('''            greedy_rules    = "milp",      # C1-C3 + D1-D2 (this module)''',
'''            greedy_rules    = rules,       # "milp" (CORR) or "bay" (CORR2)'''),
('''    _p(f"  GREEDY (CORRECTED, MILP rules) START   "''',
'''    _p(f"  GREEDY (CORRECTED, rules={rules}) START   "'''),
])

DSP = "src/simulation/runner_dispatch.py"
patch(DSP, [
('''    greedy_corrected: bool           = False,
''',
'''    greedy_corrected: bool           = False,
    # rule set of the corrected greedy: "bay" (variant CORR2, default) lets
    # the truck stay at the bay to complete a break; "milp" (variant CORR)
    # applies the MILP's sequential rule.  Ignored unless greedy_corrected.
    greedy_rules: str                = "bay",
'''),
('''    if greedy_corrected and alg == "GREEDY" and not variant:
        from src.methods.greedy_corrected import VARIANT as _GC_VARIANT
        variant = _GC_VARIANT
''',
'''    if greedy_corrected and alg == "GREEDY" and not variant:
        from src.methods.greedy_corrected import VARIANTS as _GC_VARIANTS
        variant = _GC_VARIANTS[greedy_rules]
'''),
('''            supervised      = supervised,
            prune_quantile  = prune_quantile,
        )

    elif alg == "GREEDY":''',
'''            supervised      = supervised,
            prune_quantile  = prune_quantile,
            rules           = greedy_rules,
        )

    elif alg == "GREEDY":'''),
('''                             "semantics), labelled with the variant 'CORR' "
                             "unless --variant is given.")
''',
'''                             "semantics), labelled with the variant of its "
                             "rule set unless --variant is given.")
    parser.add_argument("--greedy_rules", choices=["bay", "milp"],
                        default="bay",
                        help="Corrected greedy rule set: 'bay' (variant CORR2, "
                             "default) may stay at the bay to complete a break "
                             "and takes a break at every charging stop; 'milp' "
                             "(variant CORR) applies the MILP's sequential "
                             "rule.")
'''),
('''        if not args.variant:
            from src.methods.greedy_corrected import VARIANT as _GC_VARIANT
            args.variant = _GC_VARIANT
''',
'''        if not args.variant:
            from src.methods.greedy_corrected import VARIANTS as _GC_VARIANTS
            args.variant = _GC_VARIANTS[args.greedy_rules]
'''),
('''        greedy_corrected = args.greedy_corrected,''',
'''        greedy_corrected = args.greedy_corrected,
        greedy_rules     = args.greedy_rules,'''),
])
