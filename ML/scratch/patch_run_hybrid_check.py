"""One-off patch (2026-10-21): run_hybrid.py gets
  --gate-model NAME   which gate file to use (default gate_<arm>; gate_<arm>_ro =
                      trained on rollout labels, gate_train.py --label rollout)
  --check K [--check-z z]  keep the LA's action only if, with the student
                      driving on, it beats the student's own action over K
                      shared futures by more than z standard errors
                      (gate_rollout.call_value).  Base routes only: the
                      rollouts do not switch charging curves per charger."""
import os

P = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "code", "run_hybrid.py")
s = open(P, encoding="utf-8").read()


def sub(old, new, count=1):
    global s
    assert s.count(old) == count, (old[:70], s.count(old))
    s = s.replace(old, new)


sub('''    def __init__(self, arm, budget, nominal=True, pay_mseq=False):
        import lightgbm as lgb
        self.meta = json.load(open(os.path.join(me.ML, "models", f"gate_{arm}_meta.json")))
        self.booster = lgb.Booster(model_file=os.path.join(me.ML, "models", f"gate_{arm}.txt"))''',
'''    def __init__(self, arm, budget, nominal=True, pay_mseq=False, model=None):
        import lightgbm as lgb
        model = model or f"gate_{arm}"
        self.model = model
        self.meta = json.load(open(os.path.join(me.ML, "models", f"{model}_meta.json")))
        self.booster = lgb.Booster(model_file=os.path.join(me.ML, "models", f"{model}.txt"))''')

sub('''def drive(fd, D, E, cv, pol, name, tag, margin_h, final_h, dry, workers, time_limit,
          gate=None, rand_p=None):''',
'''def drive(fd, D, E, cv, pol, name, tag, margin_h, final_h, dry, workers, time_limit,
          gate=None, rand_p=None, check=None):''')

sub('''    calls, la_s, fallback, margins, keys, t_arr = [], 0.0, 0, [], [], []''',
'''    calls, la_s, fallback, margins, keys, t_arr = [], 0.0, 0, [], [], []
    vetoed, check_s = [], 0.0''')

sub('''            else:
                act, plan = got
                key = action_key(act.get("y", 0), act.get("break_type"),
                                 act.get("rest_type"))''',
'''            else:
                la_act, la_plan = got
                la_key = action_key(la_act.get("y", 0), la_act.get("break_type"),
                                    la_act.get("rest_type"))
                accept = True
                if check is not None and la_key != key:
                    # does the LA's action still win once the STUDENT drives on?
                    from gate_rollout import call_value
                    k_fut, z = check
                    sol0 = (la_plan or {}).get("sol") or [{}]
                    t1 = time.time()
                    mean, se, _fl, _fs = call_value(
                        fd, pre, veh, cv, pol, la_act, float(sol0[0].get("tauc", 0.0) or 0.0),
                        act, tc, k_fut, zlib.crc32(f"{name}|{tag}|{s}|check".encode()))
                    check_s += time.time() - t1
                    accept = mean - z * se > 0
                    if not accept:
                        vetoed.append(s)
                if accept:
                    act, plan, key = la_act, la_plan, la_key''')

sub('''                decisions=len(keys), n_calls=len(calls), la_s=round(la_s, 1),
                call_stops=calls, fallback=fallback), margins''',
'''                decisions=len(keys), n_calls=len(calls), la_s=round(la_s, 1),
                call_stops=calls, fallback=fallback, vetoed=vetoed,
                check_s=round(check_s, 1)), margins''')

sub('''    ap.add_argument("--random", type=float, default=None,''',
'''    ap.add_argument("--gate-model", default=None,
                    help="gate file name (default gate_<arm>; gate_<arm>_ro = rollout labels)")
    ap.add_argument("--check", type=int, default=None,
                    help="K futures: keep the LA's action only if it beats the student's own "
                         "with the student driving on (gate_rollout.call_value); base routes")
    ap.add_argument("--check-z", type=float, default=1.0)
    ap.add_argument("--random", type=float, default=None,''')

sub('''           + (f"g{args.gate}{args.budget:g}" if args.gate and args.budget is not None else "")''',
'''           + (f"g{args.gate}{args.budget:g}" if args.gate and args.budget is not None else "")
           + ("ro" if args.gate_model and args.gate_model.endswith("_ro") else "")''')

sub('''    if cfg:
        cfg += ("_ms" if args.mseq else "") + ("_N" if args.nominal_features else "")''',
'''    if args.check is not None:
        if args.routes != "base":
            ap.error("--check rolls out without per-charger curves: base routes only")
        cfg += f"c{args.check}z{args.check_z:g}"
    if cfg:
        cfg += ("_ms" if args.mseq else "") + ("_N" if args.nominal_features else "")''')

sub('''        gate = Gate(args.gate, args.budget, nominal=args.nominal_features)''',
'''        gate = Gate(args.gate, args.budget, nominal=args.nominal_features, model=args.gate_model)
    check = None if args.check is None else (args.check, args.check_z)''')

sub('''                             args.dry, args.workers, args.time_limit, gate, rand_p)''',
'''                             args.dry, args.workers, args.time_limit, gate, rand_p, check)''')

sub('''              f"{row['n_calls']} LA calls ({row['la_s']:.0f} s), "''',
'''              f"{row['n_calls']} LA calls ({row['la_s']:.0f} s), "
              f"{len(row['vetoed'])} vetoed ({row['check_s']:.0f} s), "''')

open(P, "w", encoding="utf-8").write(s)
print("patched", P)
