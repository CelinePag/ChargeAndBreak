"""One-off patch (2026-10-20): run_hybrid.py gets the gate / random triggers,
base routes, --mseq and --nominal-features.  Replaces drive() and main()."""
import os

P = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "code", "run_hybrid.py")
s = open(P, encoding="utf-8").read()

NEW_DRIVE = '''def drive(fd, D, E, cv, pol, name, tag, margin_h, final_h, dry, workers, time_limit,
          gate=None, rand_p=None):
    """One route.  Returns the result row (and, dry, the per-stop margins).

    Triggers (any one fires the LA): the rest/no-rest margin (margin_h), the
    last `final_h` hours of driving, the learned gate at its budget threshold
    (`gate`), or a random call with probability `rand_p` (the control at an
    equal call budget, seeded by route and stop).  Gate and random fire only
    where the student has at least two candidates, as the gate was trained."""
    pre = Precomp(fd)
    veh = BEHDV(fd)
    N = int(fd["N"])
    T0 = float(fd.get("T_START", 8.0))
    cum = np.r_[0.0, np.cumsum([float(fd["D"].get(i, 0.0)) for i in range(N)])]
    curves, tbar = fd.get("TbarK"), fd["Tbar"]
    calls, la_s, fallback, margins, keys, t_arr = [], 0.0, 0, [], [], []
    while veh.stop < N:
        s = veh.stop
        if curves:
            fd["Tbar"] = charger_curve(fd, s if s in pre.K else int(pre.next_cs[s]))
        # policy_core.candidates()[0], unrolled to keep the cost/feas vectors
        legal, rows, cost, feas, flags = pol._score(fd, pre, s, veh, cv)
        lkeys = list(pol._legal_keys)
        score = cost + 1e6 * (feas < pol.feas_thr)
        order = np.argsort(score, kind="stable")
        j = int(order[0])
        act, key = legal[j], lkeys[j]
        tc = pol._charge_hours(fd, s, veh, act, rows[j:j + 1], flags)
        cand = [(legal[i], None, lkeys[i], float(score[i])) for i in order]
        m = rest_margin(cand)
        left = cum[N] - cum[s]
        n_ok = int((score < 1e5).sum())
        g = (gate.score(fd, pre, s, veh, cv, cost, feas, lkeys, j)
             if gate is not None and n_ok >= 2 else None)
        u = (zlib.crc32(f"{name}|{tag}|{s}|rand".encode()) / 2**32
             if rand_p is not None and n_ok >= 2 else None)
        margins.append((m, left, len(cand), g, u))
        fire = ((margin_h is not None and m is not None and m <= margin_h)
                or (final_h is not None and left <= final_h and len(cand) > 1)
                or (g is not None and gate.thr is not None and g >= gate.thr)
                or (u is not None and u < rand_p))
        plan = None
        if fire and not dry:
            fd["Tbar"] = tbar                         # the LA sees the route as in its own runs
            t0 = time.time()
            got = ask_la(fd, s, veh, cv, zlib.crc32(f"{name}|{tag}|{s}".encode()),
                         workers, time_limit)
            la_s += time.time() - t0
            calls.append(s)
            if got is None:
                fallback += 1
                if curves:
                    fd["Tbar"] = charger_curve(fd, s if s in pre.K else int(pre.next_cs[s]))
            else:
                act, plan = got
                key = action_key(act.get("y", 0), act.get("break_type"),
                                 act.get("rest_type"))
        keys.append(key)
        t_arr.append(veh.t_arr)
        sol = plan if plan is not None else dict(
            feasible=True, sol=[dict(i=0, **durations(fd, s, act, tc, pol.pay_mseq))])
        veh.advance(action=act, D_next=float(D[s]), E_next=float(E[s]), milp_sol=sol)
        if veh.is_halted:
            break
    fd["Tbar"] = tbar
    done = (not veh.is_halted) and veh.stop >= N
    # rests from the dwell, as mixed_eval.la_row counts the LA's: what the
    # vehicle did, whoever chose it
    t_arr.append(veh.t_arr)
    dwell = [t_arr[k + 1] - float(D[k]) - t_arr[k] for k in range(len(keys))]
    rests = sum(1 for d in dwell if d >= me.REST_DWELL_H)
    return dict(instance=name, method=tag, completed=done,
                duration_h=(veh.t_arr - T0) if done else None,
                tw=len(veh.tw_misses), rests=rests, violation=veh.halt_reason,
                decisions=len(keys), n_calls=len(calls), la_s=round(la_s, 1),
                call_stops=calls, fallback=fallback), margins


def base_routes(split):
    """Base-case routes with a stored LA run (the LA dataset's), by seed."""
    seeds = range(20, 22) if split == "val" else range(22, 26)
    d = np.load(os.path.join(me.ML, "data", "dataset.npz"), allow_pickle=True)
    names = sorted(str(x) for x in d["instances"])
    return [("uniform", n, os.path.join(me._ROOT, "instances", n + ".json")) for n in names
            if int(n.rsplit("_", 1)[1]) in seeds]


'''

NEW_MAIN = '''def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--routes", default="mixed", choices=["mixed", "base"],
                    help="mixed: the mixed-charger routes (--variants); base: base-case "
                         "routes with an LA run, seeds 20-21 (--set val) or 22-25 (test)")
    ap.add_argument("--variants", default="pmix,mix")
    ap.add_argument("--set", default="test", choices=["test", "val"])
    ap.add_argument("--models", default=DEFAULT_MODELS,
                    help="student tags, comma-separated; 'kind:tag' or a bare torch tag")
    ap.add_argument("--margin", type=float, default=None, help="minutes")
    ap.add_argument("--final", type=float, default=None, help="hours of driving left")
    ap.add_argument("--gate", default=None, choices=["gbt", "torch"],
                    help="learned gate of this arm (gate_train.py); the student must be its arm's")
    ap.add_argument("--budget", type=float, default=None,
                    help="gate budget in %% of decisions (a threshold stored by gate_train.py)")
    ap.add_argument("--random", type=float, default=None,
                    help="control: call the LA at random on this %% of decisions")
    ap.add_argument("--mseq", action="store_true",
                    help="students pay the charge-then-rest overhead, as the LA does")
    ap.add_argument("--nominal-features", action="store_true",
                    help="student inputs at nominal travel times, as in training")
    ap.add_argument("--dry", action="store_true",
                    help="no LA: drive the student, report how often each trigger fires")
    ap.add_argument("--slice", default="0/1")
    ap.add_argument("--workers", type=int, default=LA_CONFIG["n_workers"])
    ap.add_argument("--time-limit", type=int, default=LA_CONFIG["time_limit"])
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    me.ROUTE_SET = args.set

    margin_h = None if args.margin is None else args.margin / 60.0
    rand_p = None if args.random is None else args.random / 100.0
    cfg = ((f"m{args.margin:g}" if args.margin is not None else "")
           + (f"f{args.final:g}" if args.final is not None else "")
           + (f"g{args.gate}{args.budget:g}" if args.gate and args.budget is not None else "")
           + (f"r{args.random:g}" if args.random is not None else ""))
    if not cfg and not args.dry:
        ap.error("give --margin, --final, --gate with --budget, or --random (or --dry)")
    if cfg:
        cfg += ("_ms" if args.mseq else "") + ("_N" if args.nominal_features else "")
    tail = "" if args.set == "test" else f"_{args.set}"
    head = "hybrid_base" if args.routes == "base" else "hybrid"
    store = os.path.join(RESULTS, f"{head}{tail}_{cfg or 'dry'}.jsonl")
    done = set()
    if os.path.exists(store) and not args.dry:
        with open(store) as fh:
            done = {(r["variant"], r["instance"], r["method"])
                    for r in map(json.loads, filter(str.strip, fh))}

    routes = (base_routes(args.set) if args.routes == "base"
              else me.instances(args.variants.split(","), uniform=False))
    jobs = [(v, n, p, t) for v, n, p in routes for t in args.models.split(",")]
    i, n = (int(x) for x in args.slice.split("/"))
    jobs = [j for j in jobs[i::n] if (j[0], j[1], j[3]) not in done]
    if args.limit:
        jobs = jobs[: args.limit]
    print(f"[hybrid] {cfg or 'dry'} on {args.routes}/{args.set}: {len(jobs)} runs "
          f"(slice {args.slice}) -> {store}", flush=True)

    gate = None
    if args.gate:
        gate = Gate(args.gate, args.budget, nominal=args.nominal_features)
    pols, all_m = {}, []
    for k, (v, name, path, spec) in enumerate(jobs):
        kind, _, tag = spec.rpartition(":")
        kind = kind or "torch"
        if gate is not None and tag != Gate.STUDENT[args.gate]:
            raise SystemExit(f"the {args.gate} gate was trained for {Gate.STUDENT[args.gate]}, "
                             f"not {tag}")
        if spec not in pols:
            pol = load_policy(kind, tag, guard_q=0.99, spread_room=True)
            if args.nominal_features:
                pol.feature_q = None
            pol.pay_mseq = args.mseq
            pols[spec] = pol
        pol = pols[spec]
        fd, D, E, cv = me.load(path)
        fd["title"] = name
        if LA_CONFIG.get("la_energy_quantile"):
            fd["la_energy_quantile"] = float(LA_CONFIG["la_energy_quantile"])
            fd["la_energy_cv"] = cv
        t0 = time.time()
        row, margins = drive(fd, D, E, cv, pol, name, tag, margin_h, args.final,
                             args.dry, args.workers, args.time_limit, gate, rand_p)
        row["variant"], row["config"] = v, cfg or "dry"
        all_m += [(v, name, tag, *x) for x in margins]
        if not args.dry:
            with open(store, "a") as fh:
                fh.write(json.dumps(row) + "\\n")
        print(f"  {k+1}/{len(jobs)} {v} {name} {tag[-2:]}: "
              f"{'done' if row['completed'] else row['violation']} "
              f"{row['duration_h'] or 0:.1f} h, {row['rests']} rests, "
              f"{row['n_calls']} LA calls ({row['la_s']:.0f} s), "
              f"{time.time()-t0:.0f} s", flush=True)

    if args.dry:
        runs = len(jobs) or 1
        dec = len(all_m) or 1
        print(f"\\ntrigger counts along the student's own trajectories "
              f"({dec / runs:.0f} decisions per route): calls per route (% of decisions)")
        for mm in (5, 10, 15, 30, 60, 120):
            c = sum(1 for x in all_m if x[3] is not None and x[3] <= mm / 60)
            print(f"  margin {mm:4d} min: {c / runs:6.2f}  ({c / dec * 100:4.1f} %)")
        for fh_ in (5, 10):
            c = sum(1 for x in all_m if x[4] <= fh_ and x[5] > 1)
            print(f"  final {fh_:4d} h  : {c / runs:6.2f}  ({c / dec * 100:4.1f} %)")
        if gate is not None:
            for b, thr in gate.meta["thresholds"].items():
                c = sum(1 for x in all_m if x[6] is not None and x[6] >= thr)
                print(f"  gate {b:>3s} % budget: {c / runs:6.2f}  ({c / dec * 100:4.1f} %)")
        if rand_p is not None:
            c = sum(1 for x in all_m if x[7] is not None and x[7] < rand_p)
            print(f"  random {args.random:g} %: {c / runs:6.2f}  ({c / dec * 100:4.1f} %)")


'''

a = s.index("def drive(fd, D, E, cv, pol, name, tag, margin_h, final_h, dry, workers, time_limit):")
b = s.index("def main():")
s = s[:a] + NEW_DRIVE + s[b:]
a = s.index("def main():")
b = s.index('if __name__ == "__main__":')
s = s[:a] + NEW_MAIN + s[b:]
open(P, "w", encoding="utf-8").write(s)
print("patched", P)
