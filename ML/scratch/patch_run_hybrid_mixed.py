"""One-off patch (2026-10-23): run_hybrid.py for the mixed-charger routes.
  - --check allowed on every route (rollouts switch charging curves now)
  - --alone: the student drives alone, rows written like any configuration
  - Gate: student / other-arm / oracle model tags read from the gate's meta
    (base gates without those keys keep their defaults); signals that the
    gate does not use are not computed (a mixed gate has no oracle signal)."""
import os

P = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "code", "run_hybrid.py")
s = open(P, encoding="utf-8").read()


def sub(old, new, count=1):
    global s
    assert s.count(old) == count, (old[:70], s.count(old))
    s = s.replace(old, new)


sub('''        self.thr = float(self.meta["thresholds"][f"{budget:g}"]) if budget is not None else None
        self.names = self.meta["features"]
        other = "torch" if arm == "gbt" else "gbt"
        self.other = load_policy(other, self.STUDENT[other], guard_q=0.99, spread_room=True)
        self.oracle = load_policy("gbt", "gbt_F95_oracle_s0", guard_q=0.99, spread_room=True)
        for p in (self.other, self.oracle):
            if nominal:
                p.feature_q = None''',
'''        self.thr = float(self.meta["thresholds"][f"{budget:g}"]) if budget is not None else None
        self.names = self.meta["features"]
        other = "torch" if arm == "gbt" else "gbt"
        self.student = self.meta.get("student", self.STUDENT[arm])
        kind = lambda t: "torch" if t.startswith("tmlp_") else "gbt"
        o_tag = self.meta.get("other", self.STUDENT[other])
        r_tag = self.meta.get("oracle", "gbt_F95_oracle_s0")
        self.other = load_policy(kind(o_tag), o_tag, guard_q=0.99, spread_room=True)
        self.oracle = (load_policy(kind(r_tag), r_tag, guard_q=0.99, spread_room=True)
                       if r_tag and "oracle_dis" in self.names else None)
        for p in (self.other, self.oracle):
            if nominal and p is not None:
                p.feature_q = None''')

sub('''        ko = self._pick(self.other, fd, pre, s, veh, cv)
        kr = self._pick(self.oracle, fd, pre, s, veh, cv)''',
'''        ko = self._pick(self.other, fd, pre, s, veh, cv)
        kr = self._pick(self.oracle, fd, pre, s, veh, cv) if self.oracle is not None else keys[j]''')

sub('''        if gate is not None and tag != Gate.STUDENT[args.gate]:
            raise SystemExit(f"the {args.gate} gate was trained for {Gate.STUDENT[args.gate]}, "
                             f"not {tag}")''',
'''        if gate is not None and tag != gate.student:
            raise SystemExit(f"the gate {gate.model} was trained for {gate.student}, not {tag}")''')

sub('''    if args.check is not None:
        if args.routes != "base":
            ap.error("--check rolls out without per-charger curves: base routes only")
        cfg += f"c{args.check}z{args.check_z:g}"''',
'''    if args.alone:
        cfg = "alone"
    if args.check is not None:
        cfg += f"c{args.check}z{args.check_z:g}"''')

sub('''    ap.add_argument("--dry", action="store_true",''',
'''    ap.add_argument("--alone", action="store_true",
                    help="the student alone (no trigger), rows written as config 'alone'")
    ap.add_argument("--dry", action="store_true",''')

sub('''           + ("ro" if args.gate_model and args.gate_model.endswith("_ro") else "")
''',
'''           + ("ro" if args.gate_model and args.gate_model.endswith("_ro") else "")
           + ("mx" if args.gate_model and args.gate_model.startswith("gate_mixed") else "")
''')

sub('''    if not cfg and not args.dry:
        ap.error("give --margin, --final, --gate with --budget, or --random (or --dry)")''',
'''    if not cfg and not args.dry and not args.alone:
        ap.error("give --margin, --final, --gate with --budget, --random or --alone (or --dry)")''')

open(P, "w", encoding="utf-8").write(s)
print("patched", P)
