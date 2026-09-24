"""
parse_logs.py — recover the LA teacher's per-action cost vectors from its logs
=============================================================================
The look-ahead writes, at every stop, the scenario-mean objective of EVERY
action it enumerated -- not just the one it picked.  That is the supervision
this project is built on: one decision yields 4-10 labelled (action, cost)
pairs instead of a single class label.

Log shape (one decision block)
------------------------------
    [LA] stop 30 (CS)  t=18.899h  soc=251kWh  cd=1.82h  sd=8.41h  sw=9.06h
         phi=0  r2=0  ext_sh=0/2  sd_lim=9h
         horizon [30->66]  travel=9.84h +2rest  8 actions x 25 scen ...
      y=0  brk=0    rst=0   41.651h (0.163h)  ok=25/25  ws=25/25  tauc=0m ...
      y=1  brk=0    rst=0   41.790h (0.197h)  ok=25/25  ws=25/25  tauc=26m ...
      ...
      -> CHOSEN y=0  brk=0  rst=0  tauc=0m  taub=0m  ta=...  (mean=41.651h)

Two cost traps this module encodes so nothing downstream has to remember them
----------------------------------------------------------------------------
1. INFEASIBLE_PENALTY (1e9) enters the MEAN.  An action feasible in only 13 of
   25 scenarios scores ~4.8e8 h, and one that fails in 2 scenarios scores
   ~8e7 h -- a plausible-looking number that is NOT a duration.  Feasibility is
   therefore read from ok=k/n, never inferred from the magnitude of the cost,
   and `cost_h` is marked clean only when k == n.
2. The CHOSEN action is not always argmin: a 5-minute tiebreak rule and a
   post-hoc "tauc >= 45m so credit a b45" rule both rewrite it.  Both are
   flagged (`tiebreak`, `post_hoc`) so the student can be trained on the cost
   and the deterministic rules re-applied online.

Resumed runs (~20% of logs) start mid-route; the missing early stops simply
yield no labels.  Callers join on `stop`, never on position.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

# The LA's penalty for an infeasible scenario (MILP.INFEASIBLE_PENALTY).
INFEASIBLE_PENALTY = 1e9

_RE_STOP = re.compile(
    r"^\[LA\] stop (?P<stop>\d+) \((?P<kind>\w+)\)\s+"
    r"t=(?P<t>[-\d.]+)h\s+soc=(?P<soc>[-\d.]+)kWh\s+"
    r"cd=(?P<cd>[-\d.]+)h\s+sd=(?P<sd>[-\d.]+)h\s+sw=(?P<sw>[-\d.]+)h\s+"
    r"phi=(?P<phi>\d+)\s+r2=(?P<r2>\d+)"
)
_RE_HORIZON = re.compile(
    r"^\s+horizon \[(?P<a>\d+)->(?P<b>\d+)\]\s+travel=(?P<travel>[-\d.]+)h"
    r"(?:\s+\+(?P<nrest>\d+)rest)?\s+(?P<nact>\d+) actions"
)
_RE_ACTION = re.compile(
    r"^  y=(?P<y>\d)\s+brk=(?P<brk>\S+)\s+rst=(?P<rst>\S+)\s+"
    r"(?P<cost>[\d.e+]+)h \((?P<std>[\d.]+)h\)\s+"
    r"ok=(?P<ok>\d+)/(?P<n>\d+)\s+ws=\d+/\d+\s+"
    r"tauc=(?P<tauc>[-\d.]+)m\s+taub=(?P<taub>[-\d.]+)m"
)
_RE_CHOSEN = re.compile(
    r"^  -> CHOSEN y=(?P<y>\d)\s+brk=(?P<brk>\S+)\s+rst=(?P<rst>\S+)\s+"
    r"tauc=(?P<tauc>[-\d.]+)m"
)
_RE_NSCEN = re.compile(r"N_scen=(?P<n>\d+)\s+H=(?P<H>[\d.]+)h\s+cv=(?P<cv>[\d.]+)")


def norm_brk(v) -> str:
    """'0' / 'None' / '-' / None  ->  'none'.  NOTE: the string '0' is TRUTHY."""
    s = str(v).strip().lower()
    return "none" if s in ("0", "none", "-", "") else s


norm_rst = norm_brk


def action_key(y, brk, rst) -> str:
    """Canonical 12-symbol action vocabulary, e.g. 'y1_b45' or 'y0_r2'."""
    b, r = norm_brk(brk), norm_rst(rst)
    tail = b if b != "none" else (r if r != "none" else "go")
    return f"y{int(y)}_{tail}"


@dataclass
class Scored:
    """One enumerated action and what the teacher's sub-MILPs said about it."""
    y: int
    brk: str
    rst: str
    cost_h: float          # scenario-mean objective, penalty-contaminated if ok<n
    std_h: float           # spread across scenarios
    ok: int
    n: int
    tauc_h: float
    taub_h: float

    @property
    def key(self) -> str:
        return action_key(self.y, self.brk, self.rst)

    @property
    def clean(self) -> bool:
        """Feasible in every scenario => cost_h is a real duration."""
        return self.ok == self.n


@dataclass
class Decision:
    """All actions scored at one stop, plus the state line the LA printed."""
    stop: int
    kind: str                       # LAYBY | CS | CUST | INT | ORIG
    t_arr: float
    soc: float
    cd: float
    sd: float
    sw: float
    phi: int
    rho2_used: int
    horizon_end: int | None = None
    horizon_travel_h: float | None = None
    horizon_n_rest: int = 0
    actions: list = field(default_factory=list)
    chosen: str | None = None       # action_key of the executed action
    tiebreak: bool = False
    post_hoc: bool = False

    @property
    def clean_actions(self) -> list:
        return [a for a in self.actions if a.clean]

    def regrets(self) -> dict:
        """action_key -> cost minus the best CLEAN cost (hours, >= 0).

        Returned only for clean actions: a penalty-contaminated mean is not a
        duration, so its difference from the best is not a regret.
        """
        cl = self.clean_actions
        if not cl:
            return {}
        best = min(a.cost_h for a in cl)
        return {a.key: a.cost_h - best for a in cl}


def parse_log(path: str) -> tuple[list, dict]:
    """Parse one LA log.  Returns (decisions, meta)."""
    decisions: list = []
    meta: dict = {}
    cur = None

    with open(path, "r", errors="ignore") as fh:
        for line in fh:
            if not meta:
                m = _RE_NSCEN.search(line)
                if m:
                    meta = dict(n_scen=int(m["n"]), horizon_h=float(m["H"]),
                                cv=float(m["cv"]))

            m = _RE_STOP.match(line)
            if m:
                if cur is not None:
                    decisions.append(cur)
                cur = Decision(
                    stop=int(m["stop"]), kind=m["kind"], t_arr=float(m["t"]),
                    soc=float(m["soc"]), cd=float(m["cd"]), sd=float(m["sd"]),
                    sw=float(m["sw"]), phi=int(m["phi"]),
                    rho2_used=int(m["r2"]),
                )
                continue
            if cur is None:
                continue

            m = _RE_HORIZON.match(line)
            if m:
                cur.horizon_end = int(m["b"])
                cur.horizon_travel_h = float(m["travel"])
                cur.horizon_n_rest = int(m["nrest"] or 0)
                continue

            m = _RE_ACTION.match(line)
            if m:
                cur.actions.append(Scored(
                    y=int(m["y"]), brk=norm_brk(m["brk"]), rst=norm_rst(m["rst"]),
                    cost_h=float(m["cost"]), std_h=float(m["std"]),
                    ok=int(m["ok"]), n=int(m["n"]),
                    tauc_h=float(m["tauc"]) / 60.0, taub_h=float(m["taub"]) / 60.0,
                ))
                continue

            m = _RE_CHOSEN.match(line)
            if m:
                cur.chosen = action_key(m["y"], m["brk"], m["rst"])
                cur.tiebreak = "[tiebreak]" in line
                decisions.append(cur)
                cur = None
                continue

            if line.startswith("     [POST-HOC]") and decisions:
                decisions[-1].post_hoc = True

    if cur is not None:          # block without a CHOSEN line (run ended)
        decisions.append(cur)
    return decisions, meta
