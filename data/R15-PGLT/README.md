# R15-PGLT benchmark instances (Peña-Arenas / Garaix)

Source: https://emse.fr/~garaix/TruckDriverSchedule/Expert_Systems/
(companion page of the Expert Systems with Applications paper on the truck
driver scheduling problem with EU Regulation 561/2006 and Directive
2002/15/EC).  Downloaded 2026-07-14.

Contents
--------
- `instances/Test_1.txt … Test_40.txt` — the 40 PGLT instances with all
  durations rounded to multiples of 15 minutes ("R15").
- `no_night/TEST_k.txt` — their optimal schedules for the **No-Night**
  configuration (night rules disabled, `GOELCONSTRAINT = -1` in their
  `main.cpp`); `resultsTotal.txt` is their aggregate.
- `Agg-NO-Night.txt` — reference table: optimal completion time (minutes)
  and solve time per instance.  This is what `pglt.py` compares against.

Instance format (all values minutes, tab-separated, n+2 columns = nodes
0 .. n+1 where 0 = start depot, n+1 = end depot):

    TEST_k
    CLIENTS      n
    TRAVEL TIME  c_0 … c_n 0      leg duration node i -> i+1 (trailing 0)
    SERVICE TIME s_0 … s_n s_{n+1} s_0 = depot loading (counts as WORK)
    READY TIME   e_0 … e_{n+1}     hard window on activity START
    DUE DATE     l_0 … l_{n+1}     latest START; 6000/120000 = unbounded

Objective semantics of the reference results: completion-time minimisation
where the schedule starts at t=0 with the depot loading and MUST end with a
terminal daily rest (11 h, or reduced 9 h counting against the 3-per-week
budget) at the end depot; the terminal rest IS included in the objective
(e.g. Test_1: arrive 870 + 540 rest = 1410).

Their reference MILP (`main.cpp`, kept on the companion page) allows
preemption of driving AND service activities at continuous split points,
plus pure idle time (APO).  Our reproduction (`pglt.py`) discretises both
on the 15-minute grid instead:

- driving legs are split into 15-min segments separated by layby stops
  (break/rest anywhere along a leg);
- each service is split into 15-min customer "chunks" (a break or a daily
  rest may preempt service, cf. their Test_27/Test_35 optima), with the
  window on the FIRST chunk (start in [ready, due]) and their per-piece
  completion bound X+P <= due+s on every later chunk;
- a zero-length layby before each customer allows the arrive->rest->serve
  pattern (cf. their Test_37 optimum);
- a delayed schedule start replaces their X[0] variable (cf. Test_40,
  READY[0]=480);
- the Directive 2002/15 working-time breaks (their C11/C12/C15) are
  enforced via MILP.py's gated `wtd_rules` block;
- the mandatory terminal daily rest (their C0.1) is added post-build.

The audit of their solution files (`python pglt.py --scan-reference`)
confirms all their optima lie on the 15-min grid.  Note their model also
has a 3-h split-rest part (B3h, their C25) that ours does not; it never
produced a better value on this set.  On Test_4 our model finds 1410 vs
their published 1425 with a schedule that satisfies every constraint in
their shipped source — their published value appears slightly suboptimal
for their own formulation there (see solutions/pglt_comparison.csv).
