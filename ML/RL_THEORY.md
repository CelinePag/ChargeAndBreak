# Reinforcement learning for the charge / break / rest policy: theory, design, limits

2026-10-26. A follow-on to [THEORY.md](THEORY.md) (the MDP and where imitation
sits in it). Numbers come from `ML/results`, the memory notes of the ML work and
the CPAIOR draft (`ML/paper/main.tex`, draft 4); new measurements made today are
marked **[new]**. References were checked online on 2026-10-26 (list at the end).

---

## 0. Short answer

1. **Imitation stops at the teacher by construction.** The students learn the
   LA's estimate `Q^LA(s,a)`: a 24 h horizon, solved with perfect information
   inside each of 25 scenarios. Its systematic errors are inherited, and its
   noise is averaged out (which is why a student can tie or slightly beat it).
   What is left (rare extra daily rests, window misses, the plateau on
   mixed-power routes) is where `Q^LA` is biased, or where the LA's choice
   does not suit the student's own continuation.
2. **What RL changes is the training signal.** Instead of the LA's horizon
   estimate, RL uses the policy's own cost over the whole route, under travel
   times it cannot see in advance (non-anticipative), in the simulator. It is
   the only signal we have that **prices slack**. The oracle, the LA's
   scenario solves and nominal MILPs all value slack at about zero.
3. **Not from scratch.** Use policy improvement starting from the student,
   inside a fixed and *complete* shield, with variance reduction designed for
   this problem. Rests are admissible at almost every stop, so a random policy
   would rest dozens of times per route.
4. **Order of work:** (0) fix the environment and measure the headroom;
   (1) low-dimensional policy search with common random numbers; (2) rollout-
   based approximate policy iteration aimed at rest decisions, iterated, with
   exploratory roll-ins; (3) PPO fine-tuning of the PyTorch student, anchored to
   it by a KL term; (4) journal scale: a learned terminal value inside the
   LA's MILP, and information-relaxation bounds.
5. **Limits:** the ceiling is the best non-anticipative policy π\*, not the
   oracle, and an unknown part of the LA's tail is plain bad luck (value of
   information). The cliff (one more rest = +9–11 h) makes the variance large.
   With 125 test routes, gains below ~0.7–1 pp cannot be detected. An RL
   agent will exploit any rule the simulator does not check. And the CPAIOR
   deadline is in 5–6 weeks.

**[new] A prerequisite found today.** The simulator the students run in does
not check the Art. 8 spread caps that the MILP enforces: a daily rest must
*begin* within 13 h of the shift start (15 h if reduced), and the last shift
must end within 13 h. Replaying the two final students (protocol g99sr +
nominal features + M_seq) on the 125 test routes with `BEHDV(strict_spread=True)`
(`ML/scratch/strict_spread_check.py`):

| model | default check | strict check |
|---|---|---|
| trees `gbt_F95_base_s1` | 0 / 125 broken | **28 / 125** (17 pre-rest, 11 terminal) |
| PyTorch `tmlp_F95_split_list_s0` | 0 / 125 | **15 / 125** (11 pre-rest, 4 terminal) |

The LA has the same blind spot (21.6 % of its stored runs overrun the pre-rest
cap; memory note *spread-cap blind spot*). The terminal cap is a modelling
choice. The pre-rest cap is the regulation itself (Art. 8(2): a new daily rest
within 24 h). For RL this decides everything: the cheapest way to avoid an
extra rest is to start the rest a little after 13 h, so an agent trained in
the default simulator **will learn exactly that**. It also bears on the draft's
claim that none of the 750 runs breaches the regulation.

---

## 1. What the experiments already establish

Each of these facts constrains the RL design.

| # | fact | number | where |
|---|---|---|---|
| F1 | Students match the LA on the base case | trees +0.68 ± 0.43 pp, PyTorch +0.50 ± 0.37 pp with window penalty; +0.23 / +0.10 duration only | main.tex Table 1 |
| F2 | The LA's tail is one extra daily rest | 60 / 836 LA runs; they are all the runs > 8 % above the oracle, all 43 on long routes; fixing them would move the mean gap 3.6 % → 2.6 % | rest-count notes |
| F3 | It is avoidable at least sometimes | Greedy reaches the oracle's rest count on 21 of the 60; on each of 6 test routes some student does | rest-count notes |
| F4 | Students inherit it | extra rest on 3 % (trees) / 2 % (PyTorch) of test runs; on mixed routes, 5–9 of ~90 runs make up 60–73 % of the gap to the LA | main.tex, mix notes |
| F5 | One-step improvement failed | plain rollout +7.6 % (noise chasing), gated −0.2 % (noise); FPI closed loop +0.004 ± 0.081 h; MILP end game worse on 8 / 10 | rollout / FPI / end-game notes |
| F6 | Why FPI failed | label sd 2.1 h; rest decisions ~3 % of 208,920 labels; G's R² 0.021; held-out one-step gain −0.0002 ± 0.0053 h | FPI notes |
| F7 | One measurable bias | the LA-trained head overprices "rest now" by 1.14 ± 0.12 h (long, y0_r1); breaks and charges are calibrated within ±0.2 h | FPI notes |
| F8 | Rest-count errors can't be fixed one step at a time | on RlongCfewTmedium_23, P(extra rest) under the student's continuation is 0.5–0.7 in shifts 1–2 and ≈ 1 from shift 3; no top-3 alternative lowers it | rollout notes |
| F9 | The LA's actions assume the LA continues | gate without the check: 2–3 hand-offs create an extra rest (+16 to +30 pp); the rollout check (student continues, 32 shared futures, z-gate) fixes it | gate notes |
| F10 | The hybrid reaches LA parity | gate 10 % + check: −0.36 ± 0.33 pp vs LA, −1.09 ± 0.30 vs the student alone, 8.1 calls per route | main.tex Table 2 |
| F11 | Copying the oracle is worse | ≈ +2 pp (trees) and +6–7 pp (PyTorch) vs LA; extra rest on 13 % / 32 % of runs | main.tex §5.2 |
| F12 | The shield is necessary | without it: 18 / 375 (trees) and 38 / 375 (PyTorch) runs fail | no-shield notes |
| F13 | Noise floor | LA run-to-run: median 0.35 %, max 18 % on identical realizations | LA non-determinism notes |
| F14 | Episodes | test routes: short 51 / medium 96 / long 164 decisions, 1 / 2.3 / 4.5 rests, 30 / 58 / 108 h **[new, from eval files]** | `eval_*_g99srNm_test.json` |
| F15 | Action mix (PyTorch) | 89.1 % drive on, 6.3 % charge + b45, ~2.4 % rests **[new]** | same |
| F16 | Cost of a simulated step | 2–5 ms per decision on one core, mostly Python features; an LA decision is ~12 s on the cluster and 70–160 s on the laptop | FPI / HPC notes |
| F17 | The teacher never scored some actions | the LA prunes `r1` while fewer than 2 reduced rests are used (`_prune_actions`, H ≤ 48); in 40 long-route logs, 0 of 4,752 scored `r1` rows have `r2_used < 2` **[new]** | `logs/basecase` |

---

## 2. Why imitation stops at the teacher

**The AggreVaTe bound.** If the student is trained on the expert's cost-to-go of
every action, then `J(π̂) ≤ J(π^E) + T·ε`, where ε is the cost-sensitive regret
of the regression on the student's own state distribution (Ross & Bagnell,
2014). Even with perfect learning (ε → 0) the student tends to the expert, and
no further.

**What `Q^LA` is, term by term.** Write the LA's score as

```
Q^LA(s,a)  =  Q*(s,a)  +  b_H(s,a)  −  EVPI_H(s,a)  +  ε(s,a)
```

* `b_H`, **truncation**. The MILP minimizes the arrival time at the horizon
  end, about 24 h ahead, with no terminal value. Two end states with the same
  clock time but different slack (cd, sd, h, battery) score the same. A rest
  needed beyond the horizon is invisible.
* `EVPI_H ≥ 0`, **clairvoyance within each scenario**. After the first decision,
  each scenario's MILP knows that scenario's future, so the average of the
  optima is at most the optimum of the average (E[min] ≤ min E). The gap is
  the value of information over the horizon. It is larger after actions that
  leave a fragile state, where knowing the future would help most. So the
  ranking of actions is biased toward **plans with little slack**, which is
  what the logs show (p90 pre-rest slack −0.01 h, "plans to the boundary").
  This is the known weakness of hindsight optimization (Yoon et al., 2008).
* `ε`, **noise**: 25 scenarios (median se ≈ 2 min per score), MIP time limits
  (incumbents accepted), non-determinism (F13).
* **Policy-structure choices** that are not in Q\*: the r1 pruning (F17), a
  5-minute tie-break toward charging, and durations executed from a nominal
  re-solve rather than from the scenario solves.

The student learns `E[Q^LA | features]`: it averages out ε and keeps `b_H` and
`EVPI_H`. **Copying the oracle (F11) is the extreme case**: the student copies
the value of perfect information over the whole route. Sinclair et al. (2023)
learn from hindsight and argue that this hindsight bias is small in their
applications. Our data show it is large here, exactly at end-of-route packing.

**Where the three sources of value stand.**

| source | horizon | information | prices slack? |
|---|---|---|---|
| oracle (hindsight MILP) | whole route | perfect | no: zero by construction |
| LA scenario solves | 24 h | perfect within each scenario | no, and truncated |
| nominal MILP (end game) | whole route | mean travel times | no: ties between rest placements |
| **closed-loop simulation of a policy** | **whole route** | **non-anticipative** | **yes**, for that policy |

The last row is the RL signal. It has no bias for the policy being evaluated,
but it has variance, and it depends on the policy: it values slack *as that
policy would use it*.

---

## 3. The MDP for RL, and what changes from THEORY.md §1

* **Episode.** An instance `c` from the training generator (base seeds 1–19,
  mixed training routes; new seeds may be generated, never 20–25), and a
  **fresh** travel-time draw ξ from the instance's own law in every episode.
  The stored realization is kept for validation and test, where the LA and
  oracle exist.
* **State and post-decision state.** The state `S_k` is BEHDV's state plus the
  instance. The post-decision state is
  `S^x_k = (k, t_dep, e_dep, cd', sd', h', φ', ρ₂', ext')`: the state after the
  action and before the leg is driven. The next state adds the leg:
  `t, cd, sd, h += D_k` and `e −= E_k(D_k)`. **The exogenous information of a
  leg is one scalar** (ξ_k; energy follows from the realized speed), with a
  known law. So any expectation over the next leg is a 1-D integral and can be
  computed by quadrature rather than sampled.
* **Actions.** The shielded enumeration (12 discrete combinations), and a charge
  duration in `[lo(s,a), hi(s,a)]`.
* **Cost per stop: the dwell, not the drive.**

  ```
  c_k = (t_dep,k − t_arr,k) + β·1{window missed at k},     β = 0.5 h
  ```

  `D_real` is indexed by leg, and travel times do not depend on the clock or
  on the action. So for a completed route, `Σ_k D_k(ξ)` is the same for every
  policy, and `J(π) = E[Σ D] + E[Σ c_k]`. Dropping the drive term changes
  neither the argmin nor the expected policy gradient; it removes variance.
  This is the exogenous-reward decomposition of Dietterich, Trimponias & Chen
  (2018). It assumes the route completes, hence §4.
* **Horizon and discount: γ = 1.** The objective is total time, and the expensive
  event is at the end of the route. With PPO's default γ = 0.99, a rest at the
  last decision of a long route is weighted 0.99¹⁶⁴ ≈ 0.19 (0.38 on a medium
  route). The agent would be nearly blind to exactly the error we want fixed.
  Use γ = 1, with remaining-route features so the value function knows how far
  the destination is.
* **The shape of the return.** `Σ c_k = service (fixed) + 9·n_r2 + 11·n_r1 +
  uncovered break and charge time + overheads (queue, M_stop, M_lay, M_seq) +
  β·misses`. Rests dominate: about 45 h of a 108 h long route. Under a fixed
  policy the return is a **mixture indexed by the rest count**. With a
  probability p = 0.05–0.10 of one extra rest (+≈10 h), its sd is about
  `10·√(p(1−p)) ≈ 2.2–3.0 h`, which matches FPI's label sd of 2.1 h (F6).
* **Why the cliff sits at the end (back-of-envelope).** Legs are ~25 km
  (~0.31 h), and the realized sd of ξ is 0.125, so a full final shift of ~29
  legs has a remaining-drive sd of about `√29 · 0.31 · 0.125 ≈ 0.2 h`. A plan
  with zero slack in the last shift overflows about half the time, and an
  overflow costs a full rest. A slack of ~0.35 h would make that rare (~5 %).
  In a nominal plan, slack is almost free to create (moving rests barely
  changes the arrival). But it is not free under uncertainty: slack moved into
  the last shift is taken from earlier shifts, which then carry the risk. So
  the decision is how to **allocate a route's total packing margin across its
  shifts**. Routes whose load sits just under a multiple of the shift capacity
  (13/15 h spread windows, which force rests on long routes, per the oracle
  stall analysis) are the knife-edge ones. That margin can be computed from the
  instance, so training can oversample those routes.

---

## 4. Safety: turning the constrained MDP into a plain MDP

**Today's shield** is a one-step guard at q = 0.99, the spread-room clamp on
charges, and the simulator's forcing rules. At the base cv, the 0.99 time-side
quantile of ξ is already the hard cap ξ = 1.6, and the energy side is 0.8953
against the hard 0.8889 **[new: `xi_quantile`]**. So on the one-step checks
the shield is already a worst-case guard. **Two rules are missing:** the
pre-rest cap and the terminal cap.

**What RL needs.** Run the environment with `strict_spread=True`, and extend the
shield with one invariant: *a legal rest exists at the next stop*. With
`D_wc` at ξ = 1.6 and energy at ξ = 8/9:

```
non-rest a at stop k is admissible iff
    cd + D_wc ≤ 4.5                         (else a break or rest is forced)
    sd + D_wc ≤ sd_lim                      (10 h while an extension is left, else 9 h)
    h + o_k(a) + M_k(a) + τ_c + D_wc + o_min,k+1 ≤ cap(s)
          cap = 15 h if a reduced rest is still available, else 13 h
          (13 h, with no o_min term, if k+1 is the destination)
    e_dep − E_wc(to the next charger) ≥ E_min
rest r at stop k is admissible iff h + o_k(r) ≤ 13 h (r1) or 15 h (r2)
    o_k = everything before the rest begins: service, queue, a charge taken
          before the rest, M_stop / M_lay / M_seq
o_min,k+1 = the least pre-rest dwell the next stop obliges: service at a
          customer, M_lay at a layby; at a charger M_stop, plus the queue,
          the charge and M_seq if the worst-case arrival energy there will
          not reach the charger after it. The rest is always the last
          activity at a stop, so a charge the battery forces happens BEFORE
          the rest and uses spread (the same look-ahead the 2026-08-14
          Greedy fix uses).
```

**Claim (recursive feasibility).** Suppose every leg's worst case is shorter
than the limits and every charger can be reached on a full pack. This should
hold for the base-case instances: legs are ~0.3 h, chargers ~60 km apart
(± 19 km), and the usable range at 90 km/h is ~295 km (400 kWh at 1.35 kWh/km).
Test 0b checks it. Then from a state that satisfies the invariant, the
shielded set is never empty. A rest is admissible at every stop, because the previous
step guaranteed it, and a rest resets the clocks. Energy to the next charger
was guaranteed at the last charger. So violations cannot occur, no failure
penalty is needed, and RL becomes an ordinary MDP over `A_shield(s)`. This is
shielding (Alshiekh et al., 2018) with the recursive-feasibility argument of
predictive safety filters (Wabersich & Zeilinger, 2021). **To verify before
any RL:** drive a uniformly random shielded policy on thousands of draws and
require 0 violations under strict checks. That takes about an hour, and it is
what makes exploration safe.

**Its price.** The shield becomes more conservative than today's, so every
student and LA number has to be re-stated under the strict rules. The LA's
stored runs would need re-checking too, which matters for any "vs LA"
comparison.

---

## 5. What RL would reinforce, in order of tractability

| target | size today | credit distance | frequency | why imitation can't fix it | difficulty for RL |
|---|---|---|---|---|---|
| **Window misses** | 111–116 vs LA 57 misses over 737 visits; ≈ 0.4 pp of the students' gap to the LA (0.68 → 0.23, 0.50 → 0.10) | a few stops before the customer | frequent | the 0.5 h penalty is small next to the ranking noise; the timing levers (charge length, break placement) were learned from the LA's nominal re-solve | easy to medium |
| **Where to charge** (mixed power) | energy share ≥ 700 kW: oracle 89, LA 82, students 79 %; the gap to the LA is 2.2–2.9 pp, mostly extra rests (F4) | 1–3 stops | frequent | needs demonstrations on every new physics, costing ~27 h of LA per 48 routes | medium; no labels, unlimited routes |
| **Rest count** (end-of-route packing) | LA: 60 / 836 runs, ~1 pp of the mean; students: 2–3 % of runs | across shifts, 20–100 decisions | rare | the teacher has the bias (`b_H`, `EVPI_H`) | hard, and partly irreducible |
| **Actions never scored** | r1 before two r2 (F17); other charge durations | varies | — | no data at all | unknown gain |
| **When to call the solver** | the gate + check reaches LA parity at 8.1 calls (F10) | one decision + continuation | — | its label must value a call *with the student continuing*, which is an RL quantity | already done as one-step rollout |

The hybrid in F10 is already a small RL step: it is a rollout algorithm
(Bertsekas, 2022) whose candidate set is {student action, LA action}, with a
significance gate against noise. It worked where plain top-3 rollout failed
because the candidate came from a better-informed planner, and the z-gate
stopped the noise chasing.

---

## 6. Why one-step improvement failed, and what that implies

**Noise.** With exact evaluation, rollout never does worse than its base policy.
With K samples it can: the minimum of noisy estimates is biased (the winner's
curse). With sd ≈ 2 h and K = 8, the se is ≈ 0.7 h, while true differences
are 0.1–0.5 h. Hence +7.6 %.

**Local optimality is not optimality.** The performance-difference lemma
(Kakade & Langford, 2002) states

```
J(π') − J(π)  =  Σ_k  E_{s ~ d_k^{π'}} [ A^π(s, π'(s)) ]
```

The expectation is over the **new** policy's states. A one-step test at π's own
states only checks the first deviation. If an improvement needs *coordinated*
deviations (keep slack in shift 1, keep it in shift 2, use it in shift 4),
each single deviation followed by π looks useless, because π spends the slack
again. A policy can be locally optimal in Chang et al.'s (2015) sense and
still far from π\*. F8 has three readings:

1. **Bad luck** (value of information): nothing to gain.
2. **Coordinated deviations** are needed: iterate the improvement, and explore
   states the student never visits.
3. **The improving action was never tested**: charge durations, r1, or actions
   ranked fourth or lower.

RL is worth doing only under (2) or (3). The first experiment must tell them
apart.

**FPI's failure, quantified.** It used one shared draw per branch, so the label
sd was 2.1 h against effects of 0.1–1 h, on 3 % of rows. To resolve δ = 0.2 h
at 2 standard errors with σ = 2.1 h takes `n ≈ (2σ/δ)² ≈ 440` independent
labels per group of comparable states. The finite-sample bounds of
regret-weighted classification-based policy iteration (Lazaric, Ghavamzadeh &
Munos, 2016; the same "weight a mistake by its regret" structure as our cost
heads) grow with `Q_range/√M`, where M is the number of rollouts per state.
They also grow with a concentrability coefficient: how well the sampled states
cover the optimal policy's. FPI round 1 was bad on both counts: a Q range of
~10 h, M = 1, and states from the student only, never the slack-keeping
states.

---

## 7. Methods, cheapest first (recommended path in bold)

### 7.0 Measure the headroom first

`J(oracle) ≤ J(π*) ≤ J(LA)`, and only `J(LA) − J(π*)` is available to RL.

* **Cheap empirical test:** from the decisive states of the LA's 60 extra-rest
  runs, draw many fresh futures and ask how often any non-anticipative policy
  (Greedy, the students, perturbed students) avoids the extra rest *in
  expectation*, not only on the realized draw (F3 is realized-draw only).
* **Information-relaxation dual bound** (Brown, Smith & Sun, 2010). For any
  function V̂, the expected penalized hindsight optimum,

  ```
  E_ξ [ min over clairvoyant plans of  Σ_k c_k + Σ_k ( V̂(S_k+1) − E[V̂(S_k+1) | S_k, x_k] ) ]
  ```

  is a lower bound on J(π\*). V̂ = 0 gives the plain oracle, and a better V̂
  (for instance the RL critic) tightens it. Kullman, Goodson & Mendoza (2021)
  report EV routing policies within 10 % of a dual bound, and El Shar & Jiang
  (2020) use information-relaxation bounds inside Q-learning. The penalty makes the oracle MILP
  non-linear unless V̂ is piecewise linear in a few state variables; a
  deterministic DP or label-setting over one realization (Goel, 2010; Garaix
  et al., 2024) is the more natural inner solver. This is research-sized.

### **7.1 Direct policy search with common random numbers**

* **Policy family.** `π_θ(s) = argmin_a  Q̂_IL(s,a) + θ·g(s,a)`, where g holds 2–4
  interpretable terms: a rest-now indicator × the rests still needed; the
  final-shift slack deficit `max(0, κ·σ_rem(s) − slack_final(s,a))`; an
  early-r1 indicator.
* **Objective.** The mean penalized dwell over fixed draws × training routes.
  Fixing the random numbers makes the objective deterministic (PEGASUS; Ng &
  Jordan, 2000). Optimize by grid, CMA-ES or Bayesian optimization, with
  paired comparisons.
* **Why first.** Each evaluation integrates every decision of a route, so the
  variance is low. θ changes the policy at every stop at once, which is the
  direct test of reading (2) in §6. It is interpretable, and it is cheap:
  ~50 θ × 600 routes × 4 draws ≈ 120 k episodes ≈ 10–15 core-hours. FPI's
  untested next step was a "flat rest-cost recalibration"; this is that idea,
  tuned on closed-loop outcomes instead of per-decision labels.

### **7.2 Rollout-based approximate policy iteration (the main arm for the trees)**

```
π_0 = student
for i = 0, 1, 2, …
    roll in with a mixture: π_i, π_i perturbed, and guide policies
        (Greedy; the LA's or the oracle's STATES, not their actions: JSRL)
    sample decisive states: a rest admissible, small student margin
        (the gate's signals), knife-edge routes oversampled
    at each, evaluate EVERY admissible action (r1 and 2–3 charge lengths
        included) by M = 32–64 rollouts of π_i to the destination, under
        common random numbers
    fit a correction by regret-weighted regression           (Lazaric et al. 2016)
    π_{i+1} = argmin of the corrected scores, accepted only where the
        change is significant (the rollout check's z-gate), or mixed
        conservatively                                       (Kakade & Langford 2002)
```

**Variance-reduction stack.**

1. Common random numbers across actions (already in FPI).
2. The dwell-only reward (§3).
3. **A hindsight control variate.** Because ξ is exogenous, *any* function of
   (current state, realized future ξ) is an unbiased baseline (Mao et al.,
   2019; Guo et al., 2021). An example is "rests needed from this state on
   these realized legs", from the nominal walk on realized D: microseconds,
   and it absorbs most of the luck.
4. **A decomposed critic.** `P(remaining rests = n | post-decision state)` by
   classification, plus `E[non-rest dwell]` by regression. This is a general
   value function predicting a cumulant (Sutton et al., 2011) and a structured
   distributional critic (Bellemare, Dabney & Munos, 2017). The rest count is
   the mixture variable, and cross-entropy is not dominated by 10 h outliers.
5. Oversampling rare events with importance weights (Frank, Mannor & Precup,
   2008).

**Exploration coverage.** Rolling in from guide states (Uchendu et al., 2023)
or from a restart distribution (CPI) gives the learner states where slack was
kept, which the student alone never reaches. Using the oracle's states in this
way is legitimate: labels still come from honest rollouts, so its
clairvoyance is not copied.

**Compute.** 5 k states × 4 actions × 48 rollouts × 0.3 s ≈ 80 core-hours per
round, about 3 h on the 32-core cluster. It reuses `fpi_collect` / `fpi_train`
/ `FPIPolicy`, and `gate_rollout`'s shared futures and `drive_fleet`.

### 7.3 Policy-gradient fine-tuning (PPO) of the PyTorch student

* **Policy.** `π_θ(a|s) ∝ exp(−Q̂_θ(s,a)/τ)` over the shielded set (masked
  softmax; Huang & Ontañón, 2022), initialized from the SplitNet cost head
  with τ₀ = 0.25 h, the listwise temperature already used in training. So at
  initialization the policy is the student. Charge duration: a Beta on
  `[lo, hi]`, or a discrete set {reach-next, break length, 80 % knee, full,
  spread room}. With fixed binaries the nominal problem is an LP, whose
  optima sit where such constraints are tight; this is a heuristic, not a
  theorem, under uncertainty.
* **Objective.** PPO-clip (Schulman et al., 2017) plus a KL term to the student
  policy, annealed (as in kickstarting and JSRL), with γ = 1 and GAE(λ)
  (Schulman et al., 2016) using a critic `V_φ` on the dwell, or Monte Carlo
  with the hindsight baseline. Potential-based shaping with Φ = −(rests still
  needed × ~10 h) preserves the optimal policy (Ng, Harada & Russell, 1999). It
  helps bootstrapped critics only: with γ = 1 it telescopes to a per-route
  constant in Monte Carlo returns.
* **Exploration.** Sample from the policy only at decisions with small margins
  (the gate's insight: 1 % of decisions hold 70–87 % of the one-step loss), and
  act greedily elsewhere.
* **Budget.** 10⁵–10⁶ episodes at ~0.3 s per medium route, about 100 core-hours
  at the upper end. Plain PyTorch, no new packages (installing a library like
  stable-baselines3 would need your approval). At least 3 seeds (Henderson et
  al., 2018).

### 7.4 Fitted value iteration with the known dynamics (critic for 7.2 / 7.3)

```
V^x(s^x) = E_ξ [ min over a ∈ A_shield(s') of  c(s',a) + V^x(s'^x(a)) ],     s' = s^x + leg(ξ)
```

The expectation is one quadrature over ξ, so the targets carry no sampling
noise (tree-based FQI: Ernst, Geurts & Wehenkel, 2005). Trees can represent
the jumps where the rest count changes. The cons: bootstrapping error over
50–160 steps, stop indices that do not align across instances (remaining-route
features are needed), and off-policy instability. Best used as a critic, not
alone.

### 7.5 Journal scale: the value inside the optimizer

* **A learned terminal cost `V̂(S_h)` in the LA's MILP** removes `b_H`: the LA would
  see rests beyond 24 h and price the slack it leaves at the horizon end. It
  stays MILP-representable if V̂ is piecewise linear: tree leaves via big-M,
  or ReLU networks (Delarue, Anderson & Tjandraatmadja, 2020; PARL, Harsha et
  al., 2025; CORL, Anand et al., 2026). MPC with a learned value is covered by
  Lowrey et al. (2019), Bhardwaj, Choudhury & Boots (2021), and Bertsekas's
  (2022) "approximation in value space + on-line play". It does **not** remove
  `EVPI_H`; that needs non-anticipative scoring, such as the rollout check,
  or a scenario tree.
* **An RL-trained gate.** Add "call the LA" as an action, at a cost λ per call.
  Today's gate + check is the one-step version of this.

### Not recommended

* **RL from scratch.** A rest is admissible at almost every stop, so a uniform
  shielded policy rests dozens of times per route, and exploration is
  hopeless. It is also unstable across seeds.
* **Offline RL on the LA logs.** There is one realized trajectory per route
  (830 runs), 89 % of actions are "drive on", and the logged Q-values are
  horizon estimates, not returns. With a free, exact simulator there is no
  reason to restrict ourselves to logs.
* **Learning a model.** The dynamics are known exactly.

---

## 8. Features the critic needs

* Remaining-route structure: the rests still needed (nominal walk;
  `rt_rests_left` exists), the daily driving left in the final shift
  (`rt_last_slack` exists, and was G's top feature), and new ones:
  `σ_rem = √(Σ_j D_j² Var ξ)`, the remaining mandatory dwell (service, the
  charging needed, breaks), the reduced-rest and extension budgets against the
  rests remaining, and which pre-rest cap is in force (13 or 15 h).
* Without them the critic cannot represent P(extra rest): states that differ
  in it look the same to the model (state aliasing). That is the same failure
  as the ferry in THEORY.md §5.

---

## 9. Evaluation protocol

* Same splits. Choose on validation (20–21), test once (22–25).
* Compare with the student paired, on thousands of fresh draws (cheap). Compare
  with the LA only on the stored realizations (re-running it costs 234 h of
  8-core compute per 125 routes). On 125 base routes the se is ≈ 0.3–0.4 pp,
  so about 0.7–1 pp is detectable at 2 se.
* Report the mean and median gap, the routes > 8 %, extra-rest counts against
  the LA and the oracle, window misses, P(extra rest) per route from fresh
  draws, and violations **under strict rules**, for the LA as well.

---

## 10. Limitations

1. **The ceiling is π\*.** An unknown part of the LA's tail is value of
   information, which no policy can remove.
2. **Rare-event variance.** A return sd of 2–3 h against effects of tenths of an
   hour; samples scale as (2σ/δ)².
3. **Detectability.** 125 test routes with one realization each; the LA itself
   varies by 0.35 % (median) between identical runs.
4. **Specification gaming.** RL optimizes the simulator, not the regulation:
   the pre-rest and terminal caps today. Other rules are out of scope: the
   weekly limits are diagnostic only, and the working-time directive
   (`wtd_rules`) is off in the base case. Every rule left out is a potential
   exploit.
5. **Model fidelity.** i.i.d. ξ (AR(1) ρ = 0), queue times known in advance,
   no time-of-day traffic, energy a deterministic function of speed, chargers
   always available, a hard travel-time cap at 1.6. A policy tuned to operate
   near the limits inside the modelled support has no margin for unmodelled
   events. The LA plans with the same model, but it re-plans and does not
   learn the model's quirks.
6. **Generalization.** RL does not fix extrapolation (150 kW); it needs
   training across physics, like imitation.
7. **Credit assignment** over 50–160 decisions at γ = 1.
8. **Local optima and policy oscillation** of approximate policy iteration
   under function approximation (Bertsekas & Tsitsiklis, 1996), hence
   conservative updates.
9. **Interpretability.** There is no plan to show a driver; the critic's
   P(extra rest) and slack can be shown instead.
10. **Engineering.** A batched environment is needed; Python features dominate
    the time per step.
11. **Guarantees.** No global optimality. Safety holds only for the modelled
    rules, within the modelled support.
12. **Timeline.** The CPAIOR abstract is due 1 Dec and the paper 7 Dec 2026.
    7.2–7.5 are journal-scale; for CPAIOR, at most 7.1 as a short result, or
    this as future work.

---

## 11. Literature map (checked 2026-10-26)

| area | work | what it gives us |
|---|---|---|
| closest application | Bai, Li, Johansson & Mårtensson (2023), IEEE L-CSS | e-truck charging under HoS; deterministic; rollout |
| | Orfanoudakis et al. (2026), arXiv 2604.26566 | RL for e-truck fleets, rule-based action mask; no driving-time rules |
| | Basso et al. (2022), TR-E 157 | safe RL for stochastic EV routing; depletion risk; no HoS |
| | Kullman, Goodson & Mendoza (2021), TS 55(3); Kullman et al. (2022), TS 56(3) | rollout with a dual bound for public charging queues; deep RL for EV ride-hailing |
| | Simão et al. (2009), TS 43(2) | ADP with driver HoS attributes at fleet scale, post-decision values; no charging |
| exogenous inputs | Dietterich, Trimponias & Chen (2018); Mao et al. (2019); Guo et al. (2021); Sinclair et al. (2023) | remove exogenous reward; input-dependent and hindsight baselines; hindsight learning and its bias |
| bounds | Brown, Smith & Sun (2010); El Shar & Jiang (2020) | information-relaxation duality; bounds inside Q-learning |
| policy iteration | Kakade & Langford (2002); Lagoudakis & Parr (2003); Lazaric et al. (2016); Chang et al. (2015) | conservative PI, restart distributions; classification-based API with regret weights; local optimality |
| rollout and MPC + value | Bertsekas (2022); Goodson, Thomas & Ohlmann (2017); Lowrey et al. (2019); Bhardwaj et al. (2021) | rollout theory, post-decision rollouts; learned terminal values |
| value inside a MILP | Delarue et al. (2020); Harsha et al. (2025); Anand et al. (2026) | ReLU or tree values in the action-selection MIP |
| from a prior policy | Uchendu et al. (2023); Silver et al. (2018); Bourgeat, Legrain & Cappart (2026) | guide-policy roll-in; residual RL; imitate, then RL |
| policy search | Ng & Jordan (2000) | common random numbers make policy evaluation deterministic |
| practicalities | Huang & Ontañón (2022); Schulman et al. (2016, 2017); Ng, Harada & Russell (1999); Frank, Mannor & Precup (2008); Henderson et al. (2018) | masking; PPO / GAE; shaping; rare events; seed variance |

**Gap.** I found no RL work that combines EU driving-time rules, charging and
uncertain travel times for a truck. The closest are deterministic (Bai et
al.), lack driving-time rules (Orfanoudakis et al., Basso et al., Kullman et
al.), or lack charging (Simão et al.).

---

## 12. Concrete next steps, if we go

| step | what | cost |
|---|---|---|
| 0a | RL environment in `ML/code`: strict spread, the invariant shield of §4, the dwell reward, γ = 1, batched like `drive_fleet` | 1–2 days |
| 0b | random-policy safety test: 0 violations on thousands of draws under strict rules | ~1 h |
| 0c | re-state the students and the LA under strict rules | ~1 h of student time; LA from stored runs |
| 0d | headroom test on the LA's 60 extra-rest states with fresh draws | a few core-hours |
| 1 | policy search over θ (§7.1) on training routes, choose on validation | 10–15 core-hours |
| 2 | API rounds 1–3 for the trees (§7.2) | ~80 core-hours per round |
| 3 | PPO fine-tuning of the PyTorch student (§7.3) | ~100 core-hours + tuning |
| 4 | journal: terminal value in the LA, dual bounds, an RL gate | months |

---

## 13. Progress log

The simulator's default rules are kept (user's decision, 2026-10-26). All
runs use the trees student `gbt_F95_base_s0` under the final protocol unless
stated otherwise.

**Environment (`ML/code/rl_env.py`, checked by `rl_check.py`).** Fresh
travel-time draws come from each route's own law (`generate_scenarios`, the
same sampler as `D_real`). Many vehicles run in lockstep with one batched
prediction per stop. Each job has its own chooser, and vehicles can restart
from mid-route clones with a forced first action (`ForceFirst`). It
reproduces the evaluated trees and PyTorch students exactly on all 66
validation routes, at 1.3–1.6 ms per vehicle-step.
`rests_needed` counts the remaining daily rests including the 10 h days. The
old `fpi_policy.walk_to_end` ignores those days, so it miscounts the rests
at the end of a route.

**Step 1, policy search (`rl_search.py`, grid v2, 639 training routes × 4
draws): negative.** No policy-wide correction improves the student:

* resting earlier is +0.33 to +2.18 % worse;
* resting later is −0.01 ± 0.05 % (medium −0.25, short +0.26);
* a window penalty is −0.06 ± 0.02 %;
* the final-shift slack term is ≈ 0.

Under fresh draws the rest count changes on 0 / 11 / 15 % of short /
medium / long routes. So the headroom cannot be reached by one bias applied
everywhere.

**Step 2, targeted rollout labels (`rl_api.py`, first 69 medium and long
routes).** At decisive states (a rest within 2 h of the best non-rest, or the
two best within 0.25 h), the student's pick is beaten by more than 2 se on
~30 % of states; chance would give ~5–7 %. "Rest now" is overpriced: rests
that keep the nominal rest count measure +0.2 / +1.3 / +3.0 h against
+1.7 / +5.8 / +8.1 h predicted (allowance lost 1–2 / 2–4 / 4–11 h). This is
the truncation bias `b_H` of §2.

**The labels are signal.** The same 36 states were re-labelled with
independent futures (`ML/scratch/label_replication.py`):

* correlation 0.995 overall, 0.95 within 1 h of the pick;
* challengers chosen on run A measure −0.52 ± 0.17 h on run B, and 91 % are
  better there too;
* acting on run A's best challenger is worth −0.18 ± 0.09 h per decisive
  state, measured on run B.

**Amortizing the labels (`rl_api_fit.py`): not yet.** Two bugs were found
and fixed on the way:

* fitting G on the pick's own row moved every challenger at once;
* the L2 fit was dominated by far-off rests (fixed by training only within
  3 h and splitting smooth from rest-count parts).

The corrected G fits in sample (−0.034 ± 0.019 h per state) but is ≈ 0 on
held-out routes, with ~360 training states. More labels are needed; the
collection is paused at 69 / 411 routes and resumable.

**Decisive-state rollout (`rl_policy.RolloutChooser`): in validation.** No
model, no solver: at each decisive state, the top-4 candidates (always the
best rest and the best non-rest) are tried with the student driving on,
under 32 shared futures. Long routes have ~50 decisive states, medium and
short ~28, so a long route takes ~20 min of one core with racing (8 futures
per round), against hours for the LA. Racing gives the same decisions in
half the time.

**Validation result** (66 routes, stored realizations, manuscript gap with
window penalty, `ML/scratch/dsr_vs_la.py`):

| policy | gap to oracle | vs LA (pp) | routes resting more than the LA |
|---|---|---|---|
| LA | 3.04 % | — | — |
| trees student | 4.40 % | +1.36 ± 0.40 | 2 |
| student + decisive-state rollout | 2.76 % | −0.29 ± 0.20 | 0 |

* vs the LA, by length: short +0.11 ± 0.47, medium −0.36 ± 0.21, long
  −0.70 ± 0.20 pp;
* vs the student: −1.64 ± 0.35 pp, better on 57 of 66 routes;
* about 5 changed decisions per route, 0 broken runs;
* 81 / 276 / 1000 s of one core per short / medium / long route;
* its parameters were fixed a priori; the test routes have not been run.

**Test result** (2026-10-27, run once on the cluster;
`ML/code/rl_test_report.py`, reports in `results/rl_test_report_*.json`).
Three training seeds, averaged per route; gap to the oracle with the window
penalty. The student, LA and Greedy numbers reproduce the paper's Table 1.

| routes | policy | gap | vs LA (pp) | vs its student (pp) |
|---|---|---|---|---|
| base, 125 | LA | 3.05 % | — | — |
| | PyTorch student | 3.55 % | +0.50 ± 0.37 | — |
| | PyTorch + rollout | **2.38 %** | **−0.67 ± 0.34** | −1.17 ± 0.19 (better on 118 / 125) |
| | trees student | 3.73 % | +0.68 ± 0.43 | — |
| | trees + rollout | **2.33 %** | **−0.72 ± 0.33** | −1.40 ± 0.26 (109 / 125) |
| | Greedy | 7.01 % | +3.95 ± 0.46 | |
| mixed, 61 certified | LA | 2.73 % | — | — |
| | mixed student | 5.52 % | +2.79 ± 0.70 | — |
| | mixed student + rollout | **2.71 %** | **−0.02 ± 0.26** | −2.81 ± 0.61 (56 / 61) |
| | equal-power student | 8.07 % | +5.34 ± 0.86 | — |
| | equal-power + rollout | 4.99 % | +2.26 ± 0.63 | −3.08 ± 0.94 (48 / 61) |

* 0 of the 1,500 base and 732 mixed runs broke a rule.
* Routes resting more than the LA: PyTorch 4 → 3, trees 5 → 2, mixed 7 → 3.
* One core per route on the cluster, both policies: PyTorch 11 / 36 / 166 s
  and trees 40 / 105 / 342 s (short / medium / long).
* For comparison, the LA-calling hybrid (gate + check) was −0.36 ± 0.33 pp vs
  the LA, with 8.1 LA calls per route.

On mixed routes, rollout with equal-power training alone (4.99 %) ends up
about where the mixed-trained student alone is (5.52 %). It does not replace
the mixed data, but adds to it.

Read with §6: improving at the student's own decisive states, with the
student continuing, is LOLS-style local improvement. It is enough to bring
a +1.36 pp student to LA parity without a solver. Most of the gain is timing
(dwell, charging, windows); on these routes the rest count moved only where
the student had taken an extra rest.

## References (links checked 2026-10-26)

* Alshiekh et al. (2018). Safe reinforcement learning via shielding. AAAI.
* Anand, Aarekol, Dalseg, Stålhane & Gros (2026). CORL: Reinforcement learning of MILP policies solved via branch-and-bound. CPAIOR.
* Bai, Li, Johansson & Mårtensson (2023). Rollout-based charging strategy for electric trucks with hours-of-service regulations. IEEE Control Systems Letters. https://arxiv.org/abs/2303.08895
* Basso, Kulcsár, Sanchez-Diaz & Qu (2022). Dynamic stochastic electric vehicle routing with safe reinforcement learning. Transportation Research Part E 157. https://research.chalmers.se/publication/527406
* Bellemare, Dabney & Munos (2017). A distributional perspective on reinforcement learning. ICML.
* Bertsekas (2022). Lessons from AlphaZero for Optimal, Model Predictive, and Adaptive Control. Athena Scientific. https://web.mit.edu/dimitrib/www/abstractdp_MIT.html
* Bertsekas & Tsitsiklis (1996). Neuro-Dynamic Programming. Athena Scientific.
* Bhardwaj, Choudhury & Boots (2021). Blending MPC & value function approximation for efficient reinforcement learning. ICLR. https://arxiv.org/abs/2012.05909
* Bourgeat, Legrain & Cappart (2026). Imitation-guided world models for multi-agent train rescheduling. CPAIOR.
* Brown, Smith & Sun (2010). Information relaxations and duality in stochastic dynamic programs. Operations Research 58(4):785–801.
* Chang, Krishnamurthy, Agarwal, Daumé III & Langford (2015). Learning to search better than your teacher. ICML.
* Delarue, Anderson & Tjandraatmadja (2020). Reinforcement learning with combinatorial actions: an application to vehicle routing. NeurIPS. https://arxiv.org/abs/2010.12001
* Dietterich, Trimponias & Chen (2018). Discovering and removing exogenous state variables and rewards for reinforcement learning. ICML. https://proceedings.mlr.press/v80/dietterich18a.html
* El Shar & Jiang (2020). Lookahead-bounded Q-learning. ICML. https://proceedings.mlr.press/v119/shar20a.html
* Ernst, Geurts & Wehenkel (2005). Tree-based batch mode reinforcement learning. JMLR 6.
* Frank, Mannor & Precup (2008). Reinforcement learning in the presence of rare events. ICML. https://cs.mcgill.ca/~jfrank8/pubs/icml08rareevents.pdf
* Garaix, Lacomme & Peña-Arenas (2024). A label-setting algorithm for the truck driver scheduling problem in accordance with European Community social legislation. Expert Systems with Applications.
* Goel (2010). Truck driver scheduling in the European Union. Transportation Science.
* Goodson, Thomas & Ohlmann (2017). A rollout algorithm framework for heuristic solutions to finite-horizon stochastic dynamic programs. EJOR 258:216–229.
* Guo et al. (2021). Hindsight value function for variance reduction in stochastic dynamic environment. IJCAI. https://arxiv.org/abs/2107.12216
* Harsha, Jagmohan, Kalagnanam, Quanz & Singhvi (2025). Deep policy iteration with integer programming for inventory management. M&SOM 27(2):369–388. https://arxiv.org/abs/2112.02215
* Henderson, Islam, Bachman, Pineau, Precup & Meger (2018). Deep reinforcement learning that matters. AAAI.
* Huang & Ontañón (2022). A closer look at invalid action masking in policy gradient algorithms. FLAIRS 35. https://arxiv.org/abs/2006.14171
* Kakade & Langford (2002). Approximately optimal approximate reinforcement learning. ICML.
* Kullman, Goodson & Mendoza (2021). Electric vehicle routing with public charging stations. Transportation Science 55(3):637–659.
* Kullman, Cousineau, Goodson & Mendoza (2022). Dynamic ride-hailing with electric vehicles. Transportation Science 56(3):775–794.
* Lagoudakis & Parr (2003). Reinforcement learning as classification: leveraging modern classifiers. ICML.
* Lazaric, Ghavamzadeh & Munos (2016). Analysis of classification-based policy iteration algorithms. JMLR 17(19):1–30.
* Lowrey, Rajeswaran, Kakade, Todorov & Mordatch (2019). Plan online, learn offline: efficient learning and exploration via model-based control. ICLR. https://arxiv.org/abs/1811.01848
* Mao, Venkatakrishnan, Schwarzkopf & Alizadeh (2019). Variance reduction for reinforcement learning in input-driven environments. ICLR. https://people.csail.mit.edu/hongzi/var-website/index.html
* Ng & Jordan (2000). PEGASUS: a policy search method for large MDPs and POMDPs. UAI. https://arxiv.org/abs/1301.3878
* Ng, Harada & Russell (1999). Policy invariance under reward transformations: theory and application to reward shaping. ICML.
* Orfanoudakis, Li, Yang, Aristov, Vergara, Fan & Dugundji (2026). Learning to route electric trucks under operational uncertainty. arXiv 2604.26566.
* Ross & Bagnell (2014). Reinforcement and imitation learning via interactive no-regret learning. arXiv 1406.5979.
* Schulman, Moritz, Levine, Jordan & Abbeel (2016). High-dimensional continuous control using generalized advantage estimation. ICLR.
* Schulman, Wolski, Dhariwal, Radford & Klimov (2017). Proximal policy optimization algorithms. arXiv 1707.06347.
* Silver, Allen, Tenenbaum & Kaelbling (2018). Residual policy learning. arXiv 1812.06298.
* Simão, Day, George, Gifford, Nienow & Powell (2009). An approximate dynamic programming algorithm for large-scale fleet management: a case application. Transportation Science 43(2):178–197.
* Sinclair et al. (2023). Hindsight learning for MDPs with exogenous inputs. ICML.
* Sutton, Modayil, Delp, Degris, Pilarski, White & Precup (2011). Horde: a scalable real-time architecture for learning knowledge from unsupervised sensorimotor interaction. AAMAS.
* Uchendu et al. (2023). Jump-start reinforcement learning. ICML. https://proceedings.mlr.press/v202/uchendu23a.html
* Wabersich & Zeilinger (2021). A predictive safety filter for learning-based control of constrained nonlinear dynamical systems. Automatica.
* Yoon, Fern, Givan & Kambhampati (2008). Probabilistic planning via determinization in hindsight. AAAI.

Entries without a link were not re-opened today (standard references, or
already in `ML/paper/references.bib`). Check volume and pages before citing.
