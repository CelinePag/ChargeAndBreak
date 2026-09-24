# Method — what was built, why, and what it means

A walkthrough of the ML arm: the idea, the theory behind it, the
implementation, the results, how they connect to the manuscript's own numbers,
and what is still missing.

---

## 1. The problem, in one paragraph

The look-ahead policy (`LA`) decides, at every stop, whether to charge, whether
to take a break or a rest, and for how long. It decides by solving a
rolling-horizon MILP over ~41 stops ahead, under 25 sampled travel-time
scenarios, once per candidate action. That costs **~68 seconds per decision**
and roughly three hours per long route. The question is whether a model can
learn to make the same decision in milliseconds, with no solver at deployment.

This is *imitation learning*: a cheap **student** clones an expensive
**teacher**.

---

## 2. The central design decision

### The obvious approach, and why it is wrong here

The obvious framing is **classification**: at each stop the teacher picked one
of 12 actions (`y ∈ {0,1}` × {nothing, b15, b30, b45, r1, r2}); train a
classifier to predict which.

Three measurements from the base-case data say this is the wrong problem:

| measurement | why it breaks classification |
|---|---|
| **88.8%** of chosen actions are "just drive on" | the loss is dominated by a class nobody needs help with |
| **13.6%** of decisions have a best-vs-second margin below twice the teacher's own sampling noise | the label is a coin flip; a classifier is asked to fit noise |
| error costs span two orders of magnitude — a spurious daily rest is **9–11 h**, a b15/b45 mix-up is **minutes** | cross-entropy weights both the same |

The previous (deleted) attempt at this project hit exactly that wall: its
class weighting gave "drive on" weight 0.035 and a two-example class weight
5.3 — a 150× ratio that made predicting an 11-hour rest *nearly free*. Fixing
that was its single largest result.

### What I did instead: score the actions, then take the argmin

**Easy version.** Instead of asking "which action did the teacher pick?", ask
"how much would *each* action cost?" — then pick the cheapest legal one. It is
the difference between memorising someone's answers and learning to grade the
options yourself.

**Detailed version.** For every enumerated action *a* at state *s*, predict the
**regret**

```
regret(s, a) = cost(s, a) − min over clean a' of cost(s, a')
```

in hours, where `cost` is the teacher's own scenario-mean horizon objective.
At deployment:

```
A  = enumerate_actions(stop, state, data)      # the simulator's legality rules
A  = [a in A if action_passes(...)]            # the supervisor's forcing rules
score[a] = cost_head(s, a) + BIG · [feas_head(s, a) < 0.5]
a*       = argmin score
tauc*    = clip(tauc_head(s, a*), reach-next-CS, charge-to-full)
```

Three consequences, all of which the data confirms:

1. **The imbalance disappears.** There are no classes to be imbalanced. The
   150× weighting bug is *structurally impossible* in this formulation.
2. **The loss is the decision cost.** A regression error of 9 hours produces
   a gradient 100× that of a 5-minute error — automatically, with no weighting.
3. **5.5× more supervision, for free.** The teacher logs the scenario-mean
   cost of **every action it enumerated**, not just the winner. That is
   **411,118 rows instead of 74,380**, and rare actions (`y1_r1`, 0.11% of
   choices) get gradient wherever they were *scored*.

### The theory

This is a **cost-sensitive reduction** (Beygelzimer, Langford and colleagues on
error-limiting reductions; and Elmachtoub & Grigas's *Smart Predict-then-
Optimize* for the same point in an optimisation setting). The relevant
guarantee: if the regressor's error on every action is at most *r*, the
argmin's decision regret is at most **2r**. Prediction error converts directly
into decision quality. Plain classification gives you no such statement —
there is no theorem connecting 1% of probability mass to hours of route time.

The counterweight is **compounding error**. Ross & Bagnell (2010) show that
behavioural cloning with per-step error ε incurs regret up to **O(εT²)** under
the learner's own state distribution, because each mistake moves the state
off-distribution. DAgger (Ross, Gordon & Bagnell, 2011) reduces this to
**O(εT)**. Here **T ≈ 88 decisions per route**, so T² ≈ 7,700 — the theory says
distribution shift should dominate, and that DAgger should matter more than
model capacity. *(The results below complicate this, interestingly.)*

**Why trees.** Grinsztajn, Oyallon & Varoquaux (2022) identify why
gradient-boosted trees still beat neural networks on tabular data: networks are
biased toward *overly smooth* functions, they degrade with uninformative
features, and their first layer is rotation-invariant, which is a bad prior
when each feature is a distinct physical quantity. All three apply. The
decisive one is smoothness: the feasible region here is *exact thresholds* —
4.5 h consecutive driving, 9 h shift driving, 13/15 h spread. A tree split
**is** such a threshold; a network approximates a step with a ramp whose error
peaks precisely at the boundary, which is where the decision flips.

---

## 3. The pipeline

### Stage 0 — Extraction (`extract.py`, `parse_logs.py`, `features.py`)

Labels come from the LA **logs**, which record every scored action:

```
[LA] stop 30 (CS)  t=18.899h  soc=251kWh  cd=1.82h  sd=8.41h ...
  y=0  brk=0    rst=0   41.651h (0.163h)  ok=25/25  tauc=0m
  y=1  brk=0    rst=0   41.790h (0.197h)  ok=25/25  tauc=26m
  y=1  brk=b45  rst=0   41.917h (0.190h)  ok=25/25  tauc=45m   ...
```

State comes from the stored **solution** JSONs. The two are joined on the stop
index.

**Two traps in the stored data.**

1. *The runs cannot be replayed naively.* `BEHDV.advance` takes the executed
   break/rest from the nominal MIP's flags, but `vehicle.actions` stores the
   action the look-ahead *selected*. They disagree whenever the re-solve moved
   the break, and the executed flags are never written to disk — a replay
   drifts silently (`phi` off by one, `cd` off by up to 2.7 h on the first
   three routes tried). So states are read from `sim_trajectory` instead.
2. *The shift spread `h` is not stored anywhere*, despite being a hard 13/15 h
   constraint. It is exactly reconstructible:
   `h[k+1] = (0 if rest else h[k] + o_dwell) + D_actual[k]`.
   *(The deleted tree found this too — not a new discovery.)*

**Four validation gates**, run on every extraction, because a silent join
error would poison everything downstream:

| gate | check | result |
|---|---|---|
| G1 | the log's state line agrees with the solution's trajectory | pass |
| G2 | `t_arr[k+1] == td[k] + D_actual[k]` | pass |
| G3 | `t_arr[last] − T_START == duration_h` | pass |
| G4 | reconstructed spread stays in [0, 15] h on feasible runs | pass |

Output: **830 usable runs, 72,595 decisions, 411,118 rows, 91 features.**

### Features — the one design that matters

**Slacks, not levels.** Feed `cd_slack = 4.5 − cd` rather than `cd`. This puts
every regulatory threshold at zero. A tree then splits at 0 instead of having
to discover 4.5; and it is also exactly where a ReLU hinges, so the same
features serve a network.

**Typed lookahead, not raw stops.** The teacher's window is a median of 41
stops, but two thirds are identical laybys. Instead of 6×41 raw columns,
describe the **next 3 charging stations** and **next 2 customers** by type,
plus a horizon summary reusing the teacher's own `find_horizon_end_stop`. 91
features instead of ~250.

**Hard rules are not learned.** Legality (`enumerate_actions`), forcing
(`compute_flags` / `action_passes`) and the charge clamp come from the
simulator itself. Reimplementing them would eventually diverge and produce a
policy that looks good and is illegal.

### Stage 1 — Three heads (`gbt_train.py`)

| head | target | rows |
|---|---|---|
| cost | regret in hours (Huber) | clean rows only |
| feasibility | P(feasible in all 25 scenarios) | all rows |
| charge duration | `tauc` in hours | `y=1` rows |

Cost trains on **clean rows only** because `INFEASIBLE_PENALTY` (1e9) enters
the teacher's scenario mean — an action failing 2 of 25 scenarios is recorded
at ~8e7 h, which is not a duration and whose difference from the best is not a
regret. Those rows are what the feasibility head is for.

**Splits by seed within family** (1–17 / 18–21 / 22–25), never by row: the ~88
decisions of one route are a Markov chain. Effective sample size is the **569
independent routes**, not 284k rows — capacity is sized against that.

### Stage 2 — Deployment (`policy_core.py`, `gbt_policy.py`)

The policy owns its simulator loop, mirroring `greedy.run_greedy`, rather than
registering a method in `runner_dispatch.py` — everything stays under `ML/`,
and the reporting pipeline globs `solutions/<bucket>/` by method name, so a
stray student run in the main tree would silently enter the manuscript's
tables.

---

## 4. Results

### Headline — held-out test split, 122 completed routes

| comparison | median | 95% CI | Wilcoxon p | verdict |
|---|---:|---|---:|---|
| vs **LA** (teacher) | **−0.01%** | [−0.20, +0.17] | 0.92 | **indistinguishable** |
| vs **GREEDY** | −2.47% | [−2.77, −2.18] | 8e−21 | faster |
| vs **ORACLE** | +2.18% | [+1.95, +2.29] | 9e−22 | 2.2% above optimum |

Faster on exactly 61 of 122 routes — a literal coin flip (sign test p = 1.00).

A **practical floor of 0.35%** applies throughout: the look-ahead's own
measured run-to-run spread on this simulator. A difference below it is not
evidence of anything, however significant.

### The check that matters most

Whenever a student matches its teacher, suspect it of cutting corners. It
isn't: **278 rests vs the teacher's 279**, identical on 115/122 routes. And
the tail has a single mechanism — correlation **+0.887** between the duration
gap and the rest-count difference. Every route more than 5% off took one rest
more (slower) or one fewer (faster).

### Ablations (validation)

| variant | vs LA | infeas. | what it shows |
|---|---:|---:|---|
| `base` | −0.20% | 4 | the design |
| `rawcost` | **+3.41%** | **14** | **centring the target is the whole ballgame** |
| `noweight` | −0.16% | 2 | margin weighting is inert |
| `shallow` | +0.16% | 9 | capacity is not the binding constraint |
| `base_g95` | +0.10% | **0** | the guard is the feasibility lever |

`rawcost` is the important row. Regressing the teacher's raw horizon objective
— whose *level* has std 29.6 h against a decision *margin* of ~40 min — means
99.95% of explained variance is "how much route is left". It produces a policy
3.4% slower than the teacher that infeasible runs 3.5× as often.

### Cost

~4 ms per decision against the teacher's ~68 s — **order 10⁴**. Read it as an
order of magnitude, not three digits: it was measured on a loaded machine and
is dominated by Python feature construction and three separate LightGBM
`predict` calls, so it is an upper bound.

### Trees vs the simplest neural network

Same rows, same targets, same splits, same decision rule (`policy_core.py`) —
only the regressor differs.

| | vs LA | vs GREEDY | infeas. /136 | p95 vs LA |
|---|---:|---:|---:|---:|
| GBT, guard 0.95 | **+0.10%** | −2.21% | **0** | +2.3% |
| MLP, guard 0.95 | +2.79% | +0.60% | 8 | +16.9% |

The network ends up level with GREEDY — it recovers little of what the
look-ahead buys over a myopic rule — and it over-rests, which is where its
heavy right tail comes from. It was **not promoted to test**: it lost on
validation, and that is what validation is for.

This is a claim about the *simplest* network (three plain MLPs, no shared
trunk, no listwise loss), not about neural networks in general.

---

## 5. How this connects to the manuscript

The ML tree reports "% vs LA" because the look-ahead is the teacher. **The
manuscript reports gap to the hindsight ORACLE** — `data_output/paper_gap_stats.csv`
and `figures/basecase/paper_gap_box*.png`.

`paper_link.py` recomputes that gap for the student using
`compile_solutions._annotate_gap_to_oracle`'s exact definition, on the same
instances (see `RESULTS_PAPER.md`):

| method | n | gap to oracle (median) | penalised |
|---|---:|---:|---:|
| **ML student (GBT)** | 122 | **+2.18%** | +2.50% |
| LA (look-ahead MILP) | 125 | +2.14% | +2.31% |
| 2SP | 104 | +2.10% | +2.18% |
| GREEDY | 120 | +4.90% | +5.68% |
| RO | 125 | +32.42% | +35.15% |

**This is the row for the paper.** The student sits within 0.04 pp of the LA
and level with 2SP, at GREEDY's online cost. It more than halves GREEDY's gap
to the oracle while needing no solver at all.

### Which figures to look at

| figure | what it answers |
|---|---|
| `ML/figures/fig1_distributions_test.png` | the headline — distribution and tail vs LA / GREEDY / ORACLE, not just a median |
| `ML/figures/fig4_arms_val.png` | trees vs network, the cleanest single picture of that comparison |
| `ML/figures/fig2_ablations_val.png` | which design choices mattered (`rawcost` is the tall bar) |
| `ML/figures/fig3_families_test.png` | is the failure concentrated or diffuse? |
| `figures/basecase/paper_gap_box.png` | **the manuscript's** existing box plot; the student's row belongs in it |

---

## 6. What still needs doing

### Blocking, if this becomes a paper

1. **Time windows.** The student misses 122 windows against the teacher's 57 —
   more than double. Duration is what the cost head trains on; window
   compliance only reaches it through the small β·δ term in the teacher's
   objective. **Fix:** train on the penalised cost, or add a window-miss head
   and include it in the score. This is the clearest known deficiency.
2. **Feasibility does not fully generalise.** The selected guard gave 0 halts
   on validation and 3 on test, all shift-spread. The guard reduces halts; it
   does not remove them. **Fix:** a spread-specific guard quantile, or fold the
   spread margin into the feasibility head's threshold.
3. **Seed variance is unmeasured.** LightGBM with fixed seed is deterministic,
   so there is one number per config. The neural arm was also a single seed.
   Train 3–5 seeds of each and report mean ± spread before publishing any
   comparison this close to the 0.35% floor.

### High value, not blocking

4. **DAgger.** Theory says this is the largest available gain (O(εT²) → O(εT),
   T = 88). Notably, the current result **does not need it** — the previous
   attempt required DAgger to reach +0.186% whereas cost-scoring reaches ≈0%
   with plain cloning. Worth testing whether DAgger still helps the tail (the
   ±1-rest routes) at ~68 s per query.
5. **Risk-sensitive policy.** The logs record the scenario **spread**, not just
   the mean, and nothing uses it yet. Learning both enables `mean + λ·std` at
   zero online cost — a policy the LA does not have, since it optimises the
   mean. This is the most interesting unexplored direction: it could make the
   student *different from and better than* its teacher rather than an
   approximation of it.
6. **Latency.** ~4 ms is an upper bound. Batch the three heads into one call
   and cache the horizon walks; sub-millisecond is achievable, which would
   restore the 10⁵ claim.
7. **The proper neural arm.** One shared trunk scoring all 12 actions in a
   single pass, trained with a **listwise** loss over the action set at a
   temperature set by the teacher's own 2.08 min sampling SEM. That is the
   version with a genuine structural advantage over trees, and it is the only
   route to RL fine-tuning. Needs `torch` (available).

### Scope

8. **Generalisation is untested.** Only the seed split was run. The deleted
   tree also had *family* splits (train on short/medium, test on long) — the
   extrapolation study. Trees cannot extrapolate beyond their training range at
   all, so this is where a network might genuinely win.
9. **Base case only.** No sensitivity, diesel, battery or charger-power axes.
10. **No LA-LP baseline** — zero `LPTAIL` runs exist on disk; it would have to
    be re-run.
