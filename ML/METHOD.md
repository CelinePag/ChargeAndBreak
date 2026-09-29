# Method — what was built, why, and what it means

A walkthrough of the ML arm: the idea, the theory behind it, the
implementation, what was found, and what is still missing. How all of it
sits in the MDP the paper simulates, and which result confirms or bends
which piece of theory, is in [THEORY.md](THEORY.md). Every number quoted
here is in [RESULTS.md](RESULTS.md), which is generated from the result stores
by `code/report.py`.

---

## 1. The problem, in one paragraph

The look-ahead policy (`LA`) decides, at every stop, whether to charge, whether
to take a break or a rest, and for how long. It decides by solving a
rolling-horizon MILP over ~41 stops ahead, under 25 sampled travel-time
scenarios, once per candidate action. That costs **~70 seconds per decision**
and hours per long route. The question is whether a model can learn to make
the same decision in milliseconds, with no solver at deployment.

This is *imitation learning*: a cheap **student** clones an expensive
**teacher**.

---

## 2. The central design decision

### The obvious approach, and why it is the wrong default here

The obvious framing is **classification**: at each stop the teacher picked one
of 12 actions (`y ∈ {0,1}` × {go, b15, b30, b45, r1, r2}); train a classifier to
predict which. Three measurements from the base-case data argue against it:

| measurement | why it hurts classification |
|---|---|
| **88.8 %** of chosen actions are "just drive on" | the loss is dominated by a class nobody needs help with |
| **13.6 %** of decisions have a best-vs-second margin below twice the teacher's own sampling noise | the label is a coin flip; a classifier is asked to fit noise |
| error costs span two orders of magnitude — a spurious daily rest is **9–11 h**, a b15/b45 mix-up is **minutes** | cross-entropy weights both the same |

### What was done instead: score the actions, then take the argmin

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
Subtracting the best cost is what "**centring**" means: the raw cost's *level*
(std ~30 h: "how much route is left") is removed, leaving only the differences
between actions (~40 min) that the decision depends on. At deployment:

```
A  = enumerate_actions(stop, state, data)      # the simulator's legality rules
A  = [a in A if action_passes(...)]            # the supervisor's forcing rules
score[a] = cost_head(s, a) + BIG · [feas_head(s, a) < 0.5]
a*       = argmin score
tauc*    = clip(tauc_head(s, a*), reach-next-CS, charge-to-full)
```

Three consequences:

1. **The imbalance disappears.** There are no classes to be imbalanced.
2. **The loss is the decision cost.** A 9-hour error produces a gradient 100×
   that of a 5-minute error — automatically, with no weighting.
3. **5.7× more supervision, for free.** The teacher logs the scenario-mean
   cost of **every action it enumerated**, not just the winner: **411,118
   rows instead of 72,595 decisions**, and rare actions (`y1_r1`, 0.11 % of
   choices) get gradient wherever they were *scored*.

The classifier was kept anyway, as its own arm (`clf_*`), because it is the
framing of the deleted 2026-08 model and the controlled test of it: same
features, instances, splits and decision rule, only the learning differs. It
reads the state only and feeds `−log P(a)` into the same argmin.

### The theory

This is a **cost-sensitive reduction** (Beygelzimer, Langford and colleagues on
error-limiting reductions; Elmachtoub & Grigas's *Smart Predict-then-Optimize*
makes the same point in an optimisation setting). If the regressor's error on
every action is at most *r*, the argmin's decision regret is at most **2r**:
prediction error converts directly into decision quality. Plain classification
gives no such statement — nothing connects 1 % of probability mass to hours of
route time.

The counterweight is **compounding error**. Ross & Bagnell (2010) show that
behavioural cloning with per-step error ε incurs regret up to **O(εT²)** under
the learner's own state distribution, because each mistake moves the state
off-distribution; DAgger (Ross, Gordon & Bagnell, 2011) reduces this to
**O(εT)**. Here **T ≈ 88 decisions per route** (~160 on long routes). DAgger was
removed from this project: plain cloning already reaches the teacher on the
base case, and at ~70 s per teacher query it is the most expensive thing one
could add.

**Why trees.** Grinsztajn, Oyallon & Varoquaux (2022) identify why
gradient-boosted trees still beat neural networks on tabular data: networks are
biased toward *overly smooth* functions, they degrade with uninformative
features, and their first layer is rotation-invariant, a poor prior when each
feature is a distinct physical quantity. The decisive one here is smoothness:
the feasible region has *exact thresholds* — 4.5 h consecutive driving, 9/10 h
shift driving, 13/15 h spread. A tree split **is** such a threshold; a network
approximates a step with a ramp whose error peaks at the boundary, which is
where the decision flips.

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

State comes from the stored **solution** JSONs, joined on the stop index.

**Two traps in the stored data.**

1. *The runs cannot be replayed naively.* `BEHDV.advance` takes the executed
   break/rest from the nominal MIP's flags, but `vehicle.actions` stores the
   action the look-ahead *selected*. They disagree whenever the re-solve moved
   the break, and the executed flags are never written to disk — a replay
   drifts silently. So states are read from `sim_trajectory` instead.
2. *The shift spread `h` is not stored*, although it is a hard 13/15 h
   constraint. It is exactly reconstructible:
   `h[k+1] = (0 if rest else h[k] + o_dwell) + D_actual[k]`.

**Four validation gates** run on every extraction (log agrees with trajectory;
`t_arr[k+1] = td[k] + D_actual[k]`; final arrival = duration; reconstructed
spread in [0, 15] h on feasible runs). Output: **830 usable runs, 72,595
decisions, 411,118 rows**, one superset of **215 columns** (197 state + 18
action).

### Features, and the feature sets

**Slacks where the limit moves.** A slack against a *constant* threshold
(`cd_slack = 4.5 − cd`) is an affine transform of the level: a tree finds the
same split either way, and a standardised network sees the same input. Such
columns are exact duplicates and set D drops them. Slacks earn their place
when the **limit itself depends on the state**: the shift-driving limit is 10 h
while an extended day is left and 9 h after, the spread cap is 13 h before a
regular rest and 15 h before a reduced one, the energy margin subtracts a
worst-case need that depends on the next legs. There the slack hands the model
an interaction it would otherwise have to discover.

**Typed lookahead.** The teacher's window is a median of 41 stops, two thirds
of them identical laybys. Instead of 6 × 41 raw columns: the **next 3 charging
stations** and **next 2 customers** by type, plus a horizon summary reusing the
teacher's own `find_horizon_end_stop`.

A **feature set** is a named selection from the superset, applied identically
at training and serving (policies look columns up by name, `fsets.py`):

| set | what it keeps | trees / MLP | classifier |
|---|---|---:|---:|
| C | top-40 by LightGBM gain, from a model fitted on seeds 1–19 only | C40 | C28 |
| D | F minus 1 constant and 17 exact linear duplicates | D77 | D60 |
| F | every engineered feature | F95 | F77 |
| L | F + the next 20 stops raw (6 numbers each) | L215 | L197 |
| R | F minus the 7 whole-route position features | R88 | R70 |

The classifier reads state features only, hence its own counts. A model's
name carries its set and input count: `<arm>_<SET><n>_<config>[_SM][_s<seed>]`,
e.g. `gbt_F95_base_s1`, `clf_R70_base_SM_s2`.

**Hard rules are not learned.** Legality (`enumerate_actions`), forcing
(`compute_flags` / `action_passes`) and the charge clamp come from the
simulator. Reimplementing them would eventually diverge and produce a policy
that looks good and is illegal. (Section 6 found one place where the shared
check is incomplete for a learned policy.)

### Stage 1 — Training (`gbt_train.py`, `nn_train.py`, `clf_train.py`)

| arm | model | heads |
|---|---|---|
| Trees `gbt_*` | LightGBM | cost (Huber on regret), feasibility (binary), charge duration |
| MLP `mlp_*` | sklearn MLP + scaler | the same three; cost on `log1p(regret)` |
| Classifier `clf_*` | sklearn MLP classifier | 12-way action, charge duration |

Cost trains on **clean rows only**: `INFEASIBLE_PENALTY` (1e9) enters the
teacher's scenario mean, so an action failing 2 of 25 scenarios is recorded at
~8e7 h — not a duration. Those rows are what the feasibility head is for. The
MLP's cost target is `log1p(regret)` because squared error on raw regret lets
the few multi-hour rows dominate the loss (the `sqerr` ablation).

**Splits by route seed within family**, never by row — the ~88 decisions of a
route are a Markov chain:

| split | route seeds | routes | used for |
|---|---|---:|---|
| fit | 1–19 | 639 | fitting |
| stop | 20–21 | 66 | early stopping only |
| test | 22–25 | 125 | every reported number |

The effective sample size is the ~639 independent routes, not the rows. A
*route* seed picks the instance; a *training* seed initialises the model — each
configuration is trained with 3 training seeds and reported as mean ± sd.

### Stage 2 — Deployment (`policy_core.py`)

`policy_core.py` owns `enumerate_actions` → legality filter → forcing →
argmin → charge clamp → `BEHDV.advance`; each arm implements only
`_predict(rows)` and `_predict_tauc(row)`. The loop mirrors `greedy.run_greedy`
and stays under `ML/`: the reporting pipeline globs `solutions/<bucket>/` by
method name, so a stray student run in the main tree would silently enter the
manuscript's tables.

---

## 4. Results on the base case

Numbers: [RESULTS.md](RESULTS.md) §1–3. Figures: `fig_gap_test.png`,
`fig_ladder.png`, `fig_ablations.png`.

* **All three arms reach the teacher.** At its best feature set, every arm is
  within the 0.35 % practical floor of the LA (the look-ahead's own run-to-run
  spread) and ~2.4 pp ahead of Greedy, at milliseconds per decision.
* **Every arm is an inverted U in the number of inputs.** Too few (C) starve
  it; the raw 20-stop lookahead (L) adds noise and infeasible runs. The best
  sets differ by arm — F95 for trees, F77 for the classifier, D77 for the MLP —
  which is why the sets are not forced to be equal.
* **Centring the target is the design choice that matters.** Regressing the raw
  horizon cost instead of the regret costs the trees ~3.5 pp and the MLP ~43 pp.
* **Time windows remain the open weakness**: 102–146 misses against the
  teacher's 57, on every arm and set.

---

## 5. Generalisation

### Trained on short and medium routes, tested on long ones

`run_length.py`, `--scope SM`; [RESULTS.md](RESULTS.md) §4,
`fig_length*.png`. Physics and rules unchanged, only the route length unseen:
long routes have ~162 stops, ~101 h and 4 daily rests against ~96 stops, ~56 h
and 2 for medium. Each model is scored on the short+medium test routes (did
dropping long routes cost anything?), on the 30 long test routes (paired with
the models trained on all lengths), and on all 239 long routes (none seen in
training).

* **Trees transfer in duration.** Trained on short+medium only, they are as
  fast relative to the LA on long routes as the trees trained on everything.
  They lose a few tenths of a percent in distribution, where they now have
  less data.
* **The classifier and the MLP transfer less well**: trained on short+medium,
  both fall behind the teacher on long routes — the classifier by a few tenths
  of a percent, the MLP by more.
* **The route-local set R** (no whole-route position features) was the
  hypothesis for why transfer might fail — those features leave the training
  range on long routes. It did not change the picture: the failures come from
  elsewhere.
* **As trained, the short+medium models become infeasible on long routes** —
  6–8 % of them for the trees and the classifier, 14–18 % for the MLP.
  Section 6 is the reason.

### Trained on the base case, tested on shifted physics

`ood_eval.py`; [RESULTS.md](RESULTS.md) §5, `fig_ood*.png`. Battery 300 /
900 kWh, chargers 150 / 1000 kW, chargers every 100 km, and the real Arendal
tour — each on the base-case test seeds, so the physics is the only change.
Charger spacing transfers well. Shifts that change charging time (150 kW) are
where, as trained, a quarter to 40 % of runs became infeasible.

**The best base-case model is not the most robust one.** The MLP is run twice:
D77, its best base-case set, and F95. Under the charger shifts D77 is roughly
twice as far from the oracle as F95 (150 kW: +42 % against +20 %; 1000 kW:
+28 % against +12 %), while the trees and the classifier degrade far less.
This is one training seed per set, so it is an observation, not a mechanism.
D77 does drop columns that are duplicates only under base-case physics
(`soc_frac = soc_kwh / 500` only while the pack is 500 kWh), but none of them
obviously carries charger power; the MLP's extrapolation may simply be fragile.

---

## 6. Why learned policies become infeasible — the spread-room check

`halt_state.py` replays an infeasible run exactly (policies and realisations
are deterministic — all 737 failures in the length and physics experiments
reproduce) and reads the simulator's state at the decision that broke the
rule; `diagnose_all.py` does this for every experiment. As trained, most
failures are the same move — two thirds to three quarters of them on long
routes, 64 of 67 with 150 kW chargers: **charge (or break), then drive on, and
the 15 h shift spread is exceeded on the next leg.** Nearly all the rest are
realised drives longer than the guard assumed. The weekly allowances (reduced rests, extended driving days)
are not the mechanism: the MLP's failures do cluster late in the route, after
they are spent, but the trees' do not, and in both the breaking decision is
the same kind — a dwell the check did not count. Nor is it the position
features: the route-local set R fails the same way.

**The cause is in the safety layer, not the model.** The shared legality check,
`supervisor._spread_with_dwell_fails`, admits a non-rest action when

```
h + o(a) + D_wc ≤ 15 h,     o(a) = service + queue + minimum break
```

It leaves out **the charge**, deliberately — the MILP models the charge-spread
coupling itself, so the LA never needed it — and **the stop overhead**: `M_stop`
for any activity at a charger, `M_lay` for a break at a layby. A learned policy
chooses its charge duration only *after* that check, so nothing stopped a
charge that did not fit. On the base case this is rare; wherever charges get
longer or shifts tighter — slow chargers, long routes — it is the dominant
failure.

`policy_core.spread_room` (opt-in: `evaluate.py --spread-room`) completes the
check on the policy's side:

```
room(a) = 15 − h − o(a) − M(a) − D_wc     for a non-rest action; M = M_stop / M_lay
drop a            if room(a) < charge needed to reach the next charger
tauc* = clip(tauc_head, reach-next-CS, min(charge-to-full, room(a*)))
```

When a charge cannot fit, the non-rest options go and the policy charges during
a rest instead. After it, no failure of that kind is left anywhere; what
remains is realised drives longer than the 0.95 quantile the guard assumes
(`drive-tail`), which a 0.99 guard removes, and the ferries.

* **The check alone is free** where it is not needed: it moves 1–22 % of
  routes and none on average, and removes most infeasible runs — 57–70 % of
  them on long routes, 76 % across the physics shifts.
* **The 0.99 guard is not free**: it touches nearly every route — negligibly
  for the trees (≈ +0.01 %), a few tenths of a percent for the classifier and
  the MLP.
* With the check and the 0.99 guard, **every model trained on short+medium
  completes all 239 long routes**. The trees stay ahead of the teacher
  (−0.11 to −0.18 % vs LA); the classifier and the MLP end up behind it
  (about +0.6 % and +1.2 % vs LA). Across all experiments, the only failures
  left are 8 ferry crossings in the use case.
* **What it cannot fix: the ferry.** A sea crossing is a forced 3.8–4.9 h
  break with exactly one legal action. The mistake — not resting before
  boarding — is made stops earlier and needs look-ahead; the base case has no
  ferries to learn it from.

The conclusion for the generalisation question: what did not transfer was
mostly the safety layer around the models, not what they learned.

---

## 7. How this connects to the manuscript

The ML code reports "% vs LA" internally because the look-ahead is the teacher.
**The manuscript reports the gap to the hindsight oracle**
(`data_output/paper_gap_stats.csv`, `figures/basecase/paper_gap_box*.png`).
`paper_link.py` recomputes that gap for every learned run with
`compile_solutions._annotate_gap_to_oracle`'s exact definition, on the same
instances, and every figure in `ML/figures/` is drawn on that axis in the
manuscript's style (`ml_style.py` extends `src/plot/paper_style.py`: the same
method colours, the same green → vermillion infeasibility ramp).

| figure | what it answers |
|---|---|
| `fig_gap_test.png` | the headline: every method on the gap-to-oracle axis |
| `fig_ladder.png` | how each arm depends on the number of inputs |
| `fig_ablations.png` | which design choices mattered |
| `fig_length.png`, `fig_length_<variant>.png` | trained on short+medium, tested on long |
| `fig_ood.png`, `fig_ood_<variant>.png` | trained on the base case, tested on shifted physics |

---

## 8. What still needs doing

1. **Adopt the spread-room check or not.** It is opt-in so every earlier number
   stays valid; making it the default changes the base-case results
   negligibly and the generalisation results a lot. The guard quantile
   (0.95 / 0.99) is a separate cost-vs-safety choice.
2. **Time windows.** 102–146 misses against the teacher's 57. Duration is what
   the cost head trains on; window compliance only reaches it through the
   small β·δ term. Train on the penalised cost, or add a window-miss head.
3. **Ferries.** Anticipating a forced crossing needs a feature for it (or a
   look-ahead check), and training data that contains one.
4. **Risk-sensitive policy.** The logs record the scenario spread, not just the
   mean; learning both enables `mean + λ·std` at zero online cost — a policy
   the LA does not have.
5. **The proper neural arm.** One shared trunk scoring all 12 actions with a
   listwise loss at the teacher's own sampling temperature; needs `torch`
   (available).
6. **LA-LP baseline.** No `LPTAIL` runs exist on disk.
7. **Latency** is measured on a loaded machine and dominated by Python feature
   construction; read it as an order of magnitude (~10⁴ faster than the LA).
