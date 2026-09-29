# Theory — the MDP, and where every piece of the ML arm sits in it

This links what was built and measured in `ML/` to the sequential-decision
model the paper simulates and to the learning theory the design relies on.
Numbers are in [RESULTS.md](RESULTS.md); the build is in [METHOD.md](METHOD.md).
Notation follows Powell (2022), the framework the paper's concept figure uses
to classify its policies (`src/plot/framework_concept_pptx.py`).

---

## 1. The MDP the simulator implements

One route is one episode; the decision epochs are its stops k = 0, …, N−1
(the LA scores ~86 decisions per route on average, up to ~190 on long routes).

| element | definition | in the code |
|---|---|---|
| **state** `S_k` | arrival time `t`, state of charge `e`; the regulatory clocks — consecutive driving `cd`, shift driving `sd`, shift spread `h`, work `sw`, split-break flag `φ`; the weekly budgets — reduced rests used `ρ₂` (≤ 3), extended days used `ext` (≤ 2); and the route itself (stop types, distances, chargers, customers, windows), known at departure | `BEHDV` state (`stop, t_arr, e_arr, cd, sd, sw, phi, rho2_used, ext_shift_used, h`) + the instance; `features.py` computes φ(S_k) from them |
| **decision** `x_k` | `(y, b, r, τ_c)`: charge or not, break (none / b15 / b30 / b45), rest (none / r1 regular / r2 reduced), and the charge duration — a hybrid discrete × continuous action | the action dict + `durations()` |
| **admissible set** `X(S_k)` | what exists at the stop and what the rules force or forbid | `enumerate_actions` ∩ `action_passes` |
| **exogenous information** `W_{k+1}` | realised drive time and energy of the next leg, `D_k = D̄_k·ξ_k`, `ξ = min(0.889 + LN, 1.6)`, CV 0.15 | `D_real`, `E_real` in the instance |
| **transition** `S_{k+1} = S^M(S_k, x_k, W_{k+1})` | departure = arrival + service + queue + `M_stop` + `τ_c` + break + rest; arrival = departure + `D_k`; clocks advance or reset; charging follows the PWL curve | `BEHDV.advance` |
| **cost** `C(S_k, x_k, W_{k+1})` | time spent (dwell + drive) + β·(window miss), β = 0.5 h | route duration + β·misses |
| **constraints** | `cd ≤ 4.5 h`, `sd ≤ 9 h` (10 h while an extension is left), `h ≤ 13 / 15 h`, weekly budgets, `e ≥ E_min` | `BEHDV` violations; the first one ends the run |

The objective is

```
min over policies π    E[ Σ_k C(S_k, X^π(S_k), W_{k+1}) ]     subject to every rule holding on the realised path
```

— a finite-horizon **constrained** MDP (Altman, 1999), in which a violation
is an absorbing failure: the run is *infeasible* and has no duration. Across
instances it is a *family* of MDPs indexed by the instance `c` (route geometry,
customers, windows, battery, charger power): training samples `c` from the
base-case distribution.

**Policies** (the paper's concept figure, Powell's four classes):

| class | here |
|---|---|
| PFA — an analytic decision rule | Greedy |
| CFA — a deterministic model with hedged parameters | RO |
| DLA — an approximate model of the future | 2SP (two-stage, solved once), **LA** (rolling-horizon lookahead) |
| VFA — a value function approximation | *"not used here"* in the paper — **this is where the learned policies go** |
| — | ORACLE is not a policy: the perfect-information (hindsight) bound |

**The gap to the oracle** decomposes as

```
J(π) − J_oracle  =  [ J(π) − J(π*) ]  +  [ J(π*) − J_oracle ]
                    suboptimality of π     value of perfect information (≥ 0)
```

with `π*` the best non-anticipative policy. The second term is irreducible
(Birge & Louveaux, 2011), so every gap is an *upper bound* on a policy's
suboptimality.

---

## 2. What the LA computes, in these terms

At each stop the LA scores every admissible decision
(`Simulation.select_best_action`):

```
Q^LA(S_k, x) = (1/25) Σ_{i=1..25}  V_H(S_k, x; ω_i)        π^LA(S_k) = argmin_x Q^LA(S_k, x)
```

where `V_H` is the optimal value of the horizon problem under sampled scenario
`ω_i`, with the first decision fixed to `x`: the stops of the next 24 h
(extended to cover mandatory rests), solved as a MILP — the "MIP tail"; its LP
relaxation is the older `LPTAIL` variant. It is a **direct lookahead (DLA)**: a
sample-average estimate of a Q-factor under an approximate model. Two
properties matter for learning from it:

* **It is optimistic.** Inside each scenario the MILP knows that scenario's
  whole future, so `Q^LA` averages perfect-information values after the first
  decision. The LA is a good heuristic, not an optimal policy, and a student
  of it inherits its biases.
* **It is noisy.** 25 scenarios give a median standard error of 2.08 min per
  score; 13.6 % of decisions have a best-vs-second margin below twice that.
  The LA's own argmin is partly chance — the *optimiser's curse* (Smith &
  Winkler, 2006). This is the 0.35 % practical floor: the LA's run-to-run
  spread.

The logs record `Q^LA(S_k, x)` for **every** admissible `x`, its spread, and in
how many of the 25 scenarios `x` stayed feasible. That is the dataset.

---

## 3. What the students are — the VFA slot

**Trees and MLP** learn the LA's *advantage*

```
A^LA(S, x) = Q^LA(S, x) − min_{x'} Q^LA(S, x')        (the "regret" target; "centring")
```

and act by

```
π̂(S) = argmin_{x ∈ X(S)}  Â(S, x) + BIG · 1{ p̂_feas(S, x) < ½ },      τ_c = clip(τ̂_c, reach next charger, charge to full)
```

In Powell's terms this is a **VFA** — an approximation of the value of
state–decision pairs — fitted by supervised learning to a DLA's Q-factors
instead of by dynamic programming: the lookahead is *amortised* into a
function evaluated in milliseconds. The **classifier** learns
`π^LA : S → x` directly: a learned **PFA**, i.e. behavioural cloning.

Three pieces of theory justify the construction:

1. **Centring is free for control.** `argmin_x Q(S, x) = argmin_x [Q(S, x) − V(S)]`
   for any baseline `V(S)`, so learning the advantage loses nothing. It is
   also far easier: the level of `Q^LA` ("how much route is left") has a
   standard deviation of ~30 h, the differences between actions ~40 min. This
   is the reasoning behind advantage learning (Baird, 1993) and dueling
   networks (Wang et al., 2016). **Observed:** regressing the raw cost
   instead costs the trees +3.5 pp and the MLP +43 pp (RESULTS §3).
2. **Prediction error bounds decision error.** If `|Â − A^LA| ≤ r` on a
   state's admissible set, the chosen decision's regret is at most `2r`:
   `A(x̂) ≤ Â(x̂) + r ≤ Â(x*) + r ≤ A(x*) + 2r = 2r` — the cost-sensitive
   reduction (Beygelzimer et al., 2005; Elmachtoub & Grigas, 2022). The
   classifier has no such link from its loss to hours of route time.
   **Observed:** in distribution it did not matter — the classifier matches
   the trees (−0.12 vs −0.04 % vs LA, both inside the floor). It matters out
   of distribution (§5).
3. **Feasibility is learned as the LA's own test.** The feasibility head
   estimates `P(x feasible in all 25 scenarios)` — the LA's `ok = 25/25` —
   and vetoes; the charge duration is a separate head inside a physics clamp.

---

## 4. From one decision to a whole route

The **performance-difference lemma** (Kakade & Langford, 2002), finite horizon:

```
J(π̂) − J(π^LA)  =  Σ_k  E_{S_k ~ d_k^π̂} [ A^{π^LA}_k(S_k, π̂(S_k)) ]
```

The route-cost gap between student and teacher is the teacher's advantage of
the student's decisions, summed over the stops **on the states the student
itself visits**. Two consequences:

* **The training target is the right quantity.** The per-stop term is exactly
  what the cost head learns (with the LA's lookahead estimate `Q^LA` in place
  of its true cost-to-go `Q^{π^LA}`).
* **It is paid on the wrong distribution.** Training data comes from the
  teacher's states `d^{π^LA}`; the bill comes on the student's `d^{π̂}`. This is
  the compounding-error problem: behavioural cloning with per-step error ε
  can cost O(T²ε) (Ross & Bagnell, 2010), DAgger — training on the student's
  own states — O(uTε), with `u` the most one wrong decision can add to the
  cost-to-go (Ross, Gordon & Bagnell, 2011).

**Observed:** in distribution there is no visible compounding without DAgger
— every arm is within the 0.35 % floor of the LA, and the trees trained on
all lengths stay at −0.27 % vs LA on long routes, where T ≈ 160 instead of 88.
The structure explains why: mistakes are *recoverable* — a missed break can be
taken at the next stop, and every daily rest resets the daily clocks (`cd`,
`sd`, `h`), so most errors stop propagating within a shift; only the weekly
budgets (reduced rests, extended days) carry a mistake further. (A student can even post a
negative gap: when the LA's own argmin was noise, a model that averages over
similar states picks the better action.)

---

## 5. Generalisation as shifts of the MDP

| experiment | what changes in the MDP | the theory's expectation | observed |
|---|---|---|---|
| **route length** (trained on short+medium) | nothing in the dynamics, costs or rules; only which states are visited — 4–5 shifts, spent weekly budgets, larger remaining quantities: **covariate shift** | a model is only evaluated off its support; trees extrapolate as a constant, ReLU networks linearly (Xu et al., 2021), so trees should degrade gracefully and the MLP least gracefully | trees keep their duration (−0.18 / −0.24 % vs LA); classifier +0.2–0.4 %; MLP +0.7–0.8 % |
| **physics** (battery, charger power, spacing) | the **transition function** changes, so `Q^LA` itself changes: **dynamics shift** | transfer only where the features make the advantage invariant across the family — e.g. `charge_time_to_full`, `e_margin_next_cs_wc` are recomputed from each instance's physics | spacing transfers; charger power degrades most, the MLP by far the most |
| **use case** (ferries) | a state component never seen: a forced 3.8–4.9 h crossing with one admissible action | the decision that matters (rest before boarding) depends on information the features do not encode: φ(S) is not sufficient for `Q^LA` there (**state aliasing**) — no amount of base-case data can fix it | the only failures left in any experiment are ferry crossings |

Two finer points:

* **Removing the features that leave the training range (set R) did not
  help.** The shift that hurt was not in those features.
* **Selecting features on one MDP can cost robustness on another.** D drops
  columns that are exact duplicates *in the base case*; under a different
  pack they are not. The MLP's best base-case set (D77) is roughly twice as
  far from the oracle as F95 under the charger shifts (one training seed
  each: an observation, not a mechanism).

---

## 6. The constraints: shield, chance constraint, and what the fix is

The admissible set `X(S)` is not learned; it is a **shield** in the sense of
safe reinforcement learning (Alshiekh et al., 2018) — the learned part only
ranks decisions the rules allow. The failures found in §5 are two MDP-level
facts about that shield.

**(a) The shield must cover the whole decision.** The decision is hybrid,
`(y, b, r) × τ_c`. The shared check tests the spread with the *minimum* dwell
of the discrete part — without `τ_c` and without the stop overhead `M_stop` —
and the student then chooses `τ_c` freely. So the continuous part of the
action space was never shielded. The LA does not need it: its MILP puts `τ_c`
into the spread constraint. `policy_core.spread_room` computes the shield on
the full decision, `X(S) ∩ { τ_c ≤ room(S, x) }`. **Observed:** as trained,
this unshielded dwell causes 64–75 % of the long-route failures and 64 of 67
at 150 kW; with the check on, no failure of that kind is left in any
experiment (all 737 failures replayed exactly).

**(b) A one-step guard is a chance constraint, and chance constraints compound.**
The guard tests each rule at the `q`-quantile of the next drive,
`P(violation on the next leg | S_k, x_k) ≤ 1 − q`. Over a route with `m`
decisions that sit near a limit, the union bound gives

```
P(the route becomes infeasible) ≤ m (1 − q)
```

— growing with the number of tight decisions, hence with route length. The LA
does not face this: it requires feasibility in all 25 scenarios over a 24 h
horizon, a scenario version of a *joint* chance constraint (Calafiore &
Campi, 2006) — far stronger than a one-step quantile. **Observed:** apart from
the ferries, every failure left after (a) is a drive longer than the guard's
quantile; they sit on long routes (39 of the 46 in the physics experiments),
and `q = 0.99` (a 5× smaller per-step risk) removes all of them.

**(c) A stricter constraint is a smaller admissible set, so a higher cost.**
**Observed:** the 0.99 guard costs the trees ≈ 0.01 %, the classifier and the
MLP 0.14–0.58 % — the more a policy operates near the limits, the more it
pays for a margin.

---

## 7. The feature ladder and the statistics

* **Episodes, not rows.** Decisions within a route form a Markov chain, so
  the information is in the ~639 independent training routes, not the 411k
  rows: splits, early stopping and every standard deviation are by route.
* **Approximation vs estimation.** The feature map must be (nearly)
  sufficient for the advantage — too few inputs alias states with different
  advantages (C) — while the model's capacity must match ~639 episodes — the
  raw 20-stop lookahead (L) adds variance. **Observed:** an inverted U on every
  arm (RESULTS §2).
* **Why trees.** The admissible region is cut by exact thresholds (4.5, 9/10,
  13/15 h); a tree split is such a threshold, while a network approximates a
  step with a ramp whose error peaks at the boundary (Grinsztajn et al.,
  2022). The same property — flat outside the data — is what makes trees
  degrade gracefully under covariate shift (§5).

---

## 8. What the theory does *not* give

* **No guarantee of feasibility.** The shield is exact only for the rules it
  checks and only at the guard's quantile; the ferry shows a constraint that
  needs look-ahead.
* **The performance-difference identity uses the LA's true cost-to-go**, which
  is unknown; the logged `Q^LA` is its lookahead estimate.
* **DAgger** is the theory's answer to covariate shift and was removed from
  this project; the length and physics results are where it would apply.

---

## References

* Alshiekh, M., et al. (2018). Safe reinforcement learning via shielding. *AAAI*.
* Altman, E. (1999). *Constrained Markov Decision Processes*. Chapman & Hall/CRC.
* Baird, L. C. (1993). Advantage updating. Tech. rep. WL-TR-93-1146, Wright-Patterson AFB.
* Beygelzimer, A., Dani, V., Hayes, T., Langford, J., & Zadrozny, B. (2005). Error limiting reductions between classification tasks. *ICML*.
* Birge, J. R., & Louveaux, F. (2011). *Introduction to Stochastic Programming* (2nd ed.). Springer.
* Calafiore, G. C., & Campi, M. C. (2006). The scenario approach to robust control design. *IEEE Trans. Automatic Control*, 51(5).
* Elmachtoub, A. N., & Grigas, P. (2022). Smart "predict, then optimize". *Management Science*, 68(1).
* Grinsztajn, L., Oyallon, E., & Varoquaux, G. (2022). Why do tree-based models still outperform deep learning on typical tabular data? *NeurIPS Datasets and Benchmarks*.
* Kakade, S., & Langford, J. (2002). Approximately optimal approximate reinforcement learning. *ICML*.
* Powell, W. B. (2022). *Reinforcement Learning and Stochastic Optimization: A Unified Framework for Sequential Decisions*. Wiley.
* Ross, S., & Bagnell, J. A. (2010). Efficient reductions for imitation learning. *AISTATS*.
* Ross, S., Gordon, G., & Bagnell, J. A. (2011). A reduction of imitation learning and structured prediction to no-regret online learning. *AISTATS*.
* Smith, J. E., & Winkler, R. L. (2006). The optimizer's curse: skepticism and postdecision surprise in decision analysis. *Management Science*, 52(3).
* Wang, Z., et al. (2016). Dueling network architectures for deep reinforcement learning. *ICML*.
* Xu, K., et al. (2021). How neural networks extrapolate: from feedforward to graph neural networks. *ICLR*.
