# The models of the ML work: what they are, what they learned from, how they did

Updated 29 October 2026. The research map (what was tried, kept, given up, in
time) is `ML/figures/research_map.png`; the running log is `ML/STATUS.html`;
the RL theory and its progress log are `ML/RL_THEORY.md`.

**How to read the numbers**

* **Gap**: route time above the hindsight oracle's, in % of the oracle's.
  **vs LA**: a policy's gap minus the LA's gap on the same routes, in
  percentage points (pp), ± one standard error over routes. Negative = better
  than the LA.
* **Final protocol** (every number marked *final*): the shield on, input
  features at nominal travel times, the 5-minute charge-then-rest overhead
  paid, and the 0.5 h window penalty included in the gap (as in the paper's
  Table 1). **Earlier protocol** (marked *earlier*): before the 20 October
  fixes, route duration only. Numbers from the two are not directly
  comparable.
* The simulator's default rules throughout (the strict 13 h rest-start check
  is off, by decision of 26 October).

---

## 1. Instance sets

| set | what is in it | routes: fit / validation / test | used for |
|---|---|---|---|
| **Base case** | 3 lengths (short 800-1,200 km, medium 1,500-2,500, long 3,000-4,000) × 3 customer counts (1-3, 4-6, 7-15) × 4 window types (none; tight ±0.5-1 h; medium ±1-3 h; large ±3-6 h) × 25 seeds. 350 kW chargers about every 60 km, 500 kWh battery | seeds 1-19 (639 with an LA run) / 20-21 (66) / 22-25 (125) | training, choosing, testing every model |
| **Other truck types** (uniform) | the same routes with one setting changed: battery 300 / 700 / 900 kWh; chargers 150 / 700 / 1000 kW; chargers every 30 / 100 km | LA runs on seeds 1-25 (about 250 per value; 30 km: 41); tests on seeds 22-25 (about 40 routes per value) | training across truck types; transfer tests |
| **Mixed chargers** | built from base short and medium routes (few or many customers, no or tight windows): each charger's power drawn from 150 / 350 / 700 / 1000 kW (*power-mixed*), the spacing varies (*spacing-mixed*), or both | training 121 power-mixed + 117 both, with LA runs / validation seeds 20-21 / test 32 of each kind (seeds 22-25; of the 64 power-mixed and both, 62 have a certified oracle, 61 of them an LA run) | mixed-route students and tests |
| **Use case** | the real Arendal - Netherlands / Germany tour, with ferry crossings | test only | robustness |

## 2. Reference methods (not learned)

| method | what it is | base test gap | mixed test gap |
|---|---|---|---|
| **LA** (the teacher) | at every stop, a MILP over the next 24 h for every candidate action and 25 travel-time scenarios; 70-160 s per decision on the laptop, about 12 s on the cluster | 3.05 % (2.77 % duration only) | 2.73 % |
| **Oracle** | the route's MILP solved with the realised travel times: the hindsight bound (gap 0) | 0 (within 0.5 % on 114 of 125) | 0 (certified within 1 %) |
| **Greedy** | the companion paper's rule-based heuristic | 7.01 % (+3.95 ± 0.46 pp vs LA; 5 of 125 runs break a rule) | 15.47 % (+12.73 ± 1.09) |
| **Shield** (around every learned policy) | the simulator's legal actions and forcing rules, a 0.99-quantile drive guard, the spread-room check, the charge clamped between "reach the next charger" and "full" | - | - |

## 3. The learned models

### A. Trees, `gbt_F95_base_s0-s2` (kept: main arm 1)
* **What**: LightGBM, three heads: the cost of each action (its regret
  against the best action, Huber loss), the probability it is feasible, and
  the charge duration. Seed 0: 1,193 / 338 / 216 trees, 63 leaves.
* **Inputs (F95)**: 77 state features (driving and duty clocks and their
  slack to every limit, battery and energy margins to the next chargers, the
  next 3 chargers and next 2 customers with their window slack, the LA's
  horizon summary, what is left of the route) and 18 action features (charge,
  break, rest type, dwell, window slack after the action).
* **Trained on**: the LA's logged cost of every candidate action at every
  decision of the 639 base training routes.
* **Validated** on the 66 validation routes; **tested** once on 125 test
  routes, 3 training seeds.

### B. PyTorch network, `tmlp_F95_split_list_s0-s2` (kept: main arm 2)
* **What**: three separate networks (cost 256-256-128, feasibility and
  charge 128-128), the cost trained with a Huber loss plus a ranking loss
  over each stop's actions (temperature 0.25 h); each kept at its own best
  epoch.
* Inputs, data and splits as A.

### C. Students across truck types (partly: works on what they have seen)
* `gbt_F95_phys_s1` (trees) and `tmlp_F95_phys_split_list_s0` (PyTorch):
  recipes A and B trained on the LA runs of all uniform truck types (1.33 M
  rows). Leave-one-out copies (`..._LO<value>`) drop one value to test it as
  unseen.

### D. Mixed-charger student, `tmlp_T144_physmix_split_list_s0-s2` (kept, for mixed routes)
* **What**: as B, with inputs **T144** = F95 + 49 charger features (the
  charger here and the next six: exists, distance, charge on arrival, queue,
  reachable, power, hours to full).
* **Trained on**: all uniform truck types + the 121 power-mixed and 117
  power-and-spacing-mixed training routes. Chosen on mixed validation
  routes; tested on the 64 mixed test routes.
* Reference: the same recipe without mixed routes, `tmlp_T144_phys_split_list_s0`
  ("equal-power student").

### E. Tried for mixed routes, not kept
Charger-power inputs (trees, P102); a charger-comparison network
(ChargerNet); DAgger, the LA asked at the student's own states (1,781 then
1,776 calls); training on mixed routes only; weighting mixed routes ×5.

### F. Copying the oracle, `gbt_F95_oracle_s0-2`, `tmlp_F95_oracle_split_list_s0-2` (given up)
Recipes A and B trained to reproduce the stored hindsight schedules (830
routes, 82,580 decisions) instead of the LA's costs.

### G. Solver-call gate, `gate_gbt`, `gate_torch`, `gate_gbt_ro`, `gate_mixed_torch` (kept, in the hybrid)
* **What**: a LightGBM model of how much calling the LA is worth at a
  decision, from 14 signals (score margins, rest margin, entropy,
  feasibility, disagreement with a second model) and the state.
* **Trained on**: out-of-fold errors of fold-trained students (5 folds by
  route seed) on the training routes; one threshold per call budget.
* **Used in the hybrid**: the student drives, the LA is called where the
  gate fires, and the LA's action is kept only if it still wins when the
  student drives on (the rollout check).

### H. Rollout at close calls, `rl_policy.RolloutChooser` (kept; planning, nothing learned)
* At a close call (a rest within 2 h of the best other action, or the two
  best actions within 0.25 h), the student's 4 best actions (always the best
  rest and the best non-rest) are each simulated to the destination with the
  student driving on, under 32 shared travel-time futures (raced in rounds
  of 8); the cheapest is taken. No solver; settings fixed before validation.

### I. RL student, expert iteration, `tmlp_F95_split_list_s0_exit1`, `..._exit2` (kept; round 2 tested)
* **What**: B's network with its cost network fine-tuned so that it prefers
  the actions H's simulations found better; anchored to the original (a KL
  term and 200,000 imitation rows).
* **Trained on**: H's evaluation of every candidate at the student's own
  close calls, on the 639 training routes × 2 fresh travel-time draws:
  49,171 labelled decisions in round 1 (original student driving), 32,998 more
  in round 2 (round-1 student driving). Early stopping on route seeds 17-19.
* **Validated** on 66 routes × 16 fresh draws; round 2 **tested** once (125
  routes). One training seed (0).

### J. Dropped along the way
The classifier of the LA's choice and the scikit-learn network (dropped
20 October); the first PyTorch network without the ranking loss; one round of
fitted policy iteration; the exact MILP for the last shifts; a single
correction applied at every stop; a learned correction for the trees (on
hold).

---

## 4. How they did

### Base case, 125 test routes (*final*)

| policy | gap | vs LA | broken runs | time |
|---|---|---|---|---|
| LA | 3.05 % | - | 0 / 125 | 70-160 s per decision (laptop) |
| Greedy | 7.01 % | +3.95 ± 0.46 | 5 / 125 | ms |
| Trees (3 seeds) | 3.73 % | +0.68 ± 0.43 | 0 / 375 | 1-5 ms per decision |
| PyTorch (3 seeds) | 3.55 % | +0.50 ± 0.37 | 0 / 375 | 1-5 ms per decision |
| Trees copying the oracle | | +2.05 ± 0.48 | | |
| PyTorch copying the oracle | | +6.42 ± 0.66 | | |
| Hybrid: trees + LA at gated calls + check (seed 0) | | −0.36 ± 0.33 | 0 / 125 | 8.1 LA calls per route |
| Trees + rollout (3 seeds) | 2.33 % | −0.72 ± 0.33 | 0 / 375 | 40 s / 1.8 min / 5.7 min per short / medium / long route |
| PyTorch + rollout (3 seeds) | 2.38 % | −0.67 ± 0.34 | 0 / 375 | 11 s / 36 s / 2.8 min per route |
| **RL student, round 2** (seed 0) | **3.08 %** | **+0.03 ± 0.35** | 0 / 125 | 1-5 ms per decision |
| **RL round 2 + rollout** (seed 0) | **1.84 %** | **−1.21 ± 0.32** | 0 / 125 | 14 s / 38 s / 2.4 min per route |

* The original PyTorch student, seed 0 only, is at 3.54 % (+0.49 ± 0.36):
  round 2 improves it by −0.46 ± 0.20 pp on the stored realisations and by
  −0.25 ± 0.05 h per route (−0.37 ± 0.07 %) on 16 fresh draws per route
  (better on 88 routes, worse on 31).
* Routes with more daily rests than the LA (seed 0): original 3, round 2 2,
  round 2 + rollout 0.
* By length, round 2 + rollout vs LA: long −2.25 ± 0.64, medium −0.52 ± 0.17,
  short −1.27 ± 0.72 pp.
* Times with "per route" are one cluster core for the whole route.

### Longer routes than trained on (*earlier*; 30 long test routes, 3 seeds)

| model trained on short + medium routes | vs LA on long routes | broken runs |
|---|---|---|
| Trees | −0.3 ± 0.6 | 0 / 90 |
| Classifier (dropped) | +0.5 ± 0.6 | 0 / 90 |
| scikit-learn network (dropped) | +2.1 ± 0.6 | 0 / 90 |
| PyTorch | not tested | |

### Other truck types (*earlier*; about 40 test routes per value; pp vs LA)

| setting | trees: base only / all types / all but this | PyTorch: base only / all types / all but this | Greedy |
|---|---|---|---|
| chargers 150 kW | +9.9 / +0.5 / +5.5 | +17.6 / +0.2 / +7.3 | +4.8 |
| chargers 700 kW | +3.0 / +1.8 / +2.5 | +4.1 / +1.0 / not finished | +3.6 |
| chargers 1000 kW | +1.8 / +2.4 / +1.9 | +15.4 / 0.0 / not finished | +8.6 |
| battery 300 kWh | +2.6 / +0.3 / +0.8 | not tested | |
| battery 700 kWh | +2.0 / +1.5 / +0.9 | not tested | |
| battery 900 kWh | +1.8 / +1.6 / +1.6 | not tested | |
| chargers every 100 km | −0.1 / +0.2 / +0.2 | not tested | |

* Training on all types costs the base case a little: trees +0.7 vs +0.3,
  PyTorch +0.4 vs +0.1 pp vs LA. 0 broken runs in every row.
* Values between seen ones transfer; a value below the trained range
  (150 kW unseen) does not.

### Mixed chargers, test (*final*; the 61 routes with a certified oracle and an LA run)

| policy | gap | vs LA | energy taken at ≥ 700 kW |
|---|---|---|---|
| Oracle | 0 | | 89 % |
| LA | 2.73 % | - | 82 % |
| Greedy | 15.47 % | +12.73 ± 1.09 | 52 % |
| Equal-power student (seed 0) | 8.07 % | +5.34 ± 0.86 | 61 % |
| Equal-power student + rollout | 4.99 % | +2.26 ± 0.63 | |
| **Mixed student** (3 seeds) | 5.52 % | +2.79 ± 0.70 | 79 % |
| **Mixed student + rollout** (3 seeds) | **2.71 %** | **−0.02 ± 0.26** | |

* By kind, mixed student vs LA: power-mixed +2.50 ± 1.09, power and spacing
  +3.09 ± 0.87; with the rollout +0.00 ± 0.34 and −0.03 ± 0.41.
* Routes with more daily rests than the LA: mixed student 7, with rollout 3.
* Energy shares are on all 64 routes the LA ran.
* *Earlier* results on these routes (pp vs LA, power-mixed / both):
  base-trained trees +6.3 / +6.2; base-trained PyTorch +15.4 / +18.2; DAgger
  round 1 +2.2 / +3.4; charger network + 47 mixed routes +3.8 / +3.1.
* **Spacing-mixed only**: every model within about 1 pp of its result on the
  uniform original route (no LA run there).
* **Hybrid on mixed routes** (validation only): calling the LA at close rest
  calls with the rollout check improves the mixed student by
  −0.45 ± 0.25 pp; the learned mixed gate was not finished.

### Use case: the Arendal tour with ferries (*earlier*)
With the full shield, the only failures left in any experiment were 8 ferry
crossings: a forced 3.8-4.9 h crossing needs a rest before boarding, which a
one-step check cannot see and the training routes never contain.

### Not yet tested
The RL student on mixed routes, other truck types and longer routes; any
RL student beyond seed 0; PyTorch on longer routes.
