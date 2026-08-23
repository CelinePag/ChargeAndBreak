# Design justification — every choice, and what a reviewer will attack

Companion to `main.tex`. One section per decision: **what** was chosen, **why**,
**what the alternative was**, and **how it was verified**. Sections marked
⚠️ are honest gaps — places where the paper is currently exposed.

---

## 1. Framing

### 1.1 Imitation learning rather than reinforcement learning
**Chosen:** behaviour cloning from a solver.
**Why:** we already possess a near-optimal expert that can be queried at any
state — an exact-tail MILP look-ahead. RL from scratch would spend most of its
budget rediscovering behaviour we can simply copy, and reward shaping around
hard regulatory infeasibility is notoriously unstable. Imitation converts the
problem into supervised learning with an unlimited, noise-free label source.
**Alternative:** PPO fine-tuning on realised cost. This is the only route that
could *exceed* the teacher, and it remains the obvious next step; it was cut
for scope, not because it is wrong.
**Reviewer attack:** *"Imitation bounds you by your teacher — so the ceiling is
+0% and you can never win."* Correct, and stated in the Limitations. The claim
is not that the student is better; it is that **teacher-grade decisions are
obtainable 10⁵× faster**, which is what makes the policy deployable at all.

### 1.2 Teacher = exact-tail look-ahead, not the oracle
**Chosen:** clone the MIP-tail LA.
**Why:** the hindsight oracle is *anticipative* — it sees the realised travel
times. Its decisions are not implementable and its costs are unachievable, so
cloning it would teach the network to act on information it will never have at
deployment. This is the standard argument against imitating a clairvoyant
expert, and it is why the oracle is used **only as a yardstick**, never as a
label source.
**Verified:** the oracle appears in §5.1 below purely as a denominator.

### 1.3 Why this problem needs learning at all
**Chosen:** attack integrality, not horizon or scenario count.
**Why:** the companion sweep shows the other two axes are empirically dead —
scenario count 10→50 moves realised duration ≤0.4%, horizon beyond one duty
cycle ≤0.2%. Only tail integrality is worth anything (~2 points of optimality),
and only it is expensive. The target was chosen by measurement, not by
fashion.
**Reviewer attack:** *"Why not just run the LP tail, which is 4× cheaper?"*
Because it costs those two points — that is exactly Table 1.

---

## 2. Data

### 2.1 767 teacher runs, deduplicated to latest per instance
**Why dedup:** a re-run must *update* a result, not add a second sample that
silently doubles that instance's weight in every statistic. This mirrors the
companion benchmark's own reporting convention.
**Why 767 and not 772:** five runs are flagged `run_infeasible`; their
trajectories contain states the teacher itself could not resolve, so they are
excluded rather than taught.

### 2.2 State read from stored trajectories, not replayed
**Chosen:** read `sim_trajectory` directly from the solution JSONs.
**Why:** the simulator records the exact decision state at every stop. Replaying
would risk drift between the recorded run and the reconstruction; reading is
exact and takes seconds rather than CPU-weeks.

### 2.3 Split by instance seed, never by row
**Why:** states within one route are strongly correlated (consecutive stops
share clocks, battery, and route context). A row-wise split would place
near-duplicate states on both sides of the train/test boundary and inflate
every held-out number. Splitting by instance is the only honest unit.
**Verified:** 530 train / 129 validation / 108 test instances, disjoint.

### 2.4 Two split *designs*, and why the paper reports the seed split
**Chosen for the paper:** all 36 instance families in training, seeds held out
("deployment study").
**Also built:** a family-holdout design (12 of 36 families trained, the rest
never seen) that measures extrapolation to unseen route lengths and window
regimes.
**Why the seed split leads:** it answers *how good can this policy be*, which is
the paper's question, and it gives 3.6× the training data. The family design
answers a different question and needs its own tables.
⚠️ **Gap:** the extrapolation study is built but not reported. A reviewer may
reasonably ask how the policy behaves on regimes it never saw. Honest answer:
we measured it under a superseded simulator semantics and did not re-run it.

### 2.5 Normalisation statistics from the training split only
**Why:** computing means/standard deviations over all data leaks information
about the held-out distribution into the model's input scaling. Subtle, and a
standard reviewer check.

---

## 3. State representation (141 features at K=20)

### 3.1 Composition: 21 "dashboard" + 6×K "road ahead"
The 21 cover everything the decision physically depends on now: state of charge
and usable margin, three regulatory clocks, split-break flag, budgets consumed,
node type, queue, energy and distance to the next charger, and route-remaining
aggregates. The 6×K describe upcoming nodes.

### 3.2 Everything forward-looking is encoded **relative to now**
**Why:** "the window closes in 1.4 h" means the same thing on every route at
every time; "closes at 14:30" does not. Absolute encodings would force the
network to learn the subtraction, and would generalise poorly across the
departure times and route lengths in the grid.

### 3.3 Global aggregates *as well as* the K-node detail
**Why (sufficiency):** if the teacher rests early because of a tight window 40
stops ahead and the student cannot see stop 40, then two states that look
identical to the student carry different correct labels. The mapping becomes
ambiguous and no model can learn it. The aggregates (tightest slack anywhere
ahead, total remaining drive time, energy, customers) are a coarse summary of
the whole tail, which bounds this ambiguity cheaply.

### 3.4 K = 20
**Why not larger:** measured. K=40 (261 features) is *worse* in closed loop
(+2.86% vs +0.56%) — with a fixed training set, more features buy overfitting.
**Why not smaller:** K=20 spans ~5.4 h of driving, which covers the 4.5 h
consecutive-driving cycle; shorter windows cannot see the break they are about
to need.
⚠️ **Gap:** under current semantics only K∈{20,40} were re-measured. The
K∈{3,10} points in the earlier sweep predate the halt-on-infeasible change and
are not comparable, so the paper reports the two-point comparison only.

### 3.5 No sequence encoder
**Why not:** a GRU/attention encoder over all remaining stops would remove the
arbitrary K entirely and is the principled version. Cut for scope. The K
ablation suggests the window is not the binding constraint, which weakens the
motivation for it.

---

## 4. Action space and output head

### 4.1 12 discrete classes, taken from the data
**Why:** the combinatorially possible set (charge × 4 break types × 3 rest
types = 24) is not what the teacher uses. Enumerating observed combinations
avoids allocating capacity to actions that never occur, and keeps the mapping
stable and inspectable.

### 4.2 Full decision from the network — no solver at deployment
**Chosen:** the network emits the discrete action *and* the charge duration.
**Alternative considered:** predict only the discrete structure and recover
durations with a sub-second LP (predict-and-search / neural diving style).
**Why rejected:** it puts a commercial solver back in the loop at deployment,
destroying both the millisecond-latency claim and the no-licence argument —
the two legs of the paper's own motivation.
**Supporting evidence:** the benchmark's two-stage stochastic plan, which
commits structure offline and adapts only durations, strands 13.7% of runs
against the look-ahead's 1.7%. The value in this problem lies in *per-stop
adaptive structure*, which the student retains.

### 4.3 Softplus on the charge-duration head
**Why:** guarantees non-negativity by construction rather than by penalty.

### 4.4 Split breaks are **allowed** in the reported configuration
An earlier experiment masked b15/b30 at deployment and appeared to help. That
result was measured under the superseded simulator semantics and was **not
carried into the final runs**: the reported student uses b30 (183 uses) and
b15 (161 uses) on the test set. The paper is written accordingly.
⚠️ **Gap:** the split-break restriction is therefore an untested idea in the
current setup, not a result. Do not cite the old numbers.

---

## 5. Constraint layers

### 5.1 Masking (what is forbidden) vs forcing (what is mandatory)
**Masking** sets structurally impossible actions to −∞ before the softmax: no
charging away from a station, no reduced rest without budget. Consequence:
zero probability *and* zero gradient, so the network never spends capacity
learning not to do the impossible, and cannot do it at deployment.
**Forcing** restricts the choice set when a one-step check declares a charge,
break or rest mandatory.

### 5.2 Forcing is a mask restriction, not an override
**Why this matters:** an override would replace the network's decision after
the fact — we would then be measuring the override, not the policy. As a
restriction, the network still chooses **freely among compliant actions**, so
the rules constrain rather than decide.
**Verified:** instrumentation shows that of 364 rests in a val rollout, only 20
were compelled by the guard and 344 were chosen freely. The network, not the
rule, is doing the work.

### 5.3 Both layers reuse the benchmark's own feasibility checks
**Why:** using `supervisor.compute_flags` / `action_passes` means the learned
policy obeys exactly the rules every other policy in the benchmark obeys.
Hand-rolling a second set of checks would make the comparison meaningless.
**Bonus finding:** the data falsified one of our rule readings — 199 teacher
decisions took a standalone 30-minute break with no split in progress, so the
`phi==1` precondition we had assumed was wrong and was removed. Masks encode
the *model's* rules, not our reading of the regulation.

### 5.4 The safety supervisor is OFF
**Why:** the benchmark's S1 supervisor can override any policy's action. Left
on, it would rescue the student's failures and we would be reporting the
supervisor's competence. `supervised=False` throughout, zero interventions;
failures are reported as they happen.

### 5.5 Guard quantiles: energy 0.5, time 0.95
**Energy 0.5** is *teacher-matched*: 730 of 772 staged teacher runs carry
`la_energy_quantile = 0.5`, which sizes the committed charge to cover the legs
to the next station at the 0.5-quantile of consumption. Using the same rule
means the student is not handicapped or advantaged relative to what it imitates.
**Time 0.95** is ours, and necessary: the teacher used no time-side guard
(`prune_quantile = None`) because its 25-scenario ensemble exposed
time-infeasible actions; the student has no ensemble.
**Direction subtlety (a real trap):** a quantile means opposite things on the
two sides. Energy rises when a leg is driven *fast*, so the q-quantile of
consumption is driven by the (1−q)-quantile of the travel-time multiplier;
time rises when a leg is *slow*. Because the multiplier has median 0.957 < 1,
**q=0.5 on the time side would be anti-conservative**. Hence two separate
knobs, not one.

---

## 6. Model

### 6.1 Two hidden layers of 128 units (36,365 parameters)
**Why small:** the input is 141 engineered features over ~52k training rows.
Capacity beyond this buys overfitting, and the K=40 result is direct evidence
in the same direction (more input dimensions, worse closed-loop).
**Why not a lookup table:** the state is continuous and instance-dependent;
tabulation cannot cover it. The requirement is *generalisation across states
and instances*, which is precisely what a function approximator provides.
⚠️ **Gap:** width/depth were not swept. The choice is defended by reasoning and
by the K result, not by a direct architecture ablation.

### 6.2 Shared trunk, two heads
**Why:** the discrete choice and the charge duration depend on the same
situational understanding (battery, clocks, deadlines). Sharing regularises
both tasks; only the final layer differs.

---

## 7. Loss and training

### 7.1 Cross-entropy, not accuracy
**Why:** accuracy is piecewise constant — its gradient is zero almost
everywhere and it carries no direction of improvement. Cross-entropy is smooth
and penalises *confident* errors proportionally.

### 7.2 Class weighting set to **unweighted** — the paper's central result
The decision distribution is 87.9% "keep driving", so inverse-frequency
weighting is the textbook response, and by its own measure it works (balanced
accuracy 0.299 → 0.477). But at exponent 0.5 it places the majority class at
weight 0.035 against 5.3 for the rarest — a 150× ratio — so predicting a
**rest** when the truth is **drive on** becomes nearly free in the loss, while
costing 9–11 h of route time in reality. Realised duration degrades from
+0.56% to +3.17%.
**Diagnosis path (worth reproducing in a rebuttal):** the failure was found by
eye in a schedule figure (student took three daily rests where the teacher took
two; the 9.3 h difference is exactly one rest), then confirmed population-wide
— runs with >5% duration gap averaged +1.02 extra rests, runs under 5% averaged
+0.00, correlation 0.864.

### 7.3 SmoothL1 on charge duration, charging rows only
**Why SmoothL1:** quadratic near zero, linear far — one outlier cannot dominate.
**Why charging rows only:** on non-charging rows there is no target.
⚠️ **Gap:** the weight λ balancing the two loss terms was fixed at 1 and never
tuned. A reviewer could reasonably ask.

### 7.4 Adam, lr 1e-3, batch 512, early stopping (patience 15)
Standard defaults; nothing here is load-bearing. Early stopping is the
regulariser that matters — training loss falls indefinitely while validation
loss turns upward, and models converge by epoch 5–10.

### 7.5 Three training seeds
**Why more than one:** neural training is stochastic (initialisation, batch
order); a single model is one sample, not a result. The base spread (±0.18)
is what showed the DAgger gain was not real.
⚠️ **Gap:** three is the minimum defensible number; five would be better and
costs ~10 s each.

---

## 8. Evaluation

### 8.1 Closed-loop, in the benchmark's own simulator
Same vehicle physics, same recorded travel-time realisations, same metrics as
every other policy. The `external_policy` hook replaces **only** the decision
step, so nothing else can differ.

### 8.2 Paired differences, medians, and the tail
**Paired:** routes differ enormously in length, so an unpaired mean is
dominated by which instances are in the set. Each run is compared to the *same
instance's* teacher run — the companion benchmark's own convention.
**Median + IQR + max:** a policy with a good median that occasionally loses 10 h
is unusable; the tail must be visible.

### 8.3 Halted runs are counted, never averaged
Under current semantics a run that breaches a regulatory limit **ends at that
stop** and has no duration. Averaging a partial duration would reward failure
(a truck that stops early looks fast). Halts are reported as a separate column.
**This invalidated 1,042 earlier runs**, which completed routes that would now
halt; they were archived and every reported number was regenerated.

### 8.4 A duration comparison is only valid at equal feasibility
An HoS-violating run is "faster" because it cheated. Any table mixing policies
with different violation rates on a duration column is misleading — hence the
halt column sits beside the duration column throughout.

### 8.5 The test set was evaluated once
All design decisions — K, class weighting, guard level, DAgger weight — were
made on validation. The test set was touched a single time after freezing.
This is why the DAgger reversal is credible rather than embarrassing.

### 8.6 Gap to the hindsight oracle (test set)
The paper reports differences from the teacher; a reviewer will want the
absolute yardstick. Computed on the test set:

| policy | median gap to oracle |
|---|---|
| greedy heuristic | +5.77% |
| **student (behaviour cloning)** | **+3.39 ± 0.15%** |
| student + DAgger | +3.39% |
| exact-tail teacher | +2.24% |

The student closes **67%** of the greedy→teacher gap. Worth adding to the
paper if space allows.

---

## 9. DAgger

**Design:** the student drives; the teacher labels a uniformly sampled 35% of
the states it reaches. Sampling is uniform *on purpose* — labelling only
crashes would bias the aggregated set toward disasters and teach the network
that rare states are common. Only training-split instances are eligible;
aggregating val/test states would leak.
**Teacher config identical to the staged runs** (S=25, H=24 h, MIP tail, no
pruning) — a cheaper teacher would silently change what is being cloned between
rounds.
**Weighting:** aggregated rows are ~19% of the merged set by count. The weight
was initially set to 5 (sized for a much smaller collection), which made them
55% of effective loss mass and over-corrected; weight 1 is right at this size.
**Result:** negative. Better on validation, reversed on test. Reported as such.

---

## 10. The attacks I would expect, ranked

1. **"The forcing rules are doing the work, not the network."**
   *Defence:* greedy has equivalent must-charge/break/rest logic and sits at
   +2.74% (oracle gap +5.77%); the student sits at +0.43% (+3.39%). Same rules,
   very different quality. Plus the instrumentation in §5.2: 344 of 364 rests
   were freely chosen, not compelled.
2. **"Only three seeds."** Fair. Cheap to extend.
3. **"You never beat your teacher, so what is the contribution?"** The
   contribution is latency, not quality — and the methodological result about
   offline metrics.
4. **"Single vehicle, synthetic uncertainty."** Stated in Limitations.
5. **"No architecture ablation, λ untuned."** ⚠️ True.
6. **"Why no extrapolation study?"** ⚠️ Built, not reported.
7. **"Report gap to the oracle, not to your own teacher."** Answered in §8.6;
   should go in the paper.
