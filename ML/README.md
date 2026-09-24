# ML — a solver-free policy for the discrete-event simulator

**Results: [RESULTS.md](RESULTS.md)** (the selected model) and
**[RESULTS_ARMS.md](RESULTS_ARMS.md)** (trees vs network).

Headline, boosted trees on the held-out test split (122 completed routes):
statistically indistinguishable from the LA teacher (median -0.01 %, Wilcoxon
p = 0.92), 2.47 % faster than GREEDY, 2.18 % above the hindsight ORACLE, at
~4 ms per decision against the teacher's ~72 s. Selected configuration:
`base` model + deployment guard 0.95.

**The simplest neural network loses, decisively, and consistently across both
splits.** Gap to the hindsight oracle — the manuscript's own metric:

| | val | test | infeasible (val / test) |
|---|---:|---:|---:|
| **GBT student** | **+2.10 %** | **+2.18 %** | 0 / 3 |
| MLP student | +5.48 % | +4.89 % | 8 / 8 |
| LA (teacher) | — | +2.14 % | 0 / 0 |
| Greedy | — | +4.90 % | 5 |

The network lands on top of Greedy: it recovers little of what the look-ahead
buys over a myopic rule. Both arms generalise consistently from validation to
test, so the ~2.7 pp gap between them is the stable finding.

**Selection vs measurement.** Validation (seeds 18–21) is where every choice
was made — guard quantile, depth, target, which arm wins. Test (seeds 22–25)
is measured once with those choices frozen. The guard 0.95 was picked because
it gave zero infeasible runs on validation; on test it gives 3. That optimism
is exactly what the held-out split exists to expose.

A learned stand-in for the look-ahead MILP (`LA_MIPTAIL`), compared
against GREEDY, LA and ORACLE inside the existing simulator. Base case only
(3 route x 3 customer x 4 window x 25 seeds, cv 0.15, H 24 h, 500 kWh,
350 kW).

Everything this project writes lives under `ML/`. Nothing is written into
`solutions/`, `logs/` or `figures/`: the reporting pipeline discovers runs by
globbing `solutions/<bucket>/` by method name, so a stray student run in the
main tree would silently enter the manuscript's tables.

## The formulation

Not "classify the teacher's action". Instead: predict what each action would
**cost**, then take an argmin over the legally enumerated actions.

```
A  = enumerate_actions(stop, state, full_data)     # the simulator's own rules
A  = [a for a in A if action_passes(...)]          # the supervisor's forcing
score[a] = cost(s,a) + BIG * [P(feasible) < 0.5]
a* = argmin score
tauc* = clip(tauc(s,a*), reach-next-CS, charge-to-full)
```

Three measured reasons, all from the base-case data:

| observation | consequence |
|---|---|
| 88.8% of chosen actions are `y0_go` | a classifier's loss is dominated by a class nobody needs help with |
| error costs span two orders of magnitude (spurious daily rest 9–11 h vs a b15/b45 mix-up at minutes) | cross-entropy cannot see that; regression on cost is exactly that scale |
| the teacher logs **every** enumerated action's cost, not just the winner | 411k labelled rows instead of 73k, and rare classes (`y1_r1`, 0.11%) get gradient wherever they were *scored* |

Legality, forcing and the charge clamp are **not learned** — they come from
`enumerate_actions`, `supervisor.compute_flags` / `action_passes`, and the PWL
charging curve.

## Pipeline

```bash
python ML/code/extract.py                     # logs + solutions -> ML/data/dataset.npz

# --- tree arm ---
python ML/code/gbt_train.py --tag base        # three LightGBM boosters
python ML/code/evaluate.py --kind gbt --tag base --split val

# --- network arm ---
python ML/code/nn_train.py --tag nn           # three sklearn MLPs + scaler
python ML/code/evaluate.py --kind nn --tag nn --split val

# --- head to head ---
python ML/code/compare_arms.py                # -> RESULTS_ARMS.md + fig4
```

## Three arms, one experiment

There are **three students**, all solving the *same* problem on the *same* 830
instances with the *same* 91 features, splits, legality rules and decision rule
(`policy_core.py`). Only the learning changes, so differences are attributable:

| arm | what it learns | rows |
|---|---|---|
| `gbt_*` | cost of **every** enumerated action (boosted trees) | 411,118 (state, action) |
| `nn_*` | cost of **every** enumerated action (sklearn MLP) | 411,118 (state, action) |
| `clf_*` | **which** action the teacher chose (sklearn MLP classifier) | 72,595 decisions |

The `clf` arm reimplements the framing of the deleted 2026-08 tree, which
reported far better numbers than this project's MLP. Restoring that code
verbatim would not have been comparable — it differed in features, instance
set, splits and decision code all at once — so the *framing* is reproduced
here instead, with everything else held fixed.

The file layout makes which is which unambiguous: `gbt_*` is trees, `nn_*` is
the regression network, `clf_*` is the classifier network, everything else is
shared.

| file | arm | role |
|---|---|---|
| `code/parse_logs.py` | shared | recovers per-action cost vectors from the LA logs |
| `code/features.py` | shared | the ONE state/action description, used by training and by driving |
| `code/extract.py` | shared | joins labels to states, runs the four validation gates |
| `code/dataset.py` | shared | loading, splits, weighting, offline metrics |
| `code/policy_core.py` | shared | **the decision rule and the simulator loop** |
| `code/evaluate.py` | shared | closed-loop comparison (`--kind gbt\|nn`) |
| `code/stats.py`, `code/figures.py`, `code/report.py` | shared | significance, figures, write-up |
| `code/gbt_train.py` | **trees** | three LightGBM boosters |
| `code/gbt_policy.py` | **trees** | loads the boosters, supplies predictions |
| `code/ablations.py` | **trees** | the pre-registered grid |
| `code/nn_train.py` | **MLP regression** | three sklearn MLPs + a fitted scaler |
| `code/nn_policy.py` | **MLP regression** | loads the MLPs, supplies predictions |
| `code/nn_ablations.py` | **MLP regression** | the grid mirroring the tree arm's |
| `code/clf_train.py` | **MLP classifier** | 12-way action classifier + tauc head |
| `code/clf_policy.py` | **MLP classifier** | returns `-log P(a)` into the shared argmin |
| `code/paper_link.py` | shared | the manuscript's gap-to-oracle, exact definition |
| `code/fig_gap.py`, `code/ml_style.py` | shared | figures in the paper's own style |

`policy_core.py` is the important one: it owns `enumerate_actions` → legality
filter → forcing rules → `argmin` → charge clamp → `BEHDV.advance`. Each arm
implements only two methods, `_predict(rows)` and `_predict_tauc(row)`. That is
the entire difference between them in the deployed policy.

### What actually differs

| | `gbt_*` | `nn_*` |
|---|---|---|
| model | LightGBM, 3 boosters | sklearn MLP, 3 nets (64, 64) |
| feature scaling | **none needed** — trees are invariant to monotone transforms | **required** — a `StandardScaler` fitted on train rows ships inside the checkpoint |
| sample weighting | `margin/(margin+2·SEM)` | not supported by sklearn's MLP (and shown inert, see below) |
| early stopping | LightGBM, on grouped val | hand-rolled `partial_fit` loop on grouped val |
| checkpoint | `<tag>_{cost,feas,tauc}.txt` | `<tag>_nn.joblib` |

The neural arm is deliberately the **simplest network that could work**: three
plain MLPs, no shared trunk, no listwise loss. `torch` is available here, and a
shared-trunk network with a listwise loss over the action set is the version
with a real structural advantage over trees — but that is not the simplest
thing that could work, so it is not what `nn_train.py` is.

One trap the neural arm must avoid and the tree arm need not: sklearn's
`early_stopping=True` holds out a **random fraction of rows**. Rows within a
route are a Markov chain, so that would put ~88 near-duplicates of every
validation row into training, making the internal score wildly optimistic.
`nn_train.py` therefore sets `early_stopping=False` and scores on the real
seed-held-out routes after every epoch.

## Two traps found in the stored data

**1. The stored runs cannot be replayed naively.** `BEHDV.advance` takes the
executed break/rest from the nominal MIP's flags
(`milp_sol["sol"][0]["b45"|"b15"|"b30"|"rho1"|"rho2"]`), but `vehicle.actions`
stores the action the look-ahead *selected*. The two disagree whenever the
nominal re-solve placed the break elsewhere, and the executed flags are never
written to disk. A replay therefore drifts silently — measured at `phi` off by
one and `cd` off by up to 2.7 h on the first three routes tried. So states are
rebuilt from `sim_trajectory` instead.

**2. The shift spread `h` is not stored** — not in `sim_trajectory`, not in the
log's state line — although it is a hard constraint (13 / 15 h). *(Not a new
finding: the deleted 2026-08 ML tree hit this too and reconstructed `h` the
same way — see `ML/code/extract_dataset.py` in git history. Recorded here
because it is easy to miss and expensive to miss.)* It is exactly
reconstructible, because BEHDV computes

```
o_dwell = td[k] - t_arr[k] - taur[k]
h[k+1]  = (0 if rest at k else h[k] + o_dwell) + D_actual[k]
```

and `td_list`, `durations_list` and `D_actual_list` are all in the solution
JSON.

## Validation gates (run on every extraction)

| gate | check |
|---|---|
| G1 | the LOG's state line agrees with the SOLUTION's trajectory, at the precision each was printed with |
| G2 | `t_arr[k+1] == td[k] + D_actual[k]` |
| G3 | `t_arr[last] - T_START == duration_h` |
| G4 | the reconstructed spread stays in [0, 15] h on runs recorded as feasible |

G2/G3 are checked to 1e-3 h, not machine precision: the JSON rounds `td_list`
and `D_actual_list`, worth ~5e-5 h on its own.

Current status: **830/840 runs usable** (10 excluded as `run_infeasible`),
**72,595 decisions**, **411,118 rows**, 91 features, 36 families, all gates
green.

## Splits

By **seed within family**, never by row — the ~88 decisions of one route are a
Markov chain and a row-level split leaks almost perfectly.

| split | seeds | instances |
|---|---|---|
| train | 1–17 | 569 |
| val | 18–21 | 136 |
| test | 22–25 | 125 |

The effective sample size is the ~569 independent *routes*, not the 284k rows.
Model capacity is sized against that.

## Sample weighting

13.6% of decisions have a best-vs-second margin below twice the teacher's own
scenario-sampling SEM (median SEM 2.08 min) — there, the teacher's preference
is not distinguishable from which 25 travel-time draws it happened to get.
Rows are weighted by `margin / (margin + 2*SEM)`, which spends capacity where
the teacher was confident. That is a regulariser read off the data, not a
tuned knob.

**It did not pay off.** The `noweight` ablation matches `base` on duration
(-0.16 % vs -0.20 %), so on its own the weighting is inert. It is kept in the
selected configuration only because `base` + guard 0.95 reached 0 infeasible runs on
validation where `noweight` + guard 0.95 did not — and at 0 vs 2 infeasible runs of 136
that is suggestive, not established. Reported as a negative result rather
than quietly dropped.

## Reproducing

```bash
python ML/code/extract.py                                    # ~45 s, runs the gates
python ML/code/gbt_train.py --tag base                       # 3 heads
python ML/code/evaluate.py --tag base --split val --guard-q 0.95
python ML/code/ablations.py --split val                      # the grid
python ML/code/evaluate.py --tag base --split test --guard-q 0.95   # ONCE
python ML/code/stats.py --file eval_base_test.json
python ML/code/figures.py test && python ML/code/report.py test
python ML/code/compare_arms.py                                      # arms
```

## Known limitations

* **Time windows.** The student misses ~2x as many windows as the teacher
  (122 vs 57 on test). Duration is what the cost head is trained on; window
  compliance reaches it only through the small beta*delta term in the
  teacher's objective. Weighting windows explicitly in the target is the
  obvious next step.
* **Feasibility does not fully generalise.** The selected guard gave 0 halts
  on validation and 3 on test, all shift-spread. The guard reduces infeasibility; it
  does not remove it.
* **Latency is an upper bound**, dominated by Python feature construction and
  three separate LightGBM `predict` calls, and it was measured on a loaded
  machine. Read the speed-up as order 10^4.
* **Base case only.** No sensitivity, diesel, battery or charger-power axes,
  and no LA-LP baseline (zero `LPTAIL` runs exist on disk; it would have to be
  re-run).
