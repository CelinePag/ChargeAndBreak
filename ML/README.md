# ML — solver-free policies for the discrete-event simulator

**Results: [RESULTS.md](RESULTS.md)** (generated) · **Method: [METHOD.md](METHOD.md)** ·
**Theory and the MDP: [THEORY.md](THEORY.md)** ·
figures in `ML/figures/`.

Learned stand-ins for the look-ahead MILP (`LA_MIPTAIL`): each one makes the
per-stop decision — charge or not, break or rest, how long to charge — in
milliseconds instead of ~70 s, inside the existing simulator, and is compared
with Greedy, the LA, 2SP, RO and the hindsight oracle on the manuscript's own
metric. Trained on the base case only (3 route lengths × 3 customer counts ×
4 window types × 25 seeds; cv 0.15, H 24 h, 500 kWh, 350 kW).

## Headline — base case, held-out test batch (seeds 22–25, 125 routes)

Gap to the hindsight oracle; each learned arm at its best feature set, median
training seed; the last column is the spread over 3 training seeds.

| policy | features | gap to oracle | infeasible | vs LA, over training seeds |
|---|---|---:|---:|---:|
| 2SP | — | +2.10 % | 19 / 123 | — |
| **Trees** | F95 | **+2.10 %** | 0 / 125 | −0.04 ± 0.02 % |
| **Classifier** | F77 | +2.13 % | 3 / 125 | −0.12 ± 0.02 % |
| LA — the teacher | — | +2.14 % | 0 / 125 | — |
| MLP 2026-08, as built | O143 | +2.23 % | 2 / 108 | one model |
| **MLP** | D77 | +2.27 % | 4 / 125 | +0.09 ± 0.12 % |
| Greedy | — | +4.90 % | 5 / 125 | — |
| RO | — | +32.4 % | 0 / 125 | — |

* **Every arm reaches the teacher**: all within the 0.35 % practical floor
  (the LA's own run-to-run spread), ~2.4 pp ahead of Greedy, ~10⁴× faster.
* **Centring the target is the choice that matters**: regressing the raw
  horizon cost instead of the per-decision regret costs the trees ~3.5 pp and
  the MLP ~43 pp.
* **Generalisation**: trees trained on short+medium routes only are as fast on
  long routes as trees trained on everything; the classifier and the MLP
  transfer less well. Most runs that became infeasible on long routes and
  shifted physics hit one hole in the safety layer, which
  `policy_core.spread_room` closes; with it and a 0.99 drive guard, the only
  failures left in any experiment are 8 ferry crossings — see METHOD.md §6.
* **Open weakness: time windows** — 102–146 misses against the teacher's 57.

## Scope and where things go

Everything this project writes lives under `ML/`; `src/` is read-only. Nothing
is written into `solutions/`, `logs/` or `figures/`: the reporting pipeline
discovers runs by globbing `solutions/<bucket>/` by method name, so a stray
student run in the main tree would silently enter the manuscript's tables.

## The formulation

Predict what each legal action would **cost** (its regret against the best
action, in hours), then take the argmin — rather than classifying the
teacher's choice. Legality, forcing and the charge clamp are **not learned**:
they come from `enumerate_actions`, `supervisor.compute_flags` /
`action_passes`, and the PWL charging curve. Details and theory: METHOD.md §2.

```
A  = enumerate_actions(stop, state, full_data)     # the simulator's own rules
A  = [a for a in A if action_passes(...)]          # the supervisor's forcing
score[a] = cost(s,a) + BIG * [P(feasible) < 0.5]
a* = argmin score
tauc* = clip(tauc(s,a*), reach-next-CS, charge-to-full)
```

## Three arms, one decision rule

| arm | what it learns | rows |
|---|---|---|
| Trees `gbt_*` | cost of every enumerated action (LightGBM) | 411,118 (state, action) |
| MLP `mlp_*` | cost of every enumerated action (sklearn MLP, `log1p` target) | 411,118 (state, action) |
| Classifier `clf_*` | which action the teacher chose (sklearn MLP classifier) | 72,595 decisions |

The classifier is the framing of the deleted 2026-08 model, reproduced with
everything else held fixed; the 2026-08 model itself is shown as it was built
(`legacy/`, 143 inputs, its own instances and rollout — not a controlled
comparison).

**Names** say what a model is: `<arm>_<SET><n>_<config>[_SM][_s<seed>]` —
`n` is the number of inputs it consumes, `_SM` means trained on short+medium
routes only, `s<seed>` is the training seed. E.g. `gbt_F95_base_s1`,
`clf_R70_base_SM_s2`. Feature sets (C, D, F, L, R): METHOD.md §3, `fsets.py`.

**Variants** of a policy at evaluation time are named in the result files:
`g95` = as trained (drive guard at the 0.95 quantile), `g95sr` = + the
spread-room check, `g99sr` = + the check and a 0.99 guard.

## Files

| file | role |
|---|---|
| `code/parse_logs.py`, `code/extract.py` | recover per-action costs from the LA logs, join to states, run the validation gates → `data/dataset.npz` |
| `code/features.py` | the ONE state/action description, used by training and by driving |
| `code/fsets.py` | the feature-set registry (C, D, F, L, R) |
| `code/dataset.py` | loading, splits, scopes (all / short+medium) |
| `code/gbt_train.py`, `code/nn_train.py`, `code/clf_train.py` | the three trainers |
| `code/gbt_policy.py`, `code/nn_policy.py`, `code/clf_policy.py` | load a model, supply predictions |
| `code/policy_core.py` | **the decision rule and the simulator loop**, shared by every arm; `spread_room`; `candidates` (the k best actions) |
| `code/evaluate.py` | closed-loop evaluation of one model on one split |
| `code/rollout_policy.py` | rollout on top of any student: its k best actions each driven to the end of the route under sampled travel times |
| `code/run_rollout.py` | the rollout policy on chosen routes, paired with the plain student, the LA and the oracle |
| `code/configs.py`, `code/run_all.py` | the ablation registry and runner |
| `code/run_ladder.py` | every arm × feature set × training seed |
| `code/run_length.py` | trained on short+medium, tested on long routes |
| `code/ood_eval.py` | base-case models on shifted physics and the use case |
| `code/halt_state.py`, `code/diagnose_all.py` | replay infeasible runs; the cause of each |
| `code/spread_compare.py` | the same models with and without the spread-room check |
| `code/paper_link.py` | the manuscript's gap-to-oracle, exact definition |
| `code/ml_style.py`, `code/fig_*.py` | figures in the manuscript's style |
| `code/report.py` | writes RESULTS.md from the result stores |
| `code/legacy_adapt.py`, `legacy/` | the 2026-08 model, restored as built |
| `code/stats.py`, `code/figures.py` | older per-model statistics and figures |

## Reproducing

```bash
python ML/code/extract.py                                  # dataset + gates
python ML/code/run_ladder.py --arms gbt,clf,mlp --fsets C,D,F,L --seeds 3
python ML/code/run_all.py                                  # ablations
python ML/code/run_length.py --arms gbt,clf,mlp --fsets F,R --seeds 3
python ML/code/run_length.py --arms gbt,clf,mlp --fsets F,R --seeds 3 --spread-room
python ML/code/run_length.py --arms gbt,clf,mlp --fsets F,R --seeds 3 --guard 0.99 --spread-room
python ML/code/ood_eval.py            # and with --spread-room, --guard-q 0.99 --spread-room
python ML/code/diagnose_all.py && python ML/code/spread_compare.py
python ML/code/fig_gap.py && python ML/code/fig_ladder.py
python ML/code/fig_length.py [--variant g95sr|g99sr] && python ML/code/fig_ood.py [--variant ...]
python ML/code/report.py                                   # -> RESULTS.md
python ML/code/run_rollout.py --set smoke --guard-q 0.99 --spread-room   # rollout, beyond the teacher
```

Every runner skips work whose output already exists, so a rerun resumes.

## Splits

By **route seed within family**, never by row — the ~88 decisions of one route
are a Markov chain and a row-level split leaks almost perfectly.

| split | route seeds | routes | used for |
|---|---|---:|---|
| fit | 1–19 | 639 | fitting |
| stop | 20–21 | 66 | early stopping only |
| test | 22–25 | 125 | every reported number |

The effective sample size is the ~639 independent routes, not the rows.

## Known limitations

* **Time windows** — 102–146 misses against the teacher's 57 on every arm.
* **Ferries** — a forced crossing needs a rest before boarding; the one-step
  checks cannot see it coming and the base case has no ferries to learn from.
* **The spread-room check is opt-in**, so all earlier numbers stay valid;
  RESULTS.md shows each result with and without it.
* **Latency** is an upper bound measured on a loaded machine; read it as an
  order of magnitude.
* **No LA-LP baseline** — zero `LPTAIL` runs exist on disk.
