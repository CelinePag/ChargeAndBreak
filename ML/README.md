# ML — workshop paper: distilling the exact-tail LA policy

Target: **ML×OR workshop @ NeurIPS 2026** (Atlanta, Dec 12/13). Submission
**Aug 31, 2026 AoE**, 4 pages NeurIPS format, non-archival, non-anonymous.
https://mlxor-2026.github.io/

Everything for this paper lives here, separate from the journal-paper tree.

## WHERE THINGS RUN (rule)
Heavy stages — DAgger collection, training sweeps — run on the **HPC**, never
on the laptop; sync with git and run the commands below in the cluster
terminal. Rollouts (~0.6 ms/decision), merging, plotting and inspection are
cheap and stay local.

Keep `OMP_NUM_THREADS=1` and `GRB_THREADS=1` for DAgger: parallelism comes
from running several slices, and letting each grab a whole node
oversubscribes it (observed locally: a 10 s training run took >10 min under a
10-worker oversubscription).

```bash
# 0) smoke test FIRST — proves the solver licence works on the compute node
python ML/code/dagger.py --model ML/models/policy_K20_seedsplit_seed0_cw0.pt \
    --round 0 --prob 0.08 --limit 1 && rm -rf ML/data/dagger/round0

# 1) DAgger round 1 — run these 16 in parallel (one per core/job).
#    Resumable: one shard per instance, existing shards skipped, so a
#    timeout costs nothing — just rerun the same line.
export OMP_NUM_THREADS=1 GRB_THREADS=1
for i in $(seq 0 15); do
  python ML/code/dagger.py --model ML/models/policy_K20_seedsplit_seed0_cw0.pt \
      --round 1 --prob 0.35 --limit 0 --slice $i/16 \
      --k 20 --split-mode seed --guard-q 0.95 --no-split \
      > ML/logs/dagger_r1_$i.log 2>&1 &
done; wait

# progress
ls ML/data/dagger/round1/*.npz | wc -l
python -c "import glob,numpy as np; fs=glob.glob('ML/data/dagger/round1/*.npz'); print(sum(len(np.load(f)['y']) for f in fs),'labels from',len(fs),'instances')"

# 2) merge + retrain (merge prints the covariate-shift class-mix table)
python ML/code/merge_dagger.py --round 1 --k 20 --split-mode seed
python ML/code/train.py --k 20 --split-mode seed --cw-power 0 \
    --dagger 1 --dagger-weight 5 --seed 0

# 3) seed sweep for the mean±spread the paper needs
for cw in 0 0.25; do for s in 0 1 2 3 4; do
  python ML/code/train.py --k 20 --split-mode seed --cw-power $cw --seed $s
done; done
```

Sizing: a teacher query costs **~54 s measured**; at `--prob 0.35` an
instance needs ~14 queries (short) to ~60 (long) = **13–75 min each**.
Timings and licence caveat: the DAgger call is the ONLY stage that needs the
MILP solver, so if the licence does not resolve on compute nodes every slice
dies instantly — which is what the smoke test catches.

## Output isolation (rule: nothing from this work leaves ML/)
`ML/code/rollout.py` calls `src.paths.redirect_outputs(ML/)` before importing
the simulator, so student runs/logs/figures land in `ML/solutions`,
`ML/logs`, `ML/figures`, `ML/data_output`. Instances stay shared (read-only).
This matters because the reporting pipeline globs `<root>/solutions` by
method name — a stray STUDENT run outside `ML/` would silently enter the
manuscript's tables. Figures: `python ML/code/plotML.py all`.

## Layout
- `data/miptail/` — teacher runs: plain-variant `LA_MIPTAIL` solution JSONs
  copied from `solutions/`, **deduped to the latest run per instance**
  (772 runs, 36 families, 75,794 state–action pairs; 5 runs flagged
  `run_infeasible` — exclude at extraction). Copied 2026-08-20.
- `data/manifest.csv` — one row per copied run: instance, family, file,
  n_stops, decision-time stats, feasibility flag, (S, H) config.
- `paper/main.tex` — draft skeleton (rough bullets, article class;
  TODO swap to `neurips_2026.sty`).
- `code/rollout.py` — student drives in the real simulator via the additive
  `external_policy` hook in `run_simulation_precomputed`.
- `code/plotML.py` — all figures (money / paired / failures / training).
- `solutions/`, `logs/`, `figures/`, `data_output/` — ML outputs only.

## Design decisions (running log)
- **Full-decision network** (discrete heads + charge-duration head with
  worst-case reachability clamp), NOT solver-in-the-loop duration recovery.
  Rationale: keeps ms latency / no solver at deployment; 2SP evidence shows
  frozen-structure + duration-recourse strands 13.7% of runs, i.e. the value
  is in per-stop adaptive decisions, which the student keeps. LP-execution
  variant demoted to optional ablation.
- Teacher config: base LA cell (S=25, H=24 h, cv=0.15), `solve_mode=mip`.
- Splits by instance seed; OOD holdouts by family (route length, TW class).

## Progress
- 2026-08-20: `code/extract_dataset.py` → `data/dataset.npz` (74,380 rows,
  81 features, 12 action classes; splits: train 14,353 / val 3,387 /
  test-ID 3,431 / OOD-route 17,338 / OOD-tw 18,027 / OOD-both 17,844).
  `code/model.py` (masked 2×128 MLP, class + tauc heads) +
  `code/train.py` (BC, class-weighted CE + SmoothL1). First model
  `models/policy_v1_seed0.pt`: val acc ≈0.88, balanced acc ≈0.59,
  tauc MAE ≈9 min. Mask lesson: b30 is legal at phi=0 (199 teacher
  decisions) — rule removed.

## K ablation (2026-08-20) — headline: flashcards and the road test disagree
Features = 21 dashboard + 6·K forward nodes. K∈{3,10,20,40} × 3 seeds,
identical everything else; closed loop on val (48 instances).

| K | feats | val_loss (3 seeds) | bal_acc | median vs teacher | IQR upper | max | infeas/48 |
|---|-------|--------------------|---------|-------------------|-----------|-----|-----------|
| 3 | 39 | **0.676**±0.004 | **0.659** | +1.73% | +16.9 | +56.8 | 18 |
| 10 | 81 | 0.701±0.008 | 0.588 | +1.56% | +18.3 | +38.2 | 15 |
| 20 | 141 | 0.734±0.012 | 0.571 | +1.82% | +16.2 | +33.0 | **12** |
| 40 | 261 | 0.738±0.006 | 0.545 | +1.89% | **+3.5** | **+29.0** | 20 |

* Supervised metrics get monotonically WORSE with K (more features, same
  14,353 rows → overfitting); the closed-loop TAIL gets monotonically BETTER
  (max +56.8 → +29.0; K=40's IQR collapses to +3.5). Medians barely move.
  This is the paper's cleanest argument for why accuracy is not the metric.
* Feasibility does NOT follow: K=20 best (12/48), K=40 worst (20/48, and 61
  HoS violations). More foresight tightens duration but does not supply the
  missing *forcing* rules.
* Duration deltas must be read feasible-only: an HoS-violating run is
  "faster" because it cheated. Feasible-only medians: K3 +2.95, K10 +1.46,
  K20 +1.65, K40 +2.18.
* Latency 0.52–0.61 ms vs teacher 62.6 s, LP-tail 16.4 s (~10^5×).
* All students 12–20/48 infeasible vs teacher 1/48, LP-tail 0/48 → forcing
  rules are the blocking issue, not representation size.

## Forcing rules (2026-08-20) — teacher-matched guards
The structural mask says what is FORBIDDEN; forcing rules say what is
MANDATORY. Implemented with the project's single source of truth
(`supervisor.compute_flags` / `action_passes`), applied as a mask
*restriction* so the network still chooses among compliant actions.

Guard levels mirror what the teacher actually ran (checked in the staged
JSONs): `la_energy_quantile = 0.5` (730/772) and `prune_quantile = None`
(772/772). So:
* **ENERGY_Q = 0.5** sizes the committed charge over the legs to the next CS
  at the 0.5-quantile of consumption — the same rule as
  `MILP._build_sub_data`. Replaces the earlier arbitrary 25 kWh buffer.
* **GUARD_Q = None** (nominal) on the time side, teacher-matched.
* DIRECTION SUBTLETY: q means opposite things per side. Energy rises on FAST
  legs (q-quantile of consumption ← (1−q)-quantile of ξ); time rises on SLOW
  legs. Since ξ has median 0.957 < 1, **q=0.5 on the time side is
  anti-conservative** — hence two separate knobs, not one.

Val (48 instances), K=20 model:

| guard | infeas | HoS | strand | median vs teacher | vs LP-tail | TW |
|-------|--------|-----|--------|-------------------|------------|-----|
| *(none — before)* | 12 | 18 | 5 | +1.82% | −0.39% | 90 |
| **None (nominal)** | **1** | 1 | 0 | **+1.74%** | **−0.41%** | 95 |
| 0.95 | **0** | 0 | 0 | +2.04% | +0.43% | 98 |
| 1.0 | **0** | 0 | 0 | +2.04% | +0.43% | 99 |
| teacher | 1 | 0 | 1 | — | — | 49 |
| LP-tail | 0 | 0 | 0 | — | — | 85 |

* Nominal forcing is a FREE LUNCH: infeasibility 12→1 *and* the median
  improves (+1.82→+1.74). Stricter guards buy the last run at +0.3 pp.
* K ablation re-run WITH forcing: K20 best (+1.74), K3 worst (+3.27); K40
  still owns the tightest tail (IQR upper +3.84 vs +17.8–29.5). K=20 is the
  configuration to carry forward.
* OPEN WEAKNESS: TW misses 88–99 vs teacher 49 (LP-tail 85). On the
  PENALISED objective (duration + β·misses, the paper's `gap_pen`) the
  student is +3.5–4.7% vs teacher, i.e. roughly double its duration-only
  gap. Windows, not energy or HoS, are now the binding deficiency.

## Schedule figure — what the student actually does differently
`python ML/code/plotML.py --method=STUDENTFK20 schedule` (add
`--instance=NAME` to pin one; default picks the run whose student/teacher gap
is closest to the MEDIAN, so it is representative, not cherry-picked).
Two Gantt lanes on a shared clock + SoC traces; a charge bar is hatched when
a declared break was fully absorbed inside it (taub = 0), which is the
coupling the whole problem is about.

Reading of `RmediumCmanyTtight_21` (76.6 h vs 75.3 h), confirmed in aggregate
over the 48 val runs:
* Both policies hide nearly EVERY break inside a charge — the student learned
  the coupling; standalone breaks are almost absent in both lanes.
* Rest placement is similar in count, shifted in time.
* The student runs the pack HIGHER and charges MORE: mean SoC 60.9% vs 57.2%,
  5th-pct SoC 34.2% vs 28.4%, 7.6 vs 6.9 charge events per route, and +6.1%
  total charging hours. Charging is time, so this over-charging is a direct
  contributor to the +1.74% duration gap — and it is the NETWORK's own
  behaviour, not the safety clamp (clamp fired 0 times).
* The student shows more window-miss ticks, consistent with the penalised-gap
  weakness noted above.

## Split breaks: REMOVED at deployment (`--no-split`) — and it HELPS
Question: the model is cloned from a teacher that uses splits; may the
student be forbidden them? Yes — restricting the POLICY is our choice of what
the truck may do; relabelling the teacher would not be legitimate. So
`--no-split` masks b15/b30 at deployment only; training data is untouched.

Prior belief "splits change duration by <0.5%" could NOT be reproduced: the
230 `__nosplit` teacher runs point at instance files that no longer exist
anywhere on disk, so that comparison is unverifiable (paired deltas had
median −0.03% but IQR ±0.57 and range −23.8%..+31.1%, i.e. noise, not a
controlled contrast). Consistent with the known integrity problems on that
axis. Do NOT cite the old no-split row.

Direct test instead (val, 48 inst, K=20 + nominal forcing):

| variant | dur vs teacher | penalised vs teacher | vs LP-tail | infeas | TW | beats teacher |
|---------|----------------|----------------------|------------|--------|-----|---------------|
| full action set | +1.74% | +3.68% | −0.41% | 1 | 95 | 5/48 |
| **--no-split** | **+1.13%** | **+2.97%** | **−1.01%** | 1 | **86** | **18/48** |

Removing options makes the student BETTER on every axis. Mechanism: b15
(1.7%) and b30 (2.1%) are the thinnest classes, so they are exactly where
imitation is worst; mispredicting them costs time. Forbidden, the policy
falls back on well-learned b45 (176 → 266 uses) and rests. The teacher can
afford a rich action set because it OPTIMISES; the student cannot because it
APPROXIMATES, and its approximation is weakest where data is thinnest.
=> Paper-worthy: the cloned policy wants a SMALLER action space than its
teacher. Carry `--no-split` forward as the default student configuration.

## Two split designs — both kept, they answer different questions
`extract_dataset.py --split-mode {family,seed}` (train.py / rollout.py /
compare_runs.py all take the same flag; rollout reads the mode from the
checkpoint and the instance list from the dataset, so a model can only be
scored under the labels it trained on).

* **family** (extrapolation study): train on 12 of 36 families
  (short+medium × none/tight), seeds 1–17 → 14,353 rows. Long routes and
  medium/large windows never trained on. Measures how far a cloned policy
  TRAVELS.
* **seed** (deployment study): train on ALL 36 families, seeds 1–17 →
  **51,931 rows (3.6×)**. Measures how good the policy can BE.

Val comparison on the 47 instances both designs share (K=20, forcing,
no-split): family-trained +1.13% vs all-families +1.21% median — a wash on
the median, but the all-families IQR is **[+0.26,+2.50] vs [−0.46,+18.24]**,
i.e. far more consistent. More data bought reliability, not typical-case
quality. Supervised metrics again moved the OTHER way (bal_acc .571→.477).

All-families model, full seed-split val (129 instances, 12 regimes):

| regime | n | median | worst | regime | n | median | worst |
|---|---|---|---|---|---|---|---|
| short/none | 12 | +1.54 | +4.9 | medium/medium | 10 | **+16.02** | +21.8 |
| short/tight | 12 | +1.87 | +36.5 | medium/large | 10 | +1.54 | +18.7 |
| short/medium | 12 | +0.92 | +37.2 | long/none | 10 | **+10.35** | +12.7 |
| short/large | 12 | +1.29 | +31.8 | long/tight | 7 | +0.83 | +11.6 |
| medium/none | 11 | +0.83 | +35.0 | long/medium | 10 | +1.05 | +11.7 |
| medium/tight | 12 | +1.46 | +21.2 | long/large | 11 | **+9.98** | +18.8 |

Aggregate: +1.50% median, 14/129 infeasible, 207 TW misses vs teacher 112
and LP-tail 193; 0.65 ms vs teacher 72.0 s. Three regimes are badly off
(medium/medium +16.0, long/none +10.4, long/large +10.0) while the other
nine sit at +0.8–1.9% — the failure is CONCENTRATED, not diffuse, so it is
worth diagnosing rather than averaging away.

## DIAGNOSIS: the failures are ONE SPURIOUS DAILY REST
The three bad regimes are not a capacity limit — they are a single, specific,
fixable error. Found by eye in `ml_schedule.png`
(`RmediumCmediumTmedium_20`: student 56.6 h vs teacher 47.3 h, and the
student's Gantt lane shows THREE rest blocks against the teacher's TWO;
56.6 − 47.3 = 9.3 h ≈ exactly one 9 h reduced rest), then confirmed
population-wide:

| group | n | mean extra rests vs teacher | ≥1 extra rest |
|-------|---|------------------------------|---------------|
| duration gap > 5% | 43 | **+1.02** | **41/43** |
| duration gap ≤ 5% | 86 | +0.00 | **0/86** |

**correlation(duration gap %, extra rests) = 0.864**

Worst cases are all "took N+1 rests where the teacher took N" (2 vs 1, 4 vs
2), and 5 of the 43 rest twice within 12 h — i.e. rest, drive briefly, rest
again. So the whole heavy tail of this policy is one decision type going
wrong, not diffuse imprecision. A rest costs 9–11 h, which is why a single
misfire dominates a 47 h route.

Likely causes to test next, cheapest first: (1) the forcing rule fires
`must_rest` where the teacher would instead have inserted a b45 and pushed
on; (2) the network itself picks r2 opportunistically because rests are rare
in training (1.6 % r2, 0.3 % r1) and class weighting over-encourages them;
(3) genuine covariate shift — after one early rest the state is off-teacher
and the error compounds (a DAgger case).

CODE TRAP (hit twice now): the solution JSONs store "no break/no rest" as
`None`, `0`, `"0"` or `"none"`, and the string `"0"` is TRUTHY. Any count of
`a.get('rest_type')` must go through `extract_dataset._norm` first — an
uncorrected count reported "teacher 43 rests" on a 25 h route.

## FIX: the culprit was the CLASS WEIGHTING, not the forcing rule
Attribution counter in rollout.py (`rests: N FORCED / M chosen freely`)
settled it: of 364 rests, **20 were forced by `must_rest`, 344 were chosen
freely by the network**. The guard was innocent.

Cause: `weight_c ∝ (N/n_c)**0.5` put the dominant "just drive on" class at
weight **0.035** and a 2-example class at **5.3** — a 150× ratio. Predicting
a REST when the truth is PASS was nearly free, while a daily rest costs
9–11 h of route time. The loss had stopped caring about the single most
expensive mistake the policy can make. `--cw-power` now exposes the exponent.

Val, seed-split, 129 instances (K=20, no-split):

| variant | cw | guard | dur vs teacher | penalised | >5% runs | infeas | HoS | strand |
|---------|----|-------|----------------|-----------|----------|--------|-----|--------|
| original | 0.5 | nom | +1.50% | +2.54% | 43 | 14 | 1 | 13 |
| unweighted | 0 | nom | **−0.10%** | — | 5 | 20 | 13 | 10 |
| mild | 0.25 | nom | +0.14% | +0.62% | 15 | 14 | 5 | 10 |
| unweighted+guard | 0 | 0.95 | +0.15% | +0.80% | **6** | **5** | 5 | **0** |
| **mild+guard** | 0.25 | 0.95 | +0.40% | +0.83% | 14 | **3** | 3 | **0** |

* Unweighted training BEATS the 72 s teacher on the median (−0.10%) and
  cuts catastrophic runs 43 → 5, but under-predicts rests, so `must_rest`
  fires 115× (vs 20) and 13 HoS breaches slip past a nominal guard.
* Pairing mild weighting with the 0.95 guard gives the best feasibility
  (3 infeasible, 0 strandings) at +0.40%; cw=0 + guard 0.95 is the
  quality/robustness sweet spot (+0.15%, 6 bad runs, 5 infeasible).
* Extra-rests-vs-teacher fell +0.34 → +0.04, confirming the mechanism.
* FOURTH disagreement between supervised and closed-loop metrics: the
  unweighted model has the WORST balanced accuracy of any model trained
  (0.277 vs 0.571) and the BEST closed-loop result. This is now the paper's
  central methodological claim, not an anecdote.

## DAgger (round 1 in progress)
`code/dagger.py` — the STUDENT drives, the exact-tail teacher labels the
states it actually reaches; `code/merge_dagger.py` folds shards into a
training set; `train.py --dagger N --dagger-weight W` trains on it.

Design notes:
* Teacher queries use the SAME config as the staged runs (S=25, H=24 h,
  `solve_mode=mip`, no pruning) — a cheaper teacher would silently change
  what is being cloned between rounds.
* `--prob` samples a subset of visited stops (a query costs ~54 s measured).
  Sampling is UNIFORM over the trajectory on purpose: labelling only the
  crashes would bias the set toward disasters.
* TRAIN-split instances only — aggregating val/test states would leak.
* One shard per instance ⇒ resumable and parallel (`--slice i/N`).
* Aggregated rows are ~2 % of the deck by count, so they are upweighted
  (default ×5). That is DAgger's β — how much of the learner's own
  distribution to mix in — as a loss weight rather than a sampling rate.
* merge_dagger.py reports teacher/student DISAGREEMENT and the class-mix
  shift between base and aggregated rows: a direct measurement of covariate
  shift, and a paper figure if it falls across rounds.

Round 1 launched: 72 train instances, prob 0.35, 4 parallel workers,
≈10 h wall clock, expected ≈2,000 aggregated labels.

WINDOWS GOTCHA: `nohup ... &` from the Bash tool does NOT survive, and Git
Bash `ps` cannot see surviving Windows processes (it reported 0 while 20 were
running). Use harness background tasks, and check with PowerShell
`Get-CimInstance Win32_Process`.

## Open items
- [ ] Recompile LA stats including the new long-route MIPTAIL batch
      (long-route rows of the motivation table).
- [ ] Download `neurips_2026.sty`; author block (non-anonymous).
- [ ] Email organizers re in-person presentation requirement.
- [ ] Extraction script: JSON → (features, labels) table; audit + dedup pass
      consistent with `compile_solutions` conventions.
