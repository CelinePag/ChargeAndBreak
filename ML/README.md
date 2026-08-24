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
      --k 20 --split-mode seed --guard-q 0.95 \
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
=> Paper-worthy IF it replicates. **CORRECTION 2026-08-23: it was NOT
carried forward.** `--no-split` in rollout.py is `store_true` with
default False, and the final val/test runs never passed it, so the
reported student DOES use split breaks (test: b30 x183, b15 x161).
These numbers were measured under the superseded pre-halt semantics and
have not been re-validated. Treat the split-break restriction as an
untested idea, not a result; main.tex is written accordingly.

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

## DAgger round 1 — RESULTS (2026-08-23)

Collected on the HPC: **530 shards, 17,941 teacher-labelled states** at the
student's own visited states = 19.4 % of the merged training set (far more
than the ~2,000 originally sized for; `--limit 0` took all train instances).

**Covariate shift, measured.** At student-visited states the teacher wants
MORE rest than in the base distribution: r1 0.33 % -> 0.81 % (2.5x),
r2 1.63 % -> 2.33 %, charge+r1 0.10 % -> 0.20 %. Consistent with the cw=0
model under-resting (must_rest fired 115x vs 20x). That table IS the
covariate-shift measurement.

**The aggregation weight had to be re-tuned.** `--dagger-weight 5` was sized
for ~2,000 rows; with 17,941 rows it makes aggregated data 55 % of effective
loss mass and over-corrects. Weight 1 is right at this collection size.

Val, 129 instances, K=20 / cw=0 / guard 0.95 / no-split, CURRENT halt
semantics, 3 training seeds each:

| variant | median vs teacher | halts /129 | max | ms/dec |
|---------|-------------------|-----------|-----|--------|
| base, 3 seeds | **+0.435 ± 0.187** | **5.7 ± 0.9** | +19.5…+36.6 | 0.66 |
| DAgger w=5 (1 seed) | +0.36 | 8 | +28.2 | 0.73 |
| DAgger w=2 (1 seed) | +0.23 | 7 | +30.6 | 0.62 |
| **DAgger w=1, 3 seeds** | **+0.186 ± 0.138** | **3.3 ± 1.7** | +14.9…+36.4 | 0.64 |
| LA-MIP teacher | — | 0 | — | 72 263 |

* Duration improvement +0.25 pp is **WITHIN 2x the base seed spread** — so on
  the median DAgger is suggestive, NOT proven. Report it that way.
* The robustness gain is the stronger signal: halts 5.7 -> 3.3 (-42 %), and
  the best DAgger seed halts on 1 of 129 routes.
* TW misses unchanged (~195-214 vs teacher 112) — DAgger did not touch the
  window weakness, as expected: it corrects state-distribution mismatch, not
  an objective the loss never emphasised.

## REPO CHANGES THAT BROKE / CHANGED THIS WORK (2026-08-22)
1. **solutions/ is bucketed** (basecase / LAconfig / sensitivity / usecase),
   ZERO files at the tree root. Never glob the root — use
   `paths.in_tree(dir, pattern)` / `paths.glob_solutions`. compare_runs.py and
   plotML.py are already migrated.
2. **halt-on-infeasible**: a breaching run ENDS at that stop and carries
   `duration_h = None` plus `route_completed`, `halted_at_stop`,
   `halt_reason`, `partial_duration_h`. A duration exists only for a
   COMPLETED route, so halted runs are counted, never averaged.
   1,042 pre-halt student runs were archived to `ML/solutions_stale_prehalt/`
   because they completed routes that would now halt — they are NOT
   comparable to anything produced after the change.
3. **LP-tail LA no longer exists for the base grid** (MIPTAIL is now the
   standard "LA"). The "vs LP-tail" column is gone; the teacher is the only
   solver baseline. This actually simplifies the paper's claim to
   "student ~= the standard look-ahead at 10^5 less online compute".

## Figures regenerated 2026-08-23 (`--method=STUDENTDAGW1`)
`python ML/code/plotML.py --method=STUDENTDAGW1 all`

plotML.py was patched for the two repo changes: `_completed()` filters rows
whose run halted (duration_h = None) out of every duration statistic — never
averaged, only counted — and the LP-tail series is replaced by GREEDY, which
is now the only other baseline on the base grid.

* **ml_money** — STUDENT sits at ~0 % penalty at 0.6 ms, on the same
  horizontal as LA-MIP at 72 s, with GREEDY at +2.4 % at 0.1 ms. Teacher
  quality at greedy speed, which is the whole paper in one panel.
* **ml_paired** — now vs. teacher AND vs. greedy, by route class. Medians sit
  on zero against the teacher and clearly below zero against greedy; the
  remaining outliers are the handful of halted/near-halted routes.
* **ml_failures** — STUDENT's 5 failures are ALL `hos_spread`; greedy has 1
  `hos_sd`; teacher 0. The failure mode is now single and specific (the 15 h
  spread ceiling), not diffuse.
* **ml_schedule** — `RlongCmediumTtight_20`: student 99.7 h vs teacher 99.7 h,
  rest blocks and SoC traces essentially superimposed.

**OVER-CHARGING IS FIXED.** It was the standing weakness (student +6.1 %
charge hours, mean SoC 60.9 vs 57.2 %). Now:

| | student | teacher |
|---|---|---|
| mean SoC at arrival | 56.9 % | 57.0 % |
| 5th-pct SoC | 29.5 % | 28.2 % |
| charge events / route | 9.2 | 9.5 |
| total charge hours | 6.74 h | **6.80 h (−0.9 %)** |

The student now charges marginally LESS than the teacher and runs the pack to
the same depth. Credit is shared between cw=0 (which stopped the loss from
ignoring the majority class) and DAgger (which labelled the states the
student actually reaches).

## PAPER DRAFTED 2026-08-23 — `paper/main.tex` + `paper/refs.bib`
Full 4-page draft, ~2,310 words, 3 tables, 2 figures, 9 references. Validated
structurally (environments/braces balanced, no dangling refs, no missing or
unused bib keys, both figure PDFs resolve). pdflatex is NOT installed locally
— compile on the HPC or Overleaf. Still TODO: `neurips_2026.sty` and the
author block (non-anonymous venue).

**TEST SET WAS SPENT** (108 instances, evaluated once, config frozen first):

| policy | duration vs teacher | halts /108 | TW | t_dec | solver |
|--------|--------------------|-----------|-----|-------|--------|
| greedy | +2.74 % | 0 | 166 | 0.03 ms | no |
| **student (cloned)** | **+0.43 ± 0.18 %** | 2.7 ± 0.5 | 128 | **0.74 ms** | no |
| student + DAgger | +0.48 ± 0.31 % | 4.7 ± 0.9 | 126 | 0.79 ms | no |
| teacher (exact tail) | reference | 0 | 53 | 70,446 ms | yes |

**DAGGER IS A NEGATIVE RESULT.** It looked better on val (+0.19 vs +0.44 %,
halts 3.3 vs 5.7) but REVERSED on test: +0.48 vs +0.43 %, and MORE halts
(4.7 vs 2.7). The val gain was inside 2x the seed spread and did not
transfer. Reported honestly in the paper — it is the reason the test set is
held back. Do not quote the val DAgger numbers as a result.

Ablations re-measured under CURRENT halt semantics (val, the earlier ones
were pre-halt and not comparable):

| decision | setting | balanced acc | duration vs teacher |
|----------|---------|--------------|---------------------|
| class weighting | inverse-freq p=0.5 | **0.477** | +3.17 % |
| class weighting | **unweighted p=0** | 0.299 | **+0.56 %** |
| features | K=40 (261) | 0.452 | +2.86 % |
| features | **K=20 (141)** | 0.299 | **+0.56 %** |

In both rows the offline metric prefers the WORSE policy — this is the
paper's spine, and it now rests on current-semantics numbers.

## COMPILED 2026-08-23 — `paper/main.pdf` (5 pp: 4 body + 1 references)
Compiled with Tectonic 0.17.0 (single binary fetched to the scratch dir; no
system TeX install). Body ends on p.4 and References run to p.5, so the
4-page main-body limit is met.

Three preamble bugs found only by compiling:
1. `allcolors=blue!60!black` needs **xcolor** — hyperref pulls in plain
   `color`, which has no colour-expression syntax.
2. `newtxtext,newtxmath` replaces the legacy `times` package, which leaves
   `TU/ptm/b/n` undefined (no bold) under XeTeX.
3. **amssymb removed** — it and newtxmath both define `\Bbbk`. No
   amssymb-only symbol was used.
4. The author block must not put `[` right after `\` — LaTeX reads it as
   the optional length argument of `\`, giving the opaque
   "Missing number, treated as zero".

Rebuild:  `tectonic -X compile main.tex`  (from `paper/build/`, flat figure
paths), or upload `paper/mlxor_paper.zip` to Overleaf.

## RL (PPO) — built 2026-08-23, NOT yet run at scale
Goal: exceed the teacher. Behaviour cloning is bounded by it by construction;
RL optimises the realised objective, and the teacher is itself +2.24% from the
hindsight oracle and myopic in known ways (24 h horizon, mean-over-scenarios).

`code/rl_env.py` — lightweight episode runner. Drives the SAME `BEHDV` through
the SAME `advance()`, but with no disk I/O (the full simulator writes a JSON,
a log, a scenario file and hits the oracle cache every call). 41-126 ms per
episode. **Validated: 10/10 exact duration matches vs the real simulator** on
recorded realisations — RL optimises the world we evaluate in.

Two design points that decide whether this works at all:
* **Fresh realisation every training episode** (via `generate_scenarios`).
  Each instance ships ONE recorded realisation and that is what evaluation
  uses; training on it repeatedly would learn a PLAN, not a policy, and look
  brilliant in training and useless out of sample.
* **Halting must never pay.** Reward is negative elapsed time, so ending the
  route early stops accruing cost — the unshaped optimum is to breach a
  regulation immediately. On a halt we charge the remaining nominal drive time
  plus `HALT_PENALTY_H = 24 h`.

`code/rl_ppo.py` — PPO from the BC checkpoint (never from scratch), with:
* a value head on the shared trunk (the features that predict the teacher's
  action also predict remaining time);
* a **KL penalty to the frozen clone** on top of PPO clipping — the standard
  stabiliser when starting from a good prior;
* per-instance return scaling (routes differ 10x in length) + batch advantage
  normalisation;
* **only the discrete head is trained**; the tau_c head is frozen (already
  within 0.9% of teacher charge hours; a continuous action would double
  gradient variance for a second-order lever);
* `gamma = 1.0` — the objective is undiscounted total time;
* the same mask AND forcing rules as evaluation, so exploration cannot
  "discover" a regulatory breach as a shortcut;
* **in-training validation against the real metric** every `--eval-every`
  iterations: deterministic rollouts on RECORDED realisations, paired against
  the stored teacher. Prints `<-- BEATS TEACHER` if the median goes negative.
  Without this we would be watching training return, which is comparable to
  nothing published.

Sizing: ~2-3 s per 8 episodes, so 64 episodes/iter x 400 iters is **~2 h**
single-core. Run several seeds in parallel.

```bash
for s in 0 1 2; do
  python ML/code/rl_ppo.py       --model ML/models/policy_K20_seedsplit_seed0_cw0.pt       --iters 400 --episodes-per-iter 64 --seed $s       --eval-every 10 --eval-n 40       > ML/logs/ppo_seed$s.log 2>&1 &
done; wait
grep -h "VAL vs teacher" ML/logs/ppo_seed0.log | tail -20
```
Then evaluate the winner exactly like any other policy:
```bash
python ML/code/rollout.py --split val --split-mode seed --guard-q 0.95     --model ML/models/policy_ppo_seed0.pt --alg STUDENTPPO
python ML/code/compare_runs.py --split val --split-mode seed
```

**Honest odds.** The baseline to beat is +0.43% (test) / +0.44% (val) and the
oracle sits 2.24% below the teacher, so headroom exists. But RL fine-tuning
from a strong clone usually yields modest gains, and the two failure modes are
symmetric: too much KL anchoring and nothing moves, too little and the policy
drifts into halting. If 400 iterations do not move the validation median, the
honest report is that the clone is already at the achievable frontier for this
action space — which is itself a result. Do NOT touch the test set for this.

## RL RESULTS 2026-08-23 — PPO works, but does NOT beat the teacher
3 seeds x 400 iters x 64 episodes, from `policy_K20_seedsplit_seed0_cw0.pt`.

**Read the training log with care.** Two artifacts make its headline
unusable: `--eval-n 40` takes the first 40 val instances alphabetically, which
is **38 Rlong + 2 Rmedium** (long routes only), and halted instances DROP OUT
of the median, so a policy that halts on its worst routes flatters itself.
The log's "-0.2%, BEATS TEACHER" is a long-route-only, survivor-biased number.

Proper evaluation, full val, on the **common set completed by all six
policies** (n=101 of 129) so halt selection bias cannot leak in:

| | duration vs teacher | penalised vs teacher | halts /129 |
|---|---|---|---|
| BC (3 seeds) | +0.453 ± 0.180 | +0.910 ± 0.193 | 5.7 ± 0.9 |
| DAgger (3 seeds) | — | +0.559 ± 0.267 | 3.3 ± 1.7 |
| **PPO (3 seeds)** | **+0.112 ± 0.044** | **+0.364 ± 0.146** | **9.7 ± 1.2** |
| teacher | 0 (reference) | 0 | 0 |

* **PPO closes 75% of the BC→teacher duration gap** (+0.453 → +0.112) and
  more than halves the penalised gap. All three PPO seeds (+0.08…+0.17) beat
  all three BC seeds (+0.20…+0.61), and the PPO spread is 4x tighter — this
  is a real effect, not seed noise.
* It also cut TW misses (202 → 188), which the reward does price.
* **It does NOT beat the teacher.** +0.112% is parity-adjacent but positive.
* **Halts got 70% worse (5.7 → 9.7)** — RL bought duration with feasibility.

**Diagnosed cause: the halt penalty is too weak.** 27 of 29 PPO halts are
`hos_spread`. A 100 h route halted at 50 h costs 50 + 24 + ~40 = 114 against
~100 for completing — only ~14% worse, which is not a deterrent given the
variance. Fix before any rerun: make the penalty multiplicative (e.g. 2x the
expected completion cost) rather than additive, and/or raise the guard
quantile for the spread check specifically, since that is the single
constraint doing the damage.

**Do not report PPO as a headline until halts are fixed** — a policy that is
faster because it abandons more routes is the same trap as comparing durations
at unequal feasibility. And RL selection has used validation only; the test
set has been spent once already and a PPO test number would be a second look.

## RERUN PREP 2026-08-23 — three fixes, one of them substantive

### 1. Halt penalty is now MULTIPLICATIVE (`rl_env.py`)
v1 charged a fixed 24 h plus the unfinished remainder. Measured, that was only
~14% worse than completing a 100 h route — no deterrent. v2 makes a halt cost
`HALT_MULT = 2.0` times an ESTIMATE of completion, where remaining nominal
drive hours are converted to full cost by `DWELL_INFLATION = 2.44` — measured,
not guessed (realised duration / nominal drive hours over completed routes:
median 2.44, p10 2.23, p90 2.56). The deterrent now scales with route length
instead of being a constant long routes can absorb.

### 2. Spread-specific guard quantile (`--spread-q`, in rollout.py and rl_env.py)
`compute_flags` folds the shift-driving limit and the 15 h spread ceiling into
one `must_rest`; re-evaluating at a stricter quantile and OR-ing tightens
exactly the spread test. **Tested on BC first, before spending 2 h retraining:**
halts 5 → 4, duration +0.56 → +0.49. Real but marginal — the spread ceiling is
breached by ACCUMULATION over a shift, which a one-step check cannot foresee.
Keep it, but it is not the lever.

### 3. THE LEVER: the policy could not see the spread clock
27 of 29 v1 PPO halts were `hos_spread` — and `h` was **not in the feature
vector at all**. The policy was being asked to respect a limit it could not
observe. `h` is also absent from `sim_trajectory`, but BEHDV's rule is
`h_new = (0 if rest else h + dwell) + D_act` and the JSON stores `td_list`, so
it is EXACTLY reconstructable offline; at deployment `rollout.py`/`rl_env.py`
read the live `vehicle.h`.
Added `h_spread` and `spread_margin` (= 15 − h) at indices **21–22**, appended
so every earlier index — and therefore every mask rule in `model.py` — keeps
its meaning. Features 141 → 143.
**Validation:** reconstructed `h` over 74,380 rows spans 0.01–14.89 and never
exceeds the ceiling, which is exactly how a teacher that never breaches spread
should look.

Result (val, common completed set n=110, 3 seeds, BC only — no RL yet):

| | duration vs teacher | halts /129 | halt causes |
|---|---|---|---|
| BC without spread features | +0.428 ± 0.221 | 5.7 ± 0.9 | 16 spread, 1 sd |
| **BC with spread features** | **+0.271 ± 0.246** | **4.0 ± 0.8** | 12 spread, 0 sd |

Better on both axes at once, before any RL. Note `t_dec` rises 0.65 → 1.15 ms
(two extra features plus the second `compute_flags` call) — still ~10^4x faster
than the teacher.

### Ready to rerun
The PPO rerun should start from a spread-aware BC checkpoint, with the
multiplicative halt penalty, and fix the eval subset (v1's `--eval-n 40` took
the first 40 val instances alphabetically = 38 Rlong + 2 Rmedium, and halted
instances dropped out of the median — its "BEATS TEACHER" was long-route-only
and survivor-biased). Always re-evaluate with rollout.py + compare_runs.py on
a COMMON completed set.

## RL RERUN RESULT 2026-08-23/24 — PPO v2 BEATS the teacher on duration
3 seeds x 400 iters x 64 episodes, from the SPREAD-AWARE clone, multiplicative
halt penalty, `--eval-n 0` (all 129 val instances, not the first 40).

Independent evaluation (rollout.py + compare_runs.py), val, on the **common
completed set n=106** so halt selection bias cannot leak in:

| policy | duration vs teacher | penalised vs teacher | halts /129 | TW | t_dec |
|--------|---------------------|----------------------|-----------|-----|-------|
| BC + spread feats | +0.294 ± 0.249 | +0.787 ± 0.134 | 4.0 ± 0.8 | 203 | 1.15 ms |
| PPO v1 (old) | +0.151 ± 0.042 | +0.412 ± 0.184 | 9.7 ± 1.2 | 188 | 0.72 ms |
| **PPO v2** | **−0.238 ± 0.049** | **−0.008 ± 0.033** | **1.7 ± 0.9** | 201 | 1.28 ms |
| teacher | 0 (reference) | 0 | 0 | 112 | 72,263 ms |

* **Duration: the student is now FASTER than its own teacher** — all three
  seeds negative (−0.18, −0.30, −0.24) with a tight 0.049 spread. Imitation
  cannot exceed its teacher; RL can, and did.
* **Halts collapsed 9.7 → 1.7**, below the BC clone's 4.0. Both fixes worked:
  the multiplicative penalty made abandonment expensive, and the spread
  feature let the policy see the limit it kept breaching. Halt causes are now
  spread evenly (2 spread / 2 sd / 1 cd) instead of 27 spread — the dominant
  failure mode is GONE, not merely reduced.
* **BUT on the PENALISED objective it is only at PARITY** (−0.008 ± 0.033,
  indistinguishable from zero). The student still misses 201 windows against
  the teacher's 112, and beta x misses eats the entire duration advantage.

**Therefore state the claim precisely:** the learned policy is faster than the
exact-tail look-ahead and matches it on the full objective, at ~5x10^4 less
online compute. It does NOT dominate the teacher — windows remain the
deficiency, exactly as they have been since the forcing rules landed.

**Test set:** untouched by any of this. It was already spent once for the BC
paper number, so a PPO test evaluation is a SECOND look and must be reported
as such if used.

## ⚠️ CORRECTION 2026-08-24 — the "beats the teacher" claim does NOT hold
The `-0.238 +/- 0.049` figure is the median-over-instances of the percentage
difference, averaged over 3 training seeds. **The +/- is seed reproducibility,
not uncertainty about whether the policy is actually better.** A paired
per-instance test tells a different story (val, n=125, 3 seeds averaged):

| | mean | median | 95% CI of median | student better on | sign test |
|---|---|---|---|---|---|
| duration (h) | −0.0617 | −0.0879 | **[−0.173, +0.053]** | 68/125 (54%) | p = 0.19 |
| penalised (h) | **+0.286** | +0.127 | [−0.062, +0.340] | 56/125 (45%) | p = 0.90 |

Both CIs cross zero. The duration win is **not significant**, and on the
penalised objective the mean is POSITIVE — slightly worse than the teacher.

**Report PARITY, not superiority.** The defensible claim is: *the learned
policy matches an exact-tail MILP look-ahead at ~5x10^4 less online compute*.
That is a strong result and it is what the data supports.

More training seeds will NOT fix this — seed variance (0.049) is an order of
magnitude smaller than the instance-level spread. Detecting a 54% win rate
would need on the order of a thousand instances; we have 129 val + 108 test
uncontaminated. Either the effect is genuinely tiny or the policy needs to be
better.

This is the paper's own thesis biting us: a headline that looked tight was
hiding the uncertainty that actually mattered. Every comparative claim in the
paper should carry a paired test, not a seed spread.

## Open items
- [ ] Recompile LA stats including the new long-route MIPTAIL batch
      (long-route rows of the motivation table).
- [ ] Download `neurips_2026.sty`; author block (non-anonymous).
- [ ] Email organizers re in-person presentation requirement.
- [ ] Extraction script: JSON → (features, labels) table; audit + dedup pass
      consistent with `compile_solutions` conventions.
