# ML — workshop paper: distilling the exact-tail LA policy

Target: **ML×OR workshop @ NeurIPS 2026** (Atlanta, Dec 12/13). Submission
**Aug 31, 2026 AoE**, 4 pages NeurIPS format, non-archival, non-anonymous.
https://mlxor-2026.github.io/

Everything for this paper lives here, separate from the journal-paper tree.

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

## Open items
- [ ] Recompile LA stats including the new long-route MIPTAIL batch
      (long-route rows of the motivation table).
- [ ] Download `neurips_2026.sty`; author block (non-anonymous).
- [ ] Email organizers re in-person presentation requirement.
- [ ] Extraction script: JSON → (features, labels) table; audit + dedup pass
      consistent with `compile_solutions` conventions.
