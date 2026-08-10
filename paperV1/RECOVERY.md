# Paper V1 — archived results and recovery notes

Snapshot taken 2026-08-10, before rebasing the charging model onto a 350 kW
AFIR-compliant charge point (Regulation (EU) 2023/1804).

Everything below was produced with the **old base case**:
`TBAR = {0: 0.0, 1: 0.55, 2: 1.367, 3: 2.50}` on energy breakpoints
0 / 40 / 80 / 100 % of a 500 kWh pack — i.e. 364 kW peak, **275 kW sustained
over the 20–80 % operating band**, 200 kW averaged over 0–100 %.
None of it is comparable with results generated after the rebase.

## What is in this folder

| Path | Contents | Files |
|---|---|---|
| `data_output/` | compiled result CSVs + `solution_summary.xlsx` | 10 |
| `figures/` | all paper and additional-experiment figures (PDF + PNG) | 45 |
| `tables/` | generated LaTeX tables (was `tex/tables/`) | 13 |

## What was deleted

Regenerable run artefacts, removed in the same change:

| Path | Files | Size |
|---|---|---|
| `instances/` | 1800 | 55 MB |
| `instances_sens/` | 1203 | 30 MB |
| `solutions/` | 13067 | 941 MB |
| `logs/` | 35090 | 9.1 GB |

`solutions/` included the `oracle_<instance>.json` bound caches.

## Recovery

The working tree was clean at the time of the snapshot, so every deleted file
is committed at:

```
70fd5342a82704b05d021c529f7d8ee49a098c70
```

Restore any of it with:

```sh
git checkout 70fd5342a82704b05d021c529f7d8ee49a098c70 -- instances
git checkout 70fd5342a82704b05d021c529f7d8ee49a098c70 -- instances_sens
git checkout 70fd5342a82704b05d021c529f7d8ee49a098c70 -- solutions
git checkout 70fd5342a82704b05d021c529f7d8ee49a098c70 -- logs
```

### Not recoverable

1366 `logs/*_scenarios.json` files were gitignored (`logs/*scenarios.json`) and
are therefore **not** in history. They are seeded scenario realizations and are
reproduced by re-running; nothing else depended on them.

## Deliberately kept in place

- `data/R15-PGLT/` — the PGLT benchmark set. Independent of the charging model.
- `archive/` — pre-existing archive, untouched.
- `_old/` — untouched.

## Known consequence

`tex/tables/` no longer exists, so the paper will not compile until the tables
are regenerated. The `\input{tables/...}` calls in `tex/sections/` are expected
to fail in the meantime.
