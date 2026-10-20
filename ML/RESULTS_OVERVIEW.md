# Learned policies: what we trained, what we tested, how it went

Updated 2026-10-20. All results are on held-out test routes (seeds 22–25) and
behind the shield unless stated otherwise.

**Result** = extra time above the look-ahead, in percentage points of the gap
to the best schedule found in hindsight, paired on the same routes. Lower is
better, 0 means as good as the look-ahead, and negative means better. Typical
noise is about ±0.4 on the base case and ±1 to ±2 elsewhere.

**Rules broken** counts runs that violated a driving-time or battery rule; a
violation ends the run.

The models:
- **PyTorch network**: the main model, with a ranking loss and one network
  per output.
- **PyTorch + charger inputs**: the same network, also given the power and
  distance of each of the next chargers.
- **ChargerNet**: a PyTorch network built to compare chargers.
- **Trees**: LightGBM.
- **Classifier**: predicts the look-ahead's chosen action directly.
- **scikit-learn network**: a plain MLP.

The base routes have 350 kW chargers about every 60 km and a 500 kWh battery.
"All uniform variants" means the base routes plus routes with one setting
changed:
- battery 300 / 700 / 900 kWh;
- chargers 150 / 700 / 1000 kW;
- chargers every 30 / 100 km.

## 1. Same kind of routes as in training (base case, 125 test routes × 3 training seeds)

| Model | Trained on | Tested on | Result | Rules broken |
|---|---|---|---|---|
| PyTorch network | base routes | base routes | **+0.2 ± 0.3** (2.9 % vs 2.8 % above the best schedule) | 0 / 375 |
| Trees | base routes | base routes | +0.5 ± 0.4 | 0 / 375 |
| Classifier | base routes | base routes | +1.0 ± 0.4 | 0 / 375 |
| scikit-learn network | base routes | base routes | +1.6 ± 0.4 | 0 / 375 |
| Greedy rule (reference) | — | base routes | +2.8 ± 0.4 | — |
| PyTorch network, **no shield** | base routes | base routes | about the same on the runs that finish | **38 / 375 (10 %)** |
| Trees, **no shield** | base routes | base routes | about the same on the runs that finish | **18 / 375 (5 %)** |
| PyTorch, first version (no ranking loss) | base routes | base routes (validation) | +2.4: one extra daily rest on six routes | — |
| Trees, trained on raw costs instead of per-stop differences | base routes | base routes | +3.5 | — |
| scikit-learn network, raw costs | base routes | base routes | +43 | — |
| Trees, 65 instead of 639 training routes | base routes | base routes | +0.9 (all 639: +0.3, same evaluation) | — |

## 2. Different route length

| Model | Trained on | Tested on | Result | Rules broken |
|---|---|---|---|---|
| Trees | short + medium base routes | long routes | **−0.3 ± 0.6** | 0 / 90 |
| Classifier | short + medium | long routes | +0.5 ± 0.6 | 0 / 90 |
| scikit-learn network | short + medium | long routes | +2.1 ± 0.6 | 0 / 90 |
| PyTorch network | — | — | *not tested* | — |

This works only with the full shield. With the earlier, weaker shield, trees
broke rules in 90 long-route runs and the scikit-learn network in 226.

## 3. Different vehicle or network, the same everywhere on the route (~40 test routes per value)

| Model | Trained on | Tested on | Result | Rules broken |
|---|---|---|---|---|
| Trees | base routes (350 kW) | chargers 150 / 700 / 1000 kW | **+9.9** / +3.0 / +1.8 | 0 |
| Trees | all uniform variants | chargers 150 / 700 / 1000 kW | +0.5 / +1.8 / +2.4 | 0 |
| Trees | all variants except the tested one | chargers 150 / 700 / 1000 kW | **+5.5** / +2.5 / +1.9 | 0 |
| PyTorch network | base routes (350 kW) | chargers 150 / 700 / 1000 kW | **+17.6** / +4.1 / **+15.4** | 0 |
| PyTorch network | all uniform variants | chargers 150 / 700 / 1000 kW | **+0.2 / +1.0 / 0.0** | 0 |
| PyTorch network | all variants except the tested one | chargers 150 / 700 / 1000 kW | **+7.3** / *running* / *running* | 0 |
| scikit-learn network | base routes | chargers 150 kW | 35 % above the best schedule (trees 7 %) | 0 |
| Greedy rule (reference) | — | chargers 150 / 700 / 1000 kW | +4.8 / +3.6 / +8.6 | — |
| Trees | base routes (500 kWh) | battery 300 / 700 / 900 kWh | +2.6 / +2.0 / +1.8 | 0 |
| Trees | all uniform variants | battery 300 / 700 / 900 kWh | +0.3 / +1.5 / +1.6 | 0 |
| Trees | all variants except the tested one | battery 300 / 700 / 900 kWh | +0.8 / +0.9 / +1.6 | 0 |
| Trees | base / all / all but tested | chargers every 100 km | −0.1 / +0.2 / +0.2 | 0 |

Training on all variants costs the base case +0.4 (trees +0.7 vs +0.3;
PyTorch +0.4 vs +0.1).

## 4. Charger power varying along the route (32 power-mixed and 30 power-and-spacing-mixed test routes)

The look-ahead itself is 2.4 % (power-mixed) and 2.7 % (power and spacing)
above the best schedule; it loses nothing from the mixing.

| Model | Trained on | Result: power-mixed | Result: power + spacing mixed |
|---|---|---|---|
| Greedy rule (reference) | — | +9.0 | +10.9 |
| Trees | base routes | +6.3 | +6.2 |
| Trees | all uniform variants | +7.5 | +7.6 |
| Trees | all uniform + 121 power-mixed routes | +4.6 | +4.2 |
| Trees | all uniform + DAgger corrections | +8.0 | +6.6 |
| PyTorch network | base routes | **+15.4** | **+18.2** |
| PyTorch network | all uniform variants | +7.1 | +4.9 |
| PyTorch + charger inputs | all uniform variants | +4.5 | +4.0 |
| ChargerNet | all uniform variants | +6.3 | +5.7 |
| ChargerNet | all uniform + 47 power-mixed routes | +3.8 | +3.1 |
| PyTorch + charger inputs | all uniform + 47 power-mixed routes | +3.4 | +3.5 |
| PyTorch + charger inputs | all uniform + 121 power-mixed routes | +3.0 | +2.9 |
| PyTorch + charger inputs | same, mixed routes weighted ×5 | +3.0 | +3.1 |
| PyTorch + charger inputs | 121 power-mixed routes only | +5.0 | +5.5 |
| PyTorch + charger inputs | all uniform + DAgger, 1 round (1,781 look-ahead calls) | **+2.2** | +3.4 |
| PyTorch + charger inputs | all uniform + DAgger, 2 rounds (3,557 calls) | +2.4 | +4.0 |
| PyTorch + charger inputs | all uniform + 121 power-mixed + 117 power-and-spacing-mixed routes | +2.5 | **+2.8** |
| PyTorch + charger inputs | those 238 mixed routes only | +4.4 | +3.5 (base case +2.5) |
| PyTorch + charger inputs | all uniform + 238 mixed routes ×5 | +2.3 | +3.0 |
| **PyTorch (238 mixed routes) + look-ahead on doubtful rest decisions** | as above; about 12 look-ahead calls per route | **+1.7 overall** (both kinds of route together), 0 of 192 runs broke a rule | — |

- The other hybrid trigger settings were:
  - rest vs no rest within 10 min: 2.7 calls, +2.5;
  - within 30 min: 8.4 calls, +1.8;
  - every stop of the last 5 h: 18.3 calls, +2.0.

  Each of those three had 1 run of 192 run out of battery after a
  look-ahead decision.
- **No other rule was broken in any row.**
  Every row with mixed-route data keeps the base case within noise of the
  look-ahead, except the mixed-only rows.
- **Spacing varying along the route (power uniform):** every model stays
  within about 1 point of its uniform-route result (some PyTorch variants up
  to +3).

## 5. Attempts to do better than the look-ahead (base case)

| Method | Result | Verdict |
|---|---|---|
| Rollout: try the student's best 3 actions forward, keep the best | +7.6 (follows scenario noise); −0.2 with a significance gate | no gain |
| Fitted policy iteration: correct the student with its own simulated outcomes | +0.004 h | no gain |
| Exact MILP for the last shifts of the route | worse on 8 of 10 routes, stopped | no gain |
