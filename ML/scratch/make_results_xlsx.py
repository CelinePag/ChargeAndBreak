"""ML/RESULTS_OVERVIEW.xlsx -- every learned-policy experiment, one row per
(model, training data, test set).  Numbers are copied from the result stores
via the scripts named in the 'Read me' sheet; rerun after new results:

    python ML/scratch/make_results_xlsx.py
"""
import os

from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

OUT = os.path.join(os.path.dirname(__file__), "..", "RESULTS_OVERVIEW.xlsx")
P = None   # pending / not applicable

S1 = "1. Same kind of routes as in training"
S2 = "2. Different route length"
S3 = "3. Other vehicle / network (same along the route)"
S4 = "4. Charger power varying along the route"
S5 = "5. Attempts to beat the look-ahead"
BASE = "base routes (350 kW, 60 km, 500 kWh)"
ALLU = "all uniform variants"
LOO = "all uniform variants except the tested one"
T125 = "base routes"

# section, model, setting, trained on, tested on, n test routes, result, se, broken, comment
ROWS = [
    (S1, "PyTorch network", "main model", BASE, T125, 125, 0.17, 0.30, "0 / 375",
     "2.9 % vs 2.8 % above the best schedule in hindsight; 2 ms vs 71 s per decision"),
    (S1, "Trees (LightGBM)", "", BASE, T125, 125, 0.48, 0.35, "0 / 375", ""),
    (S1, "Classifier", "predicts the look-ahead's chosen action", BASE, T125, 125, 1.02, 0.36, "0 / 375", ""),
    (S1, "scikit-learn network", "", BASE, T125, 125, 1.56, 0.35, "0 / 375", ""),
    (S1, "Greedy rule", "reference", "—", T125, 125, 2.78, 0.35, "—", ""),
    (S1, "PyTorch network", "no shield", BASE, T125, 125, -0.18, 0.39, "38 / 375",
     "result on the 96 routes all 3 seeds finished; failures mostly on long routes"),
    (S1, "Trees (LightGBM)", "no shield", BASE, T125, 125, -0.16, 0.38, "18 / 375",
     "result on the 109 routes all 3 seeds finished"),
    (S1, "PyTorch network", "first version, no ranking loss", BASE, "base routes (validation)", P, 2.37, P, "—",
     "one extra daily rest on six routes; fixed by the ranking loss"),
    (S1, "Trees (LightGBM)", "trained on raw costs instead of per-stop differences", BASE, T125, 125, 3.54, P, "—", ""),
    (S1, "scikit-learn network", "trained on raw costs instead of per-stop differences", BASE, T125, 125, 43.0, P, "—", ""),
    (S1, "Trees (LightGBM)", "65 instead of 639 training routes", "base routes, 10 % of them", T125, 125, 0.87, 0.35, "—",
     "all 639 routes: +0.26 ± 0.43 in the same evaluation"),

    (S2, "Trees (LightGBM)", "", "short + medium base routes", "long base routes", 30, -0.31, 0.55, "0 / 90", ""),
    (S2, "Classifier", "", "short + medium base routes", "long base routes", 30, 0.45, 0.59, "0 / 90", ""),
    (S2, "scikit-learn network", "", "short + medium base routes", "long base routes", 30, 2.11, 0.61, "0 / 90", ""),
    (S2, "Trees (LightGBM)", "reference: trained on all lengths", BASE, "long base routes", 30, -0.05, 0.53, "0 / 90", ""),
    (S2, "PyTorch network", "reference: trained on all lengths", BASE, "long base routes", 30, -0.37, 0.54, "0 / 90",
     "PyTorch trained on short + medium only: not tested"),
]

# section 3: (model, setting, trained on, [(tested on, n, result, se)], comment)
PHYS = [
    ("Trees (LightGBM)", "", BASE,
     [("chargers 150 kW", 40, 9.9, 1.7), ("chargers 700 kW", 40, 3.0, 0.9), ("chargers 1000 kW", 36, 1.8, 0.8),
      ("battery 300 kWh", 40, 2.6, 1.0), ("battery 700 kWh", 40, 2.0, 0.8), ("battery 900 kWh", 40, 1.8, 1.0),
      ("chargers every 100 km", 40, -0.1, 1.0)], "one training seed"),
    ("Trees (LightGBM)", "", ALLU,
     [("chargers 150 kW", 40, 0.5, 1.0), ("chargers 700 kW", 40, 1.8, 0.8), ("chargers 1000 kW", 36, 2.4, 1.0),
      ("battery 300 kWh", 40, 0.3, 1.2), ("battery 700 kWh", 40, 1.5, 0.9), ("battery 900 kWh", 40, 1.6, 1.2),
      ("chargers every 100 km", 40, 0.2, 1.0), ("base routes", 125, 0.7, 0.4)], "one training seed"),
    ("Trees (LightGBM)", "", LOO,
     [("chargers 150 kW", 40, 5.5, 1.4), ("chargers 700 kW", 40, 2.5, 1.1), ("chargers 1000 kW", 36, 1.9, 0.9),
      ("battery 300 kWh", 40, 0.8, 1.4), ("battery 700 kWh", 40, 0.9, 0.4), ("battery 900 kWh", 40, 1.6, 1.2),
      ("chargers every 100 km", 40, 0.2, 0.9)], "one training seed"),
    ("PyTorch network", "", BASE,
     [("chargers 150 kW", 40, 17.6, 1.6), ("chargers 700 kW", 40, 4.1, 0.6), ("chargers 1000 kW", 36, 15.4, 1.2),
      ("base routes", 125, 0.1, 0.4)], "3 training seeds; an unnecessary daily rest on 16–35 of 48 routes, varies by seed"),
    ("PyTorch network", "", ALLU,
     [("chargers 150 kW", 40, 0.2, 0.9), ("chargers 700 kW", 40, 1.0, 0.8), ("chargers 1000 kW", 36, 0.0, 0.9),
      ("base routes", 125, 0.4, 0.5)], "one training seed; battery and spacing variants not tested"),
    ("PyTorch network", "", LOO,
     [("chargers 150 kW", 40, 7.3, 1.7), ("chargers 700 kW", 40, P, P), ("chargers 1000 kW", 36, P, P),
      ("base routes (150 kW held out)", 125, 0.04, 0.36)], "one training seed; 700 / 1000 kW still training"),
    ("Greedy rule", "reference", "—",
     [("chargers 150 kW", 37, 4.8, 1.2), ("chargers 700 kW", 40, 3.6, 0.4), ("chargers 1000 kW", 29, 8.6, 1.0),
      ("battery 300 kWh", 40, 8.6, 1.8), ("battery 700 kWh", 40, 6.3, 0.9), ("battery 900 kWh", 40, 3.8, 0.9),
      ("chargers every 100 km", 40, 2.8, 1.2)], ""),
]
for model, setting, trained, tests, comment in PHYS:
    for tested, n, res, se in tests:
        c = comment if res is not None else "still training (results expected 2026-10-20)"
        ROWS.append((S3, model, setting, trained, tested, n, res, se, "0" if model != "Greedy rule" else "—", c))
ROWS.append((S3, "scikit-learn network", "", BASE, "chargers 150 kW", 40, P, P, "0",
             "35 % above the best schedule in hindsight (trees 7 %): not converted to pp"))

# section 4: (model, setting, trained on, power-mixed (res, se), power+spacing (res, se))
PM, MX = "power-mixed routes", "power + spacing mixed routes"
U = "all uniform variants"
MIXED = [
    ("Greedy rule", "reference", "—", (9.0, 1.0), (10.9, 1.4)),
    ("Trees (LightGBM)", "", BASE, (6.3, 1.3), (6.2, 1.5)),
    ("Trees (LightGBM)", "", U, (7.5, 1.2), (7.6, 2.0)),
    ("Trees (LightGBM)", "", U + " + 47 power-mixed routes", (5.4, 1.2), (5.0, 1.3)),
    ("Trees (LightGBM)", "", U + " + 121 power-mixed routes", (4.6, 1.5), (4.2, 1.2)),
    ("Trees (LightGBM)", "", U + " + DAgger corrections", (8.0, 1.9), (6.6, 1.7)),
    ("PyTorch network", "", BASE, (15.4, 2.7), (18.2, 2.6)),
    ("PyTorch network", "", U, (7.1, 1.3), (4.9, 1.4)),
    ("PyTorch network", "+ charger inputs", U, (4.5, 1.1), (4.0, 1.2)),
    ("ChargerNet", "built to compare chargers", U, (6.3, 2.1), (5.7, 1.7)),
    ("ChargerNet", "built to compare chargers", U + " + 47 power-mixed routes", (3.8, 0.9), (3.1, 1.3)),
    ("PyTorch network", "+ charger inputs", U + " + 47 power-mixed routes", (3.4, 0.6), (3.5, 1.1)),
    ("PyTorch network", "+ charger inputs", U + " + 121 power-mixed routes", (3.0, 1.2), (2.9, 0.8)),
    ("PyTorch network", "+ charger inputs", U + " + 121 power-mixed routes, weighted ×5", (3.0, 1.2), (3.1, 0.8)),
    ("PyTorch network", "+ charger inputs", "121 power-mixed routes only", (5.0, 1.1), (5.5, 1.2)),
    ("PyTorch network", "+ charger inputs", U + " + DAgger, 1 round (1,781 look-ahead calls)", (2.2, 0.6), (3.4, 1.1)),
    ("PyTorch network", "+ charger inputs", U + " + DAgger, 2 rounds (3,557 calls)", (2.4, 0.7), (4.0, 1.3)),
    ("PyTorch network", "+ charger inputs", U + " + 121 power-mixed + 117 power&spacing-mixed routes", (2.5, 1.2), (2.8, 1.1)),
    ("PyTorch network", "+ charger inputs", "those 238 mixed routes only", (4.4, 1.4), (3.5, 1.1)),
    ("PyTorch network", "+ charger inputs", U + " + 238 mixed routes, weighted ×5", (2.3, 0.9), (3.0, 1.1)),
]
for model, setting, trained, pm, mx in MIXED:
    note = "base case +2.5 (worse)" if "only" in trained else ""
    for tested, n, (res, se) in ((PM, 32, pm), (MX, 30, mx)):
        ROWS.append((S4, model, setting, trained, tested, n, res, se, "0" if model != "Greedy rule" else "1–4 / 32", note))
HYB = "PyTorch + look-ahead (hybrid)"
H_TR = U + " + 238 mixed routes (the model chosen above)"
for setting, res, se, broken, note in (
        ("look-ahead when rest vs no rest is within 10 min", 2.46, 0.49, "1 / 192", "2.7 look-ahead calls per route; the battery failure followed a look-ahead decision"),
        ("look-ahead when rest vs no rest is within 30 min", 1.81, 0.36, "1 / 192", "8.4 calls per route"),
        ("look-ahead when rest vs no rest is within 60 min", 1.68, 0.35, "0 / 192", "12.1 calls per route (the look-ahead alone makes about 72 decisions)"),
        ("look-ahead at every stop of the last 5 h", 1.99, 0.46, "1 / 192", "18.3 calls per route")):
    ROWS.append((S4, HYB, setting, H_TR, "both kinds of mixed routes together", 64, res, se, broken, note))
ROWS.append((S4, "All models", "", "", "spacing-mixed routes (power uniform)", 32, P, P, "0",
             "no look-ahead runs there; every model within about 1 pp of its uniform-route result (some PyTorch variants up to +3)"))

ROWS += [
    (S5, "Rollout on the trees", "try the best 3 actions forward, keep the best", BASE, T125, P, 7.6, P, "—", "follows scenario noise"),
    (S5, "Rollout on the trees", "same, with a significance gate", BASE, T125, P, -0.2, P, "—", "within noise; no extra rest fixed"),
    (S5, "Fitted policy iteration", "trees corrected with their own simulated outcomes", BASE, "1,000 fresh draws", P, P, P, "—", "+0.004 ± 0.081 h vs the trees: no change"),
    (S5, "MILP end game", "exact solver for the last shifts", BASE, "routes where the look-ahead rests once too often", 10, P, P, "—", "worse than the student on 8 of 10 routes; stopped"),
]

HEAD = ["Section", "Model", "Setting", "Trained on", "Tested on", "Test routes",
        "Result vs look-ahead (pp)", "± (std. error)", "Runs that broke a rule", "Comment"]
WIDTH = [34, 26, 40, 52, 34, 11, 14, 11, 13, 70]

wb = Workbook()
ws = wb.active
ws.title = "Results"
f_body = Font(name="Arial", size=10)
f_head = Font(name="Arial", size=10, bold=True)
fill_head = PatternFill("solid", fgColor="D9D9D9")
fill_sec = [PatternFill("solid", fgColor="FFFFFF"), PatternFill("solid", fgColor="F2F2F2")]
thin = Side(style="thin", color="BFBFBF")
border = Border(bottom=thin)

ws.append(HEAD)
for c, w in enumerate(WIDTH, start=1):
    cell = ws.cell(row=1, column=c)
    cell.font, cell.fill = f_head, fill_head
    cell.alignment = Alignment(wrap_text=True, vertical="center")
    ws.column_dimensions[get_column_letter(c)].width = w
ws.row_dimensions[1].height = 30

sections = []
for r in ROWS:
    ws.append(list(r))
    if r[0] not in sections:
        sections.append(r[0])
for i in range(2, ws.max_row + 1):
    shade = fill_sec[sections.index(ws.cell(row=i, column=1).value) % 2]
    for c in range(1, len(HEAD) + 1):
        cell = ws.cell(row=i, column=c)
        cell.font, cell.fill, cell.border = f_body, shade, border
        cell.alignment = Alignment(wrap_text=c in (3, 4, 5, 10), vertical="top")
    ws.cell(row=i, column=7).number_format = "+0.0;-0.0;0.0"
    ws.cell(row=i, column=8).number_format = "0.0"
    if ws.cell(row=i, column=7).value is None and "still training" in str(ws.cell(row=i, column=10).value):
        ws.cell(row=i, column=7).value = "pending"
ws.freeze_panes = "B2"
ws.auto_filter.ref = f"A1:{get_column_letter(len(HEAD))}{ws.max_row}"

rd = wb.create_sheet("Read me")
NOTES = [
    ("What the result means",
     "Extra time above the look-ahead on the same test routes, in percentage points of the gap to the best "
     "schedule found in hindsight (the manuscript's gap). Lower is better; 0 = as good as the look-ahead; "
     "negative = better. Paired route by route; models with several training seeds are averaged per route first."),
    ("Test routes", "Held-out seeds 22–25. Base case: 125 routes (short, medium, long). Other vehicle / network: "
     "about 40 routes per value. Mixed charger routes: 32 power-mixed, 30 power+spacing-mixed (short and medium)."),
    ("Noise", "Typical standard error about ±0.4 pp on the base case and ±1–2 pp elsewhere; differences smaller "
     "than about twice the ± are not meaningful."),
    ("Runs that broke a rule", "Runs that violated a driving-time rule or ran out of battery; such a run ends "
     "there. All results use the shield unless the setting says 'no shield'."),
    ("Look-ahead itself", "2.8 % above the best schedule on base routes, 2.4 % on power-mixed and 2.7 % on "
     "power+spacing-mixed routes; it loses nothing when charger power varies along the route."),
    ("Base routes", "Chargers of 350 kW about every 60 km, 500 kWh battery."),
    ("All uniform variants", "Base routes plus routes with one setting changed: battery 300 / 700 / 900 kWh, "
     "chargers 150 / 700 / 1000 kW, chargers every 30 / 100 km."),
    ("Models", "PyTorch network = main model (ranking loss, one network per output). '+ charger inputs' = also "
     "told the power and distance of each of the next chargers. ChargerNet = PyTorch network built to compare "
     "chargers. Trees = LightGBM. Classifier = predicts the look-ahead's chosen action. scikit-learn network = MLP. "
     "Hybrid = PyTorch network that hands doubtful rest decisions to the look-ahead."),
    ("Sources", "ML/results/*.json(l), computed with ML/scratch/cmp_test.py, cmp_noshield.py, cmp_power_torch.py, "
     "overview_mixed.py, cmp_hybrid.py and ML/code/phys_eval.py; summary in ML/RESULTS_OVERVIEW.md."),
]
rd.column_dimensions["A"].width = 26
rd.column_dimensions["B"].width = 110
for i, (k, v) in enumerate(NOTES, start=1):
    a, b = rd.cell(row=i, column=1, value=k), rd.cell(row=i, column=2, value=v)
    a.font, b.font = f_head, f_body
    a.alignment = Alignment(vertical="top")
    b.alignment = Alignment(wrap_text=True, vertical="top")

wb.save(OUT)
print(f"[xlsx] {os.path.abspath(OUT)}: {ws.max_row - 1} rows")
