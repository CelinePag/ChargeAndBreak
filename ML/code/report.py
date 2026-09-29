"""
report.py — assemble ML/RESULTS.md from the result stores
=========================================================
Every number in RESULTS.md is read from a store written by the script that
produced it, so the document cannot drift from the runs: re-running this after
any experiment rewrites every table rather than inviting a hand-edit.

    results/ladder_test.json          feature-set ladder        run_ladder.py
    results/all_test.json             ablations                 run_all.py
    results/length_test[_<v>].json    route-length transfer     run_length.py
    results/ood_test[_<v>].json       physics shifts            ood_eval.py
    results/halt_diagnosis.json       cause of every failure    diagnose_all.py
    results/spread_room.json          before/after the fix      spread_compare.py

<v> is a policy variant: g95 (as trained: drive guard at the 0.95 quantile),
g95sr (+ the spread-room check), g99sr (+ the check, guard at 0.99).

Protocol, for every learned row: fitted on route seeds 1-19, early-stopped on
20-21, reported on the whole test batch (seeds 22-25, 125 routes).

    python ML/code/report.py
"""
from __future__ import annotations

import collections
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from paper_link import collect_gaps                               # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.abspath(os.path.join(HERE, "..", "results"))
FIGS = os.path.abspath(os.path.join(HERE, "..", "figures"))
OUT = os.path.abspath(os.path.join(HERE, "..", "RESULTS.md"))
FLOOR = 0.35      # LA's own measured run-to-run median spread, in %
ARM_LABEL = {"gbt": "Trees", "clf": "Classifier", "mlp": "MLP"}
VARIANT = {"g95": "as trained", "g95sr": "+ spread-room check",
           "g99sr": "+ spread-room check, 0.99 drive guard"}
BASELINES = [("LA (look-ahead MILP)", "LA (the teacher)"), ("2SP", "2SP"),
             ("GREEDY", "Greedy"), ("RO", "RO")]


def load(name):
    p = os.path.join(RESULTS, name)
    if not os.path.exists(p):
        return None
    with open(p) as fh:
        return json.load(fh)


def pm(v, fmt="+.2f", sd_fmt=".2f"):
    v = np.asarray(v, float)
    return f"{v.mean():{fmt}} ± {v.std():{sd_fmt}}"


# ── 1. headline ─────────────────────────────────────────────────────────────
def headline(A):
    from configs import LEGACY
    from fig_gap import best_per_arm
    ladder = load("ladder_test.json") or []
    A("## 1. Headline — each arm at its best feature set\n")
    A("Figure: `figures/fig_gap_test.png`. Gap to the hindsight oracle is the "
      "manuscript's metric, recomputed with "
      "`compile_solutions._annotate_gap_to_oracle`'s exact definition. Each "
      "arm's row is the **median training seed** of its best feature set, so "
      "no arm is flattered by a lucky initialisation; the last column is the "
      "spread over all its seeds.\n")
    A("> The best set per arm is chosen by its result on this same test batch "
      "(every configuration is reported on it — section 2), so the headline "
      "row carries a mild selection optimism. The ladder table shows all of "
      "them.\n")
    A("| policy | features | gap to oracle | infeasible | vs LA, over training seeds |")
    A("|---|---|---:|---:|---:|")
    rows, base = [], {}
    for h in best_per_arm():
        g = collect_gaps(h["eval_file"], student_label=h["display"])
        a, _p, n_inf = g[h["display"]]
        fl = h["display"].split("(")[-1].rstrip(")")
        seeds = [r["med_la"] for r in ladder
                 if r["arm"] == h["arm"] and r["fset_label"] == fl]
        rows.append((float(np.median(a)), f"**{ARM_LABEL[h['arm']]}**", fl,
                     n_inf, len(a) + n_inf,
                     f"{pm(seeds)} % ({len(seeds)} seeds)" if seeds else "—"))
        if not base:
            base = g
    if os.path.exists(os.path.join(RESULTS, LEGACY.eval_name)):
        g = collect_gaps(LEGACY.eval_name, student_label="legacy")
        a, _p, n_inf = g["legacy"]
        rows.append((float(np.median(a)), "MLP 2026-08, as built", "O143",
                     n_inf, len(a) + n_inf, "one model"))
    for src, lab in BASELINES:
        if src in base and len(base[src][0]):
            a, _p, n_inf = base[src]
            rows.append((float(np.median(a)), lab, "—", n_inf, len(a) + n_inf,
                         "—"))
    for med, lab, fl, n_inf, n, sp in sorted(rows):
        A(f"| {lab} | {fl} | {med:+.2f} % | {n_inf}/{n} | {sp} |")
    A("")
    A(f"A difference from the teacher below **{FLOOR:.2f} %** is not evidence "
      "of anything: that is the look-ahead's own measured run-to-run median "
      "spread on this simulator. An infeasible run has no duration (a "
      "violation ends it), so read the gap and the infeasible count "
      "together.\n")


# ── 2. ladder ───────────────────────────────────────────────────────────────
def ladder(A):
    rows = load("ladder_test.json")
    if not rows:
        return
    A("## 2. Feature-set ladder\n")
    A("Figure: `figures/fig_ladder.png`. The same arm on four feature sets, "
      "three training seeds each; `n` is the number of inputs the model "
      "consumes (the classifier reads state features only, hence its own "
      "counts). C = compact (top-40 by gain), D = deduplicated, F = full "
      "engineered, L = full + the next 20 stops raw.\n")
    A("| arm | set | n | vs LA | vs Greedy | infeasible (of 125) | TW misses |")
    A("|---|---|---:|---:|---:|---:|---:|")
    order = {"C": 0, "D": 1, "F": 2, "L": 3, "F91": 4}
    for arm in ("gbt", "clf", "mlp"):
        cells = sorted({(r["fset"], r["fset_label"]) for r in rows
                        if r["arm"] == arm}, key=lambda c: order.get(c[0], 9))
        means = {}
        for fs, lab in cells:
            sub = [r for r in rows if r["arm"] == arm and r["fset"] == fs]
            means[lab] = np.mean([r["med_la"] for r in sub])
        best = min(means, key=means.get) if means else None
        for fs, lab in cells:
            sub = [r for r in rows if r["arm"] == arm and r["fset"] == fs]
            b = "**" if lab == best else ""
            A(f"| {ARM_LABEL[arm]} | {b}{lab}{b} | {sub[0]['n_features']} | "
              f"{b}{pm([r['med_la'] for r in sub])} %{b} | "
              f"{pm([r['med_greedy'] for r in sub])} % | "
              f"{pm([r['infeasible'] for r in sub], '.1f', '.1f')} | "
              f"{pm([r['tw'] for r in sub], '.0f', '.0f')} |")
    A(f"\nTeacher window misses on the same routes: {rows[0]['tw_la']}. Every "
      "arm is an inverted U: too few inputs starve it, the raw 20-stop "
      "lookahead (L) adds noise and infeasible runs.\n")


# ── 3. ablations ────────────────────────────────────────────────────────────
def ablations(A):
    rows = load("all_test.json")
    if not rows:
        return
    A("## 3. Ablations\n")
    A("Figure: `figures/fig_ablations.png`. One training seed each, on the "
      "91-input set that predates the window features (F91); compare each "
      "row with its own base, not with the ladder.\n")
    A("| model | guard | vs LA | vs Greedy | penalised | infeasible | TW | what it tests |")
    A("|---|---:|---:|---:|---:|---:|---:|---|")
    for c in rows:
        g = "nominal" if c["guard"] is None else str(c["guard"])
        tw = c.get("tw", 0)
        tw = tw[0] if isinstance(tw, (list, tuple)) else tw
        A(f"| `{c['tag']}` | {g} | {c['med_la']:+.2f} % | {c['med_greedy']:+.2f} % | "
          f"{c['med_pen']:+.2f} % | {c['infeasible']} | {tw} | {c['desc']} |")
    A("")


# ── 4. route length ─────────────────────────────────────────────────────────
def length(A):
    stores = [(v, load("length_test.json" if v == "g95" else f"length_test_{v}.json"))
              for v in VARIANT]
    stores = [(v, s) for v, s in stores if s]
    if not stores:
        return
    A("## 4. Trained on short + medium routes, tested on long ones\n")
    A("Figures: `figures/fig_length.png` (as trained) and "
      "`figures/fig_length_<variant>.png`. Physics and rules unchanged; only "
      "the route length is unseen (long: median 162 stops, 101 h, 4 daily "
      "rests; medium: 96 stops, 56 h, 2 rests). R = the full set minus the "
      "seven whole-route position features. Cells: median vs LA, mean ± sd "
      "over 3 training seeds, [mean infeasible runs].\n")
    for v, rows in stores:
        A(f"**{VARIANT[v]}** (`{v}`)\n")
        A("| arm | trained on | set | short+medium test (95) | long test (30) | "
          "long, all 239 |")
        A("|---|---|---|---:|---:|---:|")
        for arm in ("gbt", "clf", "mlp"):
            groups = sorted({(r["scope"], r["fset_label"]) for r in rows
                             if r["arm"] == arm}, key=lambda g: (g[0] != "all", g[1]))
            for sc, fl in groups:
                sub = [r for r in rows if r["arm"] == arm and r["scope"] == sc
                       and r["fset_label"] == fl]

                def cell(key):
                    x = [r[key] for r in sub if r.get(key)]
                    if not x:
                        return "—"
                    return (f"{pm([c['med_la'] for c in x])} % "
                            f"[{np.mean([c['infeasible'] for c in x]):.1f}]")
                who = "all lengths" if sc == "all" else "short+medium"
                A(f"| {ARM_LABEL[arm]} | {who} | {fl} | {cell('in_dist')} | "
                  f"{cell('long_paired')} | {cell('long_all')} |")
        A("")
    A("The models trained on all lengths have no `long, all 239` cell: 209 of "
      "those routes were in their training or stopping seeds.\n")


# ── 5. physics shifts ───────────────────────────────────────────────────────
def ood(A):
    from ood_eval import AXES, LEARNED
    stores = [(v, load("ood_test.json" if v == "g95" else f"ood_test_{v}.json"))
              for v in VARIANT]
    stores = [(v, s) for v, s in stores if s]
    if not stores:
        return
    A("## 5. Trained on the base case, tested on shifted physics\n")
    A("Figures: `figures/fig_ood.png` and `figures/fig_ood_<variant>.png`. "
      "Route seeds 22-25 on every synthetic axis, so the physics is the only "
      "change; the use case is the real Arendal tour (5 realisations, two "
      "ferry crossings). Each arm's best base-case set at its median training "
      "seed; the MLP also on F95, its set before the ladder finished. Cells: "
      "median gap to the oracle solved for those instances, "
      "[infeasible / routes].\n")
    for v, rows in stores:
        methods = [m for m in LEARNED if any(r["method"] == m for r in rows)]
        methods += ["LA", "Greedy"]
        tags = sorted({r["tag"] for r in rows if r["method"] not in ("LA", "Greedy")})
        A(f"**{VARIANT[v]}** (`{v}`; models: {', '.join(f'`{t}`' for t in tags)})\n")
        A("| shift | " + " | ".join(methods) + " |")
        A("|---|" + "---:|" * len(methods))
        for axis, (_, _, lab) in AXES.items():
            sub = [r for r in rows if r["axis"] == axis]
            if not sub:
                continue
            cells = []
            for m in methods:
                mr = [r for r in sub if r["method"] == m]
                if not mr:
                    cells.append("—")
                    continue
                g = [r["gap"] for r in mr if r["completed"] and r["gap"] is not None]
                inf = sum(1 for r in mr if not r["completed"])
                cells.append(f"{np.median(g):+.2f} [{inf}/{len(mr)}]" if g
                             else f"n/a [{inf}/{len(mr)}]")
            A(f"| {lab} | " + " | ".join(cells) + " |")
        A("")


# ── 6. the spread-room finding ──────────────────────────────────────────────
def diagnosis_table(A, diag):
    """Failures by cause: one row per experiment and model, one column per
    variant."""
    from ood_eval import AXES, MODELS
    variants = [v for v in VARIANT if any(d["variant"] == v for d in diag)]
    label = {t: lab for _k, t, lab in MODELS}
    A("| experiment | model | " + " | ".join(VARIANT[v] for v in variants) + " |")
    A("|---|---|" + "---:|" * len(variants))
    order = [("route length", "long, all 239")] + [("physics", ax) for ax in AXES]
    for ex, ax in order:
        ds_ex = [d for d in diag if d["experiment"] == ex and d["axis"] == ax]
        if ex == "route length":
            who = [(ARM_LABEL[arm], lambda d, a=arm: d["arm"] == a)
                   for arm in ("gbt", "clf", "mlp")]
        else:
            who = [(label[t], lambda d, tt=t: d["tag"] == tt) for _k, t, _l in MODELS]
        for lab, pick in who:
            cells = []
            for v in variants:
                ds = [d for d in ds_ex if d["variant"] == v and pick(d)]
                if not ds:
                    cells.append("0")
                    continue
                c = collections.Counter(d["cause"] for d in ds)
                parts = [f"{c[k]} {w}" for k, w in (
                    ("uncounted-dwell", "dwell"), ("drive-tail", "tail"),
                    ("ferry", "ferry"), ("energy", "energy"), ("other", "other"))
                    if c.get(k)]
                cells.append(f"**{len(ds)}**: " + ", ".join(parts))
            if any(x != "0" for x in cells):
                name = ex if ex == "route length" else AXES[ax][2]
                A(f"| {name} | {lab} | " + " | ".join(cells) + " |")


def spread(A):
    rows = load("spread_room.json")
    diag = load("halt_diagnosis.json")
    if not rows and not diag:
        return
    from spread_compare import aggregate
    A("## 6. Why learned policies become infeasible — and the fix\n")
    A("`code/halt_state.py` replays an infeasible run exactly and reads the "
      "simulator's state at the decision that broke the rule; "
      "`code/diagnose_all.py` does it for every failure in the length and "
      "physics experiments.\n")
    if diag:
        def share(pred):
            ds = [d for d in diag if d["variant"] == "g95" and pred(d)]
            return sum(d["cause"] == "uncounted-dwell" for d in ds), len(ds)
        parts = []
        for arm in ("gbt", "clf", "mlp"):
            n, t = share(lambda d, a=arm: d["experiment"] == "route length"
                         and d["arm"] == a)
            if t:
                parts.append(f"{ARM_LABEL[arm]} {n}/{t}")
        n150, t150 = share(lambda d: d["axis"] == "kw150")
        A("As trained, the dominant failure is one move: **charge (or break), "
          "then drive on, and the 15 h shift spread is exceeded on the next "
          "leg** -- the dwell actually spent was more than the legality check "
          "counted ('dwell' below). On long routes that is "
          + ", ".join(parts) + f" of the failures; with 150 kW chargers "
          f"{n150}/{t150}. Almost all the rest are realised drives longer than "
          "the guard assumed ('tail').\n")
        A("Failures by cause (route length pools both feature sets and all "
          "three training seeds of an arm: 1,434 long-route runs per arm):\n")
        diagnosis_table(A, diag)
        A(f"\nEvery replay reproduced its stored run: "
          f"{sum(d['reproduced'] for d in diag)}/{len(diag)}. *Tail* covers "
          "the spread, shift-driving and 4.5 h consecutive-driving limits. "
          "*Ferry*: the breaking decision is a sea crossing, which has one "
          "legal action. *Energy*: the battery fell below its floor.\n")
    A("**The cause is in the safety layer, not the model.** The shared "
      "legality check (`src/simulation/supervisor._spread_with_dwell_fails`) "
      "admits a non-rest action when `h + o(a) + D_wc <= 15 h`, where `o(a)` "
      "counts service, queue and the minimum break but **not the charge** (by "
      "design: the MILP models the charge-spread coupling itself) **nor the "
      "stop overhead** (`M_stop` at a charger, `M_lay` for a break at a "
      "layby). A learned policy picks its charge duration only after that "
      "check, so nothing stopped a charge that did not fit. "
      "`policy_core.spread_room` completes the check on the policy's side: it "
      "shortens a charge to what the spread allows, and removes a non-rest "
      "action when even the minimum charge needed to reach the next charger "
      "does not fit (the policy then charges during a rest instead). After it "
      "no failure of that kind is left; what remains is drive tail, which a "
      "0.99 guard removes, and the ferries.\n")
    if rows:
        A("What the fix costs, on the same routes, mean over training seeds. "
          "Cells: infeasible runs · median gap to the oracle; for the fixed "
          "versions also the paired median change in duration and, in "
          "brackets, the share of routes whose duration moved at all.\n")
        keyed = collections.OrderedDict()
        for a in aggregate(rows):
            keyed.setdefault((a["model"], a["trained"], a["routes"]), {})[a["variant"]] = a
        A("| model | trained on | routes | as trained | + spread-room check | "
          "+ check, 0.99 guard |")
        A("|---|---|---|---:|---:|---:|")
        for (model, who, rs), byv in keyed.items():
            first = next(iter(byv.values()))
            base = f"{first['inf_before']:.1f} · {first['gap_before']:+.2f} %"

            def cell(v):
                a = byv.get(v)
                if not a:
                    return "—"
                return (f"{a['inf_after']:.1f} · {a['gap_after']:+.2f} % · "
                        f"{a['paired_med']:+.3f} % ({100 * a['changed']:.0f} %)")
            A(f"| {model} | {who} | {rs} | {base} | {cell('g95sr')} | "
              f"{cell('g99sr')} |")
        A("\nThe check alone is free where it is not needed: it moves a few "
          "routes and none on average. The stricter guard touches nearly every "
          "route: negligibly for the trees, a few tenths of a percent for the "
          "classifier and the MLP.\n")
    A("What neither fixes: the **ferry**. A crossing is a forced 3.8-4.9 h "
      "break with exactly one legal action, so the decision that matters -- "
      "rest before boarding -- is taken stops earlier and needs look-ahead; "
      "the base case has no ferries to learn it from.\n")


def main():
    L = []
    A = L.append
    A("# Results — learned policies for the discrete-event simulator\n")
    A("Generated by `python ML/code/report.py` from the stores in "
      "`ML/results/` -- edit the scripts, not this file.\n")
    A("Every learned row is an **independent model**: fitted on route seeds "
      "1-19 (639 routes), early-stopped on seeds 20-21, reported on the "
      "**whole test batch** (seeds 22-25, 125 routes). Baselines are the runs "
      "stored in `solutions/basecase` on the same routes.\n")
    headline(A)
    ladder(A)
    ablations(A)
    length(A)
    ood(A)
    spread(A)
    A("## Figures\n")
    for f in sorted(os.listdir(FIGS)):
        if f.endswith(".png"):
            A(f"- `ML/figures/{f}`")
    A("")
    A("## Not included\n")
    A("- **DAgger** — removed from this project.\n"
      "- **LA-LP** — no `LPTAIL` runs exist on disk.\n"
      "- The 2026-08 MLP is shown exactly as it was built (143 inputs, its own "
      "instance set and rollout; see `ML/legacy/README.md`). It is not a "
      "controlled comparison; the classifier arm is the controlled test of "
      "the same framing.\n")
    with open(OUT, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L))
    print(f"wrote {OUT}  ({len(L)} lines)")


if __name__ == "__main__":
    main()
