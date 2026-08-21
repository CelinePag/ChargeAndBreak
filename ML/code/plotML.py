"""
plotML.py — figures for the ML×OR workshop paper
================================================
All figures for the learned-policy work live here, and everything written
lands in ML/figures/ (never the manuscript's figures/ tree).

Style is inherited from src/plot/paper_style.py when available, so the
workshop figures look like the journal ones; the method colours are fixed so
a method keeps its colour across every panel.

Figures
-------
  money    : gap/duration penalty vs. per-decision latency (log-x).  THE
             figure: student sits at ~1 ms with near-teacher quality while
             the solver policies sit at 10-100 s.
  paired   : per-instance paired deltas (student - teacher, student - LP
             tail) as box plots by route class — shows the median AND the
             tail, which a median-only table hides.
  failures : stacked failure counts by cause (stranding / HoS cd / sd /
             spread) per policy — the honest half of the story, and the
             before/after panel once forcing rules land.
  training : loss + balanced-accuracy curves from a training history JSON.

CLI (repo root):
    python ML/code/plotML.py money paired failures
    python ML/code/plotML.py all
Figures are skipped with a message when their input data is not present yet,
so the script is safe to run at any point in the project.
"""
from __future__ import annotations
import glob, json, os, re, sys
import collections
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
ML       = os.path.join(ROOT, "ML")
SOLS     = os.path.join(ML, "solutions")       # student runs (ML-only)
REF_SOLS = os.path.join(ROOT, "solutions")     # baselines (read-only)
FIGS     = os.path.join(ML, "figures")
DATA     = os.path.join(ML, "data")

try:                                            # match the journal figures
    from src.plot import paper_style            # noqa: F401
    paper_style.apply() if hasattr(paper_style, "apply") else None
except Exception:
    plt.rcParams.update({"figure.dpi": 150, "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False})

# Okabe-Ito, colour-blind safe; one fixed colour per method everywhere.
COLOR = {"GREEDY": "#999999", "LA-LP": "#0072B2", "STUDENT": "#D55E00",
         "LA-MIP": "#009E73", "ORACLE": "#000000"}
ROUTE_ORDER = ["short", "medium", "long"]


# ── data loading ─────────────────────────────────────────────────────────────
def _latest(pattern: str):
    fs = [f for f in glob.glob(pattern)
          if "nosplit" not in f and "LOCAL" not in f]
    return sorted(fs)[-1] if fs else None


def collect_runs(method: str | None = None) -> list[dict]:
    """One record per student run, joined with the baselines for the same
    instance (teacher = LA_MIPTAIL, LP tail = plain LA, greedy if present).

    `method` selects one student variant (e.g. STUDENTFK20); with several
    present and none requested we take the one with the most runs, because
    mixing variants in a single figure would silently average different
    policies.  Dedups to the LATEST run per instance, matching the reporting
    pipeline: a re-run updates a result, it does not add a second sample.
    """
    by_method: dict[str, dict[str, str]] = collections.defaultdict(dict)
    for f in sorted(glob.glob(os.path.join(SOLS, "*.json"))):
        mm = re.match(r".*_(STUDENT[A-Za-z0-9]*)_S\d", os.path.basename(f))
        if mm:
            by_method[mm.group(1)][json.load(open(f))["instance"]] = f
    if not by_method:
        return []
    if method is None:
        method = max(by_method, key=lambda k: len(by_method[k]))
        if len(by_method) > 1:
            print(f"note: {len(by_method)} student variants present "
                  f"({', '.join(sorted(by_method))}); using {method}")
    latest_by_inst = by_method.get(method, {})

    rows = []
    for inst, f in sorted(latest_by_inst.items()):
        d = json.load(open(f))
        m = d.get("metrics", {})
        rec = dict(instance=inst, _file=f,
                   family=inst.rsplit("_", 1)[0],
                   route=("long" if inst.startswith("Rlong") else
                          "medium" if inst.startswith("Rmedium") else "short"),
                   student=d.get("duration_h"),
                   student_dec=m.get("decision_time_mean_s"),
                   student_infeas=bool(m.get("run_infeasible")),
                   student_tw=m.get("tw_n_misses"),
                   student_viol=m.get("violations_by_type") or {})
        for tag, key in (("LA_MIPTAIL", "teacher"), ("LA_2026", "la_lp"),
                         ("GREEDY", "greedy")):
            g = _latest(os.path.join(REF_SOLS, f"{inst}_{tag}*.json"))
            if g:
                t = json.load(open(g))
                rec[key] = t.get("duration_h")
                rec[key + "_dec"] = t.get("metrics", {}).get("decision_time_mean_s")
                rec[key + "_infeas"] = bool(t.get("metrics", {}).get("run_infeasible"))
                rec[key + "_viol"] = t.get("metrics", {}).get("violations_by_type") or {}
        rows.append(rec)
    return rows


# ── figures ──────────────────────────────────────────────────────────────────
def fig_money(rows):
    """Quality vs. online latency — the paper's headline figure.

    y: median paired duration penalty relative to the exact-tail teacher
       (0% = teacher quality; higher = worse).
    x: median seconds per decision, log scale.
    """
    series = [("STUDENT", "student"), ("LA-LP", "la_lp"),
              ("LA-MIP", "teacher"), ("GREEDY", "greedy")]
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    for label, key in series:
        pts = [(r[key + "_dec"], 100 * (r[key] - r["teacher"]) / r["teacher"])
               for r in rows
               if r.get(key) and r.get("teacher") and r.get(key + "_dec") is not None]
        if not pts:
            continue
        x = np.median([p[0] for p in pts]); y = np.median([p[1] for p in pts])
        lo, hi = np.percentile([p[1] for p in pts], [25, 75])
        ax.errorbar(max(x, 1e-4), y, yerr=[[y - lo], [hi - y]], fmt="o",
                    ms=9, capsize=3, color=COLOR[label], label=label, zorder=3)
        ax.annotate(label, (max(x, 1e-4), y), textcoords="offset points",
                    xytext=(8, 6), fontsize=8, color=COLOR[label])
    ax.axhline(0, ls="--", lw=0.8, c=COLOR["LA-MIP"], zorder=1)
    ax.set_xscale("log")
    ax.set_xlabel("median decision time per stop (s, log scale)")
    ax.set_ylabel("duration penalty vs. exact-tail teacher (%)")
    ax.set_title("Solution quality against online decision latency")
    ax.grid(alpha=.25, which="both", lw=.4)
    _save(fig, "ml_money")


def fig_paired(rows):
    """Paired per-instance deltas by route class: median AND tail."""
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), sharey=True)
    for ax, (ref, title) in zip(axes, [("teacher", "vs. exact-tail teacher"),
                                       ("la_lp", "vs. LP-tail look-ahead")]):
        data, labels = [], []
        for rc in ROUTE_ORDER:
            d = [100 * (r["student"] - r[ref]) / r[ref] for r in rows
                 if r["route"] == rc and r.get(ref) and r.get("student")]
            if d:
                data.append(d); labels.append(f"{rc}\n(n={len(d)})")
        if not data:
            continue
        bp = ax.boxplot(data, tick_labels=labels, showfliers=True, widths=.55,
                        patch_artist=True, medianprops=dict(color="black"))
        for patch in bp["boxes"]:
            patch.set_facecolor(COLOR["STUDENT"]); patch.set_alpha(.55)
        ax.axhline(0, ls="--", lw=.8, c="k")
        ax.set_title(title, fontsize=9)
        ax.grid(axis="y", alpha=.25, lw=.4)
    axes[0].set_ylabel("student duration difference (%)\n(negative = student better)")
    _save(fig, "ml_paired")


def fig_failures(rows):
    """Failure counts by cause — the honest half of the story."""
    causes = ["stranding", "hos_cd", "hos_sd", "hos_spread"]
    methods = [("STUDENT", "student_viol"), ("LA-MIP", "teacher_viol"),
               ("LA-LP", "la_lp_viol")]
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    width, xs = .25, np.arange(len(causes))
    for i, (label, key) in enumerate(methods):
        tot = collections.Counter()
        for r in rows:
            for c, n in (r.get(key) or {}).items():
                tot[c] += n
        if not tot and label != "STUDENT":
            continue
        ax.bar(xs + (i - 1) * width, [tot.get(c, 0) for c in causes],
               width, label=label, color=COLOR[label], alpha=.9)
    ax.set_xticks(xs); ax.set_xticklabels(causes, fontsize=8)
    ax.set_ylabel("violations over evaluated runs")
    ax.set_title("Execution failures by cause")
    ax.legend(fontsize=8, frameon=False)
    ax.grid(axis="y", alpha=.25, lw=.4)
    _save(fig, "ml_failures")


def fig_training(_rows=None):
    """Loss / balanced-accuracy curves (needs ML/models/history_*.json)."""
    hs = sorted(glob.glob(os.path.join(ML, "models", "history_*.json")))
    if not hs:
        print("[skip] training: no ML/models/history_*.json yet "
              "(train.py must dump per-epoch history)")
        return
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    for h in hs:
        H = json.load(open(h)); lab = os.path.basename(h)[8:-5]
        axes[0].plot(H["val_loss"], lw=1.2, label=lab)
        if "train_loss" in H:
            axes[0].plot(H["train_loss"], lw=.8, ls=":", alpha=.7)
        axes[1].plot(H["bal_acc"], lw=1.2, label=lab)
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("loss")
    axes[0].set_title("validation loss (dotted = train)", fontsize=9)
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("balanced accuracy")
    axes[1].set_title("mean per-class recall", fontsize=9)
    for a in axes:
        a.grid(alpha=.25, lw=.4); a.legend(fontsize=7, frameon=False)
    _save(fig, "ml_training")


ACT_COLOR = {"charge": "#E69F00", "break": "#0072B2", "rest": "#661100",
             "queue": "#999999"}


def _schedule_segments(run: dict):
    """Yield (t_start, duration, kind, absorbed) activity bars for one run.

    Order follows BEHDV.advance: queue -> charge -> break -> rest.  Because
    the model credits a break taken INSIDE a charging dwell (taub is only the
    EXTRA time beyond tauc), a declared break with taub == 0 was fully
    absorbed by the charge — we mark that charge bar as hatched rather than
    drawing a zero-width break, since that absorption is the central
    coupling this problem is about.
    """
    traj, acts, durs = run["sim_trajectory"], run["actions"], run["durations_list"]
    for i, (a, d) in enumerate(zip(acts, durs)):
        t = float(traj[i]["t_arr"])
        brk, rst = a.get("break_type"), a.get("rest_type")
        for kind, dur in (("queue", d.get("tauq", 0.0)),
                          ("charge", d.get("tauc", 0.0)),
                          ("break", d.get("taub", 0.0)),
                          ("rest", d.get("taur", 0.0))):
            dur = float(dur or 0.0)
            if dur <= 1e-6:
                continue
            absorbed = (kind == "charge" and brk and float(d.get("taub", 0)) <= 1e-6)
            yield t, dur, kind, bool(absorbed), i, brk, rst
            t += dur


def fig_schedule(rows, instance: str | None = None):
    """Student vs. exact-tail teacher: the realised schedule along the route.

    Two Gantt lanes on a shared clock plus the SoC traces, so the behavioural
    difference (when each policy stops, how long it charges, whether it hides
    the break inside the charge) is visible rather than inferred from a
    duration delta.
    """
    cand = [r for r in rows if r.get("teacher") and r.get("student")]
    if not cand:
        print("[skip] schedule: need student and teacher runs for one instance")
        return
    if instance:
        cand = [r for r in cand if r["instance"] == instance] or cand
        pick = cand[0]
    else:
        # "representative" = the instance whose student/teacher gap is closest
        # to the median, so the figure is typical rather than cherry-picked.
        deltas = [(r, 100 * (r["student"] - r["teacher"]) / r["teacher"])
                  for r in cand]
        med = np.median([d for _, d in deltas])
        pick = min(deltas, key=lambda kv: abs(kv[1] - med))[0]
    inst = pick["instance"]
    stu_f = pick.get("_file") or _latest(os.path.join(SOLS, f"{inst}_*STUDENT*.json"))
    tea_f = _latest(os.path.join(REF_SOLS, f"{inst}_LA_MIPTAIL*.json"))
    runs = [("STUDENT", json.load(open(stu_f))), ("LA-MIP (teacher)",
                                                  json.load(open(tea_f)))]
    raw = json.load(open(os.path.join(ROOT, "instances", f"{inst}.json")))["instance"]
    Wha = {int(k): float(v) for k, v in (raw.get("Wha") or {}).items()}
    Whf = {int(k): float(v) for k, v in (raw.get("Whf") or {}).items()}
    Ecap = float(raw["Ecap"]); Emin = float(raw["Emin"])

    fig, axes = plt.subplots(2, 1, figsize=(9.5, 4.6), sharex=True,
                             gridspec_kw=dict(height_ratios=[1.15, 1]))
    axg, axs = axes
    tmax = max(float(r["sim_trajectory"][-1]["t_arr"]) for _, r in runs)

    # customer windows as vertical spans (shared by both policies)
    for c in sorted(set(Wha) | set(Whf)):
        a, b = Wha.get(c), Whf.get(c)
        if a is None or b is None or b > tmax * 5:
            continue
        axg.axvspan(a, min(b, tmax), color="#009E73", alpha=.10, zorder=0)

    for lane, (label, run) in enumerate(runs):
        y = 1 - lane
        for t, dur, kind, absorbed, i, brk, rst in _schedule_segments(run):
            axg.barh(y, dur, left=t, height=.42,
                     color=ACT_COLOR[kind], alpha=.95, zorder=3,
                     hatch="///" if absorbed else None,
                     edgecolor="white" if absorbed else "none", linewidth=.4)
        traj = run["sim_trajectory"]
        axs.plot([s["t_arr"] for s in traj], [s["e_arr"] / Ecap * 100 for s in traj],
                 lw=1.3, label=label,
                 color=COLOR["STUDENT"] if lane == 0 else COLOR["LA-MIP"])
        # missed windows: red ticks on the lane
        miss = [c for c in Wha
                if (traj[c]["t_arr"] > Whf.get(c, 1e9) + 1e-3
                    or traj[c]["t_arr"] < Wha.get(c, -1e9) - 1e-3)] if Wha else []
        if miss:
            axg.plot([traj[c]["t_arr"] for c in miss], [y] * len(miss),
                     "v", ms=5, color="#CC3311", zorder=5)

    axg.set_yticks([1, 0]); axg.set_yticklabels([r[0] for r in runs], fontsize=8)
    axg.set_ylim(-.55, 1.55)
    axg.set_title(f"Realised schedule — {inst}   "
                  f"(student {pick['student']:.1f} h vs teacher {pick['teacher']:.1f} h)",
                  fontsize=9)
    handles = [plt.Rectangle((0, 0), 1, 1, color=ACT_COLOR[k]) for k in ACT_COLOR]
    handles.append(plt.Rectangle((0, 0), 1, 1, facecolor=ACT_COLOR["charge"],
                                 hatch="///", edgecolor="white"))
    handles.append(plt.Line2D([], [], marker="v", ls="", color="#CC3311"))
    axg.legend(handles, list(ACT_COLOR) + ["break inside charge", "window miss"],
               ncol=6, fontsize=7, frameon=False,
               loc="upper center", bbox_to_anchor=(.5, -.02))
    axg.grid(axis="x", alpha=.25, lw=.4)

    axs.axhline(Emin / Ecap * 100, ls="--", lw=.8, c="#CC3311")
    axs.text(0, Emin / Ecap * 100 + 2, "$E_{min}$", fontsize=7, color="#CC3311")
    axs.set_ylabel("state of charge (%)"); axs.set_xlabel("time (h)")
    axs.set_xlim(0, tmax * 1.02)
    axs.legend(fontsize=8, frameon=False); axs.grid(alpha=.25, lw=.4)
    _save(fig, "ml_schedule")


def _save(fig, stem):
    os.makedirs(FIGS, exist_ok=True)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIGS, f"{stem}.{ext}"), bbox_inches="tight")
    plt.close(fig)
    print(f"wrote ML/figures/{stem}.png|pdf")


FIGURES = {"money": fig_money, "paired": fig_paired,
           "failures": fig_failures, "training": fig_training,
           "schedule": fig_schedule}


def main(argv):
    inst = method = None
    argv = list(argv)
    for flag in list(argv):
        if flag.startswith("--instance="):
            inst = flag.split("=", 1)[1]; argv.remove(flag)
        elif flag.startswith("--method="):
            method = flag.split("=", 1)[1]; argv.remove(flag)
    want = argv or ["all"]
    if want == ["all"]:
        want = list(FIGURES)
    rows = collect_runs(method)
    print(f"{len(rows)} student runs found in ML/solutions/")
    if not rows and want != ["training"]:
        print("nothing to plot yet — run ML/code/rollout.py first")
    for w in want:
        if w not in FIGURES:
            print(f"unknown figure '{w}'; choose from {list(FIGURES)} or 'all'")
            continue
        try:
            FIGURES[w](rows, inst) if w == "schedule" else FIGURES[w](rows)
        except Exception as e:
            print(f"[skip] {w}: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main(sys.argv[1:])
