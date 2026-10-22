"""
greedy_corrected_summary.py — one-document summary of the corrected-greedy results.

Reads the outputs of src/output_analysis/greedy_corrected_report.py (new files
with the suffix _greedyC) next to the existing ones, and writes
review/TRD_R1/greedy_corrected_results.html; print it to PDF with Edge.

    python -m review.TRD_R1.greedy_corrected_summary
"""
from __future__ import annotations

import csv
import os
import statistics as st
from collections import defaultdict

from src import paths as P
from src.output_analysis import greedy_corrected_report as gr

HERE = os.path.dirname(os.path.abspath(__file__))
FIG_B = P.figures(P.BASECASE)
FIG_S = P.figures(P.SENSITIVITY)


def _csv(name):
    path = P.data_output(name)
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _f(x, nd=1, sign=False):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return "–"
    if x != x:
        return "–"
    return f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"


def _img(path, width="100%"):
    url = "file:///" + path.replace("\\", "/")
    return f'<img src="{url}" style="width:{width}">'


def table8() -> str:
    """Gap to oracle (%) by route x customers: other methods + both greedy versions."""
    new = gr._base_rows(swap=True)
    old = [r for r in gr._base_rows(swap=False) if r.get("method") == "greedy"]
    cell = defaultdict(list)
    for r, tag in [(r, r.get("method")) for r in new] + [(r, "greedy_old") for r in old]:
        if r.get("outcome") == "feasible" and r.get("gap_pen") is not None:
            cell[(r["route_class"], r["customers_class"], tag)].append(100 * r["gap_pen"])
    cols = [("RO", "RO"), ("2SP", "2SP"), ("greedy_old", "Greedy (submitted)"),
            ("greedy", "Greedy (corrected)"), ("LA", "LA")]
    h = ["<table><tr><th>Route</th><th>Cust.</th>"
         + "".join(f"<th class=n>{l}</th>" for _, l in cols) + "<th class=n>Δ Greedy</th></tr>"]
    for rc in gr.ROUTES:
        for cc in ("few", "medium", "many"):
            v = {m: (st.mean(cell[(rc, cc, m)]) if cell[(rc, cc, m)] else None) for m, _ in cols}
            d = (v["greedy"] - v["greedy_old"]) if (v["greedy"] is not None and v["greedy_old"] is not None) else None
            h.append(f"<tr><td>{rc.capitalize() if cc == 'few' else ''}</td><td>{cc.capitalize()}</td>"
                     + "".join(f"<td class=n>{_f(v[m])}</td>" for m, _ in cols)
                     + f"<td class=n>{_f(d, 2, True)}</td></tr>")
    h.append("</table>")
    return "\n".join(h)


def table9() -> str:
    path = P.tex_tables(f"feasibility{gr.SUFFIX}.tex")
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip().endswith(r"\\") and "&" in line and "textbf{Method}" not in line:
                cells = [c.strip().replace(r"\textit{", "").replace("}", "")
                         for c in line.strip()[:-2].split("&")]
                rows.append(cells)
    hdr = ["Method", "Infeasible (%)", "95% CI", "HoS", "Reserve", "TW hit (%)",
           "TW hit, windowed (%)", "Not opt. (%)", "Opt. gap (%)"]
    h = ["<table><tr>" + "".join(f"<th>{x}</th>" for x in hdr) + "</tr>"]
    for c in rows:
        h.append("<tr>" + "".join(f"<td>{x.replace('[', '').replace(']', '')}</td>" for x in c) + "</tr>")
    h.append("</table>")
    return "\n".join(h)


def compare_table() -> str:
    rows = _csv("greedy_corrected_comparison.csv")
    h = ["<table><tr><th>Route</th><th class=n>Infeasible old / new (%)</th>"
         "<th class=n>Gap old / new (%)</th><th class=n>TW hit, windowed old / new (%)</th>"
         "<th class=n>Directive ok old / new (%)</th><th class=n>Δ duration, paired (%)</th>"
         "<th class=n>Sequential stops per run, new (rest / break)</th></tr>"]
    for r in rows:
        h.append(f"<tr><td>{r['route'].capitalize()}</td>"
                 f"<td class=n>{_f(r['infeas_old'])} / {_f(r['infeas_new'])}</td>"
                 f"<td class=n>{_f(r['gap_old'],2)} / {_f(r['gap_new'],2)}</td>"
                 f"<td class=n>{_f(r['tw_hit_old'])} / {_f(r['tw_hit_new'])}</td>"
                 f"<td class=n>{_f(r['directive_old'])} / {_f(r['directive_new'])}</td>"
                 f"<td class=n>{_f(r['dur_change_mean'],2,True)} (n={r['n_paired']})</td>"
                 f"<td class=n>{_f(r['seq_rest_per_run_new'],2)} / {_f(r['seq_break_per_run_new'],2)}</td></tr>")
    h.append("</table>")
    return "\n".join(h)


def sens_table() -> str:
    old = {r["tag"]: r for r in _csv("additional_sens_stats.csv")}
    new = {r["tag"]: r for r in _csv(f"additional_sens_stats{gr.SUFFIX}.csv")}
    h = ["<table><tr><th>Axis level</th><th class=n>Oracle</th><th class=n>LA</th>"
         "<th class=n>Greedy submitted (n)</th><th class=n>Greedy corrected (n)</th></tr>"]
    for tag, o in old.items():
        n = new.get(tag, {})
        h.append(f"<tr><td>{o['axis']}</td><td class=n>{_f(o['oracle_delta_%'],2,True)}</td>"
                 f"<td class=n>{_f(n.get('la_delta_%', o['la_delta_%']),2,True)}</td>"
                 f"<td class=n>{_f(o['greedy_delta_%'],2,True)} ({o['n_greedy']})</td>"
                 f"<td class=n>{_f(n.get('greedy_delta_%'),2,True)} ({n.get('n_greedy','–')})</td></tr>")
    h.append("</table>")
    return "\n".join(h)


def grid_table() -> str:
    old = {(r["route_class"], r["battery_kwh"], r["charger_kw"]): r for r in _csv("additional_grid_stats.csv")}
    new = {(r["route_class"], r["battery_kwh"], r["charger_kw"]): r
           for r in _csv(f"additional_grid_stats{gr.SUFFIX}.csv")}
    out = []
    for route in ("short", "medium"):
        out.append(f"<h4>{route.capitalize()} routes — change in Greedy route duration vs base (%), "
                   f"submitted → corrected</h4>")
        h = ["<table><tr><th>Battery \\ power</th>"
             + "".join(f"<th class=n>{p} kW</th>" for p in ("150", "350", "700", "1000")) + "</tr>"]
        for b in ("900", "700", "500", "300"):
            tds = []
            for p in ("150", "350", "700", "1000"):
                o = old.get((route, b, p), {})
                n = new.get((route, b, p), {})
                tds.append(f"<td class=n>{_f(o.get('greedy_duration_vs_base_%'),1,True)} → "
                           f"<b>{_f(n.get('greedy_duration_vs_base_%'),1,True)}</b></td>")
            h.append(f"<tr><td>{b} kWh</td>{''.join(tds)}</tr>")
        h.append("</table>")
        out.append("\n".join(h))
    return "\n".join(out)


def diesel_table() -> str:
    # The CSV labels both "700 kW" and "700 kWh" as "700", so rows are grouped
    # by their ORDER (oracle, LA, greedy per level, levels in figure order).
    old = [r for r in _csv("additional_diesel_oat.csv") if r["route"] == "all"]
    new = [r for r in _csv(f"additional_diesel_oat{gr.SUFFIX}.csv") if r["route"] == "all"]
    names = ["Base", "150 kW", "700 kW", "1000 kW", "300 kWh", "700 kWh", "900 kWh"]
    groups_o, groups_n = defaultdict(dict), defaultdict(dict)
    for rows, groups in ((old, groups_o), (new, groups_n)):
        lvl_idx, prev = -1, None
        for r in rows:
            if r["method"] == "oracle" or r["level"] != prev:
                if r["method"] == "oracle":
                    lvl_idx += 1
            prev = r["level"]
            groups[lvl_idx][r["method"]] = r
    h = ["<table><tr><th>Level</th><th class=n>Oracle</th><th class=n>LA</th>"
         "<th class=n>Greedy submitted (n)</th><th class=n>Greedy corrected (n)</th></tr>"]
    for i in sorted(groups_o):
        lvl = names[i] if i < len(names) else groups_o[i]["oracle"]["level"]
        o_or, o_la, o_g = (groups_o[i].get("oracle", {}), groups_o[i].get("LA", {}),
                           groups_o[i].get("greedy", {}))
        n_g = groups_n.get(i, {}).get("greedy", {})
        h.append(f"<tr><td>{lvl}</td><td class=n>{_f(o_or.get('total_pct'))}</td>"
                 f"<td class=n>{_f(o_la.get('total_pct'))}</td>"
                 f"<td class=n>{_f(o_g.get('total_pct'))} ({o_g.get('n','–')})</td>"
                 f"<td class=n><b>{_f(n_g.get('total_pct'))}</b> ({n_g.get('n','–')})</td></tr>")
    h.append("</table>")
    return "\n".join(h)


CSS = """
@page { size: A4; margin: 15mm 14mm 16mm 14mm; }
*{ -webkit-print-color-adjust: exact; print-color-adjust: exact; }
body{ font-family: Cambria, Georgia, serif; font-size:9.8pt; line-height:1.36; color:#1b1f23; background:#fff; max-width:182mm; margin:0 auto; }
h1,h2,h3,h4{ font-family:"Segoe UI",Calibri,Arial,sans-serif; color:#14324f; }
h1{ font-size:16pt; margin:0 0 4px; } h2{ font-size:12pt; border-bottom:1.5px solid #14324f; padding-bottom:2px; margin:16px 0 6px; page-break-after:avoid; }
h4{ font-size:9.6pt; margin:8px 0 3px; }
table{ border-collapse:collapse; width:100%; font-size:8.6pt; margin:4px 0 8px; }
th,td{ border:1px solid #d5d9de; padding:3px 5px; text-align:left; }
th{ background:#e8ecf1; font-family:"Segoe UI",Calibri,Arial,sans-serif; font-weight:600; }
td.n{ text-align:right; white-space:nowrap; } th.n{ text-align:right; white-space:normal; }
.box{ background:#edf3fb; border-left:3px solid #2b6cb0; padding:5px 9px; margin:6px 0; }
.note{ background:#fdf2e7; border-left:3px solid #c05621; padding:5px 9px; margin:6px 0; font-size:9.2pt; }
.pb{ page-break-before:always; } .small{ font-size:8.5pt; color:#5b6470; }
figure{ margin:6px 0 10px; page-break-inside:avoid; } figcaption{ font-size:8.6pt; color:#3a4048; }
"""


def main():
    parts = [f"<!doctype html><html><head><meta charset='utf-8'><title>Corrected greedy results</title>"
             f"<style>{CSS}</style></head><body>"]
    parts.append("<h1>Corrected Greedy — new results for the paper’s tables and figures</h1>"
                 "<div class=small>Generated from stored runs by "
                 "<code>src/output_analysis/greedy_corrected_report.py</code> and "
                 "<code>review/TRD_R1/greedy_corrected_summary.py</code>. The submitted Greedy runs, tables "
                 "and figures are unchanged; every new file carries the suffix <code>_greedyC</code>.</div>")
    parts.append("""<h2>What the correction changes</h2><div class=box>
<p>The corrected Greedy (<code>src/methods/greedy_corrected.py</code>, runs labelled <code>GREEDY_CORR</code>)
keeps every decision rule of the submitted Greedy — stop only when forced, charge to full when stopping,
95%-quantile one-step guard, 10% battery buffer, queue avoidance — but executes each stop exactly as the MILP does:</p>
<ul>
<li><b>C1</b> A charge before a daily rest is sequential (Eq. 14): the 5-min repositioning is paid and the charge is working time.</li>
<li><b>C2</b> A charge is credited as break only if it lasts at least the break (Eq. 15); otherwise the full break follows the charge, with the repositioning. The truck no longer idles at the bay.</li>
<li><b>C3</b> No charge shorter than the 15-min minimum (Eq. 9).</li>
<li><b>D1</b> At a mandatory break, a charge is added only if it is free (covers the break) or forced by the energy state.</li>
<li><b>D2</b> The spread rules use the exact dwell the simulator executes (incl. repositioning and access time).</li>
</ul></div>""")
    parts.append("<h2>Baseline vs corrected Greedy, base case (900 instances)</h2>" + compare_table())
    parts.append("<h2>Table 8 — gap to oracle (%), window penalties included</h2>" + table8()
                 + "<div class=small>Means over each method’s feasible runs, time-window classes pooled. "
                   "All non-Greedy columns are identical to the current Table 8.</div>")
    parts.append("<h2>Table 9 — feasibility (base case only)</h2>" + table9()
                 + "<div class=small>Base-case runs only; the submitted Table 9 pooled the sensitivity-axis "
                   "runs into its Greedy row (8.9%).</div>")
    parts.append("<h2 class=pb>Figure 6 — corrected Greedy</h2><figure>"
                 + _img(os.path.join(FIG_B, f"paper_gap_box_pooledtw{gr.SUFFIX}.png"))
                 + "<figcaption>Gap to oracle by route class and customer count, time-window classes pooled; "
                   "Greedy = corrected Greedy.</figcaption></figure>")
    parts.append("<h2>Figure 9 — one-at-a-time sensitivity</h2>" + sens_table() + "<figure>"
                 + _img(os.path.join(FIG_S, f"additional_sens_effects{gr.SUFFIX}.png"))
                 + "<figcaption>Paired change in mean route duration vs the base case; Greedy bars = corrected "
                   "Greedy. Oracle unchanged. LA unchanged except the 30-km bar, which now uses the 30-km LA "
                   "runs completed on 2 Oct 2026 (n = 209; the existing figure, made on 27 Aug, had 16 pairs)."
                   "</figcaption></figure>")
    parts.append("<h2 class=pb>Figure 10 — battery × power grid</h2>" + grid_table() + "<figure>"
                 + _img(os.path.join(FIG_S, f"additional_grid_battery_power{gr.SUFFIX}.png"), "88%")
                 + "<figcaption>Corrected Greedy route duration against the base case.</figcaption></figure>")
    parts.append("<h2 class=pb>Figure 11 — electrification penalty vs diesel (%)</h2>" + diesel_table()
                 + "<figure>" + _img(os.path.join(FIG_S, f"additional_diesel_oat{gr.SUFFIX}.png"))
                 + "<figcaption>Each EV method against its diesel counterpart; Greedy = corrected Greedy on both "
                   "legs.</figcaption></figure>")
    parts.append("</body></html>")
    out = os.path.join(HERE, "greedy_corrected_results.html")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))
    print(out)


if __name__ == "__main__":
    main()
