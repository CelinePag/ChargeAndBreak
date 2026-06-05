"""
compile_solutions.py — Compile solution JSON files into an Excel summary
=========================================================================
Reads all *.json files in a solutions directory (default: "solutions/")
and writes a formatted Excel workbook with:

  Sheet "Results"  — one row per solution file, columns:
    run_id, instance, method, algorithm, n_scenarios, horizon_hours,
    delta, criterion, solve_mode,
    sim_arrival_h, duration_h, wall_clock_s,
    oracle_obj, oracle_gap_pct, oracle_feasible, oracle_optimal, oracle_status

  Sheet "Summary"  — pivot-style aggregates per (instance, method):
    mean / min / max of sim_arrival_h and oracle_obj

Usage
-----
  python compile_solutions.py                        # reads ./solutions/
  python compile_solutions.py --dir path/to/sols     # custom directory
  python compile_solutions.py --out results.xlsx     # custom output name
  python compile_solutions.py --dir solutions --out results.xlsx
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

import openpyxl
from openpyxl.styles import (
    Alignment, Font, PatternFill, Border, Side,
    numbers as xl_numbers,
)
from openpyxl.utils import get_column_letter

# ── colour palette ──────────────────────────────────────────────────────────
_HEADER_BG   = "1F4E79"   # dark blue
_HEADER_FG   = "FFFFFF"   # white
_ALT_ROW_BG  = "D6E4F0"   # light blue
_SUMMARY_BG  = "E2EFDA"   # light green
_BORDER_COL  = "B8CCE4"

_THIN = Side(style="thin", color=_BORDER_COL)
_BORDER = Border(left=_THIN, right=_THIN, top=_THIN, bottom=_THIN)

# ── column specification for "Results" sheet ────────────────────────────────
# (header, json_path_as_tuple, number_format, width)
_COLS = [
    ("run_id",          ("run_id",),                   "@",        38),
    ("instance",        ("instance",),                 "@",        28),
    ("method",          ("method",),                   "@",        14),
    ("n_scenarios",     ("n_scenarios",),               "0",        13),
    ("horizon_h",       ("horizon_hours",),             "0.0",      11),
    ("delta",           ("delta",),                    "0.00",     8),
    ("criterion",       ("criterion",),                "@",        11),
    ("solve_mode",      ("solve_mode",),               "@",        12),
    ("sim_arrival_h",   ("sim_arrival_h",),            "0.000",    14),
    ("duration_h",      ("duration_h",),               "0.000",    13),
    ("wall_clock_s",    ("wall_clock_s",),             "0.0",      13),
    ("oracle_obj",      ("oracle", "obj"),              "0.000",    13),
    ("oracle_gap_%",    ("oracle", "gap"),              "0.00%",    13),
    ("oracle_feasible", ("oracle", "feasible"),        "@",        14),
    ("oracle_optimal",  ("oracle", "optimal"),         "@",        14),
    ("oracle_status",   ("oracle", "status"),          "@",        16),
]


def _get(d: dict, path: tuple):
    """Safely walk a nested dict using a key-path tuple."""
    val = d
    for key in path:
        if not isinstance(val, dict):
            return None
        val = val.get(key)
    return val


def _safe_float(v):
    """Return float or None; treat inf/nan as None."""
    try:
        f = float(v)
        return None if (math.isnan(f) or math.isinf(f)) else f
    except (TypeError, ValueError):
        return None


def load_solutions(solutions_dir: str) -> list[dict]:
    """Load all *.json solution files from solutions_dir."""
    if not os.path.isdir(solutions_dir):
        print(f"  ERROR: directory not found: '{solutions_dir}'", file=sys.stderr)
        sys.exit(1)

    paths = sorted(
        os.path.join(solutions_dir, f)
        for f in os.listdir(solutions_dir)
        if f.endswith(".json")
    )
    if not paths:
        print(f"  WARNING: no .json files found in '{solutions_dir}'")
        return []

    rows = []
    for p in paths:
        try:
            with open(p, encoding="utf-8") as fh:
                data = json.load(fh)
            data["_file"] = os.path.basename(p)
            rows.append(data)
        except Exception as e:
            print(f"  SKIP {p}: {e}", file=sys.stderr)

    print(f"  Loaded {len(rows)} solution file(s) from '{solutions_dir}/'")
    return rows


def _style_header(cell, bg=_HEADER_BG, fg=_HEADER_FG):
    cell.font      = Font(bold=True, color=fg, name="Arial", size=10)
    cell.fill      = PatternFill("solid", start_color=bg)
    cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    cell.border    = _BORDER


def _style_data(cell, row_idx: int, num_fmt: str):
    bg = _ALT_ROW_BG if row_idx % 2 == 0 else "FFFFFF"
    cell.fill      = PatternFill("solid", start_color=bg)
    cell.font      = Font(name="Arial", size=10)
    cell.alignment = Alignment(horizontal="center", vertical="center")
    cell.border    = _BORDER
    cell.number_format = num_fmt


def build_results_sheet(ws, rows: list[dict]):
    ws.title = "Results"
    ws.freeze_panes = "A2"

    # header row
    for col_idx, (header, _, _, width) in enumerate(_COLS, start=1):
        cell = ws.cell(row=1, column=col_idx, value=header)
        _style_header(cell)
        ws.column_dimensions[get_column_letter(col_idx)].width = width
    ws.row_dimensions[1].height = 30

    # data rows
    for row_idx, rec in enumerate(rows, start=2):
        for col_idx, (_, path, num_fmt, _) in enumerate(_COLS, start=1):
            raw = _get(rec, path)

            # coerce type
            if num_fmt in ("0", "0.0", "0.00", "0.000", "0.00%"):
                val = _safe_float(raw)
                if num_fmt == "0.00%" and val is not None:
                    # openpyxl stores percentages as fractions
                    val = val  # gap is already a fraction (0.0..1.0) from oracle
            elif num_fmt == "@":
                val = str(raw) if raw is not None else ""
            else:
                val = raw

            cell = ws.cell(row=row_idx, column=col_idx, value=val)
            _style_data(cell, row_idx, num_fmt)

    # auto-filter
    ws.auto_filter.ref = (
        f"A1:{get_column_letter(len(_COLS))}{len(rows) + 1}"
    )


def build_summary_sheet(ws, rows: list[dict]):
    ws.title = "Summary"
    ws.freeze_panes = "C2"

    # aggregate: key = (instance, method)
    from collections import defaultdict
    groups: dict[tuple, list] = defaultdict(list)
    for rec in rows:
        key = (
            rec.get("instance", "?"),
            rec.get("method", "?"),
        )
        sim_h   = _safe_float(_get(rec, ("sim_arrival_h",)))
        ora_obj = _safe_float(_get(rec, ("oracle", "obj")))
        groups[key].append((sim_h, ora_obj))

    headers = [
        "instance", "method",
        "n_runs",
        "sim_arrival_h mean", "sim_arrival_h min", "sim_arrival_h max",
        "oracle_obj mean",    "oracle_obj min",    "oracle_obj max",
        "gap_to_oracle mean",
    ]
    col_widths = [28, 14, 8, 18, 18, 18, 16, 16, 16, 20]
    num_fmts   = ["@", "@", "0",
                  "0.000", "0.000", "0.000",
                  "0.000", "0.000", "0.000",
                  "0.00%"]

    for ci, (h, w) in enumerate(zip(headers, col_widths), start=1):
        cell = ws.cell(row=1, column=ci, value=h)
        _style_header(cell, bg="375623", fg="FFFFFF")
        ws.column_dimensions[get_column_letter(ci)].width = w
    ws.row_dimensions[1].height = 30

    for ri, ((instance, method), vals) in enumerate(
            sorted(groups.items()), start=2):
        sims  = [v[0] for v in vals if v[0] is not None]
        orajs = [v[1] for v in vals if v[1] is not None]

        def _mean(lst): return sum(lst) / len(lst) if lst else None
        def _min(lst):  return min(lst) if lst else None
        def _max(lst):  return max(lst) if lst else None

        sim_mean  = _mean(sims)
        ora_mean  = _mean(orajs)
        gap_mean  = ((sim_mean - ora_mean) / ora_mean
                     if (sim_mean and ora_mean and ora_mean != 0) else None)

        data_row = [
            instance, method, len(vals),
            sim_mean, _min(sims), _max(sims),
            ora_mean, _min(orajs), _max(orajs),
            gap_mean,
        ]
        bg = _SUMMARY_BG if ri % 2 == 0 else "FFFFFF"
        for ci, (val, fmt) in enumerate(zip(data_row, num_fmts), start=1):
            cell = ws.cell(row=ri, column=ci, value=val)
            cell.fill      = PatternFill("solid", start_color=bg)
            cell.font      = Font(name="Arial", size=10)
            cell.alignment = Alignment(horizontal="center", vertical="center")
            cell.border    = _BORDER
            cell.number_format = fmt


def compile_to_excel(solutions_dir: str, output_path: str):
    rows = load_solutions(solutions_dir)

    wb = openpyxl.Workbook()
    ws_results = wb.active
    build_results_sheet(ws_results, rows)

    ws_summary = wb.create_sheet()
    build_summary_sheet(ws_summary, rows)

    wb.save(output_path)
    print(f"  Excel saved : {output_path}")
    print(f"  Sheets      : Results ({len(rows)} rows), Summary")


# ── CLI ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compile solution JSON files into an Excel summary."
    )
    parser.add_argument(
        "--dir", default="solutions",
        help="Directory containing solution .json files (default: 'solutions')"
    )
    parser.add_argument(
        "--out", default="solution_summary.xlsx",
        help="Output Excel file path (default: 'solution_summary.xlsx')"
    )
    args = parser.parse_args()
    compile_to_excel(args.dir, args.out)