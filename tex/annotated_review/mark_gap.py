"""Highlight low-n cells in the generated gap table, keeping its layout intact.

Reads tex/tables/gap.tex and data_output/paper_gap_stats.csv, wraps every cell
whose sample count is below 45 in \\prov{} (yellow) and below 20 in \\provlow{}
(yellow + red bold).  Writes gap_marked.tex.
"""
import csv
import re

BASE = r'C:\Users\celinep\Documents\GitHub\ChargeAndBreak'
rows = list(csv.DictReader(open(BASE + r'\data_output\paper_gap_stats.csv', encoding='utf-8')))
N = {(r['route_class'], r['customers_class'], r['window_class'], r['method']):
     int(r['n_gap_samples']) for r in rows}

RO = ['short', 'medium', 'long']
CO = ['few', 'medium', 'many']
TO = ['none', 'tight', 'medium', 'large']
M = ['greedy', 'RO', 'ROBU', 'LA', '2SP']

src = open('gap.tex', encoding='utf-8').read().replace('\r\n', '\n')
lines = src.split('\n')
out = []
k = 0
keys = [(r, c, t) for r in RO for c in CO for t in TO]

for line in lines:
    parts = line.split('&')
    is_data = (len(parts) == 9 and line.rstrip().endswith(r'\\')
               and re.search(r'\{\\small (None|Tight|Medium|Large)\}', parts[2]))
    if not is_data:
        out.append(line)
        continue
    r, c, t = keys[k]
    k += 1
    head, cells = parts[:4], parts[4:]
    new = []
    for j, cell in enumerate(cells):
        body = cell.replace(r'\\', '').strip()
        n = N.get((r, c, t, M[j]), 0)
        if body == '--' or n == 0 or n >= 45:
            new.append(cell)
            continue
        wrap = r'\provlow{%s}' if n < 20 else r'\prov{%s}'
        rep = ' ' + (wrap % body) + ' '
        new.append(rep + r'\\' if cell.rstrip().endswith(r'\\') else rep)
    out.append('&'.join(head + new))

assert k == 36, f'expected 36 data rows, matched {k}'
open('gap_marked.tex', 'w', encoding='utf-8', newline='\n').write('\n'.join(out))

flagged = sum(1 for key, n in N.items() if 0 < n < 45)
print(f'gap_marked.tex written; {k} rows; {flagged} cells flagged '
      f'({sum(1 for _, n in N.items() if 0 < n < 20)} below n=20)')
