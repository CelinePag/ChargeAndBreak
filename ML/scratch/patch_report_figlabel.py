import io
p = "src/output_analysis/greedy_corrected_report.py"
s = io.open(p, encoding="utf-8").read()
reps = [
('LABEL  = "Greedy (corrected)"\n',
 'LABEL  = "Greedy (corrected)"   # table row label (Table 9 shows both versions)\n'
 'FIG_LABEL = "Greedy"           # figures: plain label, no mention of the variant\n'),
('    ps.METHOD_LBL["greedy"] = LABEL\n    try:\n        gaps,',
 '    ps.METHOD_LBL["greedy"] = FIG_LABEL\n    try:\n        gaps,'),
('''        fig.text(0.995, 0.004, f"Greedy = corrected greedy (variant {VAR}, "
                 "MILP execution rules)", ha="right", va="bottom",
                 fontsize=6, color="#555555")
''', ''),
('    ps.METHOD_LBL["greedy"] = LABEL\n    try:\n        af._paths',
 '    ps.METHOD_LBL["greedy"] = FIG_LABEL\n    try:\n        af._paths'),
]
for a, b in reps:
    assert s.count(a) == 1, a[:60]
    s = s.replace(a, b)
io.open(p, "w", encoding="utf-8").write(s)
print("ok")
