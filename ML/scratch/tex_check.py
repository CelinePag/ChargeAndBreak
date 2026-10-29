"""Checks for ML/paper/main.tex without LaTeX: missing / unused / duplicate
citation keys, refs without labels, and a rough word count of the body
(comments and tabulars removed) and per section.
    python ML/scratch/tex_check.py [tex file]"""
import os, re, sys
HERE = os.path.dirname(os.path.abspath(__file__))
P = os.path.join(HERE, "..", "paper")
tex = open(sys.argv[1] if len(sys.argv) > 1 else os.path.join(P, "main.tex"), encoding="utf-8").read()
bib = open(os.path.join(P, "lib.bib"), encoding="utf-8").read()
BS = chr(92)
keys = re.findall(r"@\w+\{([^,]+),", bib)
cited = set(k.strip() for c in re.findall(re.escape(BS + "cite") + r"\{([^}]+)\}", tex) for k in c.split(","))
print("missing:", sorted(cited - set(keys)))
print("unused:", sorted(set(keys) - cited))
print("duplicate keys:", sorted({k for k in keys if keys.count(k) > 1}))
labels = set(re.findall(re.escape(BS + "label") + r"\{([^}]+)\}", tex))
refs = set(re.findall(re.escape(BS + "ref") + r"\{([^}]+)\}", tex))
print("refs without label:", sorted(refs - labels))


def words(t):
    t = "\n".join(l.split("%")[0] if not l.strip().startswith("%") else "" for l in t.splitlines())
    t = re.sub(re.escape(BS + "begin{tabular}") + ".*?" + re.escape(BS + "end{tabular}"), " ", t, flags=re.S)
    t = re.sub(re.escape(BS) + r"[a-zA-Z]+", " ", t)
    return len(re.findall(r"[A-Za-z0-9][A-Za-z0-9.,%'-]*", t))


body = tex.split(BS + "begin{document}")[1].split(BS + "bibliographystyle")[0]
print("body words:", words(body))
parts = re.split(re.escape(BS) + r"(section|subsection)\{", body)
print("  front (title/abstract):", words(parts[0]))
for i in range(1, len(parts), 2):
    title = parts[i + 1].split("}")[0]
    print(f"  {parts[i]:10s} {title[:45]:45s} {words(parts[i + 1])}")
