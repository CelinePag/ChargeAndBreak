"""Static checks of ML/paper/main.tex (no LaTeX on this machine): braces,
environments, labels vs refs, cite keys vs references.bib, word count."""
import os
import re

P = os.path.join(os.path.dirname(__file__), "..", "paper")
tex = open(os.path.join(P, "main.tex"), encoding="utf-8").read()
body = re.sub(r"(?<!\\)%.*", "", tex)                      # drop comments

depth, line = 0, 1
for ch in body:
    if ch == "\n":
        line += 1
    elif ch == "{":
        depth += 1
    elif ch == "}":
        depth -= 1
        if depth < 0:
            print("unbalanced } near line", line)
            depth = 0
print("brace depth at end:", depth)

stack = []
for m in re.finditer(r"\\(begin|end)\{([^}]*)\}", body):
    if m.group(1) == "begin":
        stack.append(m.group(2))
    elif not stack or stack.pop() != m.group(2):
        print("environment mismatch at", m.group(0))
print("open environments at end:", stack)

labels = set(re.findall(r"\\label\{([^}]*)\}", body))
refs = set(re.findall(r"\\ref\{([^}]*)\}", body))
print("refs without label:", refs - labels, "| labels never referenced:", labels - refs)

bib = open(os.path.join(P, "references.bib"), encoding="utf-8").read()
keys = set(re.findall(r"@\w+\{([^,]+),", bib))
cites = set(k.strip() for grp in re.findall(r"\\cite\{([^}]*)\}", body) for k in grp.split(","))
print("cited but missing from bib:", cites - keys, "| in bib, never cited:", keys - cites)

text = re.sub(r"\\begin\{tabular\}.*?\\end\{tabular\}", " ", body, flags=re.S)
text = re.sub(r"\\[a-zA-Z]+\*?(\[[^]]*\])?", " ", text)
print("approx. words (text, no tables):", len(re.findall(r"[A-Za-z][A-Za-z'-]+", text)))
