"""Rough word count of a LaTeX body (abstract to credits), for the page budget."""
import re
import sys

t = open(sys.argv[1], encoding="utf-8").read()
a = t.find("\\begin{abstract}")
b = t.find("\\begin{credits}")
body = t[a:b]
body = re.sub(r"(?m)%.*$", "", body)
body = re.sub(r"\\(begin|end)\{[^}]*\}", " ", body)
body = re.sub(r"\\[a-zA-Z]+\*?", " ", body)
body = re.sub(r"[{}$\\^_&~]", " ", body)
print(sys.argv[1], "words (approx.):", len(body.split()))
