"""Tables of contents of the example proceedings (PyMuPDF, already installed)."""
import os
import sys

import fitz

D = os.path.join(os.path.dirname(__file__), "..", "paper", "examples")
for f in sorted(os.listdir(D)):
    if not f.endswith(".pdf"):
        continue
    doc = fitz.open(os.path.join(D, f))
    print(f"\n=== {f}: {doc.page_count} pages, title: {doc.metadata.get('title')}")
    toc = doc.get_toc(simple=True)
    for lvl, title, page in toc:
        if lvl <= 2:
            print(f"{'  ' * (lvl - 1)}{page:5d}  {title[:110]}")
    sys.stdout.flush()
