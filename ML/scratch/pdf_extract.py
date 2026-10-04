"""Extract the text of selected example papers (page ranges from the TOC)."""
import os

import fitz

D = os.path.join(os.path.dirname(__file__), "..", "paper", "examples")
OUT = os.path.join(D, "txt")
os.makedirs(OUT, exist_ok=True)
PICK = [("978-3-030-58942-4.pdf", 126, 138, "2020_learning_primal_stochastic_ip"),
        ("978-3-032-27242-3.pdf", 84, 94, "new_branching_rules_minlp"),
        ("978-3-032-27242-3.pdf", 95, 113, "new_imitation_train_rescheduling"),
        ("978-3-032-27242-3.pdf", 204, 219, "new_stochastic_tasks_prediction")]
for f, a, b, name in PICK:
    doc = fitz.open(os.path.join(D, f))
    text = "\n".join(doc[p - 1].get_text() for p in range(a, b + 1))
    with open(os.path.join(OUT, name + ".txt"), "w", encoding="utf-8") as fh:
        fh.write(text)
    print(name, len(text.split()), "words")
