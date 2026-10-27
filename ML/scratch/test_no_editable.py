"""
test_no_editable.py — import an ML/code module the way the cluster sees it
==========================================================================
The laptop's .venv has the project installed in editable mode (a meta-path
finder), which makes `src` importable from anywhere; the cluster has no such
install, so a script must put the repository root on sys.path itself
(rl_search failed there on 2026-10-27, as torch_train did on 2026-10-02).
This removes the editable finder and the repository root, checks that `src`
is then NOT importable, and imports the module given.  Run one module per
process:

    python ML/scratch/test_no_editable.py rl_search
"""
from __future__ import annotations

import importlib
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.abspath(os.path.join(HERE, "..", "code"))
ROOT = os.path.normcase(os.path.abspath(os.path.join(HERE, "..", "..")))

sys.meta_path = [f for f in sys.meta_path
                 if "editable" not in getattr(type(f), "__module__", "").lower()
                 and "editable" not in repr(f).lower()]
sys.path = [p for p in sys.path
            if os.path.normcase(os.path.abspath(p or os.getcwd())) != ROOT]
try:
    import src  # noqa: F401
    raise SystemExit("src is still importable: the test would prove nothing")
except ModuleNotFoundError:
    pass
sys.path.insert(0, CODE)
name = sys.argv[1]
importlib.import_module(name)
print(f"{name}: imports without the editable install")
