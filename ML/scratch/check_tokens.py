"""After adding features.py section J: old columns must be bit-identical, the
new token columns present and sensible.  Run after
    python ML/code/extract.py --limit 15 --out _check_tokens.npz"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code"))
DATA = os.path.join(os.path.dirname(__file__), "..", "data")

a = np.load(os.path.join(DATA, "_check_tokens.npz"), allow_pickle=True)
b = np.load(os.path.join(DATA, "dataset.npz"), allow_pickle=True)
na, nb = [str(x) for x in a["feature_names"]], [str(x) for x in b["feature_names"]]
n = len(a["stop"])
print("rows", n, "| columns new", len(na), "old", len(nb), "| added",
      len([c for c in na if c not in nb]), "| missing", [c for c in nb if c not in na])
bad = [c for c in nb if not np.array_equal(a["X"][:, na.index(c)], b["X"][:n, nb.index(c)],
                                            equal_nan=True)]
print("old columns identical:", "ALL" if not bad else f"NO: {bad[:8]}")
for k in ("action_ix", "regret", "cost", "clean", "chosen", "stop"):
    assert np.array_equal(a[k], b[k][:n], equal_nan=True), k
print("labels identical")

from fsets import resolve                                         # noqa: E402
for sid in ("F", "P", "T", "D", "L", "R"):
    names, ns = resolve(sid, na, int(a["n_state"]), "mlp")
    print(f"set {sid}: {len(names)} ({ns} state)")

X = a["X"]
for t in range(3):
    p = f"tok{t}_"
    cols = [p + s for s in ("exists", "drive", "soc_frac", "queue", "reach", "kw", "tfull")]
    v = X[:, [na.index(c) for c in cols]]
    ex = v[:, 0] > 0.5
    print(f"{p}: exists {ex.mean() * 100:5.1f}%  kw {np.unique(v[ex, 5])[:4]}  "
          f"tfull h median {np.median(v[ex, 6]):.2f} [{v[ex, 6].min():.2f}, {v[ex, 6].max():.2f}]  "
          f"drive h median {np.median(v[ex, 1]):.2f}")
