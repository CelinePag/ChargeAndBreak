"""Pairwise OKLab distance (x100) under normal vision and simulated protan /
deutan / tritan (Vienot 1999 matrices in linear sRGB), plus contrast on white."""
import itertools
import sys

import numpy as np


def lin(c):
    c = np.array([int(c[i:i + 2], 16) / 255 for i in (1, 3, 5)])
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def oklab(rgb):
    M1 = np.array([[0.4122214708, 0.5363325363, 0.0514459929],
                   [0.2119034982, 0.6806995451, 0.1073969566],
                   [0.0883024619, 0.2817188376, 0.6299787005]])
    M2 = np.array([[0.2104542553, 0.7936177850, -0.0040720468],
                   [1.9779984951, -2.4285922050, 0.4505937099],
                   [0.0259040371, 0.7827717662, -0.8086757660]])
    return M2 @ np.cbrt(M1 @ rgb)


SIM = {
    "normal": np.eye(3),
    "protan": np.array([[0.11238, 0.88762, 0.0], [0.11238, 0.88762, 0.0], [0.00401, -0.00401, 1.0]]),
    "deutan": np.array([[0.29275, 0.70725, 0.0], [0.29275, 0.70725, 0.0], [-0.02234, 0.02234, 1.0]]),
    "tritan": np.array([[1.0, 0.14461, -0.14461], [0.0, 0.85924, 0.14076], [0.0, 0.85924, 0.14076]]),
}


def contrast(c):
    L = 0.2126 * lin(c)[0] + 0.7152 * lin(c)[1] + 0.0722 * lin(c)[2]
    return 1.05 / (L + 0.05)


cols = sys.argv[1].split(",")
print("contrast on white: " + ", ".join(f"{c} {contrast(c):.1f}:1" for c in cols))
for a, b in itertools.combinations(cols, 2):
    d = {k: 100 * np.linalg.norm(oklab(np.clip(S @ lin(a), 0, 1)) - oklab(np.clip(S @ lin(b), 0, 1)))
         for k, S in SIM.items()}
    worst = min(d[k] for k in ("protan", "deutan", "tritan"))
    flag = "PASS" if d["normal"] >= 15 and worst >= 8 else ("FLOOR (needs 2nd encoding)" if worst >= 6 else "FAIL")
    print(f"{a} vs {b}: normal {d['normal']:5.1f}  protan {d['protan']:5.1f}  deutan {d['deutan']:5.1f}  tritan {d['tritan']:5.1f}  -> {flag}")
