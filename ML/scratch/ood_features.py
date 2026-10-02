"""Which inputs does the torch model see OUTSIDE its training range on mixed
routes?  Drives the model on mixed routes and their uniform originals,
captures every row it scores, and compares each feature with the range of the
FIT rows (dataset.npz).  python ood_features.py [pmix|dmix|mix]"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "code"))
from dataset import load, split_masks                             # noqa: E402
from mixed_eval import instances, load as load_route              # noqa: E402
from policy_core import load_policy, run_student                  # noqa: E402

VARIANT = sys.argv[1] if len(sys.argv) > 1 else "pmix"
TAG = "tmlp_F95_split_list_s0"

d = load()
names = [str(x) for x in d["feature_names"]]
fit = split_masks(d)[0]
pol = load_policy("torch", TAG, guard_q=0.99, spread_room=True)
cols = pol.state_names + pol.action_names
Xf = d["X"][fit][:, [names.index(c) for c in cols]]
lo, hi = np.percentile(Xf, 0.1, axis=0), np.percentile(Xf, 99.9, axis=0)
mn, mx = Xf.min(0), Xf.max(0)
sd = Xf.std(0) + 1e-9

captured = []
orig = pol._predict


def spy(rows):
    captured.append(np.array(rows))
    return orig(rows)


pol._predict = spy


def drive(variant):
    captured.clear()
    for v, name, path in instances([VARIANT], uniform=True):
        if v != variant:
            continue
        fd, D, E, cv = load_route(path)
        run_student(fd, D, E, pol, cv=cv)
    return np.concatenate(captured)


out = {}
for v in ("uniform", VARIANT):
    R = drive(v)
    beyond = np.maximum((R - mx) / sd, (mn - R) / sd)          # std units past the range
    out[v] = dict(n=len(R), frac=(beyond > 0).mean(0), worst=beyond.max(0),
                  frac_any=(beyond > 0).any(1).mean())

print(f"rows the model scored: uniform {out['uniform']['n']}, {VARIANT} {out[VARIANT]['n']}")
print(f"rows with ANY feature outside the training min/max: uniform "
      f"{out['uniform']['frac_any'] * 100:.1f}%, {VARIANT} {out[VARIANT]['frac_any'] * 100:.1f}%")
order = np.argsort(-out[VARIANT]["frac"])
print(f"\n{'feature':28s} {'% rows outside':>15s} {'(uniform)':>10s} {'worst, in std':>14s} {'(uniform)':>10s}")
for j in order[:15]:
    if out[VARIANT]["frac"][j] <= 0:
        break
    print(f"{cols[j]:28s} {out[VARIANT]['frac'][j] * 100:14.1f}% {out['uniform']['frac'][j] * 100:9.1f}% "
          f"{out[VARIANT]['worst'][j]:14.1f} {out['uniform']['worst'][j]:10.1f}")
