"""
strict_spread_check.py — do the final students break the Art. 8 spread caps?

Students are driven in BEHDV(fd), whose default (strict_spread=False) never
tests the two spread rules the MILP enforces: a daily rest must BEGIN within
13 h of the shift start (15 h if reduced), and the final shift must end within
13 h (BEHDV docstring; memory note project-spread-cap-blindspot: 21.6 % of
stored LA runs overrun the pre-rest cap undetected).

This replays the final-protocol students (g99sr + nominal features + M_seq,
as eval_*_g99srNm_test.json) on the 125 test routes twice: as evaluated, and
with BEHDV forced to strict_spread=True.  Read-only: nothing is saved; src/ is
untouched (BEHDV is swapped inside policy_core only, in this process).

Usage (.venv python):  python ML/scratch/strict_spread_check.py
"""
from __future__ import annotations

import os
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.abspath(os.path.join(HERE, "..", "code"))
sys.path.insert(0, CODE)

import numpy as np                                                   # noqa: E402

import policy_core                                                   # noqa: E402
from evaluate import DATA, INST, SPLITS                              # noqa: E402
from src.instance_gen.instance_io import load_instance_json          # noqa: E402
from src.simulation.BEHDV import BEHDV                               # noqa: E402


class StrictBEHDV(BEHDV):
    def __init__(self, full_data, strict_spread=True):
        super().__init__(full_data, strict_spread=True)


MODELS = [("gbt", "gbt_F95_base_s1"), ("torch", "tmlp_F95_split_list_s0")]


def main():
    d = np.load(os.path.join(DATA, "dataset.npz"), allow_pickle=True)
    insts = [str(x) for x in d["instances"]]
    test = [i for i in insts if int(i.rsplit("_", 1)[1]) in SPLITS["test"]]
    print(f"{len(test)} test routes")
    for kind, tag in MODELS:
        for strict in (False, True):
            policy_core.BEHDV = StrictBEHDV if strict else BEHDV
            pol = policy_core.load_policy(kind, tag, guard_q=0.99,
                                          spread_room=True)
            pol.feature_q = None
            pol.pay_mseq = True
            fails, kinds, per_len = 0, Counter(), Counter()
            for inst in test:
                fd, D_real, E_real, cv = load_instance_json(
                    os.path.join(INST, inst + ".json"))
                fd["_horizon_h"] = 24.0
                r = policy_core.run_student(fd, D_real, E_real, pol, cv=cv)
                if not r["route_completed"]:
                    fails += 1
                    kinds[r["halt_reason"]] += 1
                    per_len[inst[:7]] += 1
            print(f"{tag:28s} strict={strict!s:5s}  broken {fails:3d}/{len(test)}"
                  f"  by rule {dict(kinds)}  by length {dict(per_len)}")
    policy_core.BEHDV = BEHDV


if __name__ == "__main__":
    main()
