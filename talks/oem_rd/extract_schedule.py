# -*- coding: utf-8 -*-
"""
Dump the hindsight-optimal EV and diesel schedules of one paired instance as
plain activity segments, for the timeline slide of the OEM talk.

Reuses the reconstruction of src/plot/diesel_vs_ev_figure.py, so the segments
are exactly the ones drawn in that figure.

Run:
    python talks/oem_rd/extract_schedule.py [RlongCfewTnone_2]
"""
from __future__ import annotations

import json
import os
import sys

from src import paths as _paths
from src.plot.diesel_vs_ev_figure import _oracle_run, reconstruct

HERE = os.path.dirname(os.path.abspath(__file__))


def segments(rec: dict) -> list[dict]:
    t0 = rec["t0"]
    out = [dict(kind="drive", t0=a - t0, t1=b - t0) for a, b in rec["drive"]]
    out += [dict(kind=k, t0=a - t0, t1=b - t0) for k, a, b in rec["blocks"]]
    return sorted(out, key=lambda s: s["t0"])


def main() -> None:
    instance = sys.argv[1] if len(sys.argv) > 1 else "RlongCfewTnone_2"
    inst = json.load(open(_paths.instances(f"{instance}.json"), encoding="utf-8"))
    fd, E_real = inst["instance"], inst["E_real"]
    fd_di = dict(fd)
    fd_di["Q"] = {k: 0.0 for k in fd["Q"]}

    ev = reconstruct(_oracle_run(instance, fd), fd, E_real)
    di = reconstruct(_oracle_run(f"{instance}__diesel", fd_di), fd_di,
                     [0.0] * len(E_real))

    soc = [(float(t - ev["t0"]), float(e)) for t, e in ev["soc"]]
    data = dict(instance=instance,
                km=sum(fd["km"].values()),
                ecap=fd.get("Ecap"), emin=fd.get("Emin"),
                ev=dict(duration=ev["t1"] - ev["t0"], segments=segments(ev),
                        budget=ev["budget"], soc=soc),
                diesel=dict(duration=di["t1"] - di["t0"], segments=segments(di),
                            budget=di["budget"]))
    out = os.path.join(HERE, f"schedule_{instance}.json")
    json.dump(data, open(out, "w", encoding="utf-8"), indent=1)
    print(out)


if __name__ == "__main__":
    main()
