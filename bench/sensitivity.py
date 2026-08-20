#!/usr/bin/env python3
"""Lambda sensitivity sweep, referenced by msme/penalty.py and docs/benchmarks.md.

Question: do the extra penalty terms change *which schedule wins*, or do they
only rescale the score?  If the ranking of SPARK against its construction-only
ablation flipped as lambda moved, the defaults would be doing real work and
would need per-instance tuning.  This sweeps each lambda over [0.25, 4] with the
others held at their defaults and reports both the penalty and the ablation gap.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from msme.instance import Instance
from msme.penalty import PenaltyWeights
from msme.spark import Spark
from run import relax_windows

CASES = [(8, 3, 0.30, 1, 0), (12, 4, 0.50, 3, 0), (50, 8, 0.25, 10, 1)]
GRID = [0.25, 0.5, 1.0, 2.0, 4.0]


def main():
    print(f"{'instance':>14} {'knob':>10} {'lambda':>7} {'SPARK':>12} "
          f"{'construction':>13} {'gap %':>7} {'ranking':>9}")
    flips = 0
    for n, K, den, seed, slack in CASES:
        inst = Instance.random(n, K, den, seed)
        if slack:
            inst = relax_windows(inst, slack)
        budget = 2000 if n <= 12 else 5000
        for knob in ("balance", "sla", "gpu"):
            for v in GRID:
                kw = {"base": 1.0, "balance": 1.0, "sla": 1.0, "gpu": 0.5}
                kw[knob] = v
                lam = PenaltyWeights(**kw)
                full = Spark(inst, lam=lam, seed=1, time_budget_ms=budget).solve()
                cons = Spark(inst, lam=lam, seed=1, time_budget_ms=budget,
                             local_search=False).solve()
                gap = 100 * (cons.penalty - full.penalty) / cons.penalty
                ok = full.penalty <= cons.penalty + 1e-9
                flips += 0 if ok else 1
                print(f"{f'n={n},K={K}':>14} {knob:>10} {v:>7.2f} "
                      f"{full.penalty:>12.4f} {cons.penalty:>13.4f} {gap:>7.2f} "
                      f"{'ok' if ok else 'FLIPPED':>9}")
    print(f"\nranking flips across the whole sweep: {flips}")


if __name__ == "__main__":
    main()
