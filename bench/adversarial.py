#!/usr/bin/env python3
"""Task 4(c) -- the tight adversarial family, verified empirically.

CONSTRUCTION  Adv(D):  K = D+1 slots, n = D+1 tasks.
    t        heavy task,  weight 1,    window [0, D]
    b_1..b_D light tasks, weight eps,  window [0, j]
    conflicts: the whole task set is a clique (every pair conflicts)
    resources: negligible, so F2 never binds

WHY THIS BREAKS SPARK'S CONSTRUCTION.  The construction ranks by surviving
palette size (MRV).  Initially |F(b_j)| = j+1 and |F(t)| = D+1, so b_1 is the
most constrained and goes first, into its cheapest slot 0.  That deletes slot 0
from every remaining palette (clique), so b_2 now has palette {1,2}, is again the
most constrained, and takes slot 1 -- and so on.  The heavy task t is always the
least constrained and is therefore placed LAST, into the only slot left: D.

MEASURED OUTCOME (see the table this script prints).  The ratio converges to
exactly D = Delta, one unit BELOW the analytic bound 1 + Delta.  The missing unit
is bought back by the third ranking key: once the b-chain has consumed slots
0..D-2, the heavy task t and the last blocker b_D are tied on palette size and on
saturation, and the urgency key w_i/(u_i-l_i+1) then favours t, so t lands in
slot D-1 rather than slot D.  Attempts to defeat that tie-break (raising the
blockers' weights until their urgency exceeds t's) inflate OPT faster than they
inflate SPARK-C and push the ratio back toward 1.  So the family below is tight
for Delta and the analytic bound 1 + Delta is tight only up to an additive 1;
closing that last unit is stated as an open point in docs/proofs.md rather than
claimed.

The script also runs the FULL solver (construction + chain repair + tabu) on the
same family, to show that the local-search phase escapes the trap -- which is the
honest reason the bound is stated for the construction phase alone.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from msme.exact import solve_exact
from msme.instance import Instance
from msme.penalty import PenaltyWeights
from msme.spark import Spark

# Pure P_base objective: the ratio theorem is stated for the given base penalty,
# so the extra terms are switched off here.
BASE_ONLY = PenaltyWeights(base=1.0, balance=0.0, sla=0.0, gpu=0.0)


def adversarial(D: int, eps: float = 1e-6) -> Instance:
    K = D + 1
    n = D + 1
    tasks = ["t"] + [f"b{j}" for j in range(1, D + 1)]
    conflicts = [(i, j) for i in range(n) for j in range(i + 1, n)]
    resources = [[0.01, 0.01, 0.0, 0.001] for _ in range(n)]
    capacities = [[32, 128, 8, 6.0] for _ in range(K)]
    windows = [(0, D)] + [(0, j) for j in range(1, D + 1)]
    weights = [1.0] + [eps] * D
    return Instance.from_dict(dict(tasks=tasks, conflicts=conflicts,
                                   resources=resources, capacities=capacities,
                                   windows=windows, weights=weights, K=K))


def main():
    print(f"{'Delta':>6} {'SPARK-C':>12} {'OPT':>12} {'ratio':>10} "
          f"{'bound 1+D':>10} {'full SPARK':>12} {'full ratio':>11}")
    for D in range(2, 10):
        inst = adversarial(D)
        cons = Spark(inst, lam=BASE_ONLY, time_budget_ms=200, restarts=1,
                     local_search=False).solve()
        full = Spark(inst, lam=BASE_ONLY, time_budget_ms=1500, restarts=4).solve()
        _, P_opt, proved = solve_exact(inst, BASE_ONLY, time_limit_s=60)
        assert proved, "exact solver did not prove optimality"
        print(f"{D:>6} {cons.penalty:>12.6f} {P_opt:>12.6f} "
              f"{cons.penalty / P_opt:>10.4f} {1 + D:>10} "
              f"{full.penalty:>12.6f} {full.penalty / P_opt:>11.4f}")


if __name__ == "__main__":
    main()
