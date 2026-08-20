"""Exact branch-and-bound solver -- ground truth for the small benchmark instances.

Used ONLY to measure the empirical approximation ratio of SPARK (Task 6).  It is
exponential and is never called on n > ~14.

The bound: at any node we have placed a prefix of tasks (ordered
most-constrained-first, which is what makes the pruning bite).  The cost already
committed is the exact own-cost of placed tasks plus the exact coupled terms
(balance + GPU) of the partial schedule.  Both coupled terms are non-negative and
the balance term can *decrease* as more tasks are added (adding load to an empty
slot flattens the profile), so we must NOT count the current coupled value as a
lower bound.  We therefore lower-bound a node by

    committed own-cost  +  sum over unplaced i of min over legal s of task_cost(i,s)

which is admissible because own-costs are additive and independent, and the two
coupled terms are >= 0.  This is weaker than counting balance, but it is *sound*
-- an inadmissible bound would silently return the wrong optimum and corrupt
every approximation ratio in the report.
"""

from __future__ import annotations

import time

from .instance import Instance
from .penalty import PenaltyModel, PenaltyWeights
from .state import ScheduleState


def solve_exact(inst: Instance, lam: PenaltyWeights | None = None,
                time_limit_s: float = 60.0):
    """Return (best_sigma | None, best_penalty, proved_optimal: bool)."""
    model = PenaltyModel(inst, lam)
    st = ScheduleState(model)
    order = sorted(range(inst.n),
                   key=lambda i: (inst.windows[i][1] - inst.windows[i][0],
                                  -inst.degree(i)))
    floor = [min((model.task_cost(i, s) for s in inst.window_slots(i)), default=0.0)
             for i in range(inst.n)]
    suffix = [0.0] * (inst.n + 1)
    for idx in range(inst.n - 1, -1, -1):
        suffix[idx] = suffix[idx + 1] + floor[order[idx]]

    best = {"P": float("inf"), "sigma": None}
    deadline = time.perf_counter() + time_limit_s
    timed_out = [False]

    def dfs(idx: int, own: float):
        if time.perf_counter() > deadline:
            timed_out[0] = True
            return
        if own + suffix[idx] >= best["P"]:
            return
        if idx == inst.n:
            P = st.penalty()
            if P < best["P"]:
                best["P"], best["sigma"] = P, st.copy_sigma()
            return
        i = order[idx]
        cands = [s for s in inst.window_slots(i) if st.feasible_slot(i, s)]
        cands.sort(key=lambda s: model.task_cost(i, s))
        for s in cands:
            st.place(i, s)
            dfs(idx + 1, own + model.task_cost(i, s))
            st.unplace(i)

    dfs(0, 0.0)
    return best["sigma"], best["P"], (not timed_out[0])
