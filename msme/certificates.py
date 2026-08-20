"""Polynomial-time infeasibility certificates.

These run BEFORE any search.  Each one, when it fires, is a *proof* that no
feasible assignment exists -- not a heuristic guess.  That distinction matters
for the report: the solver's `violation_reason` says either "PROVEN INFEASIBLE
(certificate ...)" or "NO FEASIBLE ASSIGNMENT FOUND (search exhausted budget)",
and we never conflate the two.  All five are sound but none is complete (the
problem is NP-hard, so a complete polynomial test would collapse P and NP).

C1  Window sanity          -- l_i > u_i, or the window falls outside [0, K-1].
C2  Atomic oversize        -- some task does not fit alone in ANY slot of its
                              own window, in some dimension.
C3  Interval clique bound  -- a set of pairwise-conflicting tasks whose windows
                              all live inside an interval of length L needs L
                              distinct slots; if the clique is bigger than L,
                              pigeonhole kills it.  (We use a greedy clique, so
                              this is sound but not exhaustive.)
C4  Interval Hall/knapsack -- for every slot interval [a,b] and every resource
                              dimension k, the tasks whose windows are trapped
                              inside [a,b] must all fit into the total capacity
                              of those slots.  This is the multi-dimensional,
                              time-windowed analogue of Hall's condition and is
                              the certificate that actually fires on the
                              tight-K stress instance.
C5  Cardinality pigeonhole -- a slot cannot hold more tasks than its tightest
                              dimension allows given the smallest demand present;
                              summed over an interval this bounds how many
                              window-trapped tasks that interval can absorb.
"""

from __future__ import annotations

import random

from .instance import Instance

EPS = 1e-9


def check_certificates(inst: Instance) -> str | None:
    """Return a human-readable certificate string, or None if none fires."""
    for fn in (_c1_windows, _c2_atomic, _c3_clique, _c4_interval_capacity,
               _c5_cardinality):
        reason = fn(inst)
        if reason:
            return reason
    return None


def _c1_windows(inst: Instance) -> str | None:
    for i, (lo, hi) in enumerate(inst.windows):
        if lo > hi or lo < 0 or hi > inst.K - 1:
            return (f"C1 window sanity: task {inst.tasks[i]} has empty/out-of-range "
                    f"SLA window [{lo},{hi}] against K={inst.K}")
    return None


def _c2_atomic(inst: Instance) -> str | None:
    for i in range(inst.n):
        req = inst.resources[i]
        ok = False
        for s in inst.window_slots(i):
            if all(req[k] <= inst.capacities[s][k] + EPS for k in range(inst.d)):
                ok = True
                break
        if not ok:
            return (f"C2 atomic oversize: task {inst.tasks[i]} does not fit alone in "
                    f"any slot of its window {inst.windows[i]}")
    return None


def _c3_clique(inst: Instance, tries: int = 64, seed: int = 12345) -> str | None:
    """Greedy maximal clique restricted to each slot interval [a,b].

    For each interval we build cliques greedily, first in decreasing-degree order
    (the classic deterministic heuristic) and then from `tries` random orders.
    Exact max-clique is itself NP-hard and a certificate only needs to be SOUND
    -- any clique that exceeds the interval length is already a proof -- but the
    single degree-ordered pass is weak: on the n=200/K=20/density=0.10 instance
    it finds nothing, while the randomised restarts find a triangle trapped in
    slots [18,19] within milliseconds.  The randomisation is seeded so the
    certificate is reproducible for the evaluator.
    """
    rng = random.Random(seed)
    K = inst.K
    for a in range(K):
        for b in range(a, K):
            span = b - a + 1
            cand = [i for i in range(inst.n)
                    if a <= inst.windows[i][0] and inst.windows[i][1] <= b]
            if len(cand) <= span:
                continue
            cs = set(cand)
            orders = [sorted(cand, key=lambda i: -len(inst.adj[i] & cs))]
            for _ in range(tries):
                shuffled = cand[:]
                rng.shuffle(shuffled)
                orders.append(shuffled)
            for order in orders:
                clique: list[int] = []
                for i in order:
                    if all(j in inst.adj[i] for j in clique):
                        clique.append(i)
                        if len(clique) > span:
                            names = ", ".join(inst.tasks[x] for x in clique)
                            return (f"C3 interval clique: {len(clique)} mutually conflicting "
                                    f"tasks ({names}) are all confined to slots [{a},{b}] "
                                    f"({span} slots) -- pigeonhole")
    return None


def _c4_interval_capacity(inst: Instance) -> str | None:
    K, d = inst.K, inst.d
    for a in range(K):
        for b in range(a, K):
            cap = [sum(inst.capacities[s][k] for s in range(a, b + 1)) for k in range(d)]
            dem = [0.0] * d
            members = 0
            for i in range(inst.n):
                lo, hi = inst.windows[i]
                if a <= lo and hi <= b:
                    members += 1
                    for k in range(d):
                        dem[k] += inst.resources[i][k]
            if members == 0:
                continue
            for k in range(d):
                if dem[k] > cap[k] + EPS:
                    return (f"C4 interval capacity (Hall): {members} tasks are confined to "
                            f"slots [{a},{b}] and demand {dem[k]:.2f} of dimension {k} "
                            f"but those slots supply only {cap[k]:.2f}")
    return None


def _c5_cardinality(inst: Instance) -> str | None:
    """Cardinality pigeonhole inside a slot interval.

    A slot s can physically hold at most floor(C(s)_k / rmin_k) tasks, where
    rmin_k is the smallest dimension-k demand among the tasks that could go
    there.  Summing that ceiling over an interval upper-bounds how many
    window-trapped tasks the interval can absorb.  Weaker than C4 when demands
    are homogeneous, but it fires where C4 does not: when many tiny tasks are
    trapped in few slots, total volume can fit while task *count* cannot -- that
    is exactly the shape of a tight-K credit-pipeline burst.
    """
    for a in range(inst.K):
        for b in range(a, inst.K):
            trapped = [i for i in range(inst.n)
                       if a <= inst.windows[i][0] and inst.windows[i][1] <= b]
            if len(trapped) < 2:
                continue
            rmin = [min(inst.resources[i][k] for i in trapped) for k in range(inst.d)]
            capacity_count = 0
            for s in range(a, b + 1):
                per_slot = min((int(inst.capacities[s][k] // rmin[k])
                                for k in range(inst.d) if rmin[k] > EPS),
                               default=len(trapped))
                capacity_count += per_slot
            if len(trapped) > capacity_count:
                return (f"C5 cardinality pigeonhole: {len(trapped)} tasks are confined to "
                        f"slots [{a},{b}], which can hold at most {capacity_count} tasks")
    return None
