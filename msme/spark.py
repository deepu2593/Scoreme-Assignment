"""SPARK -- Slack-Prioritised Adaptive Repacking with Kempe-chain repair.

    Phase 0  CERTIFY    polynomial infeasibility certificates (certificates.py)
    Phase 1  CONSTRUCT  Window-Saturation MRV greedy with regret-aware slot choice
    Phase 2  REPAIR     depth-bounded ejection chains for tasks left unplaced
    Phase 3  REFINE     feasibility-preserving tabu local search over
                        {relocate, swap, chain} neighbourhoods, with
                        ruin-and-recreate perturbation on stagnation

WHY THIS SHAPE (design rationale in one paragraph; the long version is in
docs/algorithm.md).  The problem is graph colouring, vector bin packing and
interval scheduling fused, and the fusion is what dictates the design.  Pure
DSATUR fails because "how constrained is this task" is not saturation degree
here: a slot can be dead for three different reasons (a neighbour sits there,
the slot is full, the slot is outside the SLA window), and only the first is
repairable by recolouring.  Pure bin packing fails because the conflict graph
makes "the fullest slot that fits" often illegal.  So SPARK's construction ranks
tasks by *surviving palette size* (MRV) with saturation as a tie-break, chooses
slots by true marginal penalty plus a regret term that prices the options the
placement destroys for un-placed neighbours, and -- the part that matters most
empirically -- its repair operator branches on WHY a slot died: conflict-dead
slots are attacked by relocating the blocking neighbours (a Kempe-style chain),
capacity-dead slots by evicting the smallest-footprint occupants that free
enough of the binding dimension.  Local search then never leaves the feasible
region, so the answer is safe to return at any moment (anytime property), which
is what a 30-second scheduling cycle actually needs.
"""

from __future__ import annotations

import random
import time

from .certificates import check_certificates
from .instance import Instance
from .penalty import PenaltyModel, PenaltyWeights
from .state import ScheduleState

EPS = 1e-9


class SparkResult:
    def __init__(self, sigma, penalty, feasible, reason, runtime_ms, stats):
        self.sigma = sigma
        self.penalty = penalty
        self.feasible = feasible
        self.reason = reason
        self.runtime_ms = runtime_ms
        self.stats = stats


class Spark:
    """The solver.  One instance of this class per problem instance."""

    def __init__(self, inst: Instance, lam: PenaltyWeights | None = None,
                 seed: int = 0, time_budget_ms: int = 2000,
                 restarts: int = 4, regret_mu: float = 1.0,
                 chain_depth: int = 3, tabu_tenure: int | None = None,
                 local_search: bool = True):
        self.inst = inst
        self.model = PenaltyModel(inst, lam)
        # Two independent RNG streams, deliberately.  With a single shared
        # stream the local-search phase consumes random numbers and thereby
        # changes which noisy constructions later restarts produce -- so the
        # construction-only ablation was not comparable with the full solver,
        # and full SPARK could finish WORSE than its own construction (measured
        # -1.3% on the relaxed n=50 instance).  Separate streams make restart r
        # produce the identical starting schedule in both modes, which restores
        # the guarantee P(full) <= P(construction-only).
        self.rng = random.Random(seed)                 # construction stream
        self.rng_search = random.Random(seed + 1_000_003)  # local-search stream
        self.time_budget_ms = time_budget_ms
        self.restarts = restarts
        self.regret_mu = regret_mu
        self.chain_depth = chain_depth
        self.local_search = local_search
        # Tenure ~ sqrt(n) is the conventional tabu scaling: short enough that a
        # slot becomes reusable within a few passes over the task set, long
        # enough to break 2-cycles between a relocate and its inverse.  I did not
        # tune it against this problem -- it is exposed as a constructor
        # argument so it can be, and the aspiration criterion below limits the
        # damage if it is too long.
        self.tabu_tenure = tabu_tenure or max(5, int(inst.n ** 0.5) + 2)
        self.stats = {"restarts": 0, "chain_repairs": 0, "moves": 0,
                      "swaps": 0, "chain_moves": 0, "perturbations": 0}

    # ================= PHASE 1: CONSTRUCTION =================

    def _palette(self, st: ScheduleState, i: int) -> list[int]:
        return st.feasible_slots(i)

    def _regret(self, st: ScheduleState, i: int, s: int, unplaced: set[int],
                palette_size: dict[int, int]) -> float:
        """How much does putting i in s hurt the un-placed neighbours of i?

        Each un-placed neighbour j that could still have used s loses one option.
        We charge 1/|F_j| for that loss, so stealing the last slot of a
        nearly-cornered task is expensive while stealing one of eight options is
        nearly free.  This is the term that stops the greedy from painting itself
        into a corner, and it is O(deg(i)) to evaluate -- cheap enough to run for
        every candidate slot of every task.
        """
        acc = 0.0
        for j in self.inst.adj[i]:
            if j in unplaced:
                lo, hi = self.inst.windows[j]
                if lo <= s <= hi and st.conf[j][s] == 0:
                    acc += 1.0 / max(1, palette_size.get(j, 1))
        return acc

    def construct(self, st: ScheduleState, noise: float = 0.0) -> list[int]:
        """Greedy construction.  Returns the list of tasks it could not place.

        Ranking key (most-constrained-first):
          1. smallest surviving palette |F_i|            -- MRV, the dominant signal
          2. largest window-restricted saturation        -- classic DSATUR, but
             counted only over slots inside i's own window, because slots outside
             it were never available and must not inflate the score
          3. largest w_i / (u_i - l_i + 1)               -- weight per unit of slack:
             an urgent Tier-1 task with a two-slot window outranks a heavy task
             that can float anywhere
        """
        inst = self.inst
        unplaced = set(range(inst.n))
        failed: list[int] = []
        while unplaced:
            palette = {i: self._palette(st, i) for i in unplaced}
            psize = {i: len(v) for i, v in palette.items()}
            empty = [i for i in unplaced if psize[i] == 0]
            if empty:
                # Cornered task: hand it to the repair phase rather than aborting.
                for i in empty:
                    unplaced.discard(i)
                    failed.append(i)
                continue
            def key(i):
                lo, hi = inst.windows[i]
                sat = sum(1 for s in range(lo, hi + 1) if st.conf[i][s] > 0)
                urgency = inst.weights[i] / (hi - lo + 1)
                jitter = self.rng.random() * noise
                return (psize[i] - jitter, -sat, -urgency)
            i = min(unplaced, key=key)
            best_s, best_score = None, float("inf")
            for s in palette[i]:
                st.place(i, s)
                dP = st.model.task_cost(i, s)
                st.unplace(i)
                # marginal penalty of the placement, measured exactly
                dP = self._marginal(st, i, s)
                score = dP + self.regret_mu * self._regret(st, i, s, unplaced - {i}, psize)
                score += self.rng.random() * noise
                if score < best_score:
                    best_s, best_score = s, score
            st.place(i, best_s)
            unplaced.discard(i)
        return failed

    def _marginal(self, st: ScheduleState, i: int, s: int) -> float:
        """Exact increase in P from inserting an unplaced task i into slot s."""
        before = st._balance() + st._gpu()
        st.place(i, s)
        after = st._balance() + st._gpu()
        st.unplace(i)
        return st.model.task_cost(i, s) + (after - before)

    # ================= PHASE 2: EJECTION-CHAIN REPAIR =================

    def repair(self, st: ScheduleState, failed: list[int]) -> list[int]:
        """Try to seat every task in `failed` by evicting blockers.

        For each candidate slot s of the cornered task i we compute the eviction
        set E(s):
          - every neighbour of i currently in s            (conflict-dead)
          - plus, while capacity is still exceeded, occupants of s in decreasing
            order of their contribution to the *binding* dimension (capacity-dead)
        The binding-dimension order is the important detail: evicting the task
        with the biggest total footprint is wrong when only RAM is tight and that
        task is CPU-heavy.  We then recursively re-seat each evicted task,
        depth-bounded.

        On chain_depth = 3: I swept depth over {1,2,3,4} on 55 generator
        instances that survive the certificates and still corner a task, and the
        number solved was identical (9/55) at every depth, with no meaningful
        runtime difference.  So depth is NOT tuned -- 3 is a conservative
        default that bounds the recursion, and on generator-shaped instances the
        repair phase is insurance rather than a workhorse.  See
        docs/benchmarks.md for the measurement.
        """
        still: list[int] = []
        for i in failed:
            if not self._seat(st, i, self.chain_depth, set()):
                still.append(i)
            else:
                self.stats["chain_repairs"] += 1
        return still

    def _eviction_set(self, st: ScheduleState, i: int, s: int) -> list[int] | None:
        inst = self.inst
        occupants = [j for j in range(inst.n) if st.sigma[j] == s]
        evict = [j for j in occupants if j in inst.adj[i]]
        load = [st.L[s][k] - sum(inst.resources[j][k] for j in evict)
                for k in range(inst.d)]
        req = inst.resources[i]
        remaining = [j for j in occupants if j not in evict]
        guard = 0
        while any(load[k] + req[k] > inst.capacities[s][k] + EPS for k in range(inst.d)):
            binding = max(range(inst.d),
                          key=lambda k: load[k] + req[k] - inst.capacities[s][k])
            remaining.sort(key=lambda j: -inst.resources[j][binding])
            if not remaining:
                return None
            j = remaining.pop(0)
            evict.append(j)
            for k in range(inst.d):
                load[k] -= inst.resources[j][k]
            guard += 1
            if guard > len(occupants):
                return None
        return evict

    def _seat(self, st: ScheduleState, i: int, depth: int, touched: set[int]) -> bool:
        if depth <= 0:
            return False
        direct = st.feasible_slots(i)
        if direct:
            best = min(direct, key=lambda s: self._marginal(st, i, s))
            st.place(i, best)
            return True
        for s in self.inst.window_slots(i):
            evict = self._eviction_set(st, i, s)
            if evict is None or any(j in touched for j in evict):
                continue
            saved = [(j, st.sigma[j]) for j in evict]
            for j, _ in saved:
                st.unplace(j)
            st.place(i, s)
            ok = True
            done = []
            for j, _ in saved:
                if self._seat(st, j, depth - 1, touched | {i} | {x for x, _ in saved}):
                    done.append(j)
                else:
                    ok = False
                    break
            if ok:
                return True
            # rollback -- exact restoration, no drift
            for j in done:
                st.unplace(j)
            st.unplace(i)
            for j, s_old in saved:
                st.place(j, s_old)
        return False

    # ================= PHASE 3: TABU LOCAL SEARCH =================

    def refine(self, st: ScheduleState, deadline: float) -> None:
        """Feasibility-preserving tabu search.

        Invariant: every accepted move keeps F1/F2/F3 satisfied, so `st` is a
        valid answer at all times (anytime property).  We take the best
        non-tabu move over the union of three neighbourhoods each iteration,
        allow a tabu move only if it beats the incumbent best (aspiration), and
        on `stall_limit` non-improving iterations we ruin-and-recreate a random
        slice of the schedule to escape.
        """
        inst = self.inst
        best_sigma = st.copy_sigma()
        best_P = st.penalty()
        tabu: dict[tuple[int, int], int] = {}
        it = 0
        stall = 0
        stall_limit = max(20, inst.n // 2)
        while time.perf_counter() < deadline:
            it += 1
            cur_P = st.penalty()  # hoisted: the aspiration test needs it once,
                                  # not once per candidate (O(n) -> O(1) per move)
            cand = None  # (delta, kind, payload)
            # --- N1: relocate ---
            order = list(range(inst.n))
            self.rng_search.shuffle(order)
            for i in order[:min(inst.n, 64)]:
                s_from = st.sigma[i]
                for s in inst.window_slots(i):
                    if s == s_from or not st.can_move(i, s):
                        continue
                    d = st.delta_move(i, s)
                    is_tabu = tabu.get((i, s), 0) > it
                    if is_tabu and cur_P + d >= best_P - EPS:
                        continue
                    if cand is None or d < cand[0]:
                        cand = (d, "move", (i, s, s_from))
            # --- N2: swap ---
            for _ in range(min(inst.n, 48)):
                i, j = self.rng_search.randrange(inst.n), self.rng_search.randrange(inst.n)
                if i == j or st.sigma[i] == st.sigma[j]:
                    continue
                if not st.can_swap(i, j):
                    continue
                d = st.delta_swap(i, j)
                if cand is None or d < cand[0]:
                    cand = (d, "swap", (i, j))
            # --- N3: chain move (relocate i by evicting one blocker) ---
            if cand is None or cand[0] >= -EPS:
                ch = self._best_chain_move(st)
                if ch is not None and (cand is None or ch[0] < cand[0]):
                    cand = ch
            if cand is None:
                # No legal improving OR sideways move exists.  Rather than
                # returning early and leaving the budget unspent (which is what
                # the first version did, and it is why the n=10 seed-2 run
                # stalled 0.01% above the optimum), kick the schedule and carry
                # on; if even the kick cannot change anything, we are genuinely
                # stuck and stop.
                snapshot = st.copy_sigma()
                self._perturb(st)
                if st.copy_sigma() == snapshot:
                    break
                stall = 0
                continue
            d, kind, payload = cand
            if kind == "move":
                i, s, s_from = payload
                st.move(i, s)
                tabu[(i, s_from)] = it + self.tabu_tenure
                self.stats["moves"] += 1
            elif kind == "swap":
                i, j = payload
                si, sj = st.sigma[i], st.sigma[j]
                st.unplace(i); st.unplace(j); st.place(i, sj); st.place(j, si)
                tabu[(i, si)] = it + self.tabu_tenure
                tabu[(j, sj)] = it + self.tabu_tenure
                self.stats["swaps"] += 1
            else:  # chain
                i, s, j, s_j = payload
                st.unplace(j); st.move(i, s); st.place(j, s_j)
                tabu[(j, s)] = it + self.tabu_tenure
                self.stats["chain_moves"] += 1
            P = st.penalty()
            if P < best_P - 1e-9:
                best_P, best_sigma = P, st.copy_sigma()
                stall = 0
            else:
                stall += 1
            if stall >= stall_limit:
                self._perturb(st)
                stall = 0
                if st.penalty() < best_P:
                    best_P, best_sigma = st.penalty(), st.copy_sigma()
        # restore incumbent
        for i, s in enumerate(best_sigma):
            if st.sigma[i] != s:
                st.move(i, s)

    def _best_chain_move(self, st: ScheduleState):
        """One-level ejection move used as an escape neighbourhood.

        Relocating i into a slot blocked by exactly one neighbour j is often the
        only way out of a local optimum on dense instances, because every
        single-task relocation is illegal there.  We only consider single
        blockers with a legal landing slot, which keeps the operator O(n*K*deg)
        and, importantly, feasibility-preserving.
        """
        inst = self.inst
        best = None
        sample = list(range(inst.n))
        self.rng_search.shuffle(sample)
        for i in sample[:min(inst.n, 40)]:
            s_from = st.sigma[i]
            for s in inst.window_slots(i):
                if s == s_from or st.conf[i][s] != 1:
                    continue
                blockers = [j for j in inst.adj[i] if st.sigma[j] == s]
                if len(blockers) != 1:
                    continue
                j = blockers[0]
                st.unplace(j)
                if st.can_move(i, s):
                    d1 = st.delta_move(i, s)
                    st.move(i, s)
                    for s_j in inst.window_slots(j):
                        if s_j == s or st.conf[j][s_j] != 0:
                            continue
                        if not all(st.L[s_j][k] + inst.resources[j][k]
                                   <= inst.capacities[s_j][k] + EPS for k in range(inst.d)):
                            continue
                        d2 = self._marginal(st, j, s_j)
                        d_total = d1 + d2 - self._removal_gain(st, j, s)
                        if best is None or d_total < best[0]:
                            best = (d_total, "chain", (i, s, j, s_j))
                    st.move(i, s_from)
                st.place(j, s)
        return best

    def _removal_gain(self, st: ScheduleState, j: int, s_old: int) -> float:
        """Penalty credited back for having removed j from s_old (j is currently
        unplaced when this is called from the chain operator)."""
        return st.model.task_cost(j, s_old)

    def _perturb(self, st: ScheduleState) -> None:
        """Ruin-and-recreate: unplace a random ~15% of tasks and greedily re-seat.

        Chosen over a random restart because construction from scratch throws away
        the resource packing that took the whole run to find, while a 15% ruin
        keeps the skeleton and still moves far enough to leave the basin.

        The whole perturbation is transactional: if any victim cannot be re-seated
        legally we roll the schedule back verbatim.  This is not defensive
        paranoia -- the first version placed an unseatable victim in an arbitrary
        slot, which broke the "state is always feasible" invariant and let the
        incumbent tracker latch onto an illegal schedule with a lower penalty.
        """
        inst = self.inst
        snapshot = st.copy_sigma()
        # min() guard: on a 1-task instance there is nothing to ruin, and the
        # required single-task unit test is what exposed this (rng.sample raised
        # ValueError for k=2 on a population of 1).
        k = min(inst.n, max(2, int(0.15 * inst.n)))
        if k < 2:
            return
        victims = self.rng_search.sample(range(inst.n), k)
        for i in victims:
            st.unplace(i)
        self.rng_search.shuffle(victims)
        for idx, i in enumerate(victims):
            if not self._seat(st, i, self.chain_depth, set()):
                for j in victims[idx:]:
                    if st.sigma[j] != -1:
                        st.unplace(j)
                for j, s_old in enumerate(snapshot):
                    if st.sigma[j] == -1:
                        st.place(j, s_old)
                    elif st.sigma[j] != s_old:
                        st.move(j, s_old)
                return
        self.stats["perturbations"] += 1

    # ================= DRIVER =================

    def solve(self) -> SparkResult:
        t0 = time.perf_counter()
        cert = check_certificates(self.inst)
        if cert:
            return SparkResult(None, float("inf"), False,
                               f"PROVEN INFEASIBLE ({cert})",
                               int((time.perf_counter() - t0) * 1000), dict(self.stats))
        deadline = t0 + self.time_budget_ms / 1000.0
        best_sigma, best_P = None, float("inf")
        last_fail = None
        r = 0
        while True:
            # Restart policy: run at least `restarts` rounds, then keep going
            # while budget remains AND we still have nothing feasible.  The
            # second clause matters: on the sparse n=200 instance the first four
            # constructions all corner a task, and giving up after four wasted
            # 93% of the budget while reporting "no solution found" -- which
            # reads like infeasibility but is only a search failure.
            if r >= self.restarts and (best_sigma is not None
                                       or time.perf_counter() > deadline):
                break
            if r > 0 and time.perf_counter() > deadline:
                break
            st = ScheduleState(self.model)
            noise = 0.0 if r == 0 else 0.35 * min(r, 6)  # r=0 is deterministic
            failed = self.construct(st, noise=noise)
            failed = self.repair(st, failed)
            self.stats["restarts"] += 1
            r += 1
            if failed:
                last_fail = failed
                continue
            if self.local_search:
                remaining = deadline - time.perf_counter()
                share = remaining / max(1, self.restarts - r + 1)
                self.refine(st, time.perf_counter() + max(0.0, share))
            P = st.penalty()
            if P < best_P:
                best_P, best_sigma = P, st.copy_sigma()
        runtime = int((time.perf_counter() - t0) * 1000)
        if best_sigma is None:
            names = ", ".join(self.inst.tasks[i] for i in (last_fail or [])[:5])
            return SparkResult(None, float("inf"), False,
                               "NO FEASIBLE ASSIGNMENT FOUND (search budget exhausted; "
                               f"unplaced: {names}) -- not a proof of infeasibility",
                               runtime, dict(self.stats))
        # Final independent audit: never trust the incremental engine on output.
        ok, why = verify(self.inst, best_sigma)
        if not ok:
            return SparkResult(best_sigma, best_P, False,
                               f"INTERNAL ERROR: produced invalid assignment ({why})",
                               runtime, dict(self.stats))
        return SparkResult(best_sigma, best_P, True, "", runtime, dict(self.stats))


def verify(inst: Instance, sigma) -> tuple[bool, str]:
    """Independent from-scratch feasibility audit of a finished assignment.

    Deliberately written without reference to ScheduleState: it re-derives the
    loads from sigma so that a bug in the incremental engine cannot hide behind
    itself.  Every returned schedule passes through here.
    """
    if sigma is None or len(sigma) != inst.n:
        return False, "assignment missing or wrong length"
    for i, s in enumerate(sigma):
        lo, hi = inst.windows[i]
        if not (lo <= s <= hi):
            return False, f"F3 violated: {inst.tasks[i]} in slot {s}, window [{lo},{hi}]"
    for (i, j) in inst.conflicts:
        if sigma[i] == sigma[j]:
            return False, (f"F1 violated: conflicting {inst.tasks[i]} and "
                           f"{inst.tasks[j]} both in slot {sigma[i]}")
    load = [[0.0] * inst.d for _ in range(inst.K)]
    for i, s in enumerate(sigma):
        for k in range(inst.d):
            load[s][k] += inst.resources[i][k]
    for s in range(inst.K):
        for k in range(inst.d):
            if load[s][k] > inst.capacities[s][k] + 1e-6:
                return False, (f"F2 violated: slot {s} dimension {k} load "
                               f"{load[s][k]:.3f} > capacity {inst.capacities[s][k]:.3f}")
    return True, ""
