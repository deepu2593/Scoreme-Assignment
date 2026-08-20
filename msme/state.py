"""Mutable schedule state with O(d) exact incremental penalty deltas.

This is the data structure the whole solver hangs off.  It maintains, for the
current assignment sigma:

    L[s][k]        absolute load of dimension k in slot s
    sumU[k]        sum over slots of utilisation U_{s,k}   (for the balance mean)
    sumU2[k]       sum over slots of U_{s,k}^2             (for the balance sum of squares)
    conf[i][s]     number of neighbours of task i currently sitting in slot s

WHY conf: the feasibility question "may task i go to slot s?" must be O(1) in
the inner loop of both construction and local search.  Recomputing it by walking
adjacency is O(deg) per query -- at density 0.6 with n=200 that is ~120
neighbours, and the refinement loop evaluates far more candidate moves than it
applies.  conf turns the query into a single array read and pays O(deg) only on
an *applied* move, which is the favourable side of that ratio.

The balance term is  scale * sum_k [ sumU2[k] - sumU[k]^2 / K ]  which is the
identity  sum (U - Ubar)^2 = sum U^2 - K*Ubar^2  with Ubar = sumU/K.  Keeping the
two running sums is what makes a relocation cost O(d) instead of O(K*d), and it
is exact -- no drift, because we add and subtract the same float quantities that
a full recompute would.  `test_incremental_penalty_matches_full_recompute` pins
this down: 3000 random relocations on a 30-task instance, asserting after every
one that the reported delta and the full recompute agree to 1e-7.
"""

from __future__ import annotations

import math

from .instance import Instance
from .penalty import GPU_DIM, PenaltyModel

EPS = 1e-9


class ScheduleState:
    def __init__(self, model: PenaltyModel, sigma: list[int] | None = None):
        self.model = model
        self.inst = model.inst
        n, K, d = self.inst.n, self.inst.K, self.inst.d
        self.sigma = [-1] * n if sigma is None else list(sigma)
        self.L = [[0.0] * d for _ in range(K)]
        self.sumU = [0.0] * d
        self.sumU2 = [0.0] * d
        self.conf = [[0] * K for _ in range(n)]
        self.count = [0] * K
        placed = [(i, s) for i, s in enumerate(self.sigma) if s >= 0]
        self.sigma = [-1] * n
        for i, s in placed:
            self.place(i, s)

    # ---------- feasibility predicates (F1, F2, F3) ----------

    def fits_window(self, i: int, s: int) -> bool:
        lo, hi = self.inst.windows[i]
        return lo <= s <= hi

    def fits_conflict(self, i: int, s: int) -> bool:
        return self.conf[i][s] == 0

    def fits_capacity(self, i: int, s: int) -> bool:
        cap, row, req = self.inst.capacities[s], self.L[s], self.inst.resources[i]
        for k in range(self.inst.d):
            if row[k] + req[k] > cap[k] + EPS:
                return False
        return True

    def feasible_slot(self, i: int, s: int) -> bool:
        return (self.fits_window(i, s) and self.fits_conflict(i, s)
                and self.fits_capacity(i, s))

    def feasible_slots(self, i: int) -> list[int]:
        lo, hi = self.inst.windows[i]
        return [s for s in range(lo, hi + 1) if self.feasible_slot(i, s)]

    # ---------- mutation ----------

    def _apply(self, i: int, s: int, sign: int) -> None:
        """Add (sign=+1) or remove (sign=-1) task i's footprint in slot s."""
        inst, row, cap = self.inst, self.L[s], self.inst.capacities[s]
        req = inst.resources[i]
        for k in range(inst.d):
            c = cap[k]
            if c > 0:
                u_old = row[k] / c
                row[k] += sign * req[k]
                u_new = row[k] / c
                self.sumU[k] += u_new - u_old
                self.sumU2[k] += u_new * u_new - u_old * u_old
            else:
                row[k] += sign * req[k]
        for j in inst.adj[i]:
            self.conf[j][s] += sign
        self.count[s] += sign

    def place(self, i: int, s: int) -> None:
        assert self.sigma[i] == -1
        self.sigma[i] = s
        self._apply(i, s, +1)

    def unplace(self, i: int) -> int:
        s = self.sigma[i]
        assert s >= 0
        self._apply(i, s, -1)
        self.sigma[i] = -1
        return s

    def move(self, i: int, s: int) -> None:
        self.unplace(i)
        self.place(i, s)

    # ---------- objective ----------

    def _balance(self) -> float:
        m = self.model
        K = self.inst.K
        acc = 0.0
        for k in range(self.inst.d):
            acc += self.sumU2[k] - (self.sumU[k] * self.sumU[k]) / K
        return m.lam.balance * m.balance_scale * max(0.0, acc)

    def _gpu(self) -> float:
        m = self.model
        if self.inst.d <= GPU_DIM:
            return 0.0
        frag = 0.0
        for s in range(self.inst.K):
            g = self.L[s][GPU_DIM]
            if g > 1e-12:
                frag += math.ceil(g) - g
        return m.lam.gpu * m.gpu_scale * frag

    def penalty(self) -> float:
        """Current P(sigma).  Only valid when every task is placed."""
        m = self.model
        own = sum(m.task_cost(i, s) for i, s in enumerate(self.sigma) if s >= 0)
        return own + self._balance() + self._gpu()

    def delta_move(self, i: int, s_to: int) -> float:
        """Exact change in P if task i relocates from its current slot to s_to.

        O(d).  Computed by simulating the two touched slots' contributions to
        sumU2/sumU and the GPU fragment, then undoing the arithmetic -- no state
        is mutated, so this is safe to call inside a candidate loop.
        """
        s_from = self.sigma[i]
        if s_from == s_to:
            return 0.0
        m, inst = self.model, self.inst
        d = inst.d
        req = inst.resources[i]
        dsum = [0.0] * d
        dsum2 = 0.0
        gpu_delta = 0.0
        for k in range(d):
            for (s, sign) in ((s_from, -1.0), (s_to, +1.0)):
                c = inst.capacities[s][k]
                if c <= 0:
                    continue
                u_old = self.L[s][k] / c
                u_new = (self.L[s][k] + sign * req[k]) / c
                dsum[k] += u_new - u_old
                dsum2 += u_new * u_new - u_old * u_old
        acc = 0.0
        K = inst.K
        for k in range(d):
            su = self.sumU[k]
            acc -= ((su + dsum[k]) ** 2 - su * su) / K
        d_balance = m.lam.balance * m.balance_scale * (dsum2 + acc)
        if d > GPU_DIM:
            for (s, sign) in ((s_from, -1.0), (s_to, +1.0)):
                g_old = self.L[s][GPU_DIM]
                g_new = g_old + sign * req[GPU_DIM]
                f_old = (math.ceil(g_old) - g_old) if g_old > 1e-12 else 0.0
                f_new = (math.ceil(g_new) - g_new) if g_new > 1e-12 else 0.0
                gpu_delta += f_new - f_old
            gpu_delta *= m.lam.gpu * m.gpu_scale
        d_own = m.task_cost(i, s_to) - m.task_cost(i, s_from)
        return d_own + d_balance + gpu_delta

    def can_move(self, i: int, s_to: int) -> bool:
        """Feasibility of relocating i to s_to, accounting for the fact that i
        currently occupies its own slot (so its own footprint must be excluded
        when s_to == s_from, and its own conflict contribution never counts --
        a task is never its own neighbour, so conf is already correct)."""
        if not self.fits_window(i, s_to):
            return False
        if self.conf[i][s_to] != 0:
            return False
        s_from = self.sigma[i]
        if s_from == s_to:
            return True
        cap, row, req = self.inst.capacities[s_to], self.L[s_to], self.inst.resources[i]
        for k in range(self.inst.d):
            if row[k] + req[k] > cap[k] + EPS:
                return False
        return True

    def can_swap(self, i: int, j: int) -> bool:
        """Feasibility of exchanging the slots of two *placed* tasks.

        Subtlety that cost us a real bug: i and j may be neighbours in the
        conflict graph, in which case conf[i][slot(j)] counts j itself.  After
        the swap j is no longer there, so that self-induced conflict must be
        discounted -- hence the `adj` check rather than a plain conf==0 test.
        """
        si, sj = self.sigma[i], self.sigma[j]
        if si == sj:
            return False
        if not (self.fits_window(i, sj) and self.fits_window(j, si)):
            return False
        adjacent = j in self.inst.adj[i]
        if self.conf[i][sj] - (1 if adjacent else 0) != 0:
            return False
        if self.conf[j][si] - (1 if adjacent else 0) != 0:
            return False
        ri, rj = self.inst.resources[i], self.inst.resources[j]
        for k in range(self.inst.d):
            if self.L[sj][k] - rj[k] + ri[k] > self.inst.capacities[sj][k] + EPS:
                return False
            if self.L[si][k] - ri[k] + rj[k] > self.inst.capacities[si][k] + EPS:
                return False
        return True

    def delta_swap(self, i: int, j: int) -> float:
        """Exact delta for a swap.

        The own-cost part is closed form (four task_cost lookups).  The coupled
        part (balance + GPU fragmentation) is measured by apply-measure-revert:
        a swap touches two slots twice and the closed form was error-prone,
        whereas apply/revert is exact by construction because it runs the same
        arithmetic a full evaluation would.  Cost is O(d + K + deg(i) + deg(j)),
        not O(n), because we only re-evaluate the coupled terms.
        """
        m = self.model
        si, sj = self.sigma[i], self.sigma[j]
        coupled_before = self._balance() + self._gpu()
        self.unplace(i)
        self.unplace(j)
        self.place(i, sj)
        self.place(j, si)
        coupled_after = self._balance() + self._gpu()
        self.unplace(i)
        self.unplace(j)
        self.place(i, si)
        self.place(j, sj)
        d_own = (m.task_cost(i, sj) - m.task_cost(i, si)
                 + m.task_cost(j, si) - m.task_cost(j, sj))
        return d_own + (coupled_after - coupled_before)

    def copy_sigma(self) -> list[int]:
        return list(self.sigma)
