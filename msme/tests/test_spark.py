"""Unit tests.  The four cases the assignment demands are marked [REQUIRED]."""

import random
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pytest

from msme.certificates import check_certificates
from msme.exact import solve_exact
from msme.instance import Instance
from msme.penalty import PenaltyModel, PenaltyWeights
from msme.spark import Spark, verify
from msme.state import ScheduleState


def make(tasks, conflicts, resources, capacities, windows, weights, K):
    return Instance.from_dict(dict(tasks=tasks, conflicts=conflicts, resources=resources,
                                   capacities=capacities, windows=windows,
                                   weights=weights, K=K))


# ---------------- [REQUIRED] all-conflict graph, chromatic number > K ----------------

def test_clique_larger_than_K_is_proven_infeasible():
    n, K = 5, 3
    inst = make([f"T{i}" for i in range(n)],
                [(i, j) for i in range(n) for j in range(i + 1, n)],
                [[1, 1, 0, 0.1]] * n, [[32, 128, 8, 6.0]] * K,
                [(0, K - 1)] * n, [1.0] * n, K)
    res = Spark(inst, time_budget_ms=200).solve()
    assert not res.feasible
    assert res.reason.startswith("PROVEN INFEASIBLE")
    assert "C3" in res.reason  # pigeonhole on the clique, not a search timeout


def test_clique_exactly_K_is_feasible():
    """Boundary companion: |clique| == K must succeed, or the certificate is
    over-firing and we would reject satisfiable production schedules."""
    n = K = 4
    inst = make([f"T{i}" for i in range(n)],
                [(i, j) for i in range(n) for j in range(i + 1, n)],
                [[1, 1, 0, 0.1]] * n, [[32, 128, 8, 6.0]] * K,
                [(0, K - 1)] * n, [1.0] * n, K)
    res = Spark(inst, time_budget_ms=500).solve()
    assert res.feasible, res.reason
    assert len(set(res.sigma)) == K


# ---------------- [REQUIRED] zero-capacity slot ----------------

def test_zero_capacity_slot_is_never_used():
    K = 3
    caps = [[32, 128, 8, 6.0], [0, 0, 0, 0.0], [32, 128, 8, 6.0]]
    inst = make(["A", "B"], [], [[4, 8, 1, 0.5], [4, 8, 1, 0.5]], caps,
                [(0, 2), (0, 2)], [3.0, 1.0], K)
    res = Spark(inst, time_budget_ms=300).solve()
    assert res.feasible, res.reason
    assert 1 not in res.sigma  # the dead slot absorbs nothing


def test_all_slots_zero_capacity_is_infeasible():
    inst = make(["A"], [], [[1, 1, 0, 0.1]], [[0, 0, 0, 0.0]] * 2, [(0, 1)], [1.0], 2)
    res = Spark(inst, time_budget_ms=200).solve()
    assert not res.feasible
    assert "C2" in res.reason  # atomic oversize, detected without search


# ---------------- [REQUIRED] tight SLA windows ----------------

def test_tight_windows_forced_assignment():
    """Every task is pinned to one slot; the only legal answer is that pinning."""
    K = 4
    inst = make(["A", "B", "C", "D"], [(0, 1), (2, 3)],
                [[2, 4, 0, 0.2]] * 4, [[32, 128, 8, 6.0]] * K,
                [(0, 0), (1, 1), (2, 2), (3, 3)], [5.0, 4.0, 3.0, 2.0], K)
    res = Spark(inst, time_budget_ms=300).solve()
    assert res.feasible, res.reason
    assert res.sigma == [0, 1, 2, 3]


def test_tight_windows_conflicting_pin_is_infeasible():
    inst = make(["A", "B"], [(0, 1)], [[1, 1, 0, 0.1]] * 2,
                [[32, 128, 8, 6.0]] * 2, [(1, 1), (1, 1)], [1.0, 1.0], 2)
    res = Spark(inst, time_budget_ms=200).solve()
    assert not res.feasible and "PROVEN INFEASIBLE" in res.reason


# ---------------- [REQUIRED] single-task instance ----------------

def test_single_task_goes_to_earliest_legal_slot():
    inst = make(["A"], [], [[4, 8, 1, 0.5]], [[32, 128, 8, 6.0]] * 5,
                [(1, 4)], [7.0], 5)
    res = Spark(inst, time_budget_ms=200).solve()
    assert res.feasible and res.sigma == [1]
    # earliest legal slot is optimal here: P_base and the SLA term both minimise
    # at s = l, and with one task the balance term is slot-independent.
    sigma_opt, P_opt, proved = solve_exact(inst)
    assert proved and abs(res.penalty - P_opt) < 1e-9


# ---------------- toy instance from the assignment (Section 3.3) ----------------

def test_assignment_toy_instance():
    inst = make(["T1", "T2", "T3", "T4", "T5", "T6"],
                [(0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 5)],
                [[8, 32, 4, 1.5], [4, 16, 0, 3.0], [2, 8, 0, 2.0],
                 [16, 64, 2, 0.5], [8, 32, 2, 1.0], [4, 16, 0, 1.5]],
                [[32, 128, 8, 6.0]] * 4,
                [(0, 2), (0, 3), (0, 3), (1, 3), (0, 3), (1, 3)],
                [5, 4, 3, 2, 3, 2], 4)
    res = Spark(inst, time_budget_ms=1000).solve()
    assert res.feasible, res.reason
    ok, why = verify(inst, res.sigma)
    assert ok, why
    sigma_opt, P_opt, proved = solve_exact(inst)
    assert proved
    assert res.penalty <= P_opt + 1e-6  # SPARK is optimal on the toy instance


# ---------------- invariants ----------------

@pytest.mark.parametrize("n,K,den,seed", [(8, 3, 0.3, 1), (10, 4, 0.4, 2),
                                          (12, 4, 0.5, 3), (25, 6, 0.2, 5)])
def test_output_is_always_verifiably_feasible_or_flagged(n, K, den, seed):
    inst = Instance.random(n, K, den, seed)
    res = Spark(inst, time_budget_ms=500, seed=seed).solve()
    if res.feasible:
        ok, why = verify(inst, res.sigma)
        assert ok, why
    else:
        assert res.reason  # never a silent failure


def test_certificate_soundness_against_exact_solver():
    """If a certificate fires, the exact solver must also find no solution.
    This is the test that would catch an over-eager (unsound) certificate."""
    for seed in range(30):
        inst = Instance.random(7, 3, 0.35, seed)
        if check_certificates(inst):
            sigma, P, proved = solve_exact(inst, time_limit_s=20)
            assert proved and sigma is None, f"seed {seed}: certificate was WRONG"


def test_incremental_penalty_matches_full_recompute():
    """Guards the O(d) delta engine against drift -- the single most dangerous
    class of bug here, because a wrong delta silently optimises the wrong
    objective while every feasibility check still passes."""
    inst = Instance.random(30, 6, 0.2, 7)
    model = PenaltyModel(inst)
    rng = random.Random(0)
    sigma = [rng.randint(*inst.windows[i]) for i in range(inst.n)]
    st = ScheduleState(model, sigma)
    for _ in range(3000):
        i = rng.randrange(inst.n)
        s = rng.choice(list(inst.window_slots(i)))
        d = st.delta_move(i, s)
        before = st.penalty()
        st.move(i, s)
        assert abs((st.penalty() - before) - d) < 1e-7
        assert abs(st.penalty() - model.total(st.sigma)) < 1e-7


def test_swap_respects_mutual_conflict_bookkeeping():
    """Two conflicting tasks in adjacent slots may legally swap; a naive
    conf == 0 test wrongly forbids it. Regression test for that exact bug."""
    inst = make(["A", "B"], [(0, 1)], [[1, 1, 0, 0.1]] * 2,
                [[32, 128, 8, 6.0]] * 2, [(0, 1), (0, 1)], [1.0, 2.0], 2)
    st = ScheduleState(PenaltyModel(inst), [0, 1])
    assert st.can_swap(0, 1)


def test_penalty_terms_are_non_negative_and_gpu_zero_when_integral():
    inst = make(["A", "B"], [], [[1, 1, 2.0, 0.1], [1, 1, 2.0, 0.1]],
                [[32, 128, 8, 6.0]] * 2, [(0, 1), (0, 1)], [1.0, 1.0], 2)
    model = PenaltyModel(inst)
    b = model.breakdown([0, 0])
    assert b["gpu_frag"] == 0.0          # 4.0 GPU units is integral -> no waste
    assert all(v >= 0 for k, v in b.items() if k != "total")


def test_penalty_weights_are_immutable():
    lam = PenaltyWeights()
    with pytest.raises(Exception):
        lam.balance = 5.0
