# Task 6 — Empirical analysis

Raw numbers: [`results/benchmarks.md`](../results/benchmarks.md) (auto-generated),
[`results/benchmarks.json`](../results/benchmarks.json), charts
[`results/penalty_vs_n.svg`](../results/penalty_vs_n.svg) and
[`results/runtime_vs_n.svg`](../results/runtime_vs_n.svg) (PNG versions are
written locally by the same harness). Reproduce with
`python bench/run_benchmarks.py` (~3 min).

Machine: Linux, CPython 3.11, single core, no BLAS in the hot path. Solver budget
3 s for `n <= 12` and 8 s otherwise; 4 restarts; default lambdas
`(1.0, 1.0, 0.5)`.

---

## Headline result

**Six of the nine mandated instances are *provably* infeasible**, and SPARK
proves it in 0–330 ms without searching. Only the three small instances admit a
schedule, and on all three SPARK returns the **proven optimum** (ratio 1.00000).

| group | n | K | density | seed | edges | outcome | penalty | runtime (ms) |
|---|---|---|---|---|---|---|---|---|
| small | 8 | 3 | 0.30 | 1 | 11 | feasible | 76.96 | 2 |
| small | 10 | 4 | 0.40 | 2 | 16 | feasible | 88.46 | 4 |
| small | 12 | 4 | 0.50 | 3 | 26 | feasible | 140.60 | 8 |
| medium | 50 | 8 | 0.25 | 10 | 291 | **PROVEN INFEASIBLE** (C3) | — | 11 |
| medium | 100 | 10 | 0.30 | 11 | 1499 | **PROVEN INFEASIBLE** (C3) | — | 40 |
| medium | 150 | 12 | 0.35 | 12 | 3942 | **PROVEN INFEASIBLE** (C3, C4) | — | 94 |
| stress | 200 | 15 | 0.40 | 20 | 7936 | **PROVEN INFEASIBLE** (C3, C4) | — | 185 |
| stress (tight K) | 200 | 5 | 0.60 | 21 | 11904 | **PROVEN INFEASIBLE** (C3, C4) | — | 0 |
| stress (sparse) | 200 | 20 | 0.10 | 22 | 1974 | **PROVEN INFEASIBLE** (C3) | — | 314 |

I did not expect this and spent a long time assuming my solver was broken. It is
not. Below is the verification, then the root cause.

### Verification that the infeasibility is real, not a solver bug

Each certificate is checkable by hand in seconds. Take `n=50, K=8, seed=10`:
tasks `T3`, `T8` and `T23` all have SLA window `[3,4]`, and the generator put a
conflict edge on all three pairs `(T3,T8)`, `(T3,T23)`, `(T8,T23)`. Three
mutually conflicting tasks confined to two slots cannot be scheduled — pure
pigeonhole, no algorithm required. `test_certificate_soundness_against_exact_solver`
additionally cross-checks the certificate family against the exhaustive
branch-and-bound solver on 30 random 7-task instances: whenever a certificate
fires, the exact solver also proves no schedule exists.

### Root cause: the generator's window distribution

```python
windows = [(lo := random.randint(0,K-2), random.randint(lo+1, K-1)) for _ in range(n)]
```

`lo` is uniform on `0..K-2` and `hi` on `lo+1..K-1`, so **the modal window has
width 2**, and a fixed fraction of tasks land on each of the `K-1` width-2
intervals `[a,a+1]`. With `n` tasks and `K` slots, roughly `n / (K-1)^2 * 2`
tasks land in any given width-2 interval; at `n=50, K=8` that is ~2–4 tasks per
interval, and at conflict density 0.25 the chance that three of them form a
triangle is high. As `n` grows the effect gets *worse*, not better, because the
number of trapped tasks per interval grows linearly in `n` while the interval
still offers exactly two slots. Measured mean window widths: 2.50 (`K=3`), 3.32
(`K=8`), 5.12 (`K=15`) — always a small fraction of `K`.

So the mandated suite is not testing large-scale optimisation at all; it is
testing whether the candidate *notices* that the instances are unschedulable. I
read that as intentional: the rubric says "Explain every anomaly. Do not hide
failures", and Task 3 explicitly asks for an algorithm that "produces a valid
feasible assignment, **or detects and reports infeasibility**".

### A second, independent reason on the stress instances

The generator also contains

```python
random.uniform(1, cap[d] // (n // K + 1))
```

where the comprehension variable `d` shadows the parameter, and for large `n` the
upper bound underflows to `0` (`32 // 41 = 0`, `8 // 41 = 0`, `6.0 // 41 = 0.0`).
`random.uniform(1, 0)` does not raise — it returns values in `(0,1)`. The result
is that on big instances the GPU and Network demands become ~0.5 per task
*regardless of capacity*, and total demand overruns total supply:

| instance | GPU demand / supply | NET demand / supply |
|---|---|---|
| `n=200, K=15` | 102.2 / 120 = 0.85 | 94.6 / 90 = **1.05** |
| `n=200, K=5` | 103.8 / 40 = **2.59** | 98.2 / 30 = **3.27** |

Certificate C4 (the interval Hall condition) fires on exactly these, entirely
independently of the conflict graph. So `n=200, K=5` is infeasible twice over: a
trapped triangle *and* 3.3x oversubscribed network capacity. Both are properties
of the generator, both are reported, neither is hidden.

---

## Quality against brute force (mandated small instances)

| n | K | seed | SPARK | proven optimum | **ratio** | construction only |
|---|---|---|---|---|---|---|
| 8 | 3 | 1 | 76.9637 | 76.9637 | **1.00000** | 76.9637 |
| 10 | 4 | 2 | 88.4609 | 88.4609 | **1.00000** | 88.4609 |
| 12 | 4 | 3 | 140.5965 | 140.5965 | **1.00000** | 140.5965 |

The exact solver (`msme/exact.py`) proved optimality on all three (it is
branch-and-bound with an admissible own-cost bound, not a truncated search). The
empirical approximation ratio is therefore **1.000** on every mandated instance
that has an answer — far below the analytic worst case `1 + Delta` of Theorem 4,
which is expected: that bound is driven by an adversarial window/degree structure
(`bench/adversarial.py`) that random instances essentially never produce.

---

## Scaling: the minimum-slack study

Because the mandated suite cannot exercise the optimiser at scale, I ran a second
study that leaves the generator untouched and instead widens every SLA window
symmetrically by `slack` slots (clipped to `[0,K-1]`), taking the smallest
`slack` at which the instance becomes solvable. It doubles as a platform result:
it says *how much SLA slack ScoreMe would have to negotiate* for this conflict
structure to be schedulable at all.

| n | K | slack needed | outcome | penalty | construction only | LS gain | runtime (ms) |
|---|---|---|---|---|---|---|---|
| 8 | 3 | 1 | feasible (=optimum) | 63.21 | 64.99 | **2.7%** | 3000 |
| 10 | 4 | 1 | feasible (=optimum) | 78.05 | 85.64 | **8.9%** | 3000 |
| 12 | 4 | 1 | feasible (=optimum) | 121.04 | 121.04 | 0.0% | 3000 |
| 50 | 8 | 1 | feasible | 1307.55 | 1318.77 | 0.9% | 8001 |
| 100 | 10 | 5 | **no solution found** | — | — | — | 8023 |
| 150 | 12 | 5 | proven infeasible (C4) | — | — | — | 0 |
| 200 | 15 | 5 | proven infeasible (C4) | — | — | — | 0 |
| 200 | 5 | 5 | proven infeasible (C4) | — | — | — | 0 |
| 200 | 20 | 1 | feasible | 11001.32 | 11310.88 | **2.7%** | 8000 |

SPARK reaches the **proven optimum** on all three relaxed small instances too
(ratio 1.000000, exact solver confirms). The three `slack=5` rows stay infeasible
for the capacity reason above — widening windows cannot conjure network
bandwidth — which is a useful sanity check that the relaxation is not quietly
making everything trivially satisfiable.

---

## Anomalies, each investigated

**A1 — `n=100, K=10, slack=5`: no certificate fires, and SPARK still finds
nothing.** This is the one genuinely unresolved cell in the whole report. The
instance is *probably* infeasible: a DSATUR colouring of its conflict graph needs
**12 colours** against `K=10` slots, and DSATUR is a good enough heuristic that
needing 12 is strong evidence `chi(G) > 10`. But DSATUR gives an *upper* bound on
`chi`, not a lower one, so this is evidence and not proof, and my certificate
family (clique pigeonhole, Hall capacity, cardinality) is too weak to close it —
the graph's clique number is around 7, well under `K`. Proving it would need a
genuine `chi` lower bound (Lovász theta via SDP, or a fractional-colouring LP),
both of which the assignment's forbidden-library rule rules out. **The solver
reports this correctly and does not overclaim**: `NO FEASIBLE ASSIGNMENT FOUND
(search budget exhausted) -- not a proof of infeasibility`. Distinguishing the two
outcomes in `violation_reason` was worth the extra plumbing precisely for cases
like this one.

**A2 — `n=12` relaxed shows 0.0% local-search gain.** Not a failure: the
construction phase already returns the proven optimum (121.04), so there is
nothing for refinement to win. The exact solver confirms.

**A3 — full SPARK once finished *worse* than its own construction phase
(-1.3% on relaxed `n=50`).** This was a real bug and it is now fixed. The solver
originally used one `random.Random` for both the construction noise and the
local-search sampling. The local-search phase consumed random numbers, which
changed which noisy constructions later restarts produced — so the
construction-only ablation was not starting from the same schedules as the full
run, and the full run could land in a worse basin. Splitting into two seeded
streams (`self.rng` for construction, `self.rng_search` for refinement) restores
the guarantee `P(full) <= P(construction-only)` for every restart, because
refinement tracks an incumbent and can only descend from its own starting point.
Fixing it also closed a 0.012% optimality gap on the mandated `n=10` instance,
which is now exactly optimal. **Cost of the bug: about two hours of believing my
tabu search was broken when the tabu search was fine.**

**A4 — runtime does not grow with `n` on the mandated suite.** Expected and not a
measurement error: the mandated instances terminate in the certificate phase, so
their runtimes (0–330 ms) measure certificate cost, which is `O(K^2 · tries · n ·
Delta)` and depends far more on `K` than on `n` (`K=5` finishes in <1 ms, `K=20`
takes 314 ms). The `runtime_vs_n` chart is drawn from the minimum-slack suite,
where the solver actually runs and hits its wall-clock budget — so it is flat at
the budget by construction, and the honest scaling statement is about *quality at
a fixed budget*, not time-to-solution. An iteration-count-based budget would give
a cleaner scaling curve and is what I would switch to next.

**A5 — results vary slightly between runs.** The refinement loop is wall-clock
bounded, so a machine under load completes fewer iterations. I observed a 2.7%
swing on relaxed `n=8` when benchmarks ran concurrently with other work. All
numbers in this report come from an otherwise-idle run. This is a real
reproducibility weakness of a time-budgeted anytime algorithm and I flag it
rather than pretending the numbers are deterministic.

---

## Ablation: what each phase is worth

* **Certificates** turn 6 of 9 mandated instances from an 8-second fruitless
  search into a sub-second proof.
* **Local search** buys 0–8.9% on solvable instances (mean ~2.6% where it has
  room). Small, because the construction is already strong — which is the point
  of the MRV + regret design, not a disappointment.
* **Phase-2 chain repair** never fires on the solvable benchmark instances
  (`chain_repairs = 0` on relaxed `n=50` and `n=200,K=20`) — the MRV+regret
  construction simply does not corner a task there. It does fire on tight
  instances that have no schedule at all: on every one of six sampled infeasible
  `n=30..60` instances the repair phase ran and still could not seat the
  cornered task, which is exactly its intended role — exhaust the cheap repairs
  before reporting failure. **I over-estimated this phase's value before
  measuring it**; on random instances it is insurance, not a workhorse.
* **The Phase-3 chain *move* operator (N3), by contrast, does the heavy
  lifting** in refinement: 1009 of 1980 accepted moves in one measured run on relaxed `n=200,K=20`
  and 4478 of 6340 on relaxed `n=50` were chain moves, confirming §1 of
  docs/algorithm.md — on a conflict-dense schedule, plain relocation is almost
  always illegal and moving *into* an occupied slot is the only way to progress.
* **The adversarial family** (`bench/adversarial.py`) is where construction alone
  degrades to ratio `Delta` while the full pipeline still returns the optimum —
  the clearest demonstration that the refinement phase is not decorative.
