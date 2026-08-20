# SPARK — MSME Credit Pipeline Slot Scheduler

Solution to the ScoreMe *Advanced Systems Design* assignment: assign `n` credit
pipeline tasks to `K` discrete compute slots subject to conflict avoidance (F1),
4-dimensional resource capacity (F2) and SLA windows (F3), minimising a weighted
penalty.

**SPARK** = **S**lack-**P**rioritised **A**daptive **R**epacking with
**K**empe-chain repair.

```
Phase 0  CERTIFY    sound polynomial infeasibility certificates (C1–C5)
Phase 1  CONSTRUCT  window-saturation MRV greedy, regret-aware slot choice
Phase 2  REPAIR     depth-bounded ejection chains, branching on WHY a slot died
Phase 3  REFINE     feasibility-preserving tabu search over relocate/swap/chain
```

## Quick start

```bash
pip install -r requirements.txt

python run.py --n 12 --K 4 --density 0.5 --seed 3 --exact   # solve + prove optimality
python run.py --input my_instance.json --output solution.json
python -m pytest msme/tests -q                              # 17 unit tests
python bench/run_benchmarks.py                              # full benchmark suite + charts
python bench/adversarial.py                                 # tight approximation-bound demo
```

Output JSON carries exactly the mandated keys — `assignment`, `penalty`,
`runtime_ms`, `feasible`, `violation_reason` — plus `penalty_breakdown`,
`utilisation` and `stats` as diagnostics.

## Headline results

* **All 9 mandated benchmark instances run.** Six of them are **provably
  infeasible**, and SPARK proves it in 0–330 ms without searching (the generator
  traps conflict triangles inside 2-slot SLA windows). See
  [docs/benchmarks.md](docs/benchmarks.md).
* On every mandated instance that *is* solvable, SPARK returns the
  **brute-force-proven optimum** — empirical approximation ratio **1.00000**.
* Proved: soundness (unconditional), completeness on the *slack* class, and
  `P_base(SPARK-C) <= (1 + max_i Delta^W_i/(l_i+1)) * P_base(OPT)`.
* The bound's tightness is demonstrated by a hand-built adversarial family that
  drives the construction phase to ratio exactly `Delta` — and the shipped
  pipeline still recovers the optimum on it.
* One cell is honestly unresolved (`n=100, K=10, slack=5`): neither a schedule
  nor a certificate. Documented as unresolved rather than papered over.

## Repository layout

| path | what |
|---|---|
| `msme/instance.py` | instance model + the **verbatim, unmodified** provided generator |
| `msme/penalty.py` | Task 2 — the extended penalty `P(sigma)` and its rationale |
| `msme/state.py` | mutable schedule with exact `O(d)` incremental penalty deltas |
| `msme/certificates.py` | C1–C5 sound infeasibility certificates |
| `msme/spark.py` | Task 3 — the algorithm, plus the independent `verify` audit |
| `msme/exact.py` | branch-and-bound ground truth for small instances |
| `msme/tests/` | 17 unit tests incl. all four mandated edge cases |
| `run.py` | CLI |
| `bench/` | benchmark suite and the adversarial tightness demo |
| `docs/proofs.md` | Task 1 (NP-hardness) + Task 4 (guarantees, ratio, tight example) |
| `docs/algorithm.md` | Task 3 — pseudocode, line-level justification, rejected alternatives |
| `docs/penalty.md` | Task 2 — penalty design and calibration |
| `docs/benchmarks.md` | Task 6 — results, ablations, every anomaly investigated |
| `docs/design_journal.md` | Task 7 — **template only; must be written by the candidate** |
| `AI_USAGE_LOG.md` | **template only; must be completed honestly by the candidate** |

## Constraints honoured

* No OR-Tools, PuLP, CPLEX, Gurobi, Z3, networkx, or any SAT/LP solver. Runtime
  dependencies are the standard library only; `numpy`/`matplotlib` are used for
  charts and `pytest` for tests.
* The provided instance generator in `msme/instance.py` is reproduced
  byte-for-byte, quirks included; its two defects are documented and analysed,
  never patched.

## ⚠️ Before you submit this

Tasks 7 (Design Journal) and 8 (Viva Voce) cannot be delegated, and the
assignment's integrity policy is explicit that inability to defend any part of
this work scores zero for the *entire* submission. Read
[`docs/design_journal.md`](docs/design_journal.md) and
[`AI_USAGE_LOG.md`](AI_USAGE_LOG.md) before submitting.
