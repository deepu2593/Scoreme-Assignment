#!/usr/bin/env python3
"""CLI entry point -- Task 5 deliverable.

    python run.py --n 50 --K 8 --density 0.25 --seed 10
    python run.py --input instance.json --output solution.json
    python run.py --n 12 --K 4 --density 0.5 --seed 3 --exact   # compare to optimum

Output JSON keys (exactly as specified by the assignment):
    assignment       {task_id: slot}
    penalty          float
    runtime_ms       int
    feasible         bool
    violation_reason string (empty when feasible)
plus `penalty_breakdown`, `stats` and `utilisation` as diagnostic extras.
"""

from __future__ import annotations

import argparse
import json
import sys

from msme.exact import solve_exact
from msme.instance import Instance
from msme.penalty import PenaltyModel, PenaltyWeights
from msme.spark import Spark


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="SPARK scheduler for the MSME credit pipeline")
    p.add_argument("--n", type=int, help="number of tasks (generated instance)")
    p.add_argument("--K", type=int, help="number of slots")
    p.add_argument("--density", type=float, default=0.3, help="conflict density")
    p.add_argument("--seed", type=int, default=42, help="generator seed")
    p.add_argument("--input", type=str, help="path to an instance JSON file")
    p.add_argument("--output", type=str, help="write the solution JSON here")
    p.add_argument("--dump-instance", type=str, help="write the generated instance here")
    p.add_argument("--budget-ms", type=int, default=2000, help="solver time budget")
    p.add_argument("--restarts", type=int, default=4)
    p.add_argument("--solver-seed", type=int, default=1)
    p.add_argument("--lam-balance", type=float, default=1.0)
    p.add_argument("--lam-sla", type=float, default=1.0)
    p.add_argument("--lam-gpu", type=float, default=0.5)
    p.add_argument("--relax", type=int, default=0,
                   help="widen every SLA window by this many slots on each side "
                        "(supplementary study only; the generator is not modified)")
    p.add_argument("--exact", action="store_true",
                   help="also solve to proven optimality (small n only)")
    p.add_argument("--quiet", action="store_true")
    return p


def relax_windows(inst: Instance, extra: int) -> Instance:
    """Widen every SLA window by `extra` slots each way, clipped to [0, K-1].

    This is NOT a modification of the provided generator; it is a separate,
    clearly-labelled transformation used in docs/benchmarks.md to study how the
    solver behaves once the generator's pathologically narrow windows (which
    make 6 of the 9 graded instances provably infeasible) are loosened.
    """
    obj = inst.to_dict()
    obj["windows"] = [[max(0, lo - extra), min(inst.K - 1, hi + extra)]
                      for lo, hi in inst.windows]
    return Instance.from_dict(obj)


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.input:
        inst = Instance.load(args.input)
    elif args.n and args.K:
        inst = Instance.random(args.n, args.K, args.density, args.seed)
    else:
        print("error: give either --input or both --n and --K", file=sys.stderr)
        return 2
    if args.relax:
        inst = relax_windows(inst, args.relax)
    if args.dump_instance:
        inst.save(args.dump_instance)

    lam = PenaltyWeights(1.0, args.lam_balance, args.lam_sla, args.lam_gpu)
    res = Spark(inst, lam=lam, seed=args.solver_seed,
                time_budget_ms=args.budget_ms, restarts=args.restarts).solve()

    out = {
        "assignment": ({inst.tasks[i]: s for i, s in enumerate(res.sigma)}
                       if res.sigma else {}),
        "penalty": res.penalty if res.feasible else None,
        "runtime_ms": res.runtime_ms,
        "feasible": res.feasible,
        "violation_reason": res.reason,
        "stats": res.stats,
    }
    if res.feasible:
        model = PenaltyModel(inst, lam)
        out["penalty_breakdown"] = model.breakdown(res.sigma)
        L = model.load_matrix(res.sigma)
        out["utilisation"] = [[round(L[s][k] / inst.capacities[s][k], 4)
                               for k in range(inst.d)] for s in range(inst.K)]
    if args.exact:
        sigma_opt, P_opt, proved = solve_exact(inst, lam, time_limit_s=120)
        out["exact"] = {"penalty": P_opt if sigma_opt else None,
                        "proved_optimal": proved,
                        "ratio": (res.penalty / P_opt) if (sigma_opt and res.feasible
                                                           and P_opt > 0) else None}
    text = json.dumps(out, indent=2)
    if args.output:
        with open(args.output, "w") as fh:
            fh.write(text)
    if not args.quiet:
        print(text)
    return 0 if res.feasible else 1


if __name__ == "__main__":
    raise SystemExit(main())
