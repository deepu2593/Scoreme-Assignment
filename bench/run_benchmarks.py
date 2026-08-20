#!/usr/bin/env python3
"""Task 6 -- benchmark harness.

Runs (a) the nine mandated instances, (b) the same instances with SLA windows
relaxed by one slot each way (a supplementary study, because six of the nine
mandated instances are provably infeasible and would otherwise leave the
optimiser untested at scale), and (c) a construction-only ablation that isolates
how much the tabu phase actually buys.

Emits results/benchmarks.json, results/benchmarks.md and two charts.
"""

from __future__ import annotations

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
# Emit SVG text as <text> elements rather than embedded glyph outlines: the
# charts are committed to the repository, and outline-embedding inflates them
# from ~4 KB to ~40 KB for no visual gain.
matplotlib.rcParams["svg.fonttype"] = "none"
import matplotlib.pyplot as plt

from msme.certificates import check_certificates
from msme.exact import solve_exact
from msme.instance import Instance
from msme.spark import Spark, verify
from run import relax_windows

SUITE = [
    ("small", 8, 3, 0.30, 1),
    ("small", 10, 4, 0.40, 2),
    ("small", 12, 4, 0.50, 3),
    ("medium", 50, 8, 0.25, 10),
    ("medium", 100, 10, 0.30, 11),
    ("medium", 150, 12, 0.35, 12),
    ("stress", 200, 15, 0.40, 20),
    ("stress (tight K)", 200, 5, 0.60, 21),
    ("stress (sparse)", 200, 20, 0.10, 22),
]

RESULTS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


def run_one(inst, budget_ms, local_search=True, seed=1):
    t0 = time.perf_counter()
    res = Spark(inst, seed=seed, time_budget_ms=budget_ms,
                local_search=local_search).solve()
    wall = int((time.perf_counter() - t0) * 1000)
    if res.feasible:
        ok, why = verify(inst, res.sigma)
        assert ok, f"solver returned an invalid schedule: {why}"
    return res, wall


def main():
    os.makedirs(RESULTS, exist_ok=True)
    rows = []
    for group, n, K, den, seed in SUITE:
        inst = Instance.random(n, K, den, seed)
        budget = 3000 if n <= 12 else 8000
        cert = check_certificates(inst)
        res, wall = run_one(inst, budget)
        row = {"group": group, "n": n, "K": K, "density": den, "seed": seed,
               "edges": len(inst.conflicts),
               "certificate": cert, "feasible": res.feasible,
               "penalty": res.penalty if res.feasible else None,
               "runtime_ms": res.runtime_ms, "wall_ms": wall,
               "reason": res.reason, "stats": res.stats}
        if n <= 12:
            sigma_opt, P_opt, proved = solve_exact(inst, time_limit_s=180)
            row["exact_penalty"] = P_opt if sigma_opt else None
            row["exact_proved"] = proved
            row["ratio"] = (res.penalty / P_opt) if (sigma_opt and res.feasible) else None
            greedy, _ = run_one(inst, budget, local_search=False)
            row["greedy_only_penalty"] = greedy.penalty if greedy.feasible else None

        # Minimum-slack study: widen every SLA window symmetrically by r slots
        # until the instance becomes solvable.  This answers the question the
        # mandated suite cannot ("how does SPARK scale?") without touching the
        # provided generator, and it doubles as a platform-capacity result: it
        # says how much SLA slack ScoreMe would have to negotiate for this
        # conflict structure to be schedulable at all.
        row["relaxed"] = {"certificate": "not reached", "feasible": False,
                          "penalty": None, "runtime_ms": 0,
                          "greedy_only_penalty": None, "reason": "", "slack": None}
        for slack in range(1, 6):
            rel = relax_windows(inst, slack)
            rcert = check_certificates(rel)
            if rcert:
                row["relaxed"] = {"certificate": rcert, "feasible": False, "slack": slack,
                                  "penalty": None, "runtime_ms": 0,
                                  "greedy_only_penalty": None, "reason": rcert}
                continue
            rres, _ = run_one(rel, budget)
            rgreedy, _ = run_one(rel, budget, local_search=False)
            row["relaxed"] = {"certificate": None, "feasible": rres.feasible,
                              "slack": slack,
                              "penalty": rres.penalty if rres.feasible else None,
                              "runtime_ms": rres.runtime_ms,
                              "greedy_only_penalty": (rgreedy.penalty
                                                      if rgreedy.feasible else None),
                              "reason": rres.reason}
            if rres.feasible:
                if n <= 12:
                    s_o, P_o, pr = solve_exact(rel, time_limit_s=180)
                    row["relaxed"]["exact_penalty"] = P_o if s_o else None
                    row["relaxed"]["exact_proved"] = pr
                    row["relaxed"]["ratio"] = (rres.penalty / P_o) if s_o else None
                break
        rows.append(row)
        print(f"done n={n} K={K}: feasible={res.feasible} "
              f"relaxed_feasible={row['relaxed']['feasible']} "
              f"slack={row['relaxed']['slack']}")

    with open(os.path.join(RESULTS, "benchmarks.json"), "w") as fh:
        json.dump(rows, fh, indent=2)
    write_markdown(rows)
    make_charts(rows)


def write_markdown(rows):
    L = ["# Benchmark results (auto-generated by bench/run_benchmarks.py)", "",
         "## A. The nine mandated instances", "",
         "| group | n | K | density | seed | edges | outcome | penalty | runtime (ms) |",
         "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        if r["certificate"]:
            outcome = "PROVEN INFEASIBLE"
        elif r["feasible"]:
            outcome = "feasible"
        else:
            outcome = "no solution found"
        pen = f"{r['penalty']:.2f}" if r["penalty"] is not None else "--"
        L.append(f"| {r['group']} | {r['n']} | {r['K']} | {r['density']} | {r['seed']} | "
                 f"{r['edges']} | {outcome} | {pen} | {r['runtime_ms']} |")
    L += ["", "## B. Small instances vs brute-force optimum", "",
          "| n | K | seed | SPARK | optimum | ratio | construction-only |",
          "|---|---|---|---|---|---|---|"]
    for r in rows:
        if "ratio" not in r:
            continue
        g = r.get("greedy_only_penalty")
        L.append(f"| {r['n']} | {r['K']} | {r['seed']} | {r['penalty']:.4f} | "
                 f"{r['exact_penalty']:.4f} | {r['ratio']:.5f} | "
                 f"{g:.4f} |" if g else
                 f"| {r['n']} | {r['K']} | {r['seed']} | {r['penalty']:.4f} | "
                 f"{r['exact_penalty']:.4f} | {r['ratio']:.5f} | -- |")
    L += ["", "## C. Minimum-slack study (windows widened by `slack` slots each side)", "",
          "| n | K | slack needed | outcome | penalty | construction-only | LS gain | runtime (ms) |",
          "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        rel = r["relaxed"]
        if rel["certificate"]:
            outcome = "PROVEN INFEASIBLE"
        elif rel["feasible"]:
            outcome = "feasible"
        else:
            outcome = "no solution found"
        pen = f"{rel['penalty']:.2f}" if rel["penalty"] is not None else "--"
        g = rel.get("greedy_only_penalty")
        gain = (f"{100*(g-rel['penalty'])/g:.1f}%"
                if (g and rel["penalty"] is not None) else "--")
        L.append(f"| {r['n']} | {r['K']} | {rel.get('slack')} | {outcome} | {pen} | "
                 f"{f'{g:.2f}' if g else '--'} | {gain} | {rel['runtime_ms']} |")
    with open(os.path.join(RESULTS, "benchmarks.md"), "w") as fh:
        fh.write("\n".join(L) + "\n")


def make_charts(rows):
    solved = [r for r in rows if r["relaxed"]["feasible"]]
    ns = [r["n"] for r in solved]
    pen = [r["relaxed"]["penalty"] for r in solved]
    gre = [r["relaxed"].get("greedy_only_penalty") for r in solved]
    rt = [r["relaxed"]["runtime_ms"] for r in solved]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(ns, pen, "o-", label="SPARK (full)")
    if all(g is not None for g in gre):
        ax.plot(ns, gre, "s--", label="construction only (ablation)")
    ax.set_xlabel("n (tasks)")
    ax.set_ylabel("penalty P(sigma)")
    ax.set_title("Penalty vs instance size (minimum-slack relaxed suite)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "penalty_vs_n.svg"))
    fig.savefig(os.path.join(RESULTS, "penalty_vs_n.png"), dpi=140)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(ns, rt, "o-", color="crimson")
    ax.set_xlabel("n (tasks)")
    ax.set_ylabel("runtime (ms)")
    ax.set_title("Runtime vs instance size (8 s budget, anytime solver)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "runtime_vs_n.svg"))
    fig.savefig(os.path.join(RESULTS, "runtime_vs_n.png"), dpi=140)


if __name__ == "__main__":
    main()
