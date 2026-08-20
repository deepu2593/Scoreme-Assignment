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
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
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


def _svg_line_chart(path, series, xlabel, ylabel, title, logy=False):
    """Minimal dependency-free SVG line chart.

    Why not just commit the matplotlib SVG: matplotlib emits ~19 KB per chart
    (marker <defs>, per-tick groups, metadata), which is noise in a repository
    and unreadable in a diff.  This writer produces ~3 KB of plain elements that
    render identically on GitHub.  The matplotlib PNGs are still written for
    local viewing -- this is a packaging choice, not an analysis one.
    """
    W, H = 640, 400
    ml, mr, mt, mb = 70, 20, 40, 55
    pw, ph = W - ml - mr, H - mt - mb
    xs = [x for _, pts in series for x, _ in pts]
    ys = [y for _, pts in series for _, y in pts]
    if logy:
        ys = [math.log10(max(y, 1e-9)) for y in ys]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    y0, y1 = min(y0, 0.0), y1 * 1.08 + 1e-9
    if x1 == x0:
        x1 = x0 + 1

    def px(x):
        return ml + pw * (x - x0) / (x1 - x0)

    def py(y):
        v = math.log10(max(y, 1e-9)) if logy else y
        return mt + ph - ph * (v - y0) / (y1 - y0 or 1)

    colours = ["#1f77b4", "#d62728", "#2ca02c"]
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
           f'viewBox="0 0 {W} {H}" font-family="Helvetica,Arial,sans-serif" '
           f'font-size="12">',
           f'<rect width="{W}" height="{H}" fill="white"/>',
           f'<text x="{W/2:.0f}" y="22" text-anchor="middle" font-size="14" '
           f'font-weight="bold">{title}</text>']
    for f in range(6):
        gy = mt + ph * f / 5
        val = y1 - (y1 - y0) * f / 5
        label = f"1e{val:.1f}" if logy else (f"{val:.0f}" if abs(val) >= 10 else f"{val:.2f}")
        out.append(f'<line x1="{ml}" y1="{gy:.1f}" x2="{ml+pw}" y2="{gy:.1f}" '
                   f'stroke="#ccc" stroke-width="0.6"/>')
        out.append(f'<text x="{ml-8}" y="{gy+4:.1f}" text-anchor="end" '
                   f'fill="#444">{label}</text>')
    for x in sorted(set(xs)):
        out.append(f'<text x="{px(x):.1f}" y="{mt+ph+18:.0f}" text-anchor="middle" '
                   f'fill="#444">{x:g}</text>')
    out.append(f'<line x1="{ml}" y1="{mt+ph}" x2="{ml+pw}" y2="{mt+ph}" stroke="#333"/>')
    out.append(f'<line x1="{ml}" y1="{mt}" x2="{ml}" y2="{mt+ph}" stroke="#333"/>')
    out.append(f'<text x="{ml+pw/2:.0f}" y="{H-14}" text-anchor="middle">{xlabel}</text>')
    out.append(f'<text x="16" y="{mt+ph/2:.0f}" text-anchor="middle" '
               f'transform="rotate(-90 16 {mt+ph/2:.0f})">{ylabel}</text>')
    for idx, (name, pts) in enumerate(series):
        c = colours[idx % len(colours)]
        dash = ' stroke-dasharray="6,4"' if idx else ""
        d = " ".join(("M" if k == 0 else "L") + f"{px(x):.1f},{py(y):.1f}"
                     for k, (x, y) in enumerate(sorted(pts)))
        out.append(f'<path d="{d}" fill="none" stroke="{c}" stroke-width="2"{dash}/>')
        for x, y in pts:
            out.append(f'<circle cx="{px(x):.1f}" cy="{py(y):.1f}" r="3.5" fill="{c}"/>')
        ly = mt + 14 + idx * 18
        out.append(f'<line x1="{ml+pw-150}" y1="{ly}" x2="{ml+pw-120}" y2="{ly}" '
                   f'stroke="{c}" stroke-width="2"{dash}/>')
        out.append(f'<text x="{ml+pw-114}" y="{ly+4}" fill="#333">{name}</text>')
    out.append("</svg>")
    with open(path, "w") as fh:
        fh.write("\n".join(out) + "\n")


def make_charts(rows):
    solved = [r for r in rows if r["relaxed"]["feasible"]]
    pen = [(r["n"], r["relaxed"]["penalty"]) for r in solved]
    gre = [(r["n"], r["relaxed"]["greedy_only_penalty"]) for r in solved
           if r["relaxed"].get("greedy_only_penalty")]
    rt = [(r["n"], r["relaxed"]["runtime_ms"]) for r in solved]

    _svg_line_chart(os.path.join(RESULTS, "penalty_vs_n.svg"),
                    [("SPARK (full)", pen), ("construction only", gre)],
                    "n (tasks)", "penalty P(sigma)",
                    "Penalty vs instance size (minimum-slack suite)")
    _svg_line_chart(os.path.join(RESULTS, "runtime_vs_n.svg"),
                    [("SPARK wall-clock", rt)],
                    "n (tasks)", "runtime (ms)",
                    "Runtime vs instance size (anytime, budget-capped)")

    # matplotlib PNGs for local viewing (gitignored)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot([x for x, _ in pen], [y for _, y in pen], "o-", label="SPARK (full)")
    if gre:
        ax.plot([x for x, _ in gre], [y for _, y in gre], "s--",
                label="construction only (ablation)")
    ax.set_xlabel("n (tasks)")
    ax.set_ylabel("penalty P(sigma)")
    ax.set_title("Penalty vs instance size (minimum-slack relaxed suite)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "penalty_vs_n.png"), dpi=140)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot([x for x, _ in rt], [y for _, y in rt], "o-", color="crimson")
    ax.set_xlabel("n (tasks)")
    ax.set_ylabel("runtime (ms)")
    ax.set_title("Runtime vs instance size (8 s budget, anytime solver)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(RESULTS, "runtime_vs_n.png"), dpi=140)


if __name__ == "__main__":
    main()
