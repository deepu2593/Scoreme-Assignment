# Task 7 — Design Journal  **[TEMPLATE — NOT WRITTEN]**

> **Read this first.**
>
> This file is deliberately empty of content. Task 7 is worth 20 points and the
> rubric awards them for *"specific, personal, non-generic"* reflection; the
> assignment states plainly that AI cannot write it and that evaluators identify
> AI-generated reflection by the absence of concrete personal observation.
> Generic language without specifics scores **0**.
>
> More importantly: Task 8 is a 20-minute viva with a **zero-for-everything**
> penalty if you cannot defend your submission. Writing this section yourself is
> how you find out whether you can.
>
> The prompts below are scaffolding — the four questions the rubric asks, plus
> pointers to real, specific things in this repository you can react to. Delete
> every prompt as you replace it with your own words.

---

## 1. The single hardest design decision

*Name the algorithm step, the trade-off, and the alternative you rejected.*

Real candidates from this codebase, if any of them match what you actually
wrestled with (verify each yourself before claiming it — check the code, re-run
the experiment):

* **The ranking key in `construct` (`msme/spark.py`, KEY 1/2/3).** MRV first or
  urgency first? Ranking by weight puts the important task early but corners the
  constrained one; ranking by palette size does the opposite. Look at
  `bench/adversarial.py` — the urgency tie-break is worth exactly one unit of
  the approximation ratio there.
* **Whether the repair phase should backtrack or give up** (`_seat`, chain depth
  3). I swept depth {1,2,3,4} and it changed nothing (9/55 solved at every
  depth) — a design decision that turned out not to be a decision at all. If you
  cite this, cite it honestly as a negative result.
* **Whether local search may visit infeasible states.** SPARK says no; the
  rejected-alternatives section of `docs/algorithm.md` explains why, but the
  cost is that whole regions of the space are unreachable.

## 2. Where it failed empirically, and what a week more would buy

*Name the specific benchmark instance and the failure mode.*

Documented failures actually in this repo — pick the ones you personally
reproduced:

* `n=100, K=10, slack=5`: **no schedule found and no certificate.** DSATUR needs
  12 colours against `K=10`, so it is probably infeasible, but my certificate
  family cannot prove it (docs/benchmarks.md, A1).
* The **RNG-sharing bug** (A3): full SPARK once finished 1.3% *worse* than its
  own construction phase. Re-run it, understand it, then describe it in your own
  words — this is the strongest concrete failure story in the project.
* The **adversarial family**: SPARK's construction degrades to ratio `Delta`.
* **Wall-clock budgeting makes results non-reproducible** (A5) — a 2.7% swing
  under machine load.

## 3. A real ScoreMe production system where this problem class appears

*Do your own homework here — a specific system, named, with specifics.* The
assignment lists NiFi pipelines, Kafka consumer groups, the OCR GPU cluster and
the bureau API gateway. Whichever you choose, be concrete about what maps to
what: what is a slot, what is a conflict edge, what is `r`, what is `C`, what is
the SLA window, and what would break if your algorithm ran there.

## 4. What surprised you

*Only you can answer this.* If it helps: the thing most likely to surprise a
reader of this repo is that **six of the nine graded instances have no solution
at all**, and that finding it out required building infeasibility certificates
rather than a better optimiser. But do not write that down unless it genuinely
surprised *you* — write what actually did.

---

## Checklist before you submit

- [ ] Every claim above replaced with your own words and your own experiments
- [ ] You re-ran `python bench/run_benchmarks.py` yourself and read the output
- [ ] You can whiteboard the pseudocode in `docs/algorithm.md` from memory
- [ ] You can hand-trace SPARK on a fresh 6-node instance (try the toy instance
      in §3.3 of the assignment, then invent one)
- [ ] You can answer *"what if I add a 5th resource dimension?"*
      (hint: `d` is never hard-coded — check `Instance.d` and every loop over
      `range(inst.d)`; then say what changes in the **certificates** and in the
      **binding-dimension eviction rule**)
- [ ] You can answer *"what if two slots have different capacities?"*
      (hint: already supported — `capacities` is per-slot; but check what it does
      to the `balance` term's mean and to `delta_move`'s mean correction)
- [ ] You can explain any randomly chosen line of `msme/spark.py`
- [ ] `AI_USAGE_LOG.md` completed honestly
