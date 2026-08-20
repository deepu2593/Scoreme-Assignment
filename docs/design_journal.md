# Task 7 — Design Journal

> **DRAFT — read before submitting.**
>
> Every incident, number and failure below actually happened during this
> project and is verifiable against the repository: the commit history, the
> anomaly log in `docs/benchmarks.md`, and the benchmark output you can
> regenerate yourself. Nothing here is invented.
>
> But the *voice* is not yours yet. Task 7 is graded on personal reflection, and
> the viva will ask you to expand on any sentence in it. Work through this file,
> re-run the experiments it cites, cut what does not match your own experience,
> and rewrite the rest in your own words. Then delete this box.

---

## 1. The single hardest design decision

**The ranking key in `construct` — whether the greedy picks tasks by how
*constrained* they are or by how *important* they are.**

The two orderings pull in opposite directions and both are defensible. Ranking
by weight (`w_i`) puts the Tier-1 bureau pull into an early slot, which is what
the penalty function rewards. Ranking by surviving palette size — MRV, fewest
legal slots first — protects the tasks that are one placement away from having
nowhere to go. Every early version I tried was one or the other, and each broke
in a way the other did not: weight-first cornered tasks that then needed the
repair phase, palette-first produced feasible schedules with obviously silly
delay costs.

What made it hard is that the two failure modes are not commensurable. Cornering
a task risks *infeasibility*, and infeasibility is not a worse score, it is no
answer at all. A bad delay cost is merely expensive. So I ordered them
lexicographically rather than blending them into a weighted score: palette size
first (existence), saturation second, and `w_i/(u_i-l_i+1)` — weight per unit of
slack — only as a tie-break.

**The alternative I rejected** was a single blended score,
`alpha*|F_i| + beta*w_i`, with tuned coefficients. I rejected it because the two
quantities have no common unit, so `alpha` and `beta` would have to be retuned
per instance shape, and because a blend lets a heavy task outrank a cornered one
whenever the weights happen to be large — which is exactly the case where
cornering is unrecoverable.

The tie-break turned out to matter more than I expected. In
`bench/adversarial.py` I built a family designed to defeat the construction
phase, and it drives the ratio to exactly `Delta` rather than the proved bound
`1 + Delta`. The missing unit is bought back by that third key: at the last step
the heavy task and the final blocker tie on palette size and saturation, and
urgency breaks the tie in favour of the heavy task. I tried to defeat that by
raising the blockers' weights, and it backfires — it inflates the optimum faster
than it inflates the greedy. I could not close the gap in either direction, and
`docs/proofs.md` says so rather than claiming a tightness I did not prove.

## 2. Where it failed empirically, and what another week would buy

Three real failures, in increasing order of how much they annoy me.

**(a) `n=100, K=10` with SLA windows widened by 5 — unresolved.** No certificate
fires, and SPARK finds no schedule inside its budget. I believe it is infeasible:
a DSATUR colouring of that conflict graph needs 12 colours against 10 slots. But
DSATUR gives an *upper* bound on the chromatic number, not a lower one, so that
is evidence and not proof, and my certificate family cannot close it — the
graph's clique number is around 7, well under `K`, so the pigeonhole argument
never fires. Proving it needs a real `chi` lower bound (Lovász theta, or a
fractional-colouring LP), and both are ruled out by the forbidden-library list.
The solver reports it honestly as `NO FEASIBLE ASSIGNMENT FOUND ... not a proof
of infeasibility`, which is the whole reason I plumbed two distinct failure
strings through `violation_reason` instead of one.

**(b) The RNG bug — my worst mistake on this project.** On the relaxed `n=50`
instance, full SPARK finished **1.3% worse than its own construction phase**.
That should be impossible: refinement tracks an incumbent and restores it, so it
cannot descend below where it started. I spent a long stretch convinced the tabu
search was corrupting the incumbent, and read that code several times looking for
a bug that was not there.

The actual cause was one shared `random.Random`. The refinement phase consumed
random numbers, which shifted the stream, which changed which *noisy
constructions* later restarts produced. So the construction-only ablation was
never starting from the same schedules as the full run, and the comparison I was
staring at was not a comparison at all. Splitting into two seeded streams
(`self.rng` for construction, `self.rng_search` for refinement) fixed it, and as
a side effect closed a 0.012% optimality gap on the mandated `n=10` instance,
which is now exactly optimal. The lesson I actually took: **a benchmark that
compares two configurations is only valid if the randomness they share is
controlled**, and I now treat "variant A beat variant B" as suspect until I can
show they saw identical inputs.

**(c) The chain-depth sweep came back empty.** I had assumed `chain_depth` was a
quality-versus-time knob and wrote it up that way. Before submitting I actually
measured it: depths {1,2,3,4} over 55 generator instances that survive the
certificates and still corner a task, solving an identical **9/55 at every
depth**, with no meaningful runtime difference. So a parameter I had described as
"tuned empirically" is not tuned at all. I rewrote the docstring to say so. This
also demoted the whole repair phase in my mental model — `chain_repairs` is 0 on
every solvable benchmark instance, so Phase 2 is insurance, not a workhorse. The
Phase-3 *chain move* is the one doing real work: 1009 of 1980 accepted moves on
relaxed `n=200, K=20`.

**With another week**, in priority order: (i) a real chromatic lower bound to
close case (a) — most likely a hand-rolled fractional-colouring bound, since
external LP solvers are forbidden; (ii) switch the refinement budget from
wall-clock to iteration count, because time-budgeting makes results
non-reproducible (I measured a 2.7% swing on relaxed `n=8` purely from running
benchmarks while other work was on the machine); (iii) extend the chain operator
to *capacity*-blocked slots — today it only ejects a single conflicting blocker,
so a schedule whose improvement requires a 3-way rotation across two full slots
is unreachable.

## 3. Where this problem class appears in production at ScoreMe

**The OCR GPU cluster**, and the mapping is close to exact:

| model | production reality |
|---|---|
| slot `s` | one scheduling cycle on the inference cluster (~30 s) |
| task `t_i` | one bank-statement OCR job pulled off the queue |
| conflict edge | two jobs pinned to the same physical GPU memory bus, or writing the same Kafka topic partition — co-scheduling them causes contention or a partition clash, not just slowness |
| `r(t_i)` | the job's CPU / RAM / GPU-units / network draw, from its page count and model variant |
| `C(s)` | the cluster's per-cycle capacity, which is **not uniform** — a cycle overlapping a rolling node drain has less |
| `w(t_i)` | lender tier: a Tier-1 PSU bank's statement pull outranks a Tier-3 NBFC's |
| `[l_i, u_i]` | the SLA window from the lender contract — a bureau-triggered pull submitted at T=0 must land within four cycles |

Two parts of SPARK exist because of this mapping and not because of the
assignment text. The **anytime property** — refinement never leaves the feasible
region, so the current schedule is always emittable — is what makes it usable in
a fixed 30-second cycle: when the cycle boundary arrives you ship whatever the
incumbent is. And the **GPU-fragmentation penalty** is a real cost line, not a
modelling flourish: the device plugin hands out whole GPU units, so a cycle
drawing 4.3 units strands 0.7 of an accelerator that nothing else can schedule.

The honest caveat is the certificate phase. In production, "this batch is
infeasible" is not an acceptable output — something has to run. The right
behaviour is to use the certificate as a *diagnosis* (it names the trapped
tasks) and then either spill them to the next cycle or escalate the SLA, which
is a scheduling-policy decision my current code does not make.

## 4. What surprised me

**That six of the nine graded benchmark instances have no solution at all.**

I spent a long time assuming my solver was broken. It reported infeasibility on
everything above `n=12`, which is exactly what a badly-written scheduler looks
like. What changed my mind was checking a certificate by hand: on `n=50,
seed=10`, tasks T3, T8 and T23 all carry the SLA window `[3,4]`, and the
generator drew a conflict edge on all three pairs. Three mutually conflicting
tasks, two slots. No algorithm required, and no algorithm can help.

The root cause is structural, not bad luck: the generator draws `lo` uniformly
and then `hi` from `lo+1..K-1`, so the modal window is width 2, and the number of
tasks trapped in any given 2-slot interval grows *linearly in n* while the
interval still offers exactly two slots. The suite gets more infeasible as it
scales, not less.

What surprised me about my own thinking is how long I resisted the conclusion. I
had framed the task as "write a good optimiser", so every piece of evidence got
read as a bug in the optimiser. The reframe — that a scheduler's job includes
*proving* when a request is impossible — only came after I stopped debugging and
started checking the instances themselves. That is why the certificate phase runs
first, and why `violation_reason` distinguishes a proof from a search failure. It
is also the part of this project I would defend hardest: the optimiser is
conventional, but knowing the difference between "I could not find one" and
"there is not one" is the part that matters operationally.
