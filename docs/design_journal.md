# Task 7 — Design Journal

## 1. The hardest design decision: calibrating the penalty weights

The algorithm itself was not the hard part. The hard part was Task 2's
consequence: once I had extended `P_base` with three additional terms — load
imbalance `B`, SLA breach risk `S`, and GPU fragmentation `F` — I had to choose
`lambda_B`, `lambda_S` and `lambda_G`, and I had no ground truth to choose
against.

This is a different kind of problem from the rest of the assignment. Everything
else has a right answer I can check: a schedule is feasible or it is not, a
certificate is sound or it is not, the exact solver either agrees with my
heuristic or it does not. But there is no correct value of `lambda_B`. There is
no test that fails when I pick 1.0 instead of 2.0. I could make the number up and
nothing in my own test suite would object, which is exactly why it worried me.

**The alternative I rejected: learning the weights from data.** The obvious
escape is to stop guessing and fit — take a set of schedules an operator has
judged good, and solve for the lambdas that rank those schedules highest. I
rejected it for two reasons. The first is that no such labelled data exists here.
I would have had to generate schedules, decide myself which were good, and then
fit weights to my own judgement — which is circular. It would have produced
confident-looking numbers whose only real input was the intuition I already had,
dressed up as measurement. The second reason is reproducibility: the evaluator
runs held-out seeds, and a fitted lambda is a number they cannot re-derive. A
stated default with a published sensitivity sweep is something they can check.

So I picked defaults by making the terms commensurate rather than by tuning
them — each term is scaled by `W = sum(w_i)` so a lambda means the same thing on
an `n=8` instance as on `n=200` — and then I did the thing that actually
defends the choice, which is to measure how much the choice matters. I swept each
lambda over `{0.25, 0.5, 1, 2, 4}` on three instances, comparing full SPARK
against its construction-only ablation at all 45 settings. **Ranking flips: zero.**
The lambdas set the score, not the winner. That is the honest claim, and it is
weaker than "I tuned them optimally" — but it is true, and it means a reviewer
who dislikes my defaults can change them without invalidating my results.

The sweep also caught something I would otherwise have shipped wrong.
`lambda_G` has *no effect whatsoever* on the relaxed `n=50` instance — the
penalty is 1307.5545 at every single setting. My first reaction was that the GPU
term was broken. It is not: the generator's `cap[d] // (n//K+1)` underflows for
large `n`, so every GPU demand on that instance is sub-1.0 and near-identical,
every schedule strands roughly the same fractional GPU mass, and the term is
genuinely constant there. On the small instances, where GPU demands span 1.0–2.0,
it moves the score from 74.70 to 108.71 across the same sweep. Had I only
benchmarked at scale I would have concluded my own penalty term was useless and
deleted it.

## 2. Where it failed, and what I would do with another week

The one genuinely unresolved cell in my report is `n=100, K=10` with SLA windows
widened by 5. No certificate fires, and SPARK finds no schedule inside its
budget. Those two facts together are the failure: I can neither solve it nor
explain why it cannot be solved.

I believe it is infeasible. A DSATUR colouring of its conflict graph needs **12
colours** against `K=10` slots. But DSATUR gives an *upper* bound on the chromatic
number, not a lower one, so needing 12 colours is evidence and not proof. My
certificate family cannot close the gap either — all three certificates are
pigeonhole arguments at heart, and this graph's clique number is around 7, well
under `K=10`. There is nothing for the pigeonhole to bite on.

**What I would do with another week: nothing to this instance.** Closing it
properly needs a real chromatic lower bound — Lovász theta via SDP, or a
fractional-colouring LP — and both are squarely inside the assignment's
forbidden-library rule. I could hand-roll a weaker version, but I would be
spending a week to move one cell of one table from "unresolved" to "probably
still unresolved", and I do not think that is the right call. The engineering
answer is already correct: the solver reports `NO FEASIBLE ASSIGNMENT FOUND (search budget
exhausted; unplaced: ...) -- not a proof of infeasibility`, which is a true
statement about what it knows. Distinguishing that string from `PROVEN
INFEASIBLE` was worth the extra plumbing precisely for cases like this one.

I would spend the week on the wall-clock budget instead, which is a smaller
problem that affects every number in my report rather than one. Budgeting
refinement by elapsed time means my results are not reproducible: I measured a
2.7% swing on relaxed `n=8` purely from running the benchmark while other work
was on the machine. Switching to an iteration count would make every figure in
the report deterministic, and would also fix the flat `runtime_vs_n` curve, which
is currently flat by construction because the solver simply spends its whole
budget. Reproducibility for everything beats a proof for one instance.

## 3. Where this appears in production at ScoreMe

The mapping I find most natural is a **Kafka consumer group** processing credit
pipeline work. I should be clear about the basis for this: I am reasoning from
what the assignment brief itself establishes — that a conflict edge means two
tasks "write to the same Kafka topic partition simultaneously" — together with
general Kafka semantics. I have no inside knowledge of ScoreMe's actual topology,
and I would not want to claim otherwise.

| Model object | Kafka consumer-group reading |
|---|---|
| slot `s` | one processing window between rebalances |
| task `t_i` | one unit of assigned partition work |
| edge `(t_i, t_j)` | two tasks writing the same topic partition, where concurrent writes clash |
| `r(t_i)` vs `C(s)` | per-consumer throughput, memory and network budget against what the window supplies |
| `[l_i, u_i]` | the lender's contractual turnaround, expressed in windows |

Two properties of SPARK exist because of this reading rather than because the
assignment asked for them. The **anytime property** — refinement never leaves the
feasible region, so the current schedule is always emittable — is what makes it
usable against a fixed window boundary: when the window closes you ship the
incumbent, whatever it is. And the **certificate phase produces a diagnosis, not
just a verdict**: it names the specific tasks that are trapped, which is what an
on-call engineer actually needs at 3am.

The honest limitation is that "infeasible" is not an acceptable production
output. Something has to run. The right behaviour is to use the certificate to
identify the trapped set and then either spill it to the next window or escalate
the SLA — a policy decision my code does not currently make.

## 4. What surprised me

That a scheduler's job includes proving when a request is impossible — and that
this is not a footnote to the optimisation, it is the more useful half.

I did not start there. I framed this task as "write a good optimiser", and for a
long stretch every piece of evidence got read through that frame. My solver
reported infeasibility on everything above `n=12`, which is exactly what a
broken scheduler looks like, so I went looking for the bug. There wasn't one.
What changed my mind was checking a certificate by hand rather than debugging the
code that produced it: on `n=50, seed=10`, tasks T3, T8 and T23 all carry the SLA
window `[3,4]`, and the generator drew a conflict edge on all three pairs. Three
mutually conflicting tasks, two slots. No algorithm required, and no algorithm
can help. **Six of the nine graded instances are unsolvable**, and the structure
that causes it gets worse as `n` grows, not better.

What surprised me about my own thinking is how long I resisted that conclusion,
and that the resistance came from the framing rather than from the evidence — the
evidence had been sitting in my terminal for hours. The reframe only arrived when
I stopped debugging my code and started interrogating my inputs. That is why the
certificate phase runs *first* in the final design, and why `violation_reason`
carries two distinct strings. My optimiser is a fairly conventional hybrid of
things that already exist in the literature. The part I would defend hardest is
the part that knows the difference between "I could not find one" and "there is
not one".
