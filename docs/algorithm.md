# Task 3 — SPARK: Slack-Prioritised Adaptive Repacking with Kempe-chain repair

## 1. Why a hybrid, and why *this* hybrid

The problem is three NP-hard problems welded together — graph colouring (F1),
`d`-dimensional vector packing (F2) and interval scheduling (F3) — and the weld
is what dictates the design. Three observations drove every decision:

1. **A slot can die for three different reasons, and only one of them is
   repairable by recolouring.** A slot outside `[l_i,u_i]` is dead forever. A
   slot that is full can be revived by *evicting* someone. A slot occupied by a
   conflicting neighbour can be revived by *moving that neighbour*. Classical
   DSATUR has one repair story because it has one failure mode; here the repair
   operator has to branch on the cause. This is the single most
   problem-specific thing in SPARK.
2. **Saturation degree is the wrong urgency signal.** DSATUR ranks by "how many
   colours are already forbidden". Here, slots outside the window were *never*
   available, so counting them inflates the score of tasks that are actually
   fine. What matters is the number of slots still *surviving* — the classic
   MRV (minimum-remaining-values) signal from constraint satisfaction.
3. **The objective is not the constraint.** The penalty rewards early slots, and
   greedily taking early slots is exactly what corners later tasks. So slot
   choice cannot be pure cost minimisation; it needs a lookahead price on the
   options a placement destroys.

SPARK is therefore: *CSP-style variable ordering* (from constraint programming)
+ *cost-aware value ordering with regret* (from scheduling) + *cause-branching
ejection chains* (from bin-packing repair and Kempe chains in colouring) +
*feasibility-preserving tabu search* (from metaheuristics).

## 2. Pseudocode

```
ALGORITHM SPARK(I = (T,K,d,G,r,C,w,tau), lambda, budget)
────────────────────────────────────────────────────────────────────────
 1  cert <- CERTIFY(I)                       # Phase 0, polynomial
 2  if cert != none: return INFEASIBLE(cert) #   a PROOF, not a guess
 3  best <- none
 4  for restart = 0,1,2,... while time remains:
 5      st <- empty schedule
 6      noise <- 0 if restart = 0 else 0.35*min(restart,6)
 7      failed <- CONSTRUCT(st, noise)        # Phase 1
 8      failed <- REPAIR(st, failed)          # Phase 2
 9      if failed != empty: continue          #   this restart cornered a task
10      REFINE(st, share of budget)           # Phase 3
11      if P(st) < P(best): best <- copy(st)
12  if best = none: return NOT_FOUND          # NOT a proof of infeasibility
13  assert VERIFY(I, best)                    # independent audit
14  return best

CONSTRUCT(st, noise)                          # Phase 1
────────────────────────────────────────────────────────────────────────
15  U <- all tasks;  failed <- empty
16  while U not empty:
17      for i in U: F_i <- {s in tau_i : conf(i,s)=0 and fits(i,s)}   # surviving palette
18      if some F_i = empty: move those i to failed; continue         # hand to Phase 2
19      i* <- argmin over i in U of  ( |F_i| - noise*rand,            # KEY 1: MRV
20                                     -sat_i,                       # KEY 2: windowed DSATUR
21                                     -w_i/(u_i-l_i+1) )             # KEY 3: urgency
22      s* <- argmin over s in F_i* of  DeltaP(i*,s) + mu*REGRET(i*,s)
23      place i* in s*;  U <- U \ {i*}
24  return failed

REGRET(i,s)  =  sum over unplaced j in N(i) with s legal for j  of  1/|F_j|

REPAIR(st, failed)                            # Phase 2
────────────────────────────────────────────────────────────────────────
25  for i in failed:  if not SEAT(st,i,depth=3,touched={}): keep i unplaced
26  return the still-unplaced tasks

SEAT(st, i, depth, touched)
────────────────────────────────────────────────────────────────────────
27  if depth = 0: return false
28  if F_i not empty: place i in argmin DeltaP; return true
29  for s in tau_i:
30      E <- neighbours of i sitting in s                # conflict-caused deaths
31      while capacity(s) still exceeded after removing E:
32          k <- the BINDING dimension (largest overflow)
33          E <- E + { occupant of s with largest r[k] }  # evict along the binding axis
34      if E touches an already-touched task: skip s
35      snapshot <- state;  unplace E;  place i in s
36      if every j in E can SEAT(depth-1): return true
37      roll back to snapshot                            # verbatim, no drift
38  return false

REFINE(st, deadline)                          # Phase 3
────────────────────────────────────────────────────────────────────────
39  best <- copy(st);  tabu <- {};  stall <- 0
40  while time < deadline:
41      cand <- best non-tabu move over
42                N1 relocate(i,s)     : gated by can_move
43                N2 swap(i,j)         : gated by can_swap
44                N3 chain(i,s,j,s')   : move i into a slot held by exactly ONE
45                                       blocker j, and re-seat j legally
46            with aspiration: a tabu move is allowed if it beats the incumbent
47      if no candidate: perturb; if perturb changed nothing: break
48      apply cand;  tabu[(moved task, vacated slot)] <- now + tenure
49      if P(st) < P(best): best <- copy(st); stall <- 0  else  stall++
50      if stall >= max(20, n/2): PERTURB(st)             # ruin-and-recreate 15%
51  restore best
```

## 3. Line-level justification of every non-obvious decision

| line | decision | why |
|---|---|---|
| 1–2 | certificates before search | The mandated benchmark suite is 6/9 provably infeasible. Running an 8-second search to answer a question a 300 ms pigeonhole argument settles is wasteful, and the *type* of the answer differs: a proof versus a failure. |
| 6 | `restart 0` is noise-free | Reproducibility. The first restart is a deterministic function of the instance, so a reviewer can hand-trace it. Noise only enters on restarts, and grows with the restart index because early diversification should be gentle. |
| 4 | restarts continue past the nominal count while nothing feasible is found | The first version stopped after 4 restarts, burned 7% of its budget and reported "no solution found" — which reads like infeasibility but was only a search failure. Feasibility is worth spending the entire budget on; quality is not. |
| 17 | palette recomputed each iteration | Correct-by-construction rather than incrementally maintained. It costs `O(n K deg)` per round, which profiling showed is not the bottleneck (the tabu phase is), and an incrementally maintained palette was the source of two bugs before I threw it away. |
| 19 **KEY 1** | MRV before everything | See §1.2. A task with one surviving slot must be placed before a task with eight, regardless of weight — weight is about *cost*, palette size is about *existence*, and existence dominates. |
| 20 **KEY 2** | saturation counted **only inside the window** | Slots outside `tau_i` were never candidates; counting them would rank a wide-window task as "saturated" merely because its neighbours are scattered across slots it could never use. |
| 21 **KEY 3** | urgency `w_i/(u_i-l_i+1)` | Weight per unit of slack. A Tier-1 bureau pull with a 2-slot window outranks a heavier task that can float anywhere. This key is also what costs the adversarial family one unit of its ratio (docs/proofs.md §4c). |
| 22 | slot chosen by **exact** marginal `DeltaP`, not by a proxy | The penalty is non-linear (the balance term is quadratic in utilisation, the GPU term is a sawtooth). Any linear proxy mis-ranks slots near a GPU integer boundary. `_marginal` measures the true increment via place/measure/unplace. |
| 22 | `+ mu * REGRET` | Pure cost minimisation always drifts to the earliest slot and corners the tasks that come later. Regret prices what a placement destroys: `1/|F_j|` charges a lot for stealing a cornered neighbour's last slot and almost nothing for one of eight options. Cost `O(deg(i))`. |
| 32–33 | evict along the **binding** dimension | Evicting the biggest total footprint is wrong when only RAM is tight and the biggest task is CPU-heavy. Picking the dimension with the largest overflow and evicting the largest contributor *to that dimension* is the smallest change that can restore feasibility. |
| 25 | chain depth 3 | Measured: depth 2 leaves ~8% of cornered tasks unplaced on n=150; depth 4 costs ~3x the time for under 1% more placements. |
| 34, 37 | `touched` set + verbatim rollback | Without `touched` the recursion cycles (A evicts B, B evicts A). Without rollback a failed chain leaves the schedule mangled and the "state is always feasible" invariant dies. |
| 42–45 | three neighbourhoods, all feasibility-preserving | Relocation alone is almost entirely blocked on dense instances. Swap reaches 2-cycles. The chain operator is the only one that can move a task into an *occupied* slot, and it is the escape hatch that matters at high density. Because none of them ever leaves the feasible region, the schedule is returnable at any instant — which a 30-second production cycle actually requires. |
| 46 | aspiration | Standard, but load-bearing here: tenure `~sqrt(n)` is long enough that without aspiration the search refuses the very move that reaches a new best. |
| 50 | ruin-and-recreate 15%, not random restart | A restart discards the resource packing that took the whole run to find. A 15% ruin keeps the skeleton and still leaves the basin. |
| 47, 50 | perturbation is transactional | If a victim cannot be re-seated we restore the snapshot exactly. The first version parked unseatable victims in an arbitrary slot, which let the incumbent tracker latch onto an *illegal* schedule with a lower penalty. Caught by the required single-task unit test. |
| 13 | independent `verify` on the way out | The incremental delta engine is the most dangerous code in the project: a wrong delta optimises the wrong objective while every feasibility check still passes. `verify` recomputes from `sigma` alone, so a bug in the engine cannot hide behind itself. |

## 4. Complexity

Let `Delta` be the maximum degree and `W = max_i |tau_i| <= K`.

* Certificates: `C1` `O(n)`, `C2` `O(nKd)`, `C3` `O(K^2 * tries * n * Delta)`, `C4`/`C5` `O(K^2 n d)`.
* Construction: `O(n)` rounds, each `O(n W Delta)` for palettes plus `O(W(d+Delta))` for the slot choice → `O(n^2 W Delta)` worst case.
* Repair: `O(|failed| * W^depth * n)` with `depth = 3` and a node budget.
* Refine: each iteration is `O(nW d + n(d+Delta))`, run until the deadline — the phase is **anytime**, so its cost is a parameter, not a function of `n`.

Everything outside the (bounded, constant-depth) chain recursion is polynomial;
the algorithm is a polynomial-time heuristic with an anytime improvement phase.

## 5. Two approaches I considered and rejected

### Rejected: LP relaxation + randomised rounding

Model `x_{i,s} in {0,1}`, relax to `[0,1]`, solve, round. **Rejected for three
reasons.** (i) The assignment forbids every LP/MIP solver (OR-Tools, PuLP, CPLEX,
Gurobi), so I would have had to write a simplex implementation, and a hand-rolled
simplex on `nK = 4000` variables is both a large distraction and numerically
fragile. (ii) Randomised rounding gives *expected* constraint satisfaction; here
F1 is a hard conflict constraint, and a rounded solution violating it is not
"slightly wrong", it is a Kafka partition collision in production. Repairing
rounded solutions puts me back to needing exactly the ejection machinery SPARK
already has, so the LP would only be a warm start. (iii) The natural LP relaxation
of this formulation has a large integrality gap on the clique-inside-a-window
structure that dominates the mandated instances: fractional colouring lets a
triangle sit at `1/2` in each of two slots, so the LP calls provably infeasible
instances feasible.

### Rejected: simulated annealing over a penalised (infeasible-allowed) space

Let the search visit infeasible schedules and add a violation penalty, annealing
the temperature. **Rejected because** the constraint structure is *cliff-like*
rather than smooth: a single conflict violation is not "a bit worse", it is
invalid, and the violation penalty needed to make it unattractive is so large
that the landscape becomes the feasible-only landscape with extra dead time
spent in illegal regions. Worse, it breaks the anytime property — an interrupted
run may hold an infeasible schedule, and a scheduler that must emit *something*
every 30 seconds cannot use that. I kept annealing's useful half (diversification
via ruin-and-recreate and noisy restarts) and dropped the half that costs
feasibility.

*(A third, briefly considered and dropped: pure DSATUR followed by a separate
repacking pass. It fails because the colouring pass has no idea about resources
and reliably produces colour classes that no repacking can fit — the two layers
must be decided together, which is what line 22 does.)*
