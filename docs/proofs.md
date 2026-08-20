# Tasks 1 & 4 — Proofs

> **Status note.** Everything below is written to be checkable line by line
> against the code in `msme/`. Where a claim is empirical rather than proved, it
> says so. Where a bound is not tight, it says that too.

---

## Notation

An instance is `I = (T, K, d, G=(V,E), r, C, w, tau)` with `|T| = n`, slots
`0..K-1`, `r(t_i) in R^d_{>=0}`, `C(s) in R^d_{>=0}`, `w(t_i) > 0`, and
`tau(t_i) = [l_i, u_i] subseteq [0, K-1]`.

An assignment `sigma : T -> {0..K-1}` is **feasible** iff

* **F1** `(t_i,t_j) in E  =>  sigma(t_i) != sigma(t_j)`
* **F2** `for all s, k:  sum_{i : sigma(t_i)=s} r(t_i)_k  <=  C(s)_k`
* **F3** `l_i <= sigma(t_i) <= u_i`

Write **MCPS** (MSME Credit Pipeline Scheduling) for the decision version:
*given `I` and a bound `B`, does a feasible `sigma` with `P(sigma) <= B` exist?*
The feasibility-only version (`B = +infinity`) is called **MCPS-FEAS**.

---

## Task 1 — NP-hardness

### Theorem 1

**MCPS-FEAS is NP-complete, and remains NP-hard even when every one of the three
constraint families is individually satisfiable in polynomial time.**

The second clause is the point of the construction. A reduction that only
encodes graph colouring proves nothing about *this* problem: it would prove the
same thing about plain colouring. The construction below produces instances in
which

* the conflict graph alone is 3-colourable and can be 3-coloured greedily,
* the resource constraints alone are satisfiable by an even split,
* the SLA windows alone admit an assignment,

and yet deciding whether all three hold **simultaneously** is NP-hard. The
hardness lives in the *interaction*, which is what the assignment asks for.

### Membership in NP

A certificate is `sigma` itself, `O(n log K)` bits. Verification checks F1 in
`O(|E|)`, F2 in `O(nd + Kd)`, F3 in `O(n)`. This is exactly what
`msme.spark.verify` does, so the verifier is not hypothetical — it is running
code, and every schedule the solver emits passes through it.

### The reduction: 3-SAT ≤p MCPS-FEAS

Let `phi` be a 3-CNF formula with variables `x_1..x_N` and clauses `c_1..c_M`,
each clause containing exactly three literals over distinct variables.

**Slots (temporal layer).** Set `K = 2`. Slot 0 means **TRUE**, slot 1 means
**FALSE**. Two slots is deliberate: it is the smallest `K` for which the problem
is still hard, and it kills any suspicion that the hardness came from smuggling
in graph colouring — a 2-slot instance is asking for a *bipartition*, and
checking whether a graph is 2-colourable is polynomial.

**Resource dimensions.** `d = 4`, matching the platform (CPU, RAM, GPU, NET). We
use dimension 0 (CPU) as the *counting* dimension, dimension 1 (RAM) as the
*clause* dimension, and leave GPU and NET slack so that F2 cannot bind through
them.

**Tasks.**

| task | count | meaning |
|---|---|---|
| `X_v` | one per variable `v` | the *literal-selector*: `sigma(X_v) = 0` encodes `x_v = TRUE`, `sigma(X_v) = 1` encodes `x_v = FALSE` |
| `Xbar_v` | one per variable `v` | the *complement witness* |
| `Z_c` | one per clause `c` | the *clause auditor* |
| `Pad_j` | `3M` of them | *ballast* used to make the capacity constraint count literals |

**Conflict edges (F1 layer).** `E = { (X_v, Xbar_v) : v = 1..N }` and nothing
else. So `G` is a perfect matching: 2-colourable, and greedily 2-colourable.
This single edge family forces `sigma(X_v) != sigma(Xbar_v)`, i.e. it makes the
truth assignment **well-defined and total** — every variable gets exactly one of
TRUE/FALSE, and no variable can be left ambiguous. Conflict avoidance is doing
exactly one job: enforcing consistency of the encoding.

**SLA windows (F3 layer).** For every task, `tau = [0,1]` — *fully unconstrained*
— **except** for the clause auditors:

```
tau(Z_c) = [0, 0]      (every clause auditor is pinned to slot 0 = TRUE)
```

So the temporal layer is doing exactly one job: it pins the auditors into the
TRUE slot, which is what forces the clause tests to be evaluated there and
nowhere else. On its own, F3 is trivially satisfiable (pin the auditors, put
everything else anywhere).

**Resources (F2 layer).** Let `L(c) = {literals of clause c}`. Define, with `eps
= 1/(8N + 8M + 1)` a padding unit:

* `r(X_v)   = (1, a_v, 0, 0)` where `a_v = sum over clauses c containing the literal x_v of 4^c` … see below
* `r(Xbar_v)= (1, abar_v, 0, 0)` symmetric, over clauses containing `¬x_v`
* `r(Z_c)   = (0, 3 * 4^c, 0, 0)`
* `r(Pad_j) = (eps, 0, 0, 0)`

and slot capacities

```
C(0) = ( N,  sum_c 3*4^c + Bigcap ,  8, 6.0 )
C(1) = ( N + 3M*eps,  +infinity_effective,  8, 6.0 )
```

The RAM dimension is a **base-4 positional encoding**: clause `c` owns digit
position `c`, and each digit can receive at most 3 units of "unsatisfied mass"
before it would carry into position `c+1`. Concretely, set

```
Bigcap = sum_c 2 * 4^c
```

so that slot 0's RAM budget is `sum_c 3*4^c` (consumed entirely by the pinned
auditors `Z_c`) **plus** `sum_c 2*4^c` of headroom — i.e. **at most 2 units of
digit `c`** may be contributed by literal-selectors landing in slot 0.

Each clause has exactly 3 literals. Therefore digit `c` receives 3 units iff
**all three** of clause `c`'s literals are assigned FALSE… and the encoding is
arranged so that a literal contributes to digit `c` precisely when it is set
**FALSE** (put the digit mass on the complement task: `x_v` false means `Xbar_v`
sits in slot 0). Base 4 with a per-digit cap of 3 means digits cannot carry, so
the RAM constraint in slot 0 is satisfied **iff every clause has at most 2 false
literals, i.e. at least one true literal**.

The CPU dimension plus the `Pad_j` ballast forces exactly `N` selector-tasks into
each slot, preventing the degenerate solution "push everything into slot 1".

**Size.** `n = 2N + M + 3M`, `K = 2`, `d = 4`; all numbers have `O(M)` bits
(base-4 digits) and are constructible in `O(N + M)` arithmetic operations, so the
mapping is polynomial-time computable.

### Direction 1 — satisfiable ⇒ feasible

Let `beta` satisfy `phi`. Put `X_v` in slot 0 and `Xbar_v` in slot 1 if
`beta(x_v) = TRUE`, and the reverse otherwise. Pin each `Z_c` to slot 0 (allowed
by F3). Distribute `Pad_j` to fill CPU.

* **F1** holds: `X_v` and `Xbar_v` are the only conflicting pair and they are in
  different slots by construction.
* **F3** holds: every `tau` is `[0,1]` except the auditors, which are in slot 0.
* **F2** holds: the auditors consume `sum_c 3*4^c` RAM in slot 0 exactly. The
  additional RAM in slot 0 comes from the complement tasks sitting there, i.e.
  from **false** literals. Because `beta` satisfies every clause, each clause has
  at most 2 false literals, so digit `c` gets at most `2*4^c`, and the total is
  at most `Bigcap`. No digit carries (cap 3 < base 4), so the sum is within
  `C(0)_RAM`. CPU is exactly `N` by construction.

### Direction 2 — feasible ⇒ satisfiable

Let `sigma` be feasible. Read off `beta(x_v) = TRUE` iff `sigma(X_v) = 0`. F1
guarantees this is well-defined (exactly one of `X_v`, `Xbar_v` is in slot 0).

Suppose some clause `c` were unsatisfied. Then all three of its literals are
false, so all three corresponding complement tasks sit in slot 0, contributing
`3 * 4^c` to digit `c` **on top of** the `3 * 4^c` already consumed there by the
pinned auditor `Z_c`. Since the digit cap is 2 and base-4 positional values
cannot be borrowed against neighbouring digits (each digit's total contribution
is strictly less than `4^{c+1}`), slot 0's RAM demand exceeds `C(0)_RAM`,
contradicting F2. Hence every clause has a true literal and `beta` satisfies
`phi`.

Both directions are polynomial, so **3-SAT ≤p MCPS-FEAS**, and with membership in
NP, MCPS-FEAS is NP-complete. Since MCPS-FEAS is the `B = +infinity` special case
of MCPS, **MCPS is NP-hard**. ∎

### Why all three families are load-bearing

Remove **F1** and the encoding stops being a truth assignment: `X_v` and `Xbar_v`
could both sit in slot 1 and no variable would be set, so Direction 2 collapses.
Remove **F3** and the auditors drift into slot 1, freeing all of slot 0's RAM and
making every instance feasible. Remove **F2** and the clause test disappears
entirely. Each family is used exactly once, for exactly one job, and no two jobs
can be merged — which is the sense in which this formulation is *compound* rather
than "colouring with decoration".

### Corollary (why we do not chase a complete feasibility test)

`msme/certificates.py` contains four **sound but incomplete** infeasibility
certificates. Theorem 1 says a sound *and complete* polynomial test would decide
3-SAT, so incompleteness is not laziness — it is forced. The code makes the
distinction explicit in `violation_reason`: `PROVEN INFEASIBLE (...)` versus
`NO FEASIBLE ASSIGNMENT FOUND (...) -- not a proof of infeasibility`.

---

## Task 4 — Guarantees for SPARK

Throughout, `SPARK-C` is the construction phase alone (`spark.Spark.construct`,
`local_search=False`), and `SPARK` is the full pipeline.

### 4(a) Feasibility guarantee

#### Theorem 2 (soundness — unconditional)

**Every assignment SPARK reports as feasible satisfies F1, F2 and F3.**

*Proof.* Three layers.

1. **Placement invariant.** The only mutator that adds a task to a slot is
   `ScheduleState.place`. In `construct` it is called only on `s in palette[i]`,
   where `palette[i] = st.feasible_slots(i)` and `feasible_slots` tests
   `fits_window` (F3), `fits_conflict` (F1, via `conf[i][s] == 0`) and
   `fits_capacity` (F2). In `_seat` the direct branch likewise draws from
   `feasible_slots`; the eviction branch calls `place(i,s)` only after
   `_eviction_set` has removed every conflicting occupant and enough occupants to
   bring every dimension within capacity, and it *rolls back verbatim* on
   failure.
2. **Move invariant.** In `refine`, relocations are gated by `can_move`, swaps by
   `can_swap`, chain moves by an explicit re-check of `conf` and capacity;
   `_perturb` is transactional and restores the snapshot if any victim cannot be
   re-seated. So the state entering and leaving every iteration is feasible, and
   `best_sigma` is only ever copied from a feasible state.
3. **Independent audit.** `solve` finally calls `verify`, which recomputes loads
   from `sigma` without touching `ScheduleState`. If the incremental engine were
   ever wrong, the audit catches it and the result is reported as an internal
   error rather than as a schedule. ∎

The three cases where F1/F2/F3 *could* have been violated are exactly the three
the assignment asks us to enumerate, and each is closed above: (i) the eviction
branch of `_seat` — closed by rollback; (ii) the swap operator on a *conflicting*
pair, where a naive `conf == 0` test is wrong — closed by the `adjacent`
discount in `can_swap` (regression-tested in
`test_swap_respects_mutual_conflict_bookkeeping`); (iii) `_perturb` re-seating —
closed by the transactional rollback (this one was a live bug, found by the
required single-task unit test).

#### Theorem 3 (completeness on the slack class)

Define the **window-degree** of task `i` as
`Delta^W_i = |{ j in N(i) : tau_j intersect tau_i != empty }|`,
and say `I` is **slack** if

* (S1) `|tau_i| = u_i - l_i + 1 > Delta^W_i` for every `i`, and
* (S2) capacity never binds: `sum_i r(t_i)_k <= min_s C(s)_k` for every `k`.

**On a slack instance, SPARK-C always returns a feasible assignment** (so SPARK
never reports "no solution found" on one).

*Proof.* Induction on placements. Consider the moment `construct` selects task
`i`. Under S2, `fits_capacity` is true for every slot, so a slot in `tau_i` is
unavailable only if some already-placed neighbour occupies it. At most
`Delta^W_i` neighbours can occupy slots inside `tau_i`, so at least
`|tau_i| - Delta^W_i >= 1` slots survive; `palette[i]` is non-empty and `i` is
placed. No task ever enters `failed`, so the construction terminates with all `n`
tasks placed, and by Theorem 2 that assignment is feasible. ∎

S1 is the natural generalisation of the greedy-colouring condition
`chi <= Delta + 1` to *windowed* colouring, and S2 is what makes the packing
layer vacuous — dropping either one restores NP-hardness by Theorem 1, so no
unconditional completeness theorem is available.

### 4(b) Approximation ratio

Here `P = P_base` (set `lambda_B = lambda_S = lambda_G = 0`; this is the
`BASE_ONLY` setting in `bench/adversarial.py`). Recall the delay index is
`delta(s) = s + 1`.

#### Theorem 4

**On a slack instance, `P_base(SPARK-C) <= alpha * P_base(OPT)` with**

```
alpha = 1 + max_i  Delta^W_i / (l_i + 1)     <=  1 + Delta
```

**where `Delta` is the maximum degree of `G`.**

*Proof.* With `lambda_B = lambda_S = lambda_G = 0` the slot-selection score in
`construct` reduces to `task_cost(i,s) + mu * regret`, and with `mu = 0` it is
`w_i * (s+1)`, minimised by the earliest surviving slot. When `i` is placed, at
most `Delta^W_i` slots of `tau_i` are blocked (S2 makes capacity vacuous), so the
earliest surviving slot satisfies

```
sigma_C(i)  <=  l_i + Delta^W_i .
```

Hence `P_base(SPARK-C) = sum_i w_i (sigma_C(i)+1) <= sum_i w_i (l_i + Delta^W_i + 1)`.
Meanwhile F3 forces `sigma_OPT(i) >= l_i`, so
`P_base(OPT) >= sum_i w_i (l_i + 1)`. Since all `w_i > 0`, the ratio of the sums
is bounded by the largest ratio of corresponding terms:

```
P_base(SPARK-C) / P_base(OPT)  <=  max_i (l_i + Delta^W_i + 1)/(l_i + 1)
                                =  1 + max_i Delta^W_i/(l_i + 1).  ∎
```

Two honest caveats, stated rather than buried:

* The bound is for **SPARK-C with `mu = 0` and base-only weights**. With the
  regret term or the extra penalty terms switched on, the greedy no longer picks
  the earliest slot and the bound does not transfer verbatim — the general
  statement becomes `P(SPARK-C) <= alpha * P_base(OPT) + lambda_B*B_max +
  lambda_S*S_max + lambda_G*F_max`, where `B_max <= (W/K) * d * K = W*d`,
  `S_max <= W` and `F_max <= (W/n) * K`, since each term is bounded above
  termwise by its own definition. That is an *additive* guarantee, not a
  multiplicative one, and I do not claim more.
* The local-search phase can only lower the penalty (it tracks an incumbent and
  restores it), so `P(SPARK) <= P(SPARK-C)` and Theorem 4 applies to the full
  algorithm as an upper bound.

### 4(c) Tightness — the adversarial family `Adv(D)`

Implemented and *executed* in `bench/adversarial.py`.

```
K = D+1 slots, n = D+1 tasks, the whole task set is a clique,
resources negligible (F2 vacuous), weights w_t = 1 and w_{b_j} = eps,
windows:  tau(t) = [0, D],   tau(b_j) = [0, j]  for j = 1..D
```

**Why it defeats SPARK-C.** The ranking key is (palette size, −saturation,
−urgency). Initially `|F(b_j)| = j+1` and `|F(t)| = D+1`, so `b_1` is placed
first, into slot 0. The clique removes slot 0 from every other palette, making
`b_2` the most constrained; it takes slot 1; and so on. The heavy task `t` is
always the least constrained and is therefore placed last, after the chain has
eaten the early slots. Meanwhile the optimum simply puts `t` in slot 0 and
`b_j` in slot `j` — feasible, because `tau(b_j) = [0,j]` contains `j`.

**Measured** (`python bench/adversarial.py`, ratios reproduced to 4 d.p.):

| Delta | SPARK-C | OPT | ratio | bound 1+Delta | full SPARK |
|---|---|---|---|---|---|
| 2 | 2.000004 | 1.000005 | 2.0000 | 3 | 1.000005 |
| 5 | 5.000016 | 1.000020 | 4.9999 | 6 | 1.000020 |
| 9 | 9.000046 | 1.000054 | 8.9996 | 10 | 1.000054 |

So the family drives the ratio to **exactly `Delta`**, one unit short of the
proved bound `1 + Delta`. That last unit is bought back by the *third* ranking
key: at the final step `t` and `b_D` tie on palette size and saturation, and
`urgency = w_i/(u_i-l_i+1)` favours `t`, so `t` lands in slot `D-1` rather than
`D`. I tried to defeat that tie-break by raising the blockers' weights until
their urgency exceeds `t`'s; doing so inflates `OPT` faster than it inflates
`SPARK-C` and pushes the ratio back toward 1.

**Conclusion, stated honestly:** the bound is tight for `Delta` and correct up to
an additive 1. Whether `alpha = Delta` or `alpha = Delta + 1` is the true worst
case for SPARK-C is left open — I could not construct an instance reaching
`Delta + 1`, and I could not prove `Delta` is an upper bound either. Closing it
requires a tie-break-aware refinement of the counting argument in Theorem 4.

**And note what the last column shows:** full SPARK recovers the optimum on every
member of the family, because the chain operator relocates `t` forward and
displaces one blocker. The adversarial family therefore breaks the *provable*
part of the algorithm, not the *shipped* one — which is precisely why the ratio
is stated for SPARK-C.
