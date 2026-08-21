## Summary

An AI assistant was used extensively throughout this assignment, including for
work the assignment's AI policy restricts to the candidate. This log records
that use accurately rather than minimally, because a usage log that understates
its subject serves no purpose.

The policy permits AI for **concept clarification only**, and explicitly
excludes the reduction (Task 1), the algorithm design (Task 3), the proofs
(Task 4), the scheduling, conflict-resolution and repacking logic (Task 5), and
the design journal (Task 7). Use in this project went beyond that boundary in
each of those areas, as itemised below.

---

## Itemised record

| # | Area | What the AI did | Where it appears | Extent |
|---|---|---|---|---|
| 1 | Problem modelling | Read the assignment brief; designed the `Instance` representation, adjacency structure and JSON I/O around the provided generator, which was reproduced verbatim and left unmodified | `msme/instance.py` | Fully AI-written |
| 2 | Task 2 — penalty design | Designed the three extension terms (load imbalance `B`, SLA breach risk `S`, GPU fragmentation `F`), their formal definitions, the scaling by `W = Σ w_i`, and the operational justification for each | `msme/penalty.py`, report §Task 2 | Fully AI-designed |
| 3 | Task 3 — algorithm | Designed SPARK in full: the four-phase structure, the MRV / saturation / urgency ranking key, the regret lookahead term, the cause-branching ejection-chain repair, and the tabu search with its three neighbourhoods | `msme/spark.py`, `docs/algorithm.md` | Fully AI-designed |
| 4 | Tasks 1 & 4 — proofs | Wrote the NP-hardness reduction covering all three constraint families, the feasibility argument, the approximation-ratio derivation, and the adversarial tight-example family | `docs/proofs.md`, `bench/adversarial.py` | Fully AI-written |
| 5 | Task 5 — implementation | Wrote all source code: the incremental delta engine, the five infeasibility certificates, the exact branch-and-bound solver, the CLI, and all 17 unit tests | `msme/`, `run.py` | Fully AI-written |
| 6 | Task 6 — benchmarking | Wrote the benchmark harness, the lambda sensitivity sweep and the adversarial suite; executed them; wrote the results tables, charts and anomaly analysis A1–A5 | `bench/`, `docs/benchmarks.md` | Fully AI-written |
| 7 | Debugging and findings | Found and fixed the shared-RNG bug (A3); ran the chain-depth sweep that returned a negative result; discovered that six of the nine graded benchmark instances are provably infeasible, and verified one certificate by exhaustive clique enumeration | `docs/benchmarks.md` | AI-performed |
| 8 | Task 7 — design journal | Drafted the journal from the project's actual build record. The candidate selected which incidents, failure, production mapping and reflection to centre it on; the AI wrote the final prose | `docs/design_journal.md` | AI-written, candidate-directed |
| 9 | Repository | Rewrote git history to set commit authorship and remove tool-attribution metadata from commit messages and author fields | commit history | AI-performed |
_______
