"""Instance representation and I/O for the MSME Credit Pipeline Scheduling Problem.

Indexing convention (important, and used consistently everywhere):
    Slots are stored 0-indexed (0 .. K-1) because the provided generator emits
    SLA windows as 0-indexed slot numbers.  The *delay index* used by the
    penalty function is delta(s) = s + 1, so the earliest slot costs 1 unit of
    delay rather than 0.  Without this offset P_base would be identically zero
    for any schedule that dumps everything into slot 0, which destroys both the
    economic meaning ("running now still costs one processing window") and the
    ability to state a multiplicative approximation ratio (division by zero).
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field

D = 4  # CPU cores, RAM GB, GPU units, Network Gbps
DIM_NAMES = ("CPU", "RAM", "GPU", "NET")


def generate_instance(n, K, d=4, conflict_density=0.3, seed=42):
    """Provided instance generator -- reproduced VERBATIM, do not modify.

    Kept byte-for-byte as issued in Section 5 of the assignment so that our
    numbers are reproducible against the evaluator's held-out seeds.  Its known
    quirks (the comprehension variable `d` shadowing the parameter `d`, and
    `random.uniform(1, x)` being called with x < 1 for large n) are deliberately
    NOT fixed here; they are documented and analysed in docs/benchmarks.md.
    """
    random.seed(seed)
    tasks = [f'T{i}' for i in range(n)]
    conflicts = [(i, j) for i in range(n) for j in range(i + 1, n)
                 if random.random() < conflict_density]
    cap = [32, 128, 8, 6.0]  # CPU, RAM, GPU, Network
    resources = [[random.uniform(1, cap[d] // (n // K + 1))
                  for d in range(4)] for _ in range(n)]
    capacities = [cap[:] for _ in range(K)]
    windows = [(lo := random.randint(0, K - 2),
                random.randint(lo + 1, K - 1)) for _ in range(n)]
    weights = [random.uniform(1, 10) for _ in range(n)]
    return dict(tasks=tasks, conflicts=conflicts,
                resources=resources, capacities=capacities,
                windows=windows, weights=weights, K=K)


@dataclass
class Instance:
    """A scheduling instance in the form the solver actually consumes.

    Design decision: we materialise the conflict graph as adjacency *sets* (not
    an edge list and not a dense matrix).  Membership tests dominate the solver's
    inner loops, and a set answers them in O(1) while an edge list needs a scan.
    Sets also cost O(deg) rather than O(n) to iterate, which matters on the
    sparse-conflict stress instance (density 0.10, mean degree ~20 against
    n=200).  The solver's hot conflict query is served by a counter array in
    ScheduleState rather than by this structure -- see msme/state.py.
    """

    tasks: list[str]
    conflicts: list[tuple[int, int]]
    resources: list[list[float]]
    capacities: list[list[float]]
    windows: list[tuple[int, int]]
    weights: list[float]
    K: int
    adj: list[set[int]] = field(default_factory=list, repr=False)

    def __post_init__(self):
        n = len(self.tasks)
        self.adj = [set() for _ in range(n)]
        for i, j in self.conflicts:
            self.adj[i].add(j)
            self.adj[j].add(i)
        # Clamp windows into range and normalise to ints; a malformed hand
        # written instance file should fail loudly here rather than silently
        # produce an "infeasible" verdict later.
        for i, (lo, hi) in enumerate(self.windows):
            if not (0 <= lo <= hi <= self.K - 1):
                raise ValueError(
                    f"task {self.tasks[i]} has window ({lo},{hi}) outside 0..{self.K-1}")

    @property
    def n(self) -> int:
        return len(self.tasks)

    @property
    def d(self) -> int:
        return len(self.capacities[0]) if self.capacities else D

    def degree(self, i: int) -> int:
        return len(self.adj[i])

    def window_slots(self, i: int) -> range:
        lo, hi = self.windows[i]
        return range(lo, hi + 1)

    @classmethod
    def from_dict(cls, obj: dict) -> "Instance":
        return cls(
            tasks=list(obj["tasks"]),
            conflicts=[tuple(e) for e in obj["conflicts"]],
            resources=[list(map(float, r)) for r in obj["resources"]],
            capacities=[list(map(float, c)) for c in obj["capacities"]],
            windows=[tuple(w) for w in obj["windows"]],
            weights=[float(w) for w in obj["weights"]],
            K=int(obj["K"]),
        )

    def to_dict(self) -> dict:
        return dict(tasks=self.tasks, conflicts=[list(e) for e in self.conflicts],
                    resources=self.resources, capacities=self.capacities,
                    windows=[list(w) for w in self.windows],
                    weights=self.weights, K=self.K)

    @classmethod
    def load(cls, path: str) -> "Instance":
        with open(path) as fh:
            return cls.from_dict(json.load(fh))

    def save(self, path: str) -> None:
        with open(path, "w") as fh:
            json.dump(self.to_dict(), fh, indent=2)

    @classmethod
    def random(cls, n, K, density=0.3, seed=42) -> "Instance":
        return cls.from_dict(generate_instance(n, K, 4, density, seed))
