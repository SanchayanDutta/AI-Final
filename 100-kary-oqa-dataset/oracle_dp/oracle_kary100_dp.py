"""
Exact information-theoretic oracle for the k-ary 100-object OQA dataset.

Usage
-----
1. Load the attribute table from JSON:

   >>> import json
   >>> with open("../data/oqa_kary100_dataset.json") as f:
   ...     table = json.load(f)

2. Build the oracle and compute a trajectory for a single target object id:

   >>> oracle = KaryOracle(table)
   >>> traj = oracle.trajectory_for_target("0000", max_steps=10)
   >>> print(traj)  # list of posterior entropies after each question

3. To reproduce the oracle curve in the released plot, choose 30 target ids
   (for example the first 30 keys of the table), run `trajectory_for_target`
   for each one, and average the entropies across targets at every step.

The dynamic program assumes:
* A uniform prior over all objects.
* Deterministic answers (no observation noise).
* Questions of the form "What is the value of attribute a?", where the
  answer selects exactly one of the discrete values of that attribute.

Under these assumptions the optimal policy is the decision tree that
minimizes the expected number of questions, which is equivalent to
maximizing expected information gain at every step.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from math import log2
from typing import Dict, Iterable, List, Mapping


AttrTable = Mapping[str, Mapping[str, str]]  # object_id -> {attribute: value}


@dataclass(frozen=True)
class State:
    """Candidate set of objects, represented by their ids.

    We keep the representation as a frozenset of ids so that it can be used
    as a key in the DP cache.
    """

    ids: frozenset

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.ids)


class KaryOracle:
    """Exact k-ary dynamic program over a finite attribute table.

    Parameters
    ----------
    table:
        Dictionary mapping object ids to dictionaries of attribute values.
        The code assumes each object has the same attribute keys and that
        attributes take on finitely many discrete values.
    """

    def __init__(self, table: AttrTable):
        if not table:
            raise ValueError("Attribute table must be non-empty.")
        self.table: Dict[str, Dict[str, str]] = {k: dict(v) for k, v in table.items()}
        # Stable ordering of object ids and attribute names
        self.object_ids: List[str] = sorted(self.table.keys())
        self.attributes: List[str] = sorted(next(iter(self.table.values())).keys())

    # ---------- Dynamic program over candidate sets ----------

    @lru_cache(maxsize=None)
    def _cost(self, state: State) -> float:
        """Expected remaining number of questions from this candidate set.

        This is the Bellman-optimal cost function C(S). For a state S and an
        attribute a, the expected residual cost is

            C_a(S) = 1 + sum_{v in values(a)} p(v | S) * C(S_v)

        where S_v is the subset of objects in S whose attribute a equals v.
        The oracle chooses the attribute a that minimizes C_a(S).

        When no attribute produces a non-trivial split we return 0, meaning
        that further querying cannot reduce the candidate set.
        """
        ids = state.ids
        n = len(ids)
        if n <= 1:
            return 0.0

        best_cost = float("inf")
        # Try every attribute as the next question.
        for attr in self.attributes:
            # Partition the candidate set by this attribute's values.
            partitions: Dict[str, set] = {}
            for obj_id in ids:
                val = self.table[obj_id][attr]
                partitions.setdefault(val, set()).add(obj_id)
            # Skip attributes that do not split the set.
            if len(partitions) <= 1:
                continue

            expected = 1.0  # cost of asking this question
            for subset in partitions.values():
                sub_state = State(frozenset(subset))
                expected += (len(subset) / n) * self._cost(sub_state)

            if expected < best_cost:
                best_cost = expected

        if best_cost == float("inf"):
            # No attribute can split this candidate set any further.
            return 0.0
        return best_cost

    def _best_attribute(self, state: State) -> str | None:
        """Return the optimal attribute to query at this state, or None."""
        ids = state.ids
        n = len(ids)
        if n <= 1:
            return None

        best_cost = float("inf")
        best_attr: str | None = None

        for attr in self.attributes:
            partitions: Dict[str, set] = {}
            for obj_id in ids:
                val = self.table[obj_id][attr]
                partitions.setdefault(val, set()).add(obj_id)
            if len(partitions) <= 1:
                continue

            expected = 1.0
            for subset in partitions.values():
                sub_state = State(frozenset(subset))
                expected += (len(subset) / n) * self._cost(sub_state)

            if expected < best_cost:
                best_cost = expected
                best_attr = attr

        return best_attr

    # ---------- Public API ----------

    def trajectory_for_target(self, target_id: str, max_steps: int = 10) -> List[float]:
        """Return posterior entropies for a particular target object.

        The list entry at index t (0-based) is the entropy after t+1 questions,
        measured as log2 of the number of remaining candidates under the
        optimal policy. Once the candidate set collapses to size 1, all
        subsequent entropies are exactly 0.0.
        """
        if target_id not in self.table:
            raise KeyError(f"Unknown target id: {target_id}")

        current_ids = set(self.object_ids)
        entropies: List[float] = []

        for _ in range(max_steps):
            if len(current_ids) <= 1:
                entropies.append(0.0)
                continue

            state = State(frozenset(current_ids))
            attr = self._best_attribute(state)
            if attr is None:
                # No attribute can split the remaining candidates.
                entropies.append(0.0)
                continue

            target_value = self.table[target_id][attr]
            # Update the candidate set to those objects matching the answer.
            current_ids = {
                obj_id
                for obj_id in current_ids
                if self.table[obj_id][attr] == target_value
            }
            entropies.append(log2(len(current_ids)))

        return entropies

    def mean_trajectory(
        self, target_ids: Iterable[str], max_steps: int = 10
    ) -> List[float]:
        """Average entropy trajectory over a collection of target ids."""
        trajectories = [
            self.trajectory_for_target(obj_id, max_steps=max_steps)
            for obj_id in target_ids
        ]
        # Transpose to step-major layout and average.
        num_targets = len(trajectories)
        if num_targets == 0:
            raise ValueError("mean_trajectory requires at least one target id.")
        return [
            sum(traj[t] for traj in trajectories) / num_targets
            for t in range(max_steps)
        ]


if __name__ == "__main__":  # simple smoke test
    import json
    from pathlib import Path

    data_path = (
        Path(__file__).resolve().parent.parent / "data" / "oqa_kary100_dataset.json"
    )
    with data_path.open() as f:
        table = json.load(f)

    oracle = KaryOracle(table)
    print("Expected optimal cost from root:",
          oracle._cost(State(frozenset(oracle.object_ids))))

    # Example: entropy trajectory for the first object id
    first_id = oracle.object_ids[0]
    print("Example trajectory for", first_id, ":", oracle.trajectory_for_target(first_id))
