"""
Exact information-theoretic oracle for the 300-object k-ary OQA dataset.

This module implements the *exact* dynamic program used for the Oracle
(Optimal) line in the k-ary 300-objects entropy plot.

The setting is:

  * There is a finite set of discrete objects X (300 in the released dataset).
  * Each object x in X is described by a dictionary of categorical attributes,
    for example the entries in ``kary300_Items.json``.
  * At each step the oracle chooses a single attribute a (e.g., "color"),
    and the environment reveals the *value* of that attribute for the
    hidden target object.  This is a k-ary question: the answer can be one
    of several attribute values (red, blue, ...), not just yes/no.
  * After observing the answer, the oracle restricts its candidate set to
    those objects whose attribute a matches the answer.

Under a uniform prior over targets and noiseless observations, the optimal
policy is the one that minimizes the *expected number of attribute
questions* needed to isolate the true object (or an equivalence class of
indistinguishable objects, if attribute vectors are not unique).  This is
equivalent to maximizing expected information gain at each step.

How to use
----------
1. Load the 300-object JSON file into a Python dictionary:

       import json
       with open("data/kary300_Items.json", "r") as f:
           items = json.load(f)

   ``items`` should now be a dict mapping string IDs to attribute dicts,
   as in the released dataset.

2. Build the oracle:

       from oracle.kary300_exact_oracle_dp import KaryOracleDP
       oracle = KaryOracleDP(items)

3. Run the oracle for a single hidden target to obtain the posterior
   entropy trajectory (in bits) as it asks optimal questions:

       entropies, asked_attributes = oracle.simulate_target("0000")
       print(entropies)

   Here ``entropies[0]`` is the prior entropy log2(#candidates) before
   any questions (step 1 in the paper plots), ``entropies[1]`` is after
   the first optimal query (step 2), and so on, until the candidate set
   cannot be split any further.

4. If you sample 30 random targets from the 300 objects and average the
   per-step entropies returned by ``simulate_target``, you recover the
   Oracle trajectory shown in the k-ary 300-objects entropy figure.

Notes
-----
* The implementation memoizes the optimal cost for every candidate set
  that appears along the search tree, so the dynamic program is exact but
  still practical for the 300-object dataset.
* The same code works for any similar k-ary attributes table; you can plug
  in any ``items`` dictionary of the same format.

"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


ItemId = str
Attributes = Mapping[str, str]
ItemsDict = Mapping[ItemId, Attributes]


def entropy_uniform(num_items: int) -> float:
    """Shannon entropy (in bits) of a uniform distribution over ``num_items`` objects."""
    if num_items <= 1:
        return 0.0
    return math.log2(num_items)


@dataclass(frozen=True)
class OracleState:
    """Represents a candidate set of objects by their IDs.

    We store the IDs as a sorted tuple so that each logical set has a
    canonical, hashable representation suitable for memoization.
    """

    ids: Tuple[ItemId, ...]

    @classmethod
    def from_iterable(cls, ids: Iterable[ItemId]) -> "OracleState":
        return cls(tuple(sorted(ids)))

    def __len__(self) -> int:  # convenience
        return len(self.ids)


class KaryOracleDP:
    """Exact dynamic-programming oracle for multi-way attribute queries.

    Parameters
    ----------
    items:
        A dictionary mapping object IDs to attribute dictionaries, as in
        ``kary300_Items.json``.  Every object must have the same set of
        attribute keys, and attribute values must be hashable (strings in
        the released dataset).

    Attributes
    ----------
    attributes:
        The ordered tuple of attribute names considered askable by the oracle.
    root_state:
        The initial candidate set containing all objects.
    """

    def __init__(self, items: ItemsDict):
        if not items:
            raise ValueError("`items` dictionary is empty.")

        self.items: ItemsDict = items
        # Fix a deterministic ordering of object IDs and attribute names.
        any_item = next(iter(items.values()))
        self.attributes: Tuple[str, ...] = tuple(sorted(any_item.keys()))
        self.root_state = OracleState.from_iterable(items.keys())

    # ---- Dynamic program -------------------------------------------------

    def _split_on_attribute(
        self, state: OracleState, attribute: str
    ) -> List[OracleState]:
        """Partition a candidate state by the value of a single attribute.

        Returns a list of non-empty child states.  If *all* candidates in
        ``state`` share the same value for ``attribute``, the list has
        length 1 and the split provides no information.
        """
        buckets: Dict[str, List[ItemId]] = {}
        for item_id in state.ids:
            v = self.items[item_id][attribute]
            buckets.setdefault(v, []).append(item_id)

        # Convert each non-empty bucket into a canonical OracleState.
        return [OracleState.from_iterable(group) for group in buckets.values()]

    def _candidate_attributes(self, state: OracleState) -> Iterable[str]:
        """Yield attributes that *could* split this state.

        We keep the logic simple and test every attribute; attributes whose
        values are already constant over the candidate set are discarded
        inside ``_solve_state``.
        """
        return self.attributes

    def _solve_state(self, state: OracleState) -> Tuple[float, Optional[str]]:
        """Internal DP: return (optimal_expected_cost, best_attribute).

        The cost is the expected number of *additional* questions needed to
        reach a leaf (an unsplittable equivalence class), starting from
        ``state``, under the optimal policy.
        """

        n = len(state)
        if n <= 1:
            # Already uniquely identified: no more questions needed.
            return 0.0, None

        best_cost = float("inf")
        best_attr: Optional[str] = None

        for attr in self._candidate_attributes(state):
            children = self._split_on_attribute(state, attr)
            if len(children) <= 1:
                # This attribute does not produce a proper split here.
                continue

            # Expected residual cost under a uniform prior.
            expected_cost = 1.0  # cost of asking this attribute
            for child in children:
                weight = len(child) / n
                sub_cost, _ = self._solve_state(child)
                expected_cost += weight * sub_cost

            if expected_cost < best_cost:
                best_cost = expected_cost
                best_attr = attr

        if best_attr is None:
            # No attribute yields a split: this is an irreducible equivalence class.
            # We treat this as a terminal node with zero additional cost.
            return 0.0, None

        return best_cost, best_attr

    # Wrap the solver in an LRU cache keyed by the canonical state ids.
    def _make_solver(self):
        @lru_cache(maxsize=None)
        def cached_solve(ids: Tuple[ItemId, ...]) -> Tuple[float, Optional[str]]:
            state = OracleState(ids)
            return self._solve_state(state)

        self._cached_solve = cached_solve  # type: ignore[attr-defined]

    @property
    def solver(self):
        """Return the memoized (cached) DP solver."""
        if not hasattr(self, "_cached_solve"):
            self._make_solver()
        return self._cached_solve  # type: ignore[attr-defined]

    # ---- Public API ------------------------------------------------------

    def optimal_root_cost(self) -> float:
        """Expected number of questions from the prior state under the optimal policy."""
        cost, _ = self.solver(self.root_state.ids)
        return cost

    def best_attribute_for_state(self, candidate_ids: Sequence[ItemId]) -> Optional[str]:
        """Return the optimal attribute to ask given the current candidate set.

        ``candidate_ids`` can be any iterable of IDs; it is canonicalized
        internally before the dynamic program is consulted.
        """
        state = OracleState.from_iterable(candidate_ids)
        _, attr = self.solver(state.ids)
        return attr

    def simulate_target(self, target_id: ItemId) -> Tuple[List[float], List[str]]:
        """Simulate the optimal question-asking policy for a fixed hidden target.

        Parameters
        ----------
        target_id:
            The ID of the hidden target object.  Must be a key in ``self.items``.

        Returns
        -------
        entropies:
            A list of posterior entropies (in bits) after each *step*,
            where step 1 is the prior (log2 of the initial candidate count),
            step 2 is after the first optimal attribute query, and so on.
        asked_attributes:
            The ordered list of attributes asked along the trajectory.

        Notes
        -----
        If the attributes in the dataset are not sufficient to uniquely
        identify the target, the trajectory ends at an equivalence class
        with size > 1, and the final entropy will be log2(class_size).
        """

        if target_id not in self.items:
            raise KeyError(f"Unknown target ID: {target_id!r}")

        # Start from the full candidate set.
        current_ids = set(self.root_state.ids)
        entropies: List[float] = [entropy_uniform(len(current_ids))]
        asked: List[str] = []

        while True:
            if len(current_ids) <= 1:
                break

            attr = self.best_attribute_for_state(tuple(current_ids))
            if attr is None:
                # No attribute can split this candidate set any further.
                break

            asked.append(attr)
            target_value = self.items[target_id][attr]

            # Restrict the candidate set to objects consistent with the answer.
            current_ids = {
                obj_id
                for obj_id in current_ids
                if self.items[obj_id][attr] == target_value
            }

            entropies.append(entropy_uniform(len(current_ids)))

        return entropies, asked


# -------------------------------------------------------------------------
# Example usage
# -------------------------------------------------------------------------

def _try_demo_with_local_json():
    """If run as a script from the dataset root, load the 300-object table and
    print a small demonstration of the oracle behavior.

    This block is intentionally lightweight: it only shows how to construct
    the oracle and run it on a single target.  Averaging over 30 random
    targets and reproducing the plot is left to the calling code.
    """
    # Expected relative location when this file lives in
    #   300-kary-oqa-dataset/oracle/kary300_exact_oracle_dp.py
    dataset_root = Path(__file__).resolve().parents[1]
    json_path = dataset_root / "data" / "kary300_Items.json"
    if not json_path.exists():
        print("[oracle demo] kary300_Items.json not found; "
              "place this module inside the dataset to run the demo.")
        return

    import json
    with json_path.open("r") as f:
        items = json.load(f)

    oracle = KaryOracleDP(items)
    print(f"[oracle demo] Optimal expected #questions from prior: "
          f"{oracle.optimal_root_cost():.3f}")

    # Pick an arbitrary target ID (first key) and show its entropy trajectory.
    some_id = next(iter(items.keys()))
    entropies, asked = oracle.simulate_target(some_id)
    print(f"[oracle demo] Example trajectory for target {some_id!r}:")
    for step, (H, attr) in enumerate(zip(entropies, ["<prior>"] + asked), start=1):
        print(f"  step {step:2d}: H = {H:5.3f} bits   (asked: {attr})")


if __name__ == "__main__":
    _try_demo_with_local_json()
