import json
from typing import Tuple, Dict, Hashable, List, Iterator, FrozenSet

import networkx as nx
import numpy as np


def yield_src_tgt_weight():
    with open('example.json') as f:
        for src_tgt, weight in json.load(f).items():
            src, tgt = (int(i) for i in src_tgt.split())
            yield (src, tgt), weight


def bisect(lst: List) -> Tuple[List, List]:
    mid = len(lst) // 2
    return lst[:mid], lst[mid:]


def revdict(d: Dict[Hashable, Hashable]):
    out = {v: k for k, v in d.items()}
    if len(d) != len(out):
        raise ValueError("Dict is not reversible (some values are identical)")
    return out


class MultiSankeySorter:
    def __init__(self, columns: List[List[Hashable]], weights: Dict[FrozenSet, float], fixed_idx=0):
        """

        :param columns: list of columns, left to right, where each column is a list of labels.
        :param weights: dict mapping label pair (as a frozenset) to a weight (number)
        :param fixed_idx: which index in the ``columns`` list is already sorted
            (i.e. is used as the foundation for the rest of the diagram)
        """
        self.columns = columns.copy()
        self.weights = weights
        self.fixed_idx = fixed_idx

    def _pair_idxs(self):
        fixed_idx = self.fixed_idx
        while fixed_idx > 0:
            yield fixed_idx, fixed_idx - 1
            fixed_idx -= 1

        fixed_idx = self.fixed_idx
        while fixed_idx + 1 < len(self.columns):
            yield fixed_idx, fixed_idx + 1
            fixed_idx += 1

    def sort(self):
        """Return the list of columns where all but the "fixed" column have been sorted to minimise crossovers"""
        for fixed_idx, sorting_idx in self._pair_idxs():
            self.columns[sorting_idx] = list(SankeySorter(
                self.columns[fixed_idx], self.columns[sorting_idx], self.weights
            ))
        return self.columns


class SankeySorter:
    def __init__(self, fixed_col: List[Hashable], sorting_col: List[Hashable], weights: Dict[frozenset, float]):
        """Given one column's ordering, sort another column to minimise crossovers in a bipartite graph.

        Uses approach from [Çakıroḡlu2008]_.

        To get the reordered ``sorting_column``, just iterate through this object.

        :param fixed_col: list of labels in column which is not to be re-ordered
        :param sorting_col: list of labels in column to be re-ordered
        :param weights: dict mapping label pair (as a frozenset) to a weight (number).
            Labels which are not in the given columns will be ignored.

        .. [Çakıroḡlu2008] https://doi.org/10.1016/j.jda.2008.08.003
        """
        self.name_to_idx = {name: idx for idx, name in enumerate(fixed_col + sorting_col, 1)}

        self.L0 = [self.name_to_idx[n] for n in fixed_col]
        self.L1 = [self.name_to_idx[n] for n in sorting_col]

        nodes = set(fixed_col) | set(sorting_col)
        self.graph = nx.Graph()
        for src_tgt, weight in weights.items():
            if not src_tgt.issubset(nodes):
                continue
            src, tgt = list(src_tgt)

            self.graph.add_edge(self.name_to_idx[src], self.name_to_idx[tgt], weight=weight)

    @property
    def n0(self):
        return len(self.L0)

    @property
    def n1(self):
        return len(self.L1)

    def _W(self, x, p):  # W(n, p)
        try:
            return self.graph.edges[x, p]["weight"]
        except KeyError:
            return 0

    def _W_between(self, x, pmin, pmax):  # W(n)_pmin^pmax
        return sum(self._W(x, p) for p in self.graph.neighbors(x) if pmin <= p <= pmax)

    def _sort(self) -> Iterator[int]:
        for r, Pr in enumerate(self._phase1()):
            yield from self._phase2(r, Pr)

    def __iter__(self) -> Iterator[Hashable]:
        idx_to_name = revdict(self.name_to_idx)
        for idx in self._sort():
            yield idx_to_name[idx]

    def _phase1(self) -> List[List[int]]:  # phase 1, coarse-grained
        P = [[] for _ in self.L0]
        for u in self.L1:
            leftsum = 0
            rightsum = self._W_between(u, 2, self.n0)
            for r in range(0, self.n0):
                if leftsum >= rightsum:
                    break
                leftsum += self._W(u, r + 1)
                rightsum += self._W(u, r + 2)
            P[r].append(u)
        return P

    def _metric(self, u, v, r):
        return self._W_between(u, 1, r) * self._W_between(v, r + 1, self.n0)

    def _phase2(self, r, Pr) -> Iterator[int]:  # phase 2, fine-grained
        if len(Pr) == 1:
            yield Pr.pop()
        if len(Pr) == 0:
            return

        pi_Pr1, pi_Pr2 = (self._phase2(r, Pri) for Pri in bisect(Pr))

        ## ???
        ## "New node `a` s.t. (such that) W_between(a, 1, r) == −1  and W_between(a, r+1, n0) == 0"
        # pi_Pr1.append(a)
        # pi_Pr2.append(a)
        ## ???

        u = next(pi_Pr1)
        v = next(pi_Pr2)

        for _ in range(1, len(Pr) + 1):
            if self._metric(v, u, r) <= self._metric(u, v, r):
                yield u
                try:
                    u = next(pi_Pr1)
                except StopIteration:
                    yield v
                    break
            else:
                yield v
                try:
                    v = next(pi_Pr2)
                except StopIteration:
                    yield u
                    break

        yield from pi_Pr1
        yield from pi_Pr2


if __name__ == '__main__':
    rand = np.random.RandomState(1)
    # s = MultiSankeySorter()
    # print(s.sort())
