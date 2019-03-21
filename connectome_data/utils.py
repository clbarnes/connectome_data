import csv
import os
import random
import re
from abc import ABC
from concurrent.futures.process import ProcessPoolExecutor
from pathlib import Path
from typing import Type, Tuple, Iterator, Iterable, Union

import networkx as nx
import numpy as np
from bct.algorithms.reference import randmio_dir, randmio_und

from connectome_data.constants import (
    Directedness, Simplicity, Weightedness, TGT_ROOT, tgt_dir, graph_type,
    StrEnum, SWAPS_PER_EDGE
)


def unpad_name(s: str) -> str:
    s = s.strip()
    out = ''
    this_element = ''
    is_digit = s[0].isdigit()

    for next_char in s:
        next_is_digit = next_char.isdigit()
        if next_is_digit != is_digit:
            this_element = unpad_element(this_element)
            out += this_element
            this_element = ''
            is_digit = next_is_digit
        this_element += next_char

    this_element = unpad_element(this_element)
    out += this_element
    return out


def unpad_element(s: str) -> str:
    if s.isdigit():
        s = str(int(s))
    return s


def re_or(e: Type[StrEnum]):
    return f"(?P<{e.__name__.lower()}>({'|'.join(str(i) for i in e)}))"


name_elements = [Simplicity, Directedness, Weightedness]
sep = os.path.sep
re_str = f"{sep}{sep.join(re_or(e) for e in name_elements)}{sep}(?P<name>.+)"
GRAPH_DIR_RE = re.compile(re_str)


class EdgesNormaliser:
    def __init__(self, simple=True, directed=False, weighted=False):
        self.simple = simple
        self.directed = directed
        self.weighted = weighted

    def get_weight(self, row, existing=0, default=1):
        if self.weighted:
            return existing + (default if len(row) < 2 else row[2])
        else:
            return default

    @staticmethod
    def csv_to_rows(fpath):
        with open(fpath) as f:
            yield from csv.reader(f)

    @staticmethod
    def rows_to_csv(rows, fpath):
        with open(fpath, 'w') as f:
            writer = csv.writer(f)
            writer.writerows(rows)

    def normalise(self, rows):
        rowlist = []
        for row in rows:
            if self.weighted:
                weight = 1 if len(row) < 2 else row[2]
            else:
                weight = 1

            if self.directed:
                src_tgt = tuple(row[:2])
            else:
                src_tgt = tuple(sorted(row[:2]))

            rowlist.append(src_tgt + (weight,))

        if not self.simple:
            return tuple(sorted(rowlist))

        row_dict = dict()
        for src, tgt, weight in rowlist:
            new_weight = row_dict.get((src, tgt), 0) + weight if self.weighted else 1
            row_dict[(src, tgt)] = new_weight

        return tuple(sorted((src, tgt, weight) for (src, tgt), weight in row_dict.items()))


def to_np_seed(hashable):
    return random.Random(hashable).randint(0, 2**32-1)


class GraphSerialiser:
    tgt_root = TGT_ROOT
    max_rand_graphs = 999

    def __init__(self, name, simplicity, directedness, weightedness):
        self.name = name
        self.directedness = directedness
        self.simplicity = simplicity
        self.weightedness = weightedness

        self.edges_normaliser = EdgesNormaliser(
            self.simplicity.is_simple, self.directedness.is_directed, self.weightedness.is_weighted
        )

        self._graph = None
        self._nodelist = None

        self.dir = tgt_dir(self.simplicity, self.directedness, self.weightedness, self.name, self.tgt_root)
        self.rand_dir = self.dir / "rand"
        self.info_json = self.dir / "info.json"
        self.real_csv = self.dir / "real.csv"

    @property
    def nodelist(self):
        if self._nodelist is None:
            self._nodelist = tuple(sorted(self.graph.nodes))
        return self._nodelist

    def _read_graph(self, fpath: Path):
        graph = self.gtype()
        edges = self.edges_normaliser.csv_to_rows(fpath)
        graph.graph["hash"] = hash(edges)
        for src, tgt, weight in edges:
            graph.add_edge(src, tgt, weight=weight)
        return graph

    def _adj_to_graph(self, adj: np.ndarray):
        """May not preserve edge order"""
        graph = self.gtype(adj)
        nx.relabel_nodes(graph, dict(enumerate(self.nodelist)), copy=False)
        return graph

    def _write_graph(self, fpath: Path, graph: Union[nx.Graph, np.ndarray]):
        if isinstance(graph, np.ndarray):
            graph = self._adj_to_graph(graph)

        fpath.parent.mkdir(parents=True, exist_ok=True)

        self.edges_normaliser.rows_to_csv(
            self.edges_normaliser.normalise(graph.edges(data='weight', default=1)),
            fpath
        )
        return graph

    def rand_paths(self, start=0, stop=None) -> Iterator[Tuple[int, Path]]:
        if stop is None:
            stop = self.max_rand_graphs
        idx = start
        while idx < stop:
            yield idx, self.rand_dir / str(idx).zfill(len(str(self.max_rand_graphs))) + '.csv'
            idx += 1

    def generate_random(self, idx_paths: Iterable[Tuple[int, Path]]):
        randomiser = randmio_dir if self.directedness.is_directed else randmio_und
        adj = nx.to_numpy_array(self.graph, sorted(self.graph.nodes))
        with ProcessPoolExecutor() as exe:
            idx_path_futs = []
            for idx, fpath in idx_paths:
                seed = to_np_seed((self.graph.graph["hash"], idx))
                idx_path_futs.append(
                    (idx, fpath, exe.submit(randomiser, adj, SWAPS_PER_EDGE, seed=seed))
                )

            for idx, fpath, fut in idx_path_futs:
                rand_adj, succ_swaps = fut.result()
                rand_g = self._write_graph(fpath, rand_adj)
                yield idx, rand_g

    def rand_graphs(self, start=0, stop=None):
        any_exist = False
        any_missing = False
        for idx, path in self.rand_paths(start, stop):
            if path.is_file():
                any_exist = True
            else:
                any_missing = True

            if any_exist and any_missing:
                raise ValueError("Only some of requested random graphs exist")

        if any_missing:  # all are missing
            yield from self.generate_random(self.rand_paths(start, stop))
        elif any_exist:  # all exist
            for idx, fpath in self.rand_paths(start, stop):
                yield idx, self._read_graph(fpath)
        else:
            raise RuntimeError("Inconsistent graph existence (do any exist? are any missing?)")

    @property
    def gtype(self) -> Type[nx.OrderedGraph]:
        return graph_type(self.simplicity, self.directedness)

    @property
    def graph(self):
        if not self._graph:
            self._graph = self._read_graph(self.real_csv)
        return self._graph

    @graph.setter
    def graph(self, g: nx.Graph):
        desired_gtype = graph_type(self.simplicity, self.directedness)
        if desired_gtype != type(g):
            raise ValueError(
                f"Given graph ({type(g).__name__}) is not of the correct networkx.Graph subtype ({desired_gtype.__name__})"
            )
        if self.real_csv.exists():
            raise ValueError(f"Cannot set graph; it already exists at {self.real_csv}")
        self._write_graph(self.real_csv, g)
        self._graph = g

    @classmethod
    def from_dir(cls, dpath: Path):
        match = GRAPH_DIR_RE.search(os.fspath(dpath))
        name_element_strs = {typ.__name__.lower(): typ for typ in name_elements}

        gtype_info = {key: name_element_strs[key](val) for key, val in match.groupdict()}
        gtype = graph_type(gtype_info["simplicity"], gtype_info["directedness"])

        instance = cls(**gtype_info)

        instance.graph = gtype()

    @classmethod
    def from_graph(cls, name, graph):
        instance = cls(name, Simplicity.from_graph(graph), Directedness.from_graph(graph), Weightedness.from_graph(graph))
        instance.graph = graph

        return instance
