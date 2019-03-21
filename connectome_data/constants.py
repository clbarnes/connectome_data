import os
from enum import Enum
from pathlib import Path
from typing import Type, Dict, Optional

import networkx as nx

PACKAGE_ROOT = Path(__file__).absolute().parent
PROJECT_ROOT = PACKAGE_ROOT.parent

SRC_ROOT = PROJECT_ROOT / "source_data"
EMMONS_ROOT = SRC_ROOT / "emmons"
BENTLEY_ROOT = SRC_ROOT / "bentley_2016_S1_dataset" / "S1 Dataset. Included are edge lists and source data for monoamine and neuropeptide networks"
AC_ROOT = SRC_ROOT / "connectome_construct" / "physical" / "src_data"

TGT_ROOT = PROJECT_ROOT / "target_data"
# UNDIR_EDGELISTS_ROOT = TGT_ROOT / "undir_simple_edgelists"

SWAPS_PER_EDGE = 10
N_RANDOM = 100

with open(AC_ROOT / "nodelist.txt") as f:
    NEURONS = tuple(sorted(line.strip() for line in f))

NEURON_COUNT = len(NEURONS)


class StrEnum(Enum):
    def __str__(self):
        return str(self.value)

    def __eq__(self, other):
        if isinstance(other, str):
            return str(self) == other
        else:
            return super().__eq__(other)

    def __hash__(self):
        return hash(self.value)


class EdgeType(StrEnum):
    CHEMICAL = "chemical"
    ELECTRICAL = "electrical"
    MONOAMINE = "monoamine"
    NEUROPEPTIDE = "neuropeptide"


class Wiring(StrEnum):
    ALBERTSON_CHKLOVSKII = "ac"
    WORMWIRING = "ww"


class Directedness(StrEnum):
    UNDIRECTED = "und"
    DIRECTED = "dir"

    @property
    def is_directed(self):
        return self == type(self).DIRECTED

    @classmethod
    def from_graph(cls, g):
        if isinstance(g, nx.DiGraph):
            return cls.DIRECTED
        elif isinstance(g, nx.Graph):
            return cls.UNDIRECTED
        raise ValueError("Not recognised graph type")


class Simplicity(StrEnum):
    MULTI = "multi"
    SIMPLE = "simple"

    @property
    def is_simple(self):
        return self == type(self).SIMPLE

    @classmethod
    def from_graph(cls, g):
        if isinstance(g, nx.MultiGraph):
            return cls.MULTI
        elif isinstance(g, nx.Graph):
            return cls.SIMPLE
        raise ValueError("Not recognised graph type")


class Weightedness(StrEnum):
    UNWEIGHTED = "unweighted"
    WEIGHTED = "weighted"

    @property
    def is_weighted(self):
        return self == type(self).WEIGHTED

    @classmethod
    def from_graph(cls, g, var_name="weight"):
        val_set = set()

        for _, _, val in g.edges(data=var_name):
            if val is None:
                return cls.UNWEIGHTED
            val_set.add(val)

        return cls.UNWEIGHTED if val_set == {1} else cls.WEIGHTED


def graph_type(simp: Simplicity = Simplicity.SIMPLE, dire: Directedness = Directedness.UNDIRECTED):
    types: Dict[Simplicity, Dict[Directedness, Type[nx.Graph]]] = {
        Simplicity.SIMPLE: {
            Directedness.UNDIRECTED: nx.OrderedGraph,
            Directedness.DIRECTED: nx.OrderedDiGraph
        },
        Simplicity.MULTI: {
            Directedness.UNDIRECTED: nx.OrderedMultiGraph,
            Directedness.DIRECTED: nx.OrderedMultiDiGraph
        }
    }
    return types[simp][dire]


def tgt_dir(
    simplicity: Optional[Simplicity] = None,
    directedness: Optional[Directedness] = None,
    weightedness: Optional[Weightedness] = None,
    tail: Optional[os.PathLike] = None,
    root: Optional[Path] = TGT_ROOT
):
    """directedness requires simplicity, weightedness requires directedness"""
    p = root
    if simplicity:
        p /= str(simplicity)
        if directedness:
            p /= str(directedness)
            if weightedness:
                p /= str(weightedness)
    if tail:
        p /= tail
    return p
