import os
from enum import Enum
from functools import total_ordering
from pathlib import Path
from typing import Type, Dict, Optional, Callable, Tuple

import networkx as nx

PACKAGE_ROOT = Path(__file__).absolute().parent
PROJECT_ROOT = PACKAGE_ROOT.parent

SRC_ROOT = PROJECT_ROOT / "source_data"
EMMONS_ROOT = SRC_ROOT / "emmons"
BENTLEY_ROOT = SRC_ROOT / "bentley_2016_S1_dataset" / "S1 Dataset. Included are edge lists and source data for monoamine and neuropeptide networks"
AC_ROOT = SRC_ROOT / "connectome_construct" / "physical" / "src_data"
DISTANCE_JSON = SRC_ROOT / "connectome_construct" / "metadata" / "tgt_data" / "dist_info.json"
NODES_DATA = SRC_ROOT / "connectome_construct" / "metadata" / "tgt_data" / "node_data.json"

TGT_ROOT = PROJECT_ROOT / "target_data"
# UNDIR_EDGELISTS_ROOT = TGT_ROOT / "undir_simple_edgelists"

SWAPS_PER_EDGE = 10
N_RANDOM = 100

RAND_DIR = 'rand'
REAL_EDGES = "real.csv"
REAL_METRICS = "real.json"
ENSEMBLE_METRICS = "ensemble.hdf5"

with open(AC_ROOT / "nodelist.txt") as f:
    NEURONS = tuple(sorted(line.strip() for line in f))

NEURON_COUNT = len(NEURONS)


@total_ordering
class StrEnum(Enum):
    def __str__(self):
        return str(self.value)

    def __eq__(self, other):
        try:
            return super().__eq__(type(self)(other))
        except Exception:
            return NotImplemented

    def __lt__(self, other):
        try:
            other = type(self)(other)
        except:
            return NotImplemented

        order = list(type(self))
        return order.index(self) < order.index(other)

    def __hash__(self):
        return hash(self.value)


class EdgeType(StrEnum):
    CHEMICAL = "chemical"
    ELECTRICAL = "electrical"
    MONOAMINE = "monoamine"
    NEUROPEPTIDE = "neuropeptide"

    @classmethod
    def physical(cls):
        return cls.CHEMICAL, cls.ELECTRICAL

    @classmethod
    def extrasynaptic(cls):
        return cls.MONOAMINE, cls.NEUROPEPTIDE


class Monoamine(StrEnum):
    DOPAMINE = "dopamine"
    OCTOPAMINE = "octopamine"
    SEROTONIN = "serotonin"
    TYRAMINE = "tyramine"

    def ellision(self):
        return str(self)[:3]

    def abbreviation(self):
        return {
            type(self).SEROTONIN: "5-HT",
            type(self).DOPAMINE: "DA",
            type(self).OCTOPAMINE: "OA",
            type(self).TYRAMINE: "TA"
        }


class NodeType(StrEnum):
    INTERNEURON = "interneuron"
    MOTOR = "motor"
    SENSORY = "sensory"


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

    def __bool__(self):
        return self.is_directed


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

    def __bool__(self):
        return self.is_simple


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

    def __bool__(self):
        return self.is_weighted


# def make_to_from_fns(falsey: str, truthy: str) -> Tuple[Callable[[bool], str], Callable[[str], bool]]:
#     false_true = [falsey, truthy]
#
#     def to_str(arg: bool) -> str:
#         return false_true[bool(arg)]
#
#     def from_str(arg: str) -> bool:
#         idx = false_true.index(arg)
#         if idx in (0, 1):
#             return bool(idx)
#         raise ValueError(f"Argument {arg} not valid")
#
#     return to_str, from_str
#
#
# simple_to_str, simple_from_str = make_to_from_fns('multi', 'simple')
# directed_to_str, directed_from_str = make_to_from_fns('und', 'dir')
# weighted_to_str, weighted_from_str = make_to_from_fns('unweighted', 'weighted')


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
    if simplicity is not None:
        p /= str(simplicity)
        if directedness is not None:
            p /= str(directedness)
            if weightedness is not None:
                p /= str(weightedness)
    if tail:
        p /= tail
    return p
