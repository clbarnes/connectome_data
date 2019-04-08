import csv
import logging
from collections import Counter, defaultdict
from typing import Dict, Tuple, Sequence, Any

import networkx as nx
from tqdm import tqdm

from connectome_data.constants import EdgeType, REAL_EDGES, NEURONS
from connectome_data.figures.constants import TGT_DIR, DEFAULT_WIRING
from connectome_data.figures.hiveplots.classes import Scaler, GraphHivePlot
from connectome_data.figures.hiveplots.common import get_node_types
from connectome_data.figures.hiveplots.node_classes import name_to_class
from connectome_data.utils import TqdmStream

logger = logging.getLogger(__name__)

# colorbrewer 3-class Dark2 http://colorbrewer2.org/#type=qualitative&scheme=Dark2&n=3
# ignore green: too similar to possible choice for MAs
etype_colors = {
    EdgeType.ELECTRICAL: "#d95f02",  # orange
    EdgeType.CHEMICAL: "#7570b3",  # purple
}


def get_phys_edges(wiring=DEFAULT_WIRING):
    d = dict()
    for etype in (EdgeType.ELECTRICAL, EdgeType.CHEMICAL):
        name = f"{wiring}-{etype}"
        fpath = TGT_DIR / name / REAL_EDGES
        with open(fpath) as f:
            d[etype] = list(tuple(row[:2]) for row in csv.reader(f))
    return d


def phys_graph(edges: Dict[Any, Sequence[Tuple[Any, ...]]] = None):
    """To unweighted, undirected graph. Edges are only duplicated if type is different"""
    edges = edges or get_phys_edges()

    g = nx.OrderedMultiGraph()
    g.add_nodes_from(NEURONS)

    for etype, edge_list in tqdm(edges.items(), desc="constructing graph", unit="edge types"):
        edge_list = sorted({tuple(sorted(pair)) for pair in edge_list})
        for src, tgt in edge_list:
            g.add_edge(src, tgt, key=f"{etype}_{src}-{tgt}", etype=etype)

    return add_ntype_offset(g)


def add_ntype_offset(g):
    degs = dict(g.degree)
    scaler = Scaler((1, max(degs.values())), (0.001, 1), log=False)

    ntypes = get_node_types()

    for name, data in g.nodes(data=True):
        data["ntype"] = ntypes[name]
        deg = degs[name]
        data["offset"] = scaler(deg) if deg else 0

    return g


def phys_cls_graph(edges: Dict[Any, Sequence[Tuple[Any, ...]]] = None):
    orig = phys_graph(edges)

    new_edges = Counter()
    cls_ntypes = dict()
    cls_nset = defaultdict(set)
    for n, data in orig.nodes(data=True):
        cls = name_to_class[n]
        cls_ntypes[cls] = data["ntype"]
        cls_nset[cls].add(n)

    for orig_src, orig_tgt, etype in orig.edges(data="etype"):
        new_src, new_tgt = (name_to_class[n] for n in (orig_src, orig_tgt))
        new_edges.update([(new_src, new_tgt, etype)])

    cls_ntypes = {name_to_class[n]: ntype for n, ntype in orig.nodes(data="ntype")}

    g = nx.OrderedMultiGraph()
    g.add_nodes_from(cls_ntypes.keys())

    for (src, tgt, etype), weight in new_edges.items():
        g.add_edge(src, tgt, key=f"{etype}_{src}-{tgt}", etype=etype, weight=weight)

    norm_degs = {n: d/len(cls_nset[n]) for n, d in dict(g.degree).items()}
    scaler = Scaler((0, max(norm_degs.values())), (0.001, 1), log=False)

    for n, data in g.nodes(data=True):
        data["neurons"] = cls_nset[n]
        data["ntype"] = cls_ntypes[n]
        deg = norm_degs[n]
        data["offset"] = scaler(deg) if deg else 0

    return g


class PhysicalGraphHivePlot(GraphHivePlot):
    seed = 2

    @classmethod
    def from_edges(cls, edges: Dict[Any, Sequence[Tuple[Any, ...]]] = None, *args, **kwargs):
        """To unweighted, undirected graphs. Edges are only duplicated if type is different"""
        edges = edges or get_phys_edges()
        g = phys_graph(edges)
        ax_names = sorted({ntype for _, ntype in g.nodes(data="ntype")})

        return cls(g, ax_names, *args, **kwargs)


class ClassPhysicalGraphHivePlot(GraphHivePlot):
    seed = 3

    @classmethod
    def from_edges(cls, edges: Dict[Any, Sequence[Tuple[Any, ...]]] = None, *args, **kwargs):
        """To unweighted, undirected graphs. Edges are only duplicated if type is different"""
        edges = edges or get_phys_edges()
        g = phys_cls_graph(edges)
        ax_names = sorted({ntype for _, ntype in g.nodes(data="ntype")})

        return cls(g, ax_names, *args, **kwargs)


def main(fpath="phys_cls.svg"):
    return ClassPhysicalGraphHivePlot.from_edges(None, etype_colors).save(fpath)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, stream=TqdmStream)
    main()

    print("ready")
