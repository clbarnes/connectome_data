import csv
import logging
import random
from collections import Counter
from typing import Dict, Any, Sequence, Tuple

import networkx as nx
from tqdm import tqdm

from connectome_data.constants import NodeType, EdgeType, BENTLEY_ROOT, Monoamine
from connectome_data.figures.hiveplots.classes import SplitHivePlot, Scaler, GraphHivePlot
from connectome_data.figures.hiveplots.node_classes import name_to_class
from connectome_data.figures.hiveplots.phys import logger, phys_cls_graph
from connectome_data.utils import TqdmStream

# colorbrewer2 4-class Set1 http://colorbrewer2.org/#type=qualitative&scheme=Set1&n=4
etype_colors = {
    Monoamine.DOPAMINE: "#e41a1c",  # red
    Monoamine.OCTOPAMINE: "#377eb8",  # blue
    Monoamine.SEROTONIN: "#4daf4a",  # green
    Monoamine.TYRAMINE: "#984ea3"  # purple
}


def get_ma_edges(dop56=False):
    fname = f"edgelist_MA{'_incl_dop-5_dop-6' if dop56 else ''}.csv"
    fpath = BENTLEY_ROOT / "edge_lists" / fname
    edges = {m: [] for m in Monoamine}
    with open(fpath) as f:
        for row in csv.reader(f):
            edges[Monoamine(row[2])].append(tuple(row[:2]))
    return edges


def plot_ma_graph(fpath, g: nx.OrderedMultiDiGraph):
    h = SplitHivePlot(
        fpath, [NodeType.INTERNEURON, NodeType.MOTOR, NodeType.SENSORY],
    )
    out_degs = dict(g.out_degree)

    scaler = Scaler((1, max(out_degs.values())), (0.001, 1), log=False)
    for name, ntype in tqdm(g.nodes(data="ntype"), desc="adding to plot", unit="nodes"):
        raw_deg = out_degs[name]
        scaled = scaler(raw_deg) if raw_deg else 0
        h.add_node(ntype, name, scaled)

    edges = list(g.edges(data="etype"))
    random.Random(1).shuffle(edges)
    for src, tgt, etype in tqdm(edges, desc="adding to plot", unit="edges"):
        h.add_edge(src, tgt, stroke=etype_colors[etype], stroke_width=1, stroke_opacity=0.5)

    label_col = [(k, etype_colors[k]) for k in EdgeType.physical()]
    h.add_legend((0.1, 0.2), label_col)

    h.save()
    logger.info("Saved")


def ma_cls_graph(edges: Dict[Any, Sequence[Tuple[Any, ...]]] = None):
    edges = edges or get_ma_edges()
    g = phys_cls_graph()
    g.remove_edges_from(list(g.edges(keys=True)))

    for etype, edge_list in edges.items():
        weights = Counter(tuple(name_to_class[n] for n in pair) for pair in edge_list)
        for (src, tgt), weight in weights.items():
            g.add_edge(src, tgt, key=f"{etype}_{src}-{tgt}", etype=etype, weight=weight)

    return g


class ClassMonoamineGraphHivePlot(GraphHivePlot):
    seed = 3

    @classmethod
    def from_edges(cls, edges: Dict[Any, Sequence[Tuple[Any, ...]]] = None, *args, **kwargs):
        """To unweighted, undirected graphs. Edges are only duplicated if type is different"""
        edges = edges or get_ma_edges()
        g = ma_cls_graph(edges)
        ax_names = sorted({ntype for _, ntype in g.nodes(data="ntype")})

        return cls(g, ax_names, *args, **kwargs)


def main(fpath="ma_cls.svg"):
    ClassMonoamineGraphHivePlot.from_edges(None, etype_colors).save(fpath)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, stream=TqdmStream)
    main()

    print("ready")
