from typing import List

import networkx as nx
from tqdm import tqdm

from connectome_data.constants import EdgeType, Directedness, Simplicity, Weightedness
from connectome_data.make_data.source_data import ac_herm_edgelist, emmons_herm_edgelist, df_to_edges, extrasyn_edges
from connectome_data.utils import GraphSerialiser

# NO CONTROLS
# ww-gap
# ww-chem
# ac-gap
# ac-chem

# CONTROLS
# ww
# ac
# ma
# np


edge_dfs = {
    'ac': ac_herm_edgelist(),
    'ww': emmons_herm_edgelist()
}


def edges_to_graph_ser(edges, name):
    graph = nx.OrderedDiGraph()
    graph.add_edges_from(edges)
    graph.graph['hash'] = hash(edges)
    graph_ser = GraphSerialiser(name, Simplicity.SIMPLE, Directedness.DIRECTED, Weightedness.UNWEIGHTED)
    graph_ser.dir.mkdir(parents=True, exist_ok=True)
    if not graph_ser.real_csv.is_file():
        graph_ser.graph = graph
    return graph_ser


def df_to_graph_ser(df, name, etype=None):
    edges = df_to_edges(df, etype, directedness=Directedness.DIRECTED)
    return edges_to_graph_ser(edges, name)


def no_controls():
    for ver, df in edge_dfs.items():
        for etype in (EdgeType.ELECTRICAL, EdgeType.CHEMICAL):
            df_to_graph_ser(df, f"{ver}-{etype}", etype)


def controls():
    serialisers: List[GraphSerialiser] = []
    for ver, df in edge_dfs.items():
        serialisers.append(df_to_graph_ser(df, ver))
    for etype in (EdgeType.MONOAMINE, EdgeType.NEUROPEPTIDE):
        edges = extrasyn_edges(etype)
        serialisers.append(edges_to_graph_ser(edges, str(etype)))

    for serialiser in tqdm(serialisers):
        for _ in tqdm(serialiser.rand_graphs(0, 1000), total=1):
            pass


if __name__ == '__main__':
    # no_controls()
    controls()
