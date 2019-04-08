import json

from connectome_data.constants import NODES_DATA, NodeType


def get_node_types():
    with open(NODES_DATA) as f:
        return {n: NodeType(info["ntype"]) for n, info in json.load(f).items()}
