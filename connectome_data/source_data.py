import csv
from typing import Tuple

import pandas as pd
import logging

from connectome_data.constants import AC_ROOT, EMMONS_ROOT, EdgeType, NEURONS, Simplicity, Directedness, Weightedness, \
    BENTLEY_ROOT
from connectome_data.utils import unpad_name

logger = logging.getLogger(__name__)


def connectome_construct_edgelists_to_df(ver='ac'):
    headers = ["Source", "Target", "Weight", "Type"]
    data = []
    with open(AC_ROOT / f"syn_edgelist_{ver}.csv") as f:
        reader = csv.reader(f)
        for src, tgt, weight in reader:
            data.append([src.strip(), tgt.strip(), int(weight), EdgeType.CHEMICAL])

    with open(AC_ROOT / f"gap_edgelist_{ver}.csv") as f:
        reader = csv.reader(f)
        for src, tgt, weight in reader:
            data.append([src.strip(), tgt.strip(), int(weight), EdgeType.ELECTRICAL])

    return pd.DataFrame(data, columns=headers)


def ac_herm_edgelist() -> pd.DataFrame:
    return connectome_construct_edgelists_to_df('ac')


def old_ww_herm_edgelist() -> pd.DataFrame:
    return connectome_construct_edgelists_to_df('ww')


def emmons_herm_edgelist() -> pd.DataFrame:
    df = pd.read_csv(EMMONS_ROOT / "herm_full_edgelist.csv")
    # df.columns = [h.lower() for h in df.columns]
    for column in ("Source", "Target"):
        df[column] = [unpad_name(s) for s in df[column]]

    df["Type"] = [EdgeType(s.strip()) for s in df["Type"]]

    neurons = set(NEURONS)
    rows_to_drop = []
    for row in df.itertuples():
        if not neurons.issuperset((row.Source, row.Target)):
            rows_to_drop.append(row.Index)
    df.drop(rows_to_drop, inplace=True)

    return df


def extrasyn_edges(etype) -> Tuple[Tuple[str, str], ...]:
    etype_abbrev = {
        EdgeType.MONOAMINE: 'MA',
        EdgeType.NEUROPEPTIDE: 'NP',
    }
    src_fpath = BENTLEY_ROOT / "edge_lists" / f"edgelist_{etype_abbrev[etype]}.csv"
    edgeset = set()
    with open(src_fpath) as f:
        for row in csv.reader(f):
            edgeset.add(tuple(row[:2]))
    return tuple(sorted(edgeset))


def df_to_edges(
    df: pd.DataFrame, etype: EdgeType=None,
    simplicity=Simplicity.SIMPLE, directedness=Directedness.DIRECTED, weightedness=Weightedness.UNWEIGHTED
) -> Tuple[Tuple[str, str], ...]:
    if not simplicity.is_simple:
        raise NotImplementedError("Multigraphs not implemented")
    if weightedness.is_weighted:
        raise NotImplementedError("Weighted graphs are not implemented")

    edgeset = set()

    for row in df.itertuples():
        if etype and etype != row.Type:
            continue
        to_add = (row.Source, row.Target)
        if not directedness.is_directed:
            edgeset.add(tuple(sorted(to_add)))
        else:
            edgeset.add(to_add)
            if row.Type == EdgeType.ELECTRICAL:
                edgeset.add(to_add[::-1])

    return tuple(sorted(edgeset))


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    df = emmons_herm_edgelist()
