import csv

import pandas as pd
import logging

from connectome_data.constants import AC_ROOT, EMMONS_ROOT, EdgeType
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

    neurons = set(neuron_list())
    rows_to_drop = []
    for row in df.itertuples():
        if not neurons.issuperset((row.Source, row.Target)):
            rows_to_drop.append(row.Index)
    df.drop(rows_to_drop, inplace=True)

    return df


def neuron_list():
    with open(AC_ROOT / "nodelist.txt") as f:
        return [line.strip() for line in f]


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    df = emmons_herm_edgelist()
