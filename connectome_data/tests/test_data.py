from collections import Counter

import pytest

from connectome_data.make_data.source_data import emmons_herm_edgelist, ac_herm_edgelist, old_ww_herm_edgelist
from connectome_data.constants import EdgeType


@pytest.fixture(params=[
    emmons_herm_edgelist,
    ac_herm_edgelist,
    old_ww_herm_edgelist
])
def phys_df(request):
    return request.param()


def test_unique_edges(phys_df):
    keys = [(row.Source, row.Target, row.Type) for row in phys_df.itertuples()]
    key_counts = Counter(keys)
    non_one = [(k, v) for k, v in key_counts.items() if v > 1]
    assert non_one == []


def test_symmetric_gap_junctions(phys_df):
    gj = phys_df.loc[phys_df["Type"] == EdgeType.ELECTRICAL]
    keys = {(row.Source, row.Target) for row in gj.itertuples()}
    symmetric = set()
    asymmetric = set()

    for k in keys:
        if k[::-1] in keys:
            symmetric.add(k)
        else:
            asymmetric.add(k)

    proportion_asymmetric = len(asymmetric) / len(gj)
    assert proportion_asymmetric == 0, f"{len(asymmetric)} asymmetric edges ({proportion_asymmetric})\n\t{sorted(asymmetric)}"


def df_to_dict(df):
    d = dict()
    for row in df.itertuples():
        d[(row.Source, row.Target, row.Type)] = row.Weight


def test_old_new_ww():
    old = old_ww_herm_edgelist()
    new = emmons_herm_edgelist()

    assert df_to_dict(old) == df_to_dict(new)
