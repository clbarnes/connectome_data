import csv
import json
import multiprocessing as mp
import os
import random
from concurrent.futures import as_completed
from concurrent.futures.process import ProcessPoolExecutor
from copy import deepcopy
from pathlib import Path
from typing import Optional, Iterator, Tuple, List, Iterable, Dict, Set
import itertools

import numpy as np
from tqdm import tqdm
import networkx as nx

from connectome_data.source_data import ac_herm_edgelist, old_ww_herm_edgelist
from connectome_data.constants import BENTLEY_ROOT, UNDIR_EDGELISTS_ROOT, SWAPS_PER_EDGE, N_RANDOM


def randomise_undir(g: nx.Graph, nswap: Optional[int]=None, seed: int=None) -> nx.Graph:
    g = deepcopy(g)
    if nswap is None:
        nswap = len(g.edges) * 10

    actual_swaps = nx.connected_double_edge_swap(g, nswap, seed)
    g.graph["nswap"] = actual_swaps

    return g


def randomise_undir_star(g_nswap_seed) -> nx.Graph:
    return randomise_undir(*g_nswap_seed)


def randomise_population_undir(g: nx.Graph, n: int=1000) -> Iterator[nx.Graph]:
    nswap = len(g.edges) * 10
    args = zip(itertools.repeat(g), itertools.repeat(nswap), range(n))
    cpus = mp.cpu_count()
    with mp.Pool(cpus) as p:
        yield from tqdm(p.imap_unordered(randomise_undir_star, args, chunksize=n//cpus), total=n)


def get_edgesets() -> Dict[str, Set[Tuple[str, str]]]:
    """Take raw data and convert it to an undirected, unweighted edge list"""
    # ac

    ac_full = ac_herm_edgelist()
    ac_set = set()
    for key in zip(ac_full["Source"], ac_full["Target"]):
        ac_set.add(tuple(sorted(key)))

    # ww

    ww_full = old_ww_herm_edgelist()
    ww_set = set()
    for key in zip(ww_full["Source"], ww_full["Target"]):
        ww_set.add(tuple(sorted(key)))

    # ma
    ma_set = set()
    with open(BENTLEY_ROOT / "edge_lists" / "edgelist_MA.csv") as f:
        reader = csv.reader(f)
        for row in reader:
            ma_set.add(tuple(sorted(row[:2])))

    # np

    np_set = set()
    with open(BENTLEY_ROOT / "edge_lists" / "edgelist_NP.csv") as f:
        reader = csv.reader(f)
        for row in reader:
            np_set.add(tuple(sorted(row[:2])))

    return {
        "ac": ac_set,
        "ww": ww_set,
        "ma": ma_set,
        "np": np_set,
        "ac_ma": ac_set.union(ma_set),
        "ac_ma_np": ac_set.union(ma_set).union(np_set),
        "ww_ma": ww_set.union(ma_set),
        "ww_ma_np": ww_set.union(ma_set).union(np_set),
    }


def save_edges(edges: Iterable[Tuple], fpath: os.PathLike):
    edgelist = sorted(edges)
    with open(fpath, 'w') as f:
        csv.writer(f).writerows(edgelist)


def randomise_edgelist(edgelist: List[Tuple[str, str]], idx, seed=None) -> Tuple[int, List[Tuple[str, str]], float]:
    nswap = len(edgelist) * SWAPS_PER_EDGE
    g = nx.Graph()
    g.add_edges_from(edgelist)
    random.seed(seed)
    true_nswap = nx.connected_double_edge_swap(g, nswap)
    return idx, hashable_edges(g.edges), true_nswap / nswap


def hashable_edges(edges: Iterable[Tuple[str, str]]) -> Tuple[Tuple[str, str], ...]:
    return tuple(sorted(
        tuple(sorted(pair)) for pair in edges
    ))


def generate_random_edgelists(edgelist, n=1000) -> Iterator[Tuple[int, List[Tuple[str, str]], float]]:
    edge_tup = hashable_edges(edgelist)
    rand = random.Random(edge_tup)
    with ProcessPoolExecutor() as exe:
        futures = [
            exe.submit(randomise_edgelist, edge_tup, idx, seed=rand.random())
            for idx in range(n)
        ]
        for fut in as_completed(futures):
            yield fut.result()


def randomise_all(n=N_RANDOM):
    edgesets = get_edgesets()
    pad_to = len(str(n))
    for name, edgeset in tqdm(sorted(edgesets.items())):
        this_dir: Path = UNDIR_EDGELISTS_ROOT / name
        this_dir.mkdir(exist_ok=True, parents=True)
        edgelist = hashable_edges(edgeset)
        save_edges(edgelist, this_dir / "real.csv")
        rand_dir = this_dir / "rand"
        rand_dir.mkdir(exist_ok=True, parents=True)
        succ = []
        for idx, rand_edgelist, prop_successful in tqdm(generate_random_edgelists(edgelist, n), total=n):
            fpath = rand_dir / (str(idx).zfill(pad_to) + ".csv")
            save_edges(rand_edgelist, fpath)
            succ.append(prop_successful)
        data = {
            "attempted_swaps_per_edge": SWAPS_PER_EDGE,
            "mean_prop_successful": np.mean(succ),
            "std_prop_successful": np.std(succ)
        }
        with open(this_dir / "info.json", 'w') as f:
            json.dump(data, f, sort_keys=True, indent=2)


if __name__ == '__main__':
    # g = nx.Graph()
    # g.add_edges_from((row.Source, row.Target) for row in ac_herm_edgelist().itertuples())
    # nswap = len(g.edges) * 10
    #
    # randomised = list(randomise_population_undir(g))
    # prop_successful_swaps = [r.graph["nswap"] / nswap for r in randomised]
    # print(f"{np.mean(prop_successful_swaps)} swaps successful; SD {np.std(prop_successful_swaps)}")

    randomise_all(N_RANDOM)
