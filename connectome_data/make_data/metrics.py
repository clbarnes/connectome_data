import csv
import json
import os
import traceback
import warnings
from abc import ABC
from concurrent.futures import as_completed
from concurrent.futures.process import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable

import h5py
import numpy as np
from bct import (
    density_dir,
    distance_bin,
    charpath,  # nb acts on distance matrix
    clustering_coef_bd,  # NB binarise first, mean after
    transitivity_bd,  # NB binarise first
    modularity_dir,
    degrees_dir,  # standardise in/out/summed
    binarize)
from tqdm import tqdm

from connectome_data.constants import tgt_dir, Simplicity, Directedness, Weightedness, NEURONS


THREADS = 25


# density (dir)
# mean_path_length (dir)
# global_efficiency (dir)
# mean_clustering_coefficient
# weighted_mean_clustering_coefficient
# clustering_coefficient
# transitivity
# maximum_modularity
# assortativity
# mean_betweenness_centrality
# betweenness_centrality
# degrees


class NumpyEncoder(json.JSONEncoder):
    """https://stackoverflow.com/a/47626762/2700168"""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.generic):
            return obj.item()
        return super().default(self, obj)


def edge_tuples(fpath):
    with open(fpath) as f:
        for row in csv.reader(f):
            src, tgt = row[:2]
            weight = float(row[2]) if len(row) >= 3 else 1
            yield src, tgt, weight


class Metrics(ABC):
    def to_json(self, fpath, **kwargs):
        json_kwargs = dict(sort_keys=True, indent=2, cls=NumpyEncoder)
        json_kwargs.update(kwargs)
        with open(fpath, 'w') as f:
            json.dump(self.__dict__, f, **json_kwargs)

    @classmethod
    def from_json(cls, fpath):
        with open(fpath) as f:
            return cls(**json.load(f))


@dataclass
class SimpleDirectedUnweightedMetrics(Metrics):
    density: float
    mean_path_length: float
    global_efficiency: float
    clustering_coefficients: Dict[str, float]
    transitivity: float
    maximum_modularity: float
    modules: Dict[str, int]
    out_degree: Dict[str, int]


class AdjacencyWrapper:
    def __init__(self, adj: np.ndarray, directedness: Directedness=None, weightedness: Weightedness=None, nodes=None):
        self.adj = adj
        self.nodes = nodes or tuple(range(len(self.adj)))
        self.node_idx = {name: idx for idx, name in enumerate(nodes)}
        self.directedness = directedness or list(Directedness)[self._is_directed()]
        self.weightedness = weightedness or list(Weightedness)[self._is_weighted()]

        self._binarised = None
        if not self.weightedness.is_weighted:
            self.adj = self.binarised

        self._undirected = None
        self._bin_und = None

    @property
    def binarised(self):
        if self._binarised is None:
            self._binarised = binarize(self.adj)
        return self._binarised

    @property
    def undirected(self):
        if self._undirected is None:
            self._undirected = (self.adj + self.adj.T) / 2
        return self._undirected

    @property
    def bin_undir(self):
        if self._bin_und is None:
            self._bin_und = binarize(self.undirected)
        return self._bin_und

    def idx(self, node):
        return self.node_idx[node]

    def _is_directed(self):
        return self.adj == self.adj.T

    def _is_weighted(self):
        return not np.allclose(np.unique(self.adj), [0, 1])

    @classmethod
    def _from_csv_no_nodes(cls, fpath, directedness: Directedness):
        raise NotImplementedError()

    @classmethod
    def from_csv(cls, fpath: Path, directedness: Directedness=None, weightedness: Weightedness=None, nodes=NEURONS):
        if not nodes:
            return cls._from_csv_no_nodes(fpath, directedness)

        adj = np.zeros((len(nodes), len(nodes)))
        node_idx = {name: idx for idx, name in enumerate(nodes)}
        for src, tgt, weight in edge_tuples(fpath):

            adj[node_idx[src], node_idx[tgt]] += weight

            if not directedness.is_directed:
                adj[node_idx[tgt], node_idx[src]] += weight

        return cls(adj, directedness=directedness, weightedness=weightedness, nodes=nodes)

    def metrics(self):
        if self.directedness.is_directed:
            density = density_dir(self.adj)[0]
            mean_path_length, global_efficiency, *_ = charpath(distance_bin(self.adj), False, False)
            out_degrees = dict(zip(self.nodes, degrees_dir(self.adj)[1]))

            if not self.weightedness.is_weighted:
                modules, modularity = modularity_dir(self.binarised)
                modules = dict(zip(self.nodes, modules))

                return SimpleDirectedUnweightedMetrics(
                    density,
                    mean_path_length,
                    global_efficiency,
                    dict(zip(self.nodes, clustering_coef_bd(self.binarised))),
                    transitivity_bd(self.binarised),
                    modularity,
                    modules,
                    out_degrees
                )
        raise NotImplementedError()

    @classmethod
    def calculate_metrics(cls, fpath, directedness, weightedness, nodes=NEURONS):
        instance = cls.from_csv(fpath, directedness, weightedness, nodes)
        metrics = instance.metrics()
        # metrics.to_json(fpath.with_suffix('.json'))
        return instance, metrics


def dict_to_list(d: Dict, order=None):
    if order is None:
        order = sorted(d)

    return [d[key] for key in order]


@dataclass
class EnsembleMetrics(Metrics):
    nodes: np.ndarray
    mean_path_length: np.ndarray
    global_efficiency: np.ndarray
    clustering_coefficients: np.ndarray
    transitivity: np.ndarray
    maximum_modularity: np.ndarray
    modules: np.ndarray
    out_degree: np.ndarray

    @classmethod
    def from_metrics(cls, metrics: Iterable[Metrics]):
        mean_path_length = []
        global_efficiency = []
        clustering_coefficients = []
        transitivity = []
        maximum_modularity = []
        modules = []
        out_degree = []

        nodes = None

        for this_metrics in metrics:
            if nodes is None:
                nodes = sorted(this_metrics.modules)
            else:
                assert nodes == sorted(this_metrics.modules)
            mean_path_length.append(this_metrics.mean_path_length)
            global_efficiency.append(this_metrics.global_efficiency)
            transitivity.append(this_metrics.transitivity)
            maximum_modularity.append(this_metrics.maximum_modularity)

            clustering_coefficients.append(dict_to_list(this_metrics.clustering_coefficients, nodes))
            modules.append(dict_to_list(this_metrics.modules, nodes))
            out_degree.append(dict_to_list(this_metrics.out_degree, nodes))

        return cls(
            np.array(nodes),
            np.array(mean_path_length),
            np.array(global_efficiency),
            np.array(clustering_coefficients),
            np.array(transitivity),
            np.array(maximum_modularity),
            np.array(modules, dtype=int),
            np.array(out_degree, dtype=int)
        )

    @classmethod
    def from_dir(cls, dpath: Path):
        h5_path = dpath / "combined.hdf5"
        if h5_path.is_file():
            return cls.from_hdf5(h5_path)

        metrics = (SimpleDirectedUnweightedMetrics.from_json(fpath) for fpath in dpath.glob('*.json'))
        return cls.from_metrics(metrics)

    def to_hdf5(self, fpath):
        with h5py.File(fpath, 'x') as f:
            for k, v in self.__dict__.items():
                if k == "nodes":
                    f.attrs['nodes'] = ' '.join(v)
                else:
                    f.create_dataset(k, data=v)

    @classmethod
    def from_hdf5(cls, fpath):
        with h5py.File(fpath, 'r') as f:
            cls(nodes=f.attrs['nodes'], **dict(f.items()))


def calc_metrics(fpath):
    _, metrics = AdjacencyWrapper.calculate_metrics(fpath, Directedness.DIRECTED, Weightedness.UNWEIGHTED)
    return fpath, metrics


def check_for_self_loops():
    directedness = Directedness.DIRECTED
    weightedness = Weightedness.UNWEIGHTED
    metric_root = tgt_dir(Simplicity.SIMPLE, directedness, weightedness)

    has_loops = set()

    for fpath in tqdm(metric_root.rglob('*.csv'), total=4004):
        wrapper = AdjacencyWrapper.from_csv(fpath, directedness, weightedness)
        if np.sum(wrapper.adj * np.eye(len(wrapper.adj))):
            has_loops.add(fpath)

    if has_loops:
        warnings.warn(f"{len(has_loops)} have loops:\n\t" + "\n\t".join(sorted(str(p) for p in has_loops)))
    else:
        print("no self-loops found")


def make_metrics():
    directedness = Directedness.DIRECTED
    weightedness = Weightedness.UNWEIGHTED
    metric_root = tgt_dir(Simplicity.SIMPLE, directedness, weightedness)

    with ProcessPoolExecutor(max_workers=THREADS) as exe:
        futs = []
        submitted = set()
        for fpath in metric_root.rglob('*.csv'):
            try:
                with open(fpath.with_suffix('.json')) as f:
                    json.load(f)
                continue
            except json.JSONDecodeError:
                os.remove(fpath.with_suffix('.json'))
            except FileNotFoundError:
                pass
            futs.append(exe.submit(calc_metrics, fpath))
            submitted.add(fpath)

        failed = 0

        for fut in tqdm(as_completed(futs), total=len(submitted)):
            try:
                fpath, metrics = fut.result()
            except Exception:
                traceback.print_exc()
                failed += 1
                continue

            metrics.to_json(fpath.with_suffix('.json'))
            submitted.remove(fpath)

    assert len(submitted) == failed
    if submitted:
        warnings.warn("The following failed:\n\t" + "\n\t".join(sorted(str(p) for p in submitted)))


def make_ensemble_metrics():
    for dpath in tgt_dir().rglob('rand'):
        ensemble = EnsembleMetrics.from_dir(dpath)
        ensemble.to_hdf5(dpath.parent / 'combined.hdf5')


if __name__ == '__main__':
    # check_for_self_loops()
    make_metrics()
    make_ensemble_metrics()
