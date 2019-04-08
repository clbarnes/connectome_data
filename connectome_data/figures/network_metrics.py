from abc import ABC, abstractmethod
from typing import Tuple, Sequence

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from connectome_data.constants import REAL_METRICS, ENSEMBLE_METRICS, Wiring, tgt_dir
from connectome_data.figures.constants import TGT_DIR, DEFAULT_WIRING
from connectome_data.make_data.make_combinations import comb_name
from connectome_data.make_data.metrics import SimpleDirectedUnweightedMetrics, EnsembleMetrics

# transitivity
# mean local clustering coefficient
# maximum modularity
# mean pairwise shortest path length


class MetricsPlot(ABC):
    x = 1, 2

    def __init__(self, figsize=None):
        self.fig: Figure = plt.figure(figsize=figsize)

        self.transitivity_ax: Axes = self.fig.add_subplot(2, 2, 1)
        self.transitivity_ax.set_ylabel("Transitivity $T$")

        self.clustering_ax: Axes = self.fig.add_subplot(2, 2, 2)
        self.clustering_ax.set_ylabel(r"Clustering coefficient $\overline{C}$")

        self.modularity_ax: Axes = self.fig.add_subplot(2, 2, 3)
        self.modularity_ax.set_ylabel("Modularity $M_{max}$")

        self.path_ax: Axes = self.fig.add_subplot(2, 2, 4)
        self.path_ax.set_ylabel("Path length $\overline{P}$")

        self.fig.tight_layout()

    @property
    def xticklabels(self):
        return self.transitivity_ax.get_xticklabels()

    @xticklabels.setter
    def xticklabels(self, arg):
        for ax in self.axes:
            ax.set_xticklabels(arg)

    @property
    def axes(self) -> Tuple[Axes, Axes, Axes, Axes]:
        return self.transitivity_ax, self.clustering_ax, self.modularity_ax, self.path_ax

    def show(self):
        self.fig.show()

    def save(self, fpath, *args, **kwargs):
        self.fig.savefig(fpath, *args, **kwargs)

    def _extract(self, *lsts, attr):
        all_out = []
        for lst in lsts:
            out = []
            for item in lst:
                if item is None:
                    out.append(None)
                elif isinstance(attr, str):
                    out.append(getattr(item, attr))
                else:
                    out.append(attr(item))
            all_out.append(out)
        return tuple(all_out)

    @abstractmethod
    def _plot(self, ax: Axes, real_vals, rand_vals):
        pass

    def plot(self, reals, rands, xticklabels):
        self._plot(
            self.transitivity_ax, *self._extract(reals, rands, attr='transitivity')
        )
        self._plot(
            self.clustering_ax,
            *self._extract(reals, attr=lambda x: np.mean(list(x.clustering_coefficients.values()))),
            *self._extract(rands, attr=lambda x: np.mean(x.clustering_coefficients, axis=1)),
        )
        self._plot(
            self.modularity_ax, *self._extract(reals, rands, attr='maximum_modularity')
        )
        self._plot(
            self.path_ax, *self._extract(reals, rands, attr='mean_path_length')
        )

        self.xticklabels = xticklabels

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        plt.close(self.fig)


class PhysicalMetricsPlot(MetricsPlot):
    def _plot(self, ax: Axes, real_vals, rand_vals):
        ax.boxplot(rand_vals, positions=self.x, whis='range')
        ax.scatter(self.x, real_vals, marker='x')

        z_vals = [(real - np.mean(rand)) / np.std(rand) for real, rand in zip(real_vals, rand_vals)]
        for x, y, z_val in zip(self.x, real_vals, z_vals):
            txt = f'\n$z={z_val:.2f}$'
            ax.annotate(txt, (x, y), horizontalalignment='center', verticalalignment='top')


class CombinedMetricsPlot(MetricsPlot):
    def _plot(self, ax: Axes, real_vals, rand_vals):
        ax.scatter(self.x, real_vals, marker='x')

        x = self.x[1]
        y = real_vals[1]
        z_val = (y - np.mean(rand_vals[1])) / np.std(rand_vals[1])

        ax.boxplot([rand_vals[1]], positions=[x], whis='range')

        txt = f'$z={z_val:.2f}$\n'
        if ax == self.modularity_ax:
            txt += '\n'
        ax.annotate(txt, (x, y), horizontalalignment='center', verticalalignment='bottom')

        ax.set_xticks(self.x)
        ax.set_xlim(min(self.x) - 1, max(self.x) + 1)


def ac_vs_ww(show=False, save=None):
    reals = []
    rands = []
    xticklabels = []

    for wiring in Wiring:
        xticklabels.append(str(wiring).upper())
        dpath = TGT_DIR / str(wiring)
        reals.append(SimpleDirectedUnweightedMetrics.from_json(dpath / REAL_METRICS))
        rands.append(EnsembleMetrics.from_hdf5(dpath / ENSEMBLE_METRICS))

    plot = PhysicalMetricsPlot()

    plot.plot(reals, rands, xticklabels)

    if save:
        plot.save(save)
    if show:
        plot.show()

    return plot


def phys_vs_phys_ma(phys_wiring=DEFAULT_WIRING, show=False, save=None):
    reals = []
    rands = []
    xticklabels = []

    wiring_str = str(phys_wiring)

    # physical
    dpath = TGT_DIR / wiring_str
    reals.append(SimpleDirectedUnweightedMetrics.from_json(dpath / REAL_METRICS))
    rands.append(None)
    xticklabels.append(wiring_str.upper())

    # combined
    name = comb_name([wiring_str], ['monoamine'])
    dpath = TGT_DIR / "combined" / name
    reals.append(SimpleDirectedUnweightedMetrics.from_json(dpath / REAL_METRICS))
    rands.append(EnsembleMetrics.from_hdf5(dpath / ENSEMBLE_METRICS))
    xticklabels.append(xticklabels[-1] + '+MA')

    plot = CombinedMetricsPlot()
    plot.plot(reals, rands, xticklabels)

    if save:
        plot.save(save)
    if show:
        plot.show()

    return plot


if __name__ == '__main__':
    # ac_vs_ww()
    phys_vs_phys_ma()
