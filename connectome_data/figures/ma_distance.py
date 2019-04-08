import csv
import json
from collections import Counter
from itertools import chain

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from connectome_data.constants import DISTANCE_JSON, NEURONS, REAL_EDGES
from connectome_data.figures.constants import TGT_DIR


def get_distances():
    with open(DISTANCE_JSON) as f:
        distances = {tuple(key.split()): value for key, value in json.load(f).items()}

    assert set(chain.from_iterable(distances)) == set(NEURONS)

    return distances


def ma_distances(distances, save=None):
    with open(TGT_DIR / 'monoamine' / REAL_EDGES) as f:
        edgetup = tuple(tuple(sorted(row[:2])) for row in csv.reader(f))

    edgedists = {e: distances[e] for e in edgetup}
    dist_counts = Counter(edgedists.values())
    dist_arr, counts = np.array(sorted(dist_counts.items(), key=lambda p: p[0])).T

    cumu_counts = np.cumsum(counts)

    fig: Figure
    ax: Axes
    fig, ax = plt.subplots(1, 1)

    ax.plot(dist_arr, cumu_counts/np.max(cumu_counts))

    ax.set_xlabel("Minimum edge distance ($\mu m$)")
    ax.set_ylabel("Cumulative frequency")
    ax.set_ylim(0, 1)

    return fig, ax


def main(show=False, save=None):
    distances = get_distances()
    fig, ax = ma_distances(distances)

    if save:
        fig.savefig(save)
    if show:
        plt.show()
    return fig, ax


if __name__ == '__main__':
    main(show=True)
