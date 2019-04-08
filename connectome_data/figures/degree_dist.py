import json
from collections import Counter, defaultdict
from typing import Tuple

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from connectome_data.constants import Simplicity, Directedness, Weightedness, Wiring, EdgeType, tgt_dir

SIMPLICITY = Simplicity.SIMPLE
DIRECTEDNESS = Directedness.DIRECTED
WEIGHTEDNESS = Weightedness.UNWEIGHTED


def get_data(simplicity=SIMPLICITY, directedness=DIRECTEDNESS, weightedness=WEIGHTEDNESS):
    root = tgt_dir(simplicity, directedness, weightedness)

    data = dict()

    for etype in EdgeType.physical():
        for wiring in Wiring:
            fpath = root / f'{wiring}-{etype}' / 'real.json'
            with open(fpath) as f:
                out_degree = json.load(f)['out_degree']
            counts = Counter(int(i) for i in out_degree.values())

            arr = np.array([(0, 0)] + sorted(counts.items())).T  # 2x302
            arr[1, :] = np.cumsum(arr[1, :])
            data[(wiring, etype)] = arr

    return data


def plot_data(data, show=False) -> Tuple[Figure, Axes]:
    fig: Figure
    ax: Axes
    fig, ax = plt.subplots(1, 1)

    if data is None:
        data = get_data()

    for (wiring, etype), this_data in data.items():
        ax.plot(this_data[0], this_data[1], label=f'{str(wiring).upper()} {etype}')

    ax.set_ylim(0, 310)
    ax.set_ylabel("Cumulative count")
    ax.set_xlabel("Out degree")
    ax.legend()

    if show:
        fig.show()
    return fig, ax


def main(save=None):
    fig, ax = plot_data(get_data(), False)
    if save:
        fig.savefig(save)


if __name__ == '__main__':
    data = get_data()
    plot_data(data, True)
