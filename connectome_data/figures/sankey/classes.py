from __future__ import annotations

import json
import multiprocessing
import random
from collections import deque, defaultdict
from concurrent.futures import as_completed
from concurrent.futures.process import ProcessPoolExecutor
from functools import wraps
from itertools import product
from numbers import Number
from typing import Dict, Any, Tuple, Optional, List, NamedTuple, Iterator, Sequence, Iterable

from colour import Color
from shapely.geometry import LineString

import matplotlib
# matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import numpy as np
from matplotlib.patches import Rectangle
from tqdm import trange, tqdm
import palettable

from connectome_data.figures.common import merge, module_name, blend_colors
from connectome_data.figures.constants import OUT_DIR
from connectome_data.figures.module_members import PHYS_DIR, COMB_DIR, Membership, get_memberships


def _fig_ax(ax: Optional[Axes]) -> Tuple[Figure, Axes]:
    if ax:
        return ax.get_figure(), ax
    else:
        fig: Figure
        fig, ax = plt.subplots(1, 1)
        fig.tight_layout()
    return fig, ax

defaults = {
    "vertical_spacing": 0.2,  # what portion of the total y space should be empty
    "bar_width": 0.1,  # how wide each solid bar should be, where each column is at an integer
    "control_bias": 0.33,  # what proportion between the straight bits the control points should lie at
    "blend_method": "hsl"
}


def from_dicts(key, *dicts, default=None):
    for d in dicts:
        out = d.get(key)
        if out is not None:
            return out
    return default


class Extent1D(NamedTuple):
    min: float = 0
    max: float = 1

    @property
    def range(self):
        return self.max - self.min

    def split(self, fractions: Sequence[Number], total_space: float = 0) -> Iterator[Extent1D]:
        """

        :param fractions: numbers proportional to what proportion of the total space should be used
        :param total_space: what proportion of the total range should be blank space, evenly divided between extents
        """
        fractions = np.array(list(fractions), dtype=float)
        if len(fractions) == 1:
            yield Extent1D(*self)
            return

        range_available = self.range - total_space
        space = total_space / (len(fractions) - 1)
        maxes = (fractions / fractions.sum()) * range_available
        this_min = self.min
        for h in maxes:
            this_max = this_min + h
            yield Extent1D(this_min, this_max)
            this_min = this_max + space

    def interp(self, bias=0.5):
        return self.range * bias + self.min


def sliding_window(iterable, wsize=2):
    q = deque([], wsize)
    for item in iterable:
        q.append(item)
        if len(q) == wsize:
            yield tuple(q)


def first_last(lst):
    return lst[0], lst[-1]


class Sankey:
    """See https://github.com/anazalea/pySankey for original implementation"""
    default_color = Color("blue")

    def __init__(
        self,
        left_labels, right_labels,
        weights: Dict[Tuple[Any, Any], float],
        ax: Axes = None,
        xticklabels=None,
        left_colors=None,
        **kwargs
    ):
        """Labels in top-to-bottom order"""
        # arbitrary n stages not supported
        self.labels = (left_labels, right_labels)
        self.weights = weights

        self.vertical_spacing = from_dicts("vertical_spacing", kwargs, defaults)
        self.bar_width = from_dicts("bar_width", kwargs, defaults)
        self.control_bias = from_dicts("control_bias", kwargs, defaults)

        self.fig, self.ax = _fig_ax(ax)
        self.ax2 = self.ax.twinx()

        half = self.bar_width / 2
        for this_ax in (self.ax, self.ax2):
            for spine in this_ax.spines.values():
                spine.set_visible(False)

            this_ax.tick_params(**{edge: False for edge in ("top", "bottom", "left", "right")})
            this_ax.set_xlim(-half, self.n_stages - 1 + half)
            this_ax.set_ylim(0, 1)

        # self.ax.set_axis_off()

        if left_colors is None:
            left_colors = {label: self.default_color for label in left_labels}
        elif isinstance(left_colors, dict):
            left_colors = {k: Color(v) for k, v in left_colors.items()}
        else:
            left_colors = {k: Color(left_colors) for k in left_labels}

        blend_method = from_dicts("blend_method", kwargs, defaults)
        colors = [left_colors]
        while len(colors) < len(self.labels):
            colors.append(self.blend_colors(colors[-1], blend_method))

        self.colors = tuple(colors)

        # draw

        y_extents = self._get_vert_extents()

        for side_labels, side_y_extents, side_ax in zip(
            first_last(self.labels), first_last(y_extents), (self.ax, self.ax2)
        ):
            side_ax.set_yticks(sorted(y.interp() for y in side_y_extents.values()))
            side_ax.set_yticklabels(list(reversed(side_labels)))

        box_x_extents = self._box_x_extents()

        if xticklabels:
            self.ax.set_xticks(list(range(self.n_stages)))
            self.ax.set_xticklabels(xticklabels)
        else:
            self.ax.set_xticks([])

        inter_box_extents = self._draw_boxes(y_extents, box_x_extents)
        # inter_straight_extents = self._draw_straight_portions(y_extents, inter_box_extents)
        self._draw_curves(y_extents, inter_box_extents)
        if not ax:
            self.fig.tight_layout()

    def blend_colors(self, left_colors, method="hsl"):
        right_to_col_weight = defaultdict(list)
        for (left, right), weight in self.weights.items():
            right_to_col_weight[right].append((left_colors[left], weight))

        return {right: blend_colors(col_weight, method) for right, col_weight in right_to_col_weight.items()}

    @property
    def n_stages(self):
        return len(self.labels)

    def _box_x_extents(self) -> Iterator[Extent1D]:
        half = self.bar_width / 2
        for idx in range(self.n_stages):
            yield Extent1D(idx - half, idx + half)

    def _draw_boxes(self, y_extents: List[Dict[Any, Extent1D]], box_x_extents: Iterable[Extent1D]) -> Iterator[Extent1D]:
        """Add solid rectangles at each stage representing fixed groupings"""
        last_max = None
        for x_extent, stage, colors in zip(box_x_extents, y_extents, self.colors):
            for label, y_extent in stage.items():
                patch = Rectangle(
                    (x_extent.min, y_extent.min),
                    x_extent.range, y_extent.range, color=colors[label].rgb, linewidth=0
                )
                self.ax.add_patch(patch)

            if last_max is not None:
                yield Extent1D(last_max, x_extent.min)
            last_max = x_extent.max

    # def _draw_straight_portions(self, y_extents: List[Dict[Any, Extent1D]], inter_box_extents: Iterable[Extent1D]) -> Iterator[Extent1D]:
    #     """Add straight, unbranched, but translucent rectangles just before and after the solid rectangles"""
    #     if not self.straight_ppn:
    #         yield from inter_box_extents
    #         return
    #
    #     for inter_box_extent, stages in zip(inter_box_extents, sliding_window(y_extents)):
    #         last_max = None
    #         for x_extent, stage in zip(inter_box_extent.split([1, 1], 1 - 2*self.straight_ppn), stages):
    #             for y_extent in stage.values():
    #                 y1 = [y_extent.max, y_extent.max]
    #                 y2 = [y_extent.min, y_extent.min]
    #                 x = [x_extent.min, x_extent.max]
    #                 self._fill_and_outline(x, y1, y2)
    #
    #             if last_max is not None:
    #                 yield Extent1D(last_max, x_extent.min)
    #             last_max = x_extent.max

    def _sort_y_weights(self, weights: Dict[Any, Dict[Any, int]], other_labels: Sequence[Any]):
        """Sort the inner dict of a {left_label: {right_label: weight}} dict based on how early in the label sequence
        the inner label appears"""
        for this_label, other_to_weight in weights.items():
            yield this_label, dict(sorted(other_to_weight.items(), key=lambda kv: other_labels.index(kv[0])))

    def _internal_y_extents(
        self, left_labels: Sequence[Any], left_extents: Dict[Any, Extent1D],
        right_labels: Sequence[Any], right_extents: Dict[Any, Extent1D]
    ):
        """For each pair of linked modules, find the y extents of their paths"""
        left_weights = defaultdict(dict)
        right_weights = defaultdict(dict)
        for (left_label, right_label), weight in self.weights.items():
            left_weights[left_label][right_label] = weight
            right_weights[right_label][left_label] = weight

        left_weights = dict(self._sort_y_weights(left_weights, list(reversed(right_labels))))
        right_weights = dict(self._sort_y_weights(right_weights, list(reversed(left_labels))))

        out = defaultdict(list)

        for left_label, right_to_weight in left_weights.items():
            extents = left_extents[left_label]
            for right_label, extent in zip(right_to_weight, extents.split(right_to_weight.values())):
                out[(left_label, right_label)].append(extent)

        for right_label, left_to_weight in right_weights.items():
            extents = right_extents[right_label]
            for left_label, extent in zip(left_to_weight, extents.split(left_to_weight.values())):
                out[(left_label, right_label)].append(extent)

        return out

    def _draw_curves(self, y_extents, inter_straight_extents: Iterable[Extent1D]):
        """Add curved tracks"""
        num = 100
        n_convolutions = 2
        kernel = np.full(20, 0.05)
        for inter_straight_extent, stage_extents, stage_labels, colors in zip(
            inter_straight_extents, sliding_window(y_extents), sliding_window(self.labels), self.colors
        ):
            for (left_label, _), (left_extent, right_extent) in self._internal_y_extents(
                stage_labels[0], stage_extents[0], stage_labels[1], stage_extents[1]
            ).items():
                ys = []
                for idx in range(2):
                    y = np.empty(num, float)
                    y[:num//2] = left_extent[idx]
                    y[num//2:] = right_extent[idx]
                    for _ in range(n_convolutions):
                        y = np.convolve(y, kernel, mode='valid')
                    ys.append(y)

                x = np.linspace(*inter_straight_extent, num=len(ys[0]))
                kwargs = {"color": colors[left_label].rgb}
                self._fill_and_outline(x, *ys, fill_kwargs=kwargs, plot_kwargs=kwargs)

    def _fill_and_outline(self, x, y1, y2, plot_kwargs=None, fill_kwargs=None):
        plot_kwargs = merge({"color": self.default_color.rgb, "alpha": 0.8, "linewidth": 0}, plot_kwargs)
        fill_kwargs = merge({"color": self.default_color.rgb, "alpha": 0.5, "linewidth": 0}, fill_kwargs)

        self.ax.fill_between(x, y1, y2, **fill_kwargs)
        # line thickness messes this up
        for y in (y1, y2):
            self.ax.plot(x, y, **plot_kwargs)

    def _get_vert_extents(self) -> List[Dict[Any, Extent1D]]:
        totals = [0, 0]
        weights = [{key: 0 for key in labels} for labels in self.labels]

        for keys, weight in self.weights.items():
            for idx, key in enumerate(keys):
                totals[idx] += weight
                weights[idx][key] += weight

        if np.count_nonzero(np.diff(totals)):
            raise NotImplementedError("Stages have different total weights")

        all_vextents = []
        for label_list, weight_dict, total in zip(self.labels, weights, totals):
            rev_labels = list(reversed(label_list))
            sorted_weights = [weight_dict[lab] for lab in rev_labels]
            if not all(sorted_weights):
                raise ValueError("Some labels have weight of zero")
            all_vextents.append(
                dict(zip(
                    rev_labels,
                    list(Extent1D(0, 1).split(sorted_weights, self.vertical_spacing))
                ))
            )

        return all_vextents

    def close(self):
        plt.close(self.fig)

    @wraps(Figure.savefig)
    def save(self, *args, **kwargs):
        self.fig.savefig(*args, **kwargs)
        return self

    # def show(self):
    #     self.fig.show()

    @classmethod
    def from_memberships(cls, m1: Membership, m2: Membership):
        labels1 = [l for l, _ in m1.as_sets(True)]
        labels2 = [l for l, _ in m2.as_sets(True)]
        weights = dict(m1.compare(m2))
        return cls(labels1, labels2, weights)

    def optimal_sort(self):
        for x in enumerate(self.labels):
            pass


class IdxSorting:
    def __init__(self, left, right, weights, rand=None, value=None):
        self.left = left
        self.right = right
        self.weights = weights
        self.random = rand if isinstance(rand, random.Random) else random.Random(rand)
        self._value = self.evaluate() if value is None else value

    @property
    def value(self):
        if self._value is None:
            self._value = self.evaluate()
        return self._value

    def copy(self):
        """Copy label dicts, keep weight and random state"""
        return type(self)(self.left.copy(), self.right.copy(), self.weights, self.random, self._value)

    def swap(self):
        labels = (self.left, self.right)
        d = labels[self.random.randint(0, 1)]
        idx1, idx2 = self.random.sample(list(d), 2)
        d[idx1], d[idx2] = d[idx2], d[idx1]
        self._value = None
        return self

    @classmethod
    def get_best(cls, left_labels, right_labels, weights, n=100, seed=None, shuffle=True, progress=False):
        rand = random.Random(seed)
        if shuffle:
            rand.shuffle(left_labels)
            rand.shuffle(right_labels)
        left = {lbl: idx for idx, lbl in enumerate(left_labels)}
        right = {lbl: idx for idx, lbl in enumerate(right_labels)}
        best = cls(left, right, weights, rand=random.Random(seed))
        best_val = best.value
        values = [best_val]
        if progress:
            it = trange(n)
        else:
            it = range(n)
        for _ in it:
            new = best.copy().swap()
            values.append(new.value)
            if values[-1] <= best_val:
                best = new
                best_val = values[-1]
        return best, values

    @classmethod
    def get_best_parallel(cls, left_labels, right_labels, weights, populations=None, swaps=100, seed=None, threads=None):
        rand = random.Random(seed)
        threads = threads or multiprocessing.cpu_count()
        populations = populations or threads
        with ProcessPoolExecutor(threads) as p:
            futs = []
            for _ in trange(populations, desc="submitting"):
                futs.append(p.submit(
                    cls.get_best,
                    left_labels.copy(), right_labels.copy(), weights.copy(),
                    n=swaps, seed=rand.random()
                ))

            best = None
            for fut in tqdm(as_completed(futs), desc="getting", total=populations):
                this_best, _ = fut.result()
                if best is None or this_best.value < best.value:
                    best = this_best
        return best

    def evaluate(self):
        line_weight = [
            (LineString([(0, self.left[l]), (1, self.right[r])]), w)
            for (l, r), w in self.weights.items()
        ]
        total_weight = 0
        for (l1, w1), (l2, w2) in product(line_weight, repeat=2):
            if l1.crosses(l2):
                total_weight += w1 + w2
        return total_weight

    @classmethod
    def from_memberships(cls, m1, m2, rand=None):
        labels1 = {l: idx for idx, (l, _) in enumerate(m1.as_sets(True))}
        labels2 = {l: idx for idx, (l, _) in enumerate(m2.as_sets(True))}
        weights = dict(m1.compare(m2))
        return cls(labels1, labels2, weights, rand)

    def to_json(self, fpath=None):
        d = {
            "left": sorted(self.left, key=self.left.get),
            "right": sorted(self.right, key=self.right.get),
            "value": self.value
        }
        if fpath:
            with open(fpath, 'w') as f:
                json.dump(d, f, sort_keys=True, indent=2)
        return d


def update_name(lst):
    other = max(lst)
    return ["other" if n == other else module_name(n) for n in lst]


if __name__ == '__main__':
    phys, comb = get_memberships()

    with open("ordering.json") as f:
        sorting = json.load(f)

    cseq = palettable.colorbrewer.qualitative.Set1_6.mpl_colors
    colors = {label: Color(rgb=c) for label, c in zip(sorting["left"], cseq)}

    sankey = Sankey(
        sorting["left"], sorting["right"], dict(phys.compare(comb)),
        xticklabels=("physical", "combined"), left_colors=colors
    )
    sankey.save(OUT_DIR / "modules.svg")
    plt.show()

    # sorting = IdxSorting.from_memberships(phys, comb)

    # best = IdxSorting.get_best_parallel(
    #     [l for l, _ in phys.as_sets(True)],
    #     [l for l, _ in comb.as_sets(True)],
    #     dict(phys.compare(comb)),
    #     populations=100, swaps=100_000
    # )
    # print(best.value)
    # best.to_json("ordering.json")

    # print(sorting.evaluate())

    # s = Sankey.from_memberships(phys, comb)
    # s.save("my_sankey.svg")
    # plt.show()
    # s.show()
