import random
from abc import ABC
from enum import IntEnum
from typing import NamedTuple, Dict, Union, Sequence, Any, Tuple, List

import networkx as nx
import numpy as np
from pyveplot import Axis, Hiveplot, Node
from tqdm import tqdm

from connectome_data.figures.common import merge

EM_DASH = "—"


def angle(a):
    a = np.asarray(a)
    while np.count_nonzero(a < 0):
        a[a<0] += 2*np.pi
    while np.count_nonzero(a >= 2*np.pi):
        a[a>=2*np.pi] -= 2*np.pi

    return a


class Orientation(IntEnum):
    CCW = 1
    CW = 2


class HiveObjIdx(NamedTuple):
    name: str
    orientation: Orientation

    @classmethod
    def make_pair(cls, name):
        return HiveObjIdx(name, Orientation.CCW), HiveObjIdx(name, Orientation.CW)


class SplitHiveAxis(Axis):
    defaults = {"stroke": "black", "stroke_width": 5}

    def __init__(self, start, end, **kwargs):
        super().__init__(start, end, **merge(self.defaults, kwargs))
        # self.idx = idx
    #
    # @property
    # def name(self):
    #     return self.idx.name

    def add_node(self, node, offset, **kwargs):
        """Adds a node with a circle"""
        node = super().add_node(node, offset)
        node.dwg.add(node.dwg.circle((node.x, node.y), **kwargs))
        return node


def angle_to_anchor(radians):
    """Convert an angle into where a text label should go so that it looks nice"""
    radians = angle(radians)
    left = "start"
    right = "end"
    top = "text-before-edge"
    bottom = "text-after-edge"
    middle = "middle"

    # x, y
    if radians == 0:
        pair = (middle, bottom)
    elif radians < np.pi / 2:
        pair = (left, bottom)
    elif radians == np.pi / 2:
        pair = (left, middle)
    elif radians < np.pi:
        pair = (left, top)
    elif radians == np.pi:
        pair = (middle, top)
    elif radians < 3 * np.pi / 2:
        pair = (right, top)
    elif radians == 3 * np.pi / 2:
        pair = (right, middle)
    elif radians < 2 * np.pi:
        pair = (right, bottom)
    else:
        raise RuntimeError("illegal angle")

    return dict(zip(("text-anchor", "dominant-baseline"), pair))


class SplitHivePlot(Hiveplot):
    axis_cls = SplitHiveAxis

    ax_split = np.radians(60)  # angle, in radians, between half-axes
    origin = 600, 600
    ax_len = 400
    ax_offset_ppn = 0.2
    edge_defaults = {"stroke": "red", "stroke_width": 1.5}
    label_offset = 0.1

    def __init__(self, ax_attrs: Union[Sequence[Any], Dict[Any, Dict[str, Any]]], dwg_attrs=None, **kwargs):
        self.__dict__.update(kwargs)

        dwg_attrs = merge({"width": 2 * self.origin[0], "height": 2 * self.origin[1]}, dwg_attrs)
        super().__init__(None, self.origin, **dwg_attrs)

        if ax_attrs is None:
            ax_attrs = [str(i) for i in range(3)]

        if len(ax_attrs) != 3:
            raise ValueError("Hive plots with other than 3 axes are not supported")

        if not isinstance(ax_attrs, dict):
            ax_attrs = {name: dict() for name in ax_attrs}

        ax_angles = np.linspace(0, 2 * np.pi, 3, endpoint=False)
        sp = self.ax_split/2
        split_ax_angles = angle([ax_angles - sp, ax_angles + sp]).T

        start_dist = self.ax_offset_ppn * self.ax_len
        stop_dist = start_dist + self.ax_len

        self.ax_pairs: Dict[Any, Tuple[SplitHiveAxis, SplitHiveAxis]] = {}

        for (ccw_angle, cw_angle), (ax_name, this_attrs) in zip(split_ax_angles, ax_attrs.items()):
            self.ax_pairs[ax_name] = (
                self.add_axis_polar((start_dist, ccw_angle), (stop_dist, ccw_angle), **this_attrs),
                self.add_axis_polar((start_dist, cw_angle), (stop_dist, cw_angle), **this_attrs)
            )

        self._add_labels(ax_angles, split_ax_angles)

        self._nodes: Dict[Any, Node] = None
        self._node_axes: Dict[HiveObjIdx, SplitHiveAxis] = None

    def _add_curve(self, *xy_pairs, **line_attrs):
        first, ctrl, *tail = xy_pairs
        pth = self.dwg.path(f"M {first[0]} {first[1]}", fill='none', **line_attrs)

        if tail:
            pth.push(f"C {ctrl[0]} {ctrl[1]}")
            for point in tail:
                pth.push(f"{point[0]} {point[1]}")
        else:
            pth.push(f"{ctrl[0]} {ctrl[1]}")

        self.dwg.add(pth)
        return pth

    def _add_curve_polar(self, *rtheta_pairs, **line_attrs):
        return self._add_curve(*(self.coords(*p) for p in rtheta_pairs), **line_attrs)

    def _add_labels(self, ax_angles, split_ax_angles, text_attrs=None, line_attrs=None):
        stop_dist = (1 + self.ax_offset_ppn) * self.ax_len
        label_offset = stop_dist + self.ax_len * self.label_offset
        root_offset = stop_dist + self.ax_len * self.label_offset * 0.9
        leaf_offset = stop_dist + self.ax_len * self.label_offset * 0.2

        text_attrs = merge({"font-size": "30px", "fill": "black"}, text_attrs)
        line_attrs = merge({"stroke": "black", "stroke_width": 2}, line_attrs)

        for ax_pair_key, root_angle, leaf_angles in zip(self.ax_pairs, ax_angles, split_ax_angles):
            label = str(ax_pair_key)
            label_xy = self.coords(label_offset, root_angle)
            text_anchor = angle_to_anchor(root_angle)

            self.dwg.add(self.dwg.text(label, label_xy, **text_attrs, **text_anchor))

            for leaf_angle in leaf_angles:
                self._add_curve_polar(
                    (leaf_offset, leaf_angle),
                    (label_offset, leaf_angle),
                    (stop_dist, root_angle),
                    (root_offset, root_angle),
                    **line_attrs
                )

    @property
    def nodes(self):
        if self._nodes is None:
            self._nodes = dict()
            for ax in self.axes:
                self._nodes.update(ax.nodes)
        return self._nodes

    @property
    def node_axes(self):
        if self._node_axes is None:
            self._node_axes = dict()
            for ax in self.axes:
                self._node_axes.update({key: ax for key in ax.nodes})
        return self._node_axes

    def _invalidate(self):
        self._nodes = None
        self._node_axes = None

    def add_node(self, ax_name: str, node_name: str, offset: float, **kwargs):
        out = []

        for node_idx, ax in zip(HiveObjIdx.make_pair(node_name), self.ax_pairs[ax_name]):
            node = ax.add_node(node_idx, offset, **kwargs)
            out.append(node_idx)

        self._invalidate()
        return tuple(out)

    def _add_edge(self, src_idx: HiveObjIdx, tgt_idx: HiveObjIdx, src_angle, tgt_angle, **kwargs):
        src_ax = self.node_axes[src_idx]
        tgt_ax = self.node_axes[tgt_idx]
        attrs = self.edge_defaults.copy()
        attrs.update(kwargs)
        return self.connect(src_ax, src_idx, src_angle, tgt_ax, tgt_idx, tgt_angle, **attrs)

    def _is_neighbour_ax(self, ax1, ax2):
        ax1_loc = self.axes.index(ax1)
        ax2_loc = self.axes.index(ax2)

        if ax1_loc < 0 or ax2_loc < 0:
            raise ValueError("One of the passed axes is not in this plot")

        return abs(ax1_loc - ax2_loc) == 1 or {ax1_loc, ax2_loc} == {0, len(self.axes) - 1}

    def _angle_bias_to_angles(self, interval, angle_bias: Union[Tuple[float, float], float]):
        """

        :param interval: angle interval to split
        :param angle_bias: float or pair of floats for how far from the source and target
            the control point should be
        :return: magnitude of deflection for source and target (multiply the ccw one by -1)
        """
        try:
            src_bias, tgt_bias = angle_bias
        except TypeError:
            src_bias, tgt_bias = angle_bias, angle_bias

        return src_bias * interval, tgt_bias * interval

    def add_edge(self, src_name, tgt_name, angle_bias=0.33, **kwargs):
        """

        :param src_name:
        :param tgt_name:
        :param angle_bias: what proportion of the angle between the two axes the control points should be placed at
        :param kwargs: to be passed to SVG line
        :return:
        """
        src_idx1, src_idx2 = HiveObjIdx.make_pair(src_name)
        tgt_idx1, tgt_idx2 = HiveObjIdx.make_pair(tgt_name)

        minor_angle = np.degrees(self.ax_split)
        if self.node_axes[src_idx1] == self.node_axes[tgt_idx1]:
            # two halves of a split axis
            src_angle, tgt_angle = self._angle_bias_to_angles(minor_angle, angle_bias)
            self._add_edge(src_idx1, tgt_idx2, src_angle, -tgt_angle, **kwargs)
            self._add_edge(src_idx2, tgt_idx1, -src_angle, tgt_angle, **kwargs)
        else:
            # two separate axes
            major_angle = (360 - 3*minor_angle) / 3
            src_angle, tgt_angle = self._angle_bias_to_angles(major_angle, angle_bias)
            if self._is_neighbour_ax(self.node_axes[src_idx2], self.node_axes[tgt_idx1]):
                # the edge goes clockwise, from the clockwise-side source half-axis
                # to the counterclockwise target half-axis
                self._add_edge(src_idx2, tgt_idx1, src_angle, -tgt_angle, **kwargs)
            elif self._is_neighbour_ax(self.node_axes[src_idx1], self.node_axes[tgt_idx2]):
                # the edge goes counterclockwise, from the counterclockwise-side source half-axis
                # to the clockwise target half-axis
                self._add_edge(src_idx1, tgt_idx2, -src_angle, tgt_angle, **kwargs)
            else:
                raise ValueError("Nodes do not seem to be on neighbouring axes")

    def add_legend(self, xy: Tuple[Union[float, str], Union[float, str]], label_col: List[Tuple[Any, str]], line_space=0.5, text_attrs=None):
        font_size = 20
        text_attrs = merge({"fill": "black", "font-size": font_size}, text_attrs)

        units = self.units.copy(font_size=text_attrs["font-size"])
        x, y = (units(i) for i in xy)

        offset_x = x + units("1em")
        offset_y_per = units(f"{1+line_space}em")

        for idx, (label, col) in enumerate(label_col):
            dash_attrs = {"fill": col, "font-weight": 900}
            this_y = y + offset_y_per*idx
            self.dwg.add(self.dwg.text(EM_DASH, x=[x], y=[this_y], **merge(text_attrs, dash_attrs)))
            self.dwg.add(self.dwg.text(str(label), x=[offset_x], y=[this_y], **text_attrs))


class Scaler:
    def __init__(self, in_interval, out_interval=(0, 1), log=False, clip=True):
        self.fn = np.log if log else lambda x: x
        self.raw_in = in_interval
        self.min_in, self.max_in = [self.fn(x) for x in in_interval]
        self.min_out, self.max_out = out_interval
        self.clip = clip

    @property
    def range_in(self):
        return self.max_in - self.min_in

    @property
    def range_out(self):
        return self.max_out - self.min_out

    def _clip(self, val):
        if not self.clip:
            return val
        val = max(val, self.raw_in[0])
        val = min(val, self.raw_in[1])
        return val

    def __call__(self, val):
        val = self.fn(self._clip(val))
        ppn = (val - self.min_in) / self.range_in
        return ppn * self.range_out + self.min_out


weight_scaler = Scaler((1, 20), (0.5, 5))


class GraphHivePlot(ABC):
    seed = 1
    node_size_range = (3, 8)

    def __init__(self, graph: nx.OrderedMultiGraph, ax_attrs, etype_colors, *args, **kwargs):
        self.graph = graph
        self.hiveplot = SplitHivePlot(ax_attrs, *args, **kwargs)
        self.random = random.Random(kwargs.get("seed", self.seed))
        self.etype_colors = etype_colors

    def populate(self):
        r_scaler = Scaler((1, 5), tuple(self.hiveplot.units(i) for i in self.node_size_range))
        ntype_name_offset_r = []
        for name, data in tqdm(self.graph.nodes(data=True), desc="adding to plot", unit="nodes"):
            count = len(data.get("neurons", 'a'))  # anything 1-length
            ntype_name_offset_r.append((data["ntype"], name, data['offset'], r_scaler(count)))

        node_kwargs = {"stroke": "black", "stroke_width": 1.5, "fill": "white"}
        for ntype, name, offset, r in sorted(ntype_name_offset_r, key=lambda x: (-x[3], x[2], x[1])):
            self.hiveplot.add_node(ntype, name, offset, r=r, **node_kwargs)

        edges = list(
            {
                tuple(sorted((src, tgt))) + (d["etype"], d.get("weight", 1))
                for src, tgt, d in self.graph.edges(data=True)
            }
        )
        random.Random(1).shuffle(edges)
        for src, tgt, etype, weight in tqdm(edges, desc="adding to plot", unit="edges"):
            self.hiveplot.add_edge(
                src, tgt, stroke=self.etype_colors[etype], stroke_width=weight_scaler(weight), stroke_opacity=0.5
            )

        etypes = sorted({etype for _, _, etype in self.graph.edges(data='etype')})

        label_col = [(etype, self.etype_colors[etype]) for etype in etypes]
        self.hiveplot.add_legend(("10vw", "20vh"), label_col)
        return self

    def save(self, fpath):
        self.populate()
        self.hiveplot.save(fpath, pretty=True)
        return self
