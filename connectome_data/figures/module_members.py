from __future__ import annotations

import itertools
import json
import string
from collections import defaultdict, Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Set, Tuple, Iterator, List, Iterable, Any, Optional

from matplotlib import pyplot as plt
import nltk
import palettable
import pandas as pd

from connectome_data.constants import NODES_DATA
from connectome_data.figures.constants import TGT_DIR, DEFAULT_WIRING, OUT_DIR
from connectome_data.figures.sankey.classes import Sankey

AGGLOM_THRESHOLD = 13

PHYS_DIR = TGT_DIR / str(DEFAULT_WIRING)
COMB_DIR = TGT_DIR / "combined" / f"{str(DEFAULT_WIRING)}+monoamine"

# todo: threshold small modules into "other"


class Membership:
    def __init__(self, cell_to_idx: Dict[str, Any]):
        self.cell_to_idx = cell_to_idx

    @property
    def modules(self):
        return set(self.cell_to_idx.values())

    def as_sets(self, len_sort=False) -> List[Tuple[Any, Set[str]]]:
        d = defaultdict(set)
        for cell, idx in self.cell_to_idx.items():
            d[idx].add(cell)

        if len_sort:
            sort_key = lambda kv: -len(kv[1])
        else:
            sort_key = lambda kv: kv[0]

        return sorted(d.items(), key=sort_key)

    def compare(self, other: Membership) -> Iterator[Tuple[Tuple[Any, Any], int]]:
        """Get how many nodes are shared between each pair of modules in this and another Membership"""
        for (this_idx, this_cells), (that_idx, that_cells) in itertools.product(self.as_sets(), other.as_sets()):
            intersection = len(this_cells.intersection(that_cells))
            if not intersection:
                continue
            yield (this_idx, that_idx), intersection

    def compare_df(self, other: Membership, ntypes=None) -> pd.DataFrame:
        """Get how many nodes are shared between each module, with optional ntypes"""
        if ntypes is None:
            ntypes = defaultdict(lambda: None)

        ntype_classes = sorted(set(ntypes.values()))

        columns = ("source", "target", "type", "value")
        data = []
        for (this_idx, this_cells), (that_idx, that_cells) in itertools.product(self.as_sets(), other.as_sets()):
            for ntype in ntype_classes:
                this_cells_ntype = {cell for cell in this_cells if ntypes[cell] == ntype}
                that_cells_ntype = {cell for cell in that_cells if ntypes[cell] == ntype}
                intersection = len(this_cells_ntype.intersection(that_cells_ntype))
                if not intersection:
                    continue
                data.append([this_idx, that_idx, ntype, intersection])

        return pd.DataFrame(data=data, columns=columns)

    @classmethod
    def from_dir(cls, dpath: Path, name_fn=None, starting_idx=0) -> Membership:
        with open(dpath / "real.json") as f:
            d = json.load(f)["modules"]

        name_fn = name_fn or (lambda x: x)
        named = {cell: name_fn(idx + starting_idx) for cell, idx in d.items()}
        return cls(named)

    @classmethod
    def from_sets(cls, sets: Iterable[Tuple[Any, Set[str]]]):
        cell_to_idx = dict()
        for idx, cells in sets:
            for cell in cells:
                cell_to_idx[cell] = idx
        return cls(cell_to_idx)

    @classmethod
    def from_membership(cls, membership: Membership):
        return cls(membership.cell_to_idx.copy())

    def agglomerate(self, threshold, new_name='other') -> Tuple[Membership, Dict[Any, Any]]:
        mapping = self.agglomeration_mapping(threshold, new_name)
        return self.remap_modules(mapping), mapping

    def agglomeration_mapping(self, threshold, new_name='other') -> Dict[int, int]:
        """Return a dict, from indices of modules smaller than ``threshold`` to a new module index"""
        to_agglomerate = []

        for label, cells in self.as_sets():
            if len(cells) < threshold:
                to_agglomerate.append(label)
        return {old: new_name for old in to_agglomerate}

    def plot_comparison(self, other):
        lefts = []
        rights = []
        weights = []
        for (left, right), weight in self.compare(other):
            lefts.append(left)
            rights.append(right)
            weights.append(weight)

        # sankey(left=lefts, right=rights, rightWeight=weights, leftWeight=weights, figureName="sankey")

    def remap_modules(self, mapping):
        cell_to_idx = {cell: mapping.get(module, module) for cell, module in self.cell_to_idx.items()}
        return type(self)(cell_to_idx)


default_stopwords = set(nltk.corpus.stopwords.words('english')) | set(string.punctuation) | {
    "neuron", "synapt", "synaps", "connect", "make", "innervat", "cell"
}


def dictzip(*dicts, check_keys=True, default=None) -> Dict[Any, Tuple[Any, ...]]:
    """Given some dicts with the same keys, return a dict combining all their values"""
    if check_keys and len(set(frozenset(d) for d in dicts)) != 1:
        raise ValueError("Dicts must have the same keys")
    for key in dicts[0]:
        yield key, tuple(d.get(key, default) for d in dicts)


@dataclass
class ModuleInfo:
    name: str
    cells: Set[str]
    ntype_weights: Dict[str, float]
    desc_words: Dict[str, float]

    def normalise_ntypes(self, popn_ntypes):
        self.ntype_weights = {ntype: this/popn for ntype, (this, popn) in dictzip(self.ntype_weights, popn_ntypes)}
        return self

    def __str__(self):
        lines = [self.name, "-" * len(self.name)]

        line_len = 79
        prefix = ' '
        lines.extend([f'Cells ({len(self.cells)}):', prefix])
        cell_lst = sorted(self.cells, reverse=True)
        while cell_lst:
            next_cell = cell_lst.pop()
            last_len = len(lines[-1])
            if last_len + len(next_cell) + 1 > line_len:
                lines.append(prefix)
            lines[-1] += ' ' + next_cell

        prefix = '  '
        lines.append('Neuron types (proportional to whole network):')
        for ntype, weight in sorted(self.ntype_weights.items(), key=lambda x: -x[1]):
            lines.append(f'{prefix}{ntype}: {weight:0.2f}')

        lines.append('Description words (mentions per cell):')
        count = 0
        for word, weight in sorted(self.desc_words.items(), key=lambda x: -x[1]):
            lines.append(f'{prefix}{word}: {weight:0.2f}')
            count += 1
            if count > 10:
                break

        return '\n'.join(lines)


def tokenise_description(desc, stopwords=None):
    if stopwords is None:
        stopwords = default_stopwords.copy()
    stemmer = nltk.stem.snowball.SnowballStemmer('english')
    words = (stemmer.stem(w.lower()) for w in nltk.word_tokenize(desc) if len(w) > 1)
    return Counter(w for w in words if w not in stopwords)


class ModuleUnderstander(Membership):
    def __init__(self, cell_to_idx: Dict[str, Any]):
        super().__init__(cell_to_idx)
        self._node_descs = None
        self._node_ntypes = None

    def _populate(self):
        with open(NODES_DATA) as f:
            d = json.load(f)

        self._node_descs = {n: d[n]["description"] for n in self.cell_to_idx}
        self._node_ntypes = {n: d[n]["ntypes"] for n in self.cell_to_idx}

    @property
    def node_descs(self):
        if self._node_descs is None:
            self._populate()
        return self._node_descs

    @property
    def node_ntypes(self):
        if self._node_ntypes is None:
            self._populate()
        return self._node_ntypes

    def modules_info(self):
        ntypes = {n: 0 for n in itertools.chain.from_iterable(self.node_ntypes.values())}
        # the proportions of the ntypes in the whole dataset
        all_ntypes = ntypes.copy()
        for cell_ntypes in self.node_ntypes.values():
            for ntype in cell_ntypes:
                all_ntypes[ntype] += 1 / len(cell_ntypes) / len(self.cell_to_idx)

        descs_tokenised = {c: tokenise_description(d) for c, d in self.node_descs.items()}

        # average number of appearances of each word per cell
        all_words = {
            word: c/len(self.cell_to_idx)
            for word, c in Counter(itertools.chain.from_iterable(descs_tokenised.values())).items()
        }

        for module, cells in self.as_sets():
            module_ntypes = ntypes.copy()
            cell_descs = []
            for cell in cells:
                for ntype in self.node_ntypes[cell]:
                    module_ntypes[ntype] += 1 / len(self.node_ntypes[cell]) / len(cells)
                cell_descs.append(descs_tokenised[cell])

            module_words = Counter(itertools.chain(*cell_descs))
            module_words = {k: v/len(cells) for k, v in module_words.items()}

            module_info = ModuleInfo(module, cells, module_ntypes, module_words)
            yield module_info.normalise_ntypes(all_ntypes)


def num2name(n: int, prefix: str = "module ") -> str:
    return f"{prefix}{n:02d}"


def get_memberships(agglomerate=10):
    phys = Membership.from_dir(PHYS_DIR, lambda x: num2name(x, "phys. "))
    comb = Membership.from_dir(COMB_DIR, lambda x: num2name(x, "comb. "))
    if agglomerate:
        phys = phys.agglomerate(agglomerate, 'phys. other')[0]
        comb = comb.agglomerate(agglomerate, 'comb. other')[0]
    return phys, comb


def write_notes(names_memberships, fpath):
    with open(fpath, 'w') as f:
        first = True
        for name, membership in names_memberships.items():
            if not first:
                f.write('\n\n\n')
            f.write(f"{name}\n{'='*len(name)}\n\n")
            u = ModuleUnderstander.from_membership(membership)
            for mi in sorted(u.modules_info(), key=lambda m: m.name):
                f.write(str(mi) + '\n\n')
            first = False


def draw_sankey(phys: Membership, comb: Membership, show=False, save=None):
    # cseq = palettable.colorbrewer.qualitative.Set1_6.mpl_colors
    # colors = {label: Color(rgb=c) for label, c in zip(sorting["left"], cseq)}
    phys_order = [m for m, _ in phys.as_sets(True)]
    comb_order = list(set(comb.cell_to_idx.values()))

    fig, ax = plt.subplots(figsize=(9, 6))

    sankey = Sankey(
        phys_order, comb_order,
        dict(phys.compare(comb)), ax,
        xticklabels=("physical", "combined"), left_colors=palettable.colorbrewer.qualitative.Set1_6
    )
    fig.tight_layout()
    if save:
        sankey.save(save)
    if show:
        plt.show()


def to_int_idx(*memberships: Membership) -> Iterator[Membership]:
    lastmax = 0
    for m in memberships:
        mapping = {old[0]: new for new, old in enumerate(m.as_sets(True), lastmax+1)}
        yield m.remap_modules(mapping)
        lastmax = max(mapping.values())


def main(show=False, save: Optional[Path] = None):
    phys, comb = get_memberships(0)

    phys = phys.remap_modules({
        num2name(n, "phys. "): name for n, name in [
            (2, "backward locomotion"),
            (7, "forward locomotion"),
            (11, "feeding"),
            (12, "head control"),
            (17, "amphid"),
        ]
    })
    phys_agg = phys.agglomerate(AGGLOM_THRESHOLD, 'phys. other')[0]

    comb = comb.remap_modules({
        num2name(n, "comb. "): name for n, name in [
            (1, "body motor"),
            (6, "core"),
            (10, "feeding + 5HT"),
            (11, "head control + TYR"),
        ]
    })
    comb_agg = comb.agglomerate(AGGLOM_THRESHOLD, 'comb. other')[0]

    draw_sankey(phys_agg, comb_agg, show, save.with_suffix(".svg") if save else None)
    if save:
        fpath = save.with_suffix('.txt')
        write_notes({"PHYSICAL": phys, "COMBINED": comb}, fpath)
        if show:
            with open(fpath) as f:
                print(f.read())


if __name__ == '__main__':
    # phys, comb = get_memberships(0)
    # phys_agg, comb_agg = (m.agglomerate(10)[0] for m in (phys, comb))
    #
    # phys_ints, comb_ints = to_int_idx(phys_agg, comb_agg)
    # d = dict()
    # for (pre, post), weight in phys_ints.compare(comb_ints):
    #     d[f"{pre} {post}"] = weight
    # with open("example.json", 'w') as f:
    #     json.dump(d, f, sort_keys=True, indent=2)
    # main(True)
    main(save=OUT_DIR / "modules.svg")
