import json
import re
from collections import defaultdict
from pathlib import Path
from typing import NamedTuple, Optional

from connectome_data.constants import StrEnum
from connectome_data.figures.hiveplots.common import get_node_types

with open(Path(__file__).absolute().parent / "nodeclasses.txt") as f:
    name_to_class = dict(tuple(s.strip().split()) for s in f.readlines() if s)


class Dorsoventrality(StrEnum):
    DORSAL = 'D'
    VENTRAL = 'V'


class Laterality(StrEnum):
    LEFT = 'L'
    RIGHT = 'R'


dv_lat_num = re.compile(r'^(?P<dv>[DV]?)(?P<lr>[LR]?)(?P<ordi>\d*)$')


def noop(x):
    return x


def and_then(x, fn=noop):
    if x:
        return fn(x)


class CellName(NamedTuple):
    base: str
    dorsoventrality: Dorsoventrality = None
    laterality: Optional[Laterality] = None
    ordinality: Optional[int] = None

    def same_class(self, other):
        return self.base == other.base

    def family(self):
        return set(class_to_cellnames[self.base])

    def __str__(self):
        s = self.base
        if self.dorsoventrality is not None:
            s += str(self.dorsoventrality)
        if self.laterality is not None:
            s += str(self.laterality)
        if self.ordinality is not None:
            s += str(self.ordinality)
        return s

    @classmethod
    def from_str(cls, name):
        base = name_to_class[name]
        assert name.startswith(base)
        suffix = name[len(name):]
        match = dv_lat_num.search(suffix)
        assert match
        d = match.groupdict()

        return CellName(
            base,
            and_then(d["dv"], Dorsoventrality),
            and_then(d["lr"], Laterality),
            and_then(d["ordi"], int)
        )


class_to_cellnames = defaultdict(set)
for name, cls in name_to_class.items():
    class_to_cellnames[cls].add(CellName(name))


def check_same_ntype():
    ntypes = get_node_types()
    cls_ntypes = dict()
    for cls, cellnames in class_to_cellnames.items():
        this_cls_ntypes = {ntypes[str(n)] for n in cellnames}
        if len(this_cls_ntypes) > 1:
            cls_ntypes[cls] = this_cls_ntypes
    assert not cls_ntypes, f"""
    Some classes had more than one neuron ntype\n
    {json.dumps(cls_ntypes, sort_keys=True, indent=2)}
    """.strip()


if __name__ == '__main__':
    check_same_ntype()
