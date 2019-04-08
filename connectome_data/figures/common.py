from typing import Dict

from colour import Color
import numpy as np


def merge(*args: Dict, **kwargs) -> Dict:
    """Merge any number of dicts.

    No dicts are mutated.
    Values from subsequent dicts overwrite earlier ones.
    Kwargs overwrite earlier values.
    """
    out = dict()
    for arg in args:
        if isinstance(arg, dict):
            out.update(arg)
    out.update(kwargs)
    return out


def module_name(idx):
    if isinstance(idx, str):
        return idx
    if int(idx) != idx:
        raise ValueError("idx should be an integer")
    if 0 > idx > 99:
        raise ValueError("idx should be between 0 and 99")
    return f"module {idx:02d}"


def blend_colors(colors_weights, method='hsl'):
    colors, weights = zip(*((getattr(Color(c), method), w) for c, w in colors_weights))
    colors = np.array(colors)
    weights = np.array(weights)  # maybe sqrt
    return Color(**{method: np.average(colors, axis=0, weights=weights)})


if __name__ == '__main__':
    val = blend_colors([
        ("red", 1),
        ("yellow", 4),
        ("blue", 5),
    ], 'hsl').hex_l
    print(val)
