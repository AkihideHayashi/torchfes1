from typing import Dict
from torch import Tensor
from ..properties import batch


def reprecate(inp: Dict[str, Tensor], n: int):
    out = {}
    for key in inp:
        if key not in batch:
            raise KeyError(key)
    for key, val in inp.items():
        size = list(val.size())
        assert size[0] == 1
        size[0] = n
        out[key] = val.expand(size)
    return out
