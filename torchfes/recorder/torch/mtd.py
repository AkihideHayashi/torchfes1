from typing import Dict, Optional
from torch import Tensor
from ...fes.mtd import add_gaussian
from .pathpair import PathPair
from .recorder import open_torch


def read_mtd(path: PathPair, inp: Optional[Dict[str, Tensor]] = None):
    if inp is None:
        out = {}
    else:
        out = inp.copy()
    with open_torch(path, 'rb') as f:
        for data in f:
            out = add_gaussian(out, data)
    return out
