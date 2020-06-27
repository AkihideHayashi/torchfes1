from typing import Dict
from torch import Tensor
from .. import properties as p


def not_tmp(inp: Dict[str, Tensor]):
    return {key: val.to('cpu') for key, val in inp.items()
            if not p.is_tmp(key)}
