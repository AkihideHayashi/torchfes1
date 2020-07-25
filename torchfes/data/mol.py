from typing import Dict, List
from torch import Tensor
from .. import properties as p
from .torch import stack as _stack
from .default import default_values


def unbind(mol: Dict[str, Tensor]):
    assert mol[p.pos].dim() == 3
    n = mol[p.pos].size(0)
    for key in mol:
        assert mol[key].size(0) == n, key
    return [{key: mol[key][i] for key in mol} for i in range(n)]


def stack(mol: List[Dict[str, Tensor]]):
    tmp: Dict[str, List[Tensor]] = {}
    for key in mol[0].keys():
        tmp[key] = []
        for m in mol:
            tmp[key].append(m[key])
    ret: Dict[str, Tensor] = {}
    for key in tmp.keys():
        ret[key] = _stack(tmp[key], default_values[key], dim=0)
    return ret
