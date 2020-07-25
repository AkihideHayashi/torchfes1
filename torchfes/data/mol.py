from typing import Dict, List, Optional, Set
import torch
from torch import Tensor
from .torch import stack as _stack, cat as _cat
from .. import properties as p
from ..properties import default_values, batch, save_trj, atoms


def filter_batch(mol: Dict[str, Tensor]):
    return {key: val for key, val in mol.items() if key in batch}


def filter_save_trj(mol: Dict[str, Tensor]):
    return {key: val for key, val in mol.items() if key in save_trj}


def _get_n_batch(mol: Dict[str, Tensor]) -> int:
    n: Optional[int] = None
    for key in mol:
        if n is None:
            n = mol[key].size(0)
        assert key in batch
        assert mol[key].size(0) == n, key
    assert n is not None
    return n


def _mask_ent(mol: Dict[str, Tensor]):
    ret: Dict[str, Tensor] = {}
    ent = mol.pop(p.ent)
    for key in mol:
        if key in atoms:
            ret[key] = mol[key][ent]
        else:
            ret[key] = mol[key]
    return ret


def unbind(mol: Dict[str, Tensor]):
    n = _get_n_batch(mol)
    ret: List[Dict[str, Tensor]] = [{} for _ in range(n)]
    for key in mol:
        for i in range(n):
            ret[i][key] = mol[key][i]
    return [_mask_ent(m) for m in ret]


def _get_keys(mol: List[Dict[str, Tensor]]):
    keys: Set[str] = set(mol[0].keys())
    for m in mol:
        if keys != set(m.keys()):
            raise RuntimeError()
    return keys


def stack(mol: List[Dict[str, Tensor]]):
    keys = _get_keys(mol)
    tmp: Dict[str, List[Tensor]] = {}
    for key in keys:
        tmp[key] = []
        for m in mol:
            tmp[key].append(m[key])
    ret: Dict[str, Tensor] = {}
    for key in tmp.keys():
        if key in atoms:
            ret[key] = _stack(tmp[key], default_values[key], dim=0)
        else:
            ret[key] = torch.stack(tmp[key], dim=0)
    ret[p.ent] = ret[p.elm] >= 0
    return ret


def cat(mol: List[Dict[str, Tensor]]):
    keys = _get_keys(mol)
    tmp: Dict[str, List[Tensor]] = {}
    for key in keys:
        tmp[key] = []
        for m in mol:
            tmp[key].append(m[key])
    ret: Dict[str, Tensor] = {}
    for key in tmp.keys():
        if key in atoms:
            ret[key] = _cat(tmp[key], default_values[key], dim=0)
        else:
            ret[key] = torch.cat(tmp[key], dim=0)
    ret[p.ent] = ret[p.elm] >= 0
    return ret
