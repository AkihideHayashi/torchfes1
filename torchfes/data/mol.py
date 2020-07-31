from typing import Dict, List, Optional, Set
import torch
from torch import Tensor
from .torch import cat as _cat
from .. import properties as p
from ..properties import default_values, batch, save_trj, atoms


def filter_batch(mol: Dict[str, Tensor]):
    return {key: val for key, val in mol.items() if key in batch}


def filter_save_trj(mol: Dict[str, Tensor]):
    return {key: val for key, val in mol.items() if key in save_trj}


def _get_n_batch(mol: Dict[str, Tensor]) -> int:
    n: Optional[int] = None
    for key in mol:
        if key not in batch:
            continue
        if n is None:
            n = mol[key].size(0)
        assert key in batch, key
        assert mol[key].size(0) in (1, n), key
    assert n is not None
    return n


def _mask_ent(mol: Dict[str, Tensor]):
    ret: Dict[str, Tensor] = {}
    ent = mol[p.ent]
    for key in mol:
        if key in atoms:
            ret[key] = mol[key][ent]
        else:
            ret[key] = mol[key]
    return ret


def _mask_mtd_ind(mol: Dict[str, Tensor]):
    if p.mtd_hgt not in mol:
        return mol
    mask = mol[p.mtd_hgt] != 0
    ret = mol.copy()
    ret[p.mtd_hgt] = mol[p.mtd_hgt][mask]
    ret[p.mtd_cen] = mol[p.mtd_cen][mask]
    ret[p.mtd_prc] = mol[p.mtd_prc][mask]
    return ret


def _mask(mol: Dict[str, Tensor]):
    return _mask_mtd_ind(_mask_ent(mol))


def _unsqueeze(mol: Dict[str, Tensor]):
    return {key: val.unsqueeze(0) for key, val in mol.items()}


def unbind(mol: Dict[str, Tensor]):
    n = _get_n_batch(mol)
    ret: List[Dict[str, Tensor]] = [{} for _ in range(n)]
    for key in mol:
        if key not in batch:
            continue
        if mol[key].size(0) == n:
            for i in range(n):
                ret[i][key] = mol[key][i]
        elif mol[key].size(0) == 1:
            for i in range(n):
                ret[i][key] = mol[key].squeeze(0)
        else:
            raise RuntimeError(key)
    return [_unsqueeze(_mask(m)) for m in ret]


def _get_keys(mol: List[Dict[str, Tensor]]):
    keys: Set[str] = set(mol[0].keys())
    for m in mol:
        keys.update(set(m.keys()))
    return keys


def cat(mol: List[Dict[str, Tensor]]):
    keys = _get_keys(mol)
    tmp: Dict[str, List[Tensor]] = {}
    for key in keys:
        assert key in batch
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
    assert (ret[p.ent] == (ret[p.elm] >= 0)).all()
    return ret


def masked_select(mol: Dict[str, Tensor], mask: Tensor):
    mol = filter_batch(mol)
    ret = {}
    for key in mol:
        ret[key] = mol[key][mask]
    return ret
