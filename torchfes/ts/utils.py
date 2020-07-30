from typing import Dict, List

import torch
from torch import Tensor

from .. import properties as p


def inter_image_vec(pos: Tensor) -> Tensor:
    diff = pos[:-1, :, :] - pos[1:, :, :]
    return diff


def inter_image_sod(pos: Tensor) -> Tensor:
    diff = pos[:-1, :, :] - pos[1:, :, :]
    sod = diff.pow(2).sum(-1).sum(-1)
    return sod


def split_spring(eng_bnd: Tensor):
    assert eng_bnd.dim() == 1
    return 0.5 * (
        torch.nn.functional.pad(eng_bnd, [0, 1]) +
        torch.nn.functional.pad(eng_bnd, [1, 0]))[:, None]


def spring_energy(eng_bnd: Tensor):
    n_bch = eng_bnd.size(0) + 1
    return eng_bnd[None, :].expand((n_bch, -1)) / n_bch


def _linear_interpolation(ini, fin, n):
    if ini.dtype == torch.bool:
        assert (fin == ini).all()
        size = list(ini.size())
        size[0] = n
        return ini.expand(size)
    if ini.dtype == torch.long:
        assert (fin == ini).all()
        size = list(ini.size())
        size[0] = n
        return ini.expand(size)
    n = n - 1
    return torch.cat([(ini * (n - i) + fin * i) / n for i in range(n + 1)])


def linear_interpolation(mid: List[Dict[str, Tensor]], n: int):
    out = {}
    for key in mid[0]:
        out[key] = _linear_interpolation(mid[0][key], mid[1][key], n)
    for key in mid[0]:
        if key not in p.batch:
            raise KeyError(key)
    out[p.ent] = out[p.elm] >= 0
    return out
