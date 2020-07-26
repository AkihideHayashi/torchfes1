from typing import Dict

import torch
from torch import Tensor, nn

from .. import properties as p
from ..functional import noise


def updt_tim(inp: Dict[str, Tensor], stp: float):
    out = inp.copy()
    out[p.tim] = out[p.tim] + out[p.dtm] * stp
    out[p.stp] = out[p.stp] + stp
    return out


def updt_mom(inp: Dict[str, Tensor], stp: float):
    out = inp.copy()
    dtm = out[p.dtm][:, None, None]
    out[p.mom] = out[p.mom] + out[p.frc] * dtm * stp
    return out


def updt_pos(inp: Dict[str, Tensor], stp: float):
    out = inp.copy()
    dtm = out[p.dtm][:, None, None]
    mas = out[p.mas][:, :, None]
    out[p.pos] = out[p.pos] + out[p.mom] * dtm / mas * stp
    return out


def update_global_langevin(inp: Dict[str, Tensor], stp: float):
    out = inp.copy()
    kbt = out[p.kbt][:, None, None]
    mas = out[p.mas][:, :, None]
    ent = out[p.ent][:, :, None]
    c1 = torch.exp(-out[p.gam_lng] * out[p.dtm] * stp)[:, None, None]
    c2 = ((torch.tensor(1, device=c1.device) - c1 * c1) * kbt * mas).sqrt()
    out[p.mom] = (c1 * out[p.mom] + c2 * noise(out[p.frc], ent))
    return out


def update_global_andersen(inp: Dict[str, Tensor], stp: float):
    out = inp.copy()
    frq = inp[p.ads_frq]
    mu_ = torch.randn_like(frq)
    ren = mu_ < torch.tensor(1) - torch.exp(-frq * inp[p.dtm] * stp)
    rnd = torch.randn_like(inp[p.mom])
    rnd = (inp[p.mas][:, :, None] * inp[p.kbt][:, None, None]).sqrt() * rnd
    mom = torch.where(ren, rnd, inp[p.mom])
    out[p.mom] = mom
    return out


def _global_nhc_g(inp: Dict[str, Tensor], ddof: Tensor):
    _, n_atm, n_dim = inp[p.pos].size()
    dof = torch.ones_like(ddof) * n_atm * n_dim - ddof
    g = torch.zeros_like(inp[p.mom_nhc])
    kbt = inp[p.kbt]
    pn = inp[p.mom_nhc]
    mn = inp[p.mas_nhc]
    pm = inp[p.mom]
    mm = inp[p.mas][:, :, None]
    g[:, 1:] = (pn * pn / mn)[:, :-1] - kbt[:, None]
    g[:, 0] = (pm * pm / mm).sum(2).sum(1) - kbt * dof
    return g


def _update_global_nhc(inp: Dict[str, Tensor], stp: float, ddof: Tensor):
    out = inp.copy()
    g = _global_nhc_g(out, ddof)
    dea = inp[p.dtm] * stp
    pos = out[p.pos_nhc].clone()
    mom = out[p.mom_nhc].clone()
    mas = out[p.mas_nhc].clone()
    mom[:, -1] += g[:, -1] * dea / 2

    for j in range(mom.size(1) - 2, -1, -1):
        g = _global_nhc_g(out, ddof)
        mom[:, j] *= torch.exp(- mom[:, j + 1] / mas[:, j + 1] * dea / 4)
        mom[:, j] += g[:, j] * dea / 2
        mom[:, j] *= torch.exp(- mom[:, j + 1] / mas[:, j + 1] * dea / 4)
    pos[:, :] += mom / mas * dea[:, None]
    out[p.mom] = out[p.mom] * torch.exp(- mom[:, 0] / mas[:, 0] * dea[:, None])

    for j in range(mom.size(1) - 2, -1, -1):
        g = _global_nhc_g(out, ddof)
        mom[:, j] *= torch.exp(- mom[:, j + 1] / mas[:, j + 1] * dea / 4)
        mom[:, j] += g[:, j] * dea / 2
        mom[:, j] *= torch.exp(- mom[:, j + 1] / mas[:, j + 1] * dea / 4)
    g = _global_nhc_g(out, ddof)
    mom[:, -1] += g[:, -1] * dea / 2
    out[p.pos_nhc] = pos
    out[p.mom_nhc] = mom
    return out


def update_global_nhc(inp: Dict[str, Tensor], stp: float, nrespa: int,
                      ddof: Tensor):
    out = inp.copy()
    w1 = w7 = 0.784513610477560
    w2 = w6 = 0.235573213359357
    w3 = w5 = -1.17767998417887
    w4 = 1 - w1 - w2 - w3 - w5 - w6 - w7
    for _ in range(nrespa):
        for w in [w1, w2, w3, w4, w5, w6, w7]:
            s = stp * w / nrespa
            out = _update_global_nhc(out, s, ddof)
    return out


class GlobalLangevin(nn.Module):
    def forward(self, inp: Dict[str, Tensor], stp: float):
        return update_global_langevin(inp, stp)


class GlobalAndersen(nn.Module):
    def forward(self, inp: Dict[str, Tensor], stp: float):
        return update_global_andersen(inp, stp)


class GlobalNHC(nn.Module):
    def __init__(self, nrespa: int):
        super().__init__()
        self.nrespa = nrespa

    def forward(self, inp: Dict[str, Tensor], stp: float):
        ddof = torch.zeros_like(inp[p.dtm], dtype=torch.int64)
        return update_global_nhc(inp, stp, self.nrespa, ddof)
