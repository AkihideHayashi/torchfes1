from typing import Dict, Union

import torch
from torch import Tensor

from . import functional as fn
from . import properties as p


def init_inp(cel, pbc, elm, pos, mas=None):
    """
    Parameters:
        cel (float[bch, dim, dim]): cell
        pbc (bool[bch, dim]): periodic boundary condition
        pos (float[bch, atm, dim]): positions
        elm (int[bch, atm])
        mas (float[bch, atm])
    """
    ret = {
        p.cel: cel,
        p.pbc: pbc,
        p.elm: elm,
        p.pos: pos,
        p.ent: elm >= 0,
    }
    if mas is not None:
        ret[p.mas] = mas
    return ret


def add_md(inp: Dict[str, Tensor], dtm: Union[float, Tensor],
           kbt: Union[float, Tensor]):
    if p.mas not in inp:
        raise RuntimeError('inp must have p.mas')
    n_bch = inp[p.pos].size(0)
    if isinstance(dtm, float):
        dtm = inp[p.pos].new_ones([n_bch]) * dtm
    if isinstance(kbt, float):
        kbt = torch.ones_like(dtm) * kbt
    inp[p.dtm] = dtm
    inp[p.tim] = torch.zeros_like(dtm)
    inp[p.stp] = torch.zeros_like(dtm, dtype=torch.long)
    inp[p.kbt] = kbt
    inp[p.mom] = fn.maxwell(inp[p.pos], inp[p.ent], inp[p.mas], kbt)
    inp[p.frc] = torch.zeros_like(inp[p.mom])
    inp[p.frc_res] = torch.zeros_like(inp[p.mom])
    inp[p.frc_mol] = torch.zeros_like(inp[p.mom])


def add_nvt(inp: Dict[str, Tensor], dtm: Union[float, Tensor],
            kbt: Union[float, Tensor]):
    """
    Parameters:
        mas: mas[elm] is mass
    """
    add_md(inp, dtm, kbt)
    if isinstance(kbt, float):
        kbt = torch.ones_like(inp[p.dtm]) * kbt
    inp[p.kbt] = kbt


def add_global_langevin(inp: Dict[str, Tensor], tau_lng: Union[Tensor, float]):
    if isinstance(tau_lng, float):
        tau_lng = torch.ones_like(inp[p.dtm]) * tau_lng
    inp[p.gam_lng] = torch.tensor(1.0) / tau_lng


def add_global_andersen(inp: Dict[str, Tensor], tau_ads: Union[Tensor, float]):
    if isinstance(tau_ads, float):
        tau_ads = torch.ones_like(inp[p.dtm]) * tau_ads
    inp[p.ads_frq] = torch.tensor(1.0) / tau_ads


def add_global_nose_hoover_chain(inp: Dict[str, Tensor],
                                 tau_nhc: Union[Tensor, float]):
    if isinstance(tau_nhc, float):
        tau_nhc = torch.ones_like(inp[p.dtm])[:, None] * tau_nhc
    if tau_nhc.dim() == 1:
        n_bch = inp[p.pos].size(0)
        tau_nhc = tau_nhc[None, :].expand([n_bch, -1])
    num_atm = inp[p.ent].sum(-1)
    n_dim = inp[p.pos].size()[-1]
    kbt = inp[p.kbt]
    x = torch.ones_like(tau_nhc) * kbt
    x[:, 0] = num_atm * n_dim * kbt
    mas_nhc = x * (tau_nhc * tau_nhc)
    pos_nhc = torch.zeros_like(mas_nhc)
    mom_nhc = torch.zeros_like(mas_nhc)
    inp[p.mas_nhc] = mas_nhc
    inp[p.pos_nhc] = pos_nhc
    inp[p.mom_nhc] = mom_nhc
