from typing import Dict, Union

import torch
from torch import Tensor

from . import functional as fn
from . import properties as p


def init_mol(cel, pbc, elm, pos, mas=None):
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


def add_basic(mol: Dict[str, Tensor]):
    new = mol.copy()
    if p.ent not in mol:
        new[p.ent] = new[p.elm] >= 0
    if p.num not in mol:
        new[p.num] = new[p.ent].sum(dim=1)
    if p.num_sqt not in mol:
        new[p.num_sqt] = new[p.num].to(new[p.pos]).sqrt()
    return new


def add_md(mol: Dict[str, Tensor],
           dtm: Union[float, Tensor], kbt: Union[float, Tensor]):
    if p.mas not in mol:
        raise RuntimeError('inp must have p.mas')
    n_bch = mol[p.pos].size(0)
    if isinstance(dtm, float):
        dtm = mol[p.pos].new_ones([n_bch]) * dtm
    if isinstance(kbt, float):
        kbt = torch.ones_like(dtm) * kbt
    new = mol.copy()
    new[p.dtm] = dtm
    new[p.tim] = torch.zeros_like(dtm)
    new[p.stp] = torch.zeros_like(dtm)
    new[p.kbt] = kbt
    new[p.mom] = fn.maxwell(mol[p.pos], mol[p.ent], mol[p.mas], kbt)
    return new


def add_nvt(mol: Dict[str, Tensor],
            dtm: Union[float, Tensor], kbt: Union[float, Tensor]):
    new = add_md(mol, dtm, kbt)
    if isinstance(kbt, float):
        kbt = torch.ones_like(new[p.dtm]) * kbt
    new[p.kbt] = kbt
    return new


def add_global_langevin(mol: Dict[str, Tensor], tau_lng: Union[Tensor, float]):
    if isinstance(tau_lng, float):
        tau_lng = torch.ones_like(mol[p.dtm]) * tau_lng
    new = mol.copy()
    new[p.gam_lng] = torch.tensor(1.0) / tau_lng
    return new


def add_global_andersen(mol: Dict[str, Tensor], tau_ads: Union[Tensor, float]):
    if isinstance(tau_ads, float):
        tau_ads = torch.ones_like(mol[p.dtm]) * tau_ads
    new = mol.copy()
    new[p.ads_frq] = torch.tensor(1.0) / tau_ads
    return new


def add_global_nose_hoover_chain(mol: Dict[str, Tensor],
                                 tau_nhc: Union[Tensor, float]):
    if isinstance(tau_nhc, float):
        tau_nhc = torch.ones_like(mol[p.dtm])[:, None] * tau_nhc
    if tau_nhc.dim() == 1:
        n_bch = mol[p.pos].size(0)
        tau_nhc = tau_nhc[None, :].expand([n_bch, -1])
    n_dim = mol[p.pos].size(2)
    x = torch.ones_like(tau_nhc) * mol[p.kbt]
    x[:, 0] = mol[p.num] * n_dim * mol[p.kbt]
    mas_nhc = x * (tau_nhc * tau_nhc)
    pos_nhc = torch.zeros_like(mas_nhc)
    mom_nhc = torch.zeros_like(mas_nhc)
    new = mol.copy()
    new[p.mas_nhc] = mas_nhc
    new[p.pos_nhc] = pos_nhc
    new[p.mom_nhc] = mom_nhc
    return new
