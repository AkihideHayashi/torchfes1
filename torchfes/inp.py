from typing import Dict
import torch
from torch import Tensor
from . import properties as p, functional as fn


def init_inp(cel, pbc, elm, pos):
    return {
        p.cel: cel,
        p.pbc: pbc,
        p.elm: elm,
        p.pos: pos,
        p.ent: elm >= 0,
    }


def add_md(inp: Dict[str, Tensor], mom, mas, tim, dtm, frc=None):
    if frc is None:
        frc = torch.zeros_like(inp[p.pos])
    inp[p.mom] = mom
    inp[p.mas] = mas
    inp[p.tim] = tim
    inp[p.dtm] = dtm
    inp[p.frc] = frc


def add_nvt(inp: Dict[str, Tensor], kbt):
    inp[p.kbt] = kbt


def add_global_langevin(inp: Dict[str, Tensor], tau_lng: Tensor):
    inp[p.gam_lng] = torch.tensor(1.0) / tau_lng


def add_global_andersen(inp: Dict[str, Tensor], tau_ads: Tensor):
    inp[p.ads_frq] = torch.tensor(1.0) / tau_ads


def add_global_nose_hoover_chain(inp: Dict[str, Tensor], tau_nhc: Tensor):
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


def init_nvt(inp: Dict[str, Tensor], mas: Tensor, dtm: Tensor, kbt: Tensor):
    mas_ = mas[inp[p.elm]]
    mom = fn.maxwell(inp[p.pos], inp[p.ent], mas_, kbt)
    tim = torch.zeros_like(dtm)
    add_md(inp, mom, mas_, tim, dtm)
    add_nvt(inp, kbt)
