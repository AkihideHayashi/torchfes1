from typing import Dict
import torch
from torch import Tensor
from .. import properties as p


def _vector_pos(pos: Tensor):
    x = pos.flatten(1).clone().requires_grad_()
    pos_ = x.view_as(pos)
    return x, pos_


def _vector_pos_lag(pos: Tensor, lag: Tensor):
    n_lag = lag.size(1)
    x = torch.cat([lag, pos.flatten(1)], dim=1).clone().requires_grad_()
    lag_ = x[:, :n_lag]
    pos_ = x[:, n_lag:].view_as(pos)
    return x, pos_, lag_


def generalize_pos_lag(inp: Dict[str, Tensor]):
    out = inp.copy()
    x, out[p.pos], out[p.con_lag] = _vector_pos_lag(inp[p.pos], inp[p.con_lag])
    out[p.gen_pos] = x
    return out


def cartesian_pos_frc_lag(inp: Dict[str, Tensor]):
    out = inp.copy()
    x = inp[p.gen_pos]
    g = inp[p.gen_grd]
    out[p.pos], out[p.con_lag] = _split_pos_lag(x, inp[p.pos], inp[p.con_lag])
    out[p.frc], _ = _split_pos_lag(-g, inp[p.pos], inp[p.con_lag])
    return out


def _split_pos_lag(x: Tensor, pos: Tensor, lag: Tensor):
    n_lag = lag.size(1)
    lag_ = x[:, :n_lag]
    pos_ = x[:, n_lag:].view_as(pos)
    return pos_, lag_
