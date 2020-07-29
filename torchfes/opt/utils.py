import torch
from torch import Tensor


def vector_pos(pos: Tensor):
    x = pos.flatten(1).clone().requires_grad_()
    pos_ = x.view_as(pos)
    return x, pos_


def vector_pos_lag(pos: Tensor, lag: Tensor):
    n_lag = lag.size(1)
    x = torch.cat([lag, pos.flatten(1)], dim=1).clone().requires_grad_()
    lag_ = x[:, :n_lag]
    pos_ = x[:, n_lag:].view_as(pos)
    return x, pos_, lag_


def split_pos_lag(x: Tensor, pos: Tensor, lag: Tensor):
    n_lag = lag.size(1)
    lag_ = x[:, :n_lag]
    pos_ = x[:, n_lag:].view_as(pos)
    return pos_, lag_
