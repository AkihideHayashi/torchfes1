import math
import warnings
import torch
from torch import Tensor
from pointneighbor import functional as fn


def _get_sft_xyz(cel_mat: Tensor, cel_inv: Tensor, pbc: Tensor, rc: float):
    num_rpt = fn.minimum_neighbor(cel_inv, pbc, math.sqrt(rc))
    max_rpt, _ = num_rpt.max(dim=0)
    sft_cel = fn.arange_prod(max_rpt * 2 + 1) - max_rpt
    sft_xyz = sft_cel.to(cel_mat) @ cel_mat
    return sft_xyz


def min_vec(cel: Tensor, pbc: Tensor, pos: Tensor, idx: Tensor) -> Tensor:
    warnings.warn('min_vec is not valified after changing pointneighbor.')
    cel_inv = cel.inverse()
    i, j = idx.t().unbind(0)
    rri = pos[:, i, :]
    rrj = pos[:, j, :]
    rrij = rrj - rri
    rc = math.sqrt(float((rrij * rrij).sum(-1).max().item()))
    sft_xyz = _get_sft_xyz(cel, cel_inv, pbc, rc)
    rrij = rrij[:, :, None, :] + sft_xyz[:, None, :, :]
    sod = (rrij * rrij).sum(-1)
    argmin = sod.min(-1)[1]
    idx0 = torch.arange(rrij.size(0), device=rrij.device)[:, None]
    idx1 = torch.arange(rrij.size(1), device=rrij.device)[None, :]
    return rrij[idx0, idx1, argmin]
