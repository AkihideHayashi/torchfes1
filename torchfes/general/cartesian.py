from typing import Dict
import torch
from torch import nn, Tensor
from .. import properties as p


def cartesian_coordinate(inp: Dict[str, Tensor], use_cel: bool = False):
    if use_cel:
        val = torch.cat([inp[p.cel].detach(), inp[p.pos].detach()], dim=1)
    else:
        val = inp[p.pos].detach()
    n_bch, n_atm_cel, n_dim = val.size()
    return val.view([n_bch, n_atm_cel * n_dim])


class CartesianCoordinate(nn.Module):
    def __init__(self, use_cel: bool = False):
        super().__init__()
        self.use_cel = use_cel

    def forward(self, inp: Dict[str, Tensor], pos_cel: Tensor):
        assert pos_cel.dim() == 2, pos_cel.dim()
        n_bch, n_atm, n_dim = inp[p.pos].size()
        out = inp.copy()
        if self.use_cel:
            pos_cel = pos_cel.view([n_bch, n_atm + n_dim, n_dim])
            cel = pos_cel[:, :n_dim, :]
            pos = pos_cel[:, n_dim:, :]
            out[p.pos] = pos
            out[p.cel] = cel
        else:
            out[p.pos] = pos_cel.view([n_bch, n_atm, n_dim])
        return out
