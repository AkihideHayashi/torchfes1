from typing import Dict
import torch
from torch import nn, Tensor
from ...forcefield import EvalEnergiesForcesGeneral
from ...utils import grad


class Hessian(nn.Module):
    def __init__(self, evl: EvalEnergiesForcesGeneral):
        super().__init__()
        self.evl = evl

    def forward(self, env: Dict[str, Tensor], pos: Tensor):
        _, pef = self.evl(env, pos, frc_grd=True)
        frc = pef.frc
        hes_lst = []
        n_bch, n_dim = pos.size()
        for i in range(n_dim):
            grd_out = pef.pos.new_zeros([n_bch, n_dim])
            grd_out[:, i] = 1.0
            hes_lst.append(
                grad(-frc, pef.pos, grd_out,
                     create_graph=False, retain_graph=True)
            )
        hes = torch.stack(hes_lst, 2)
        return hes
