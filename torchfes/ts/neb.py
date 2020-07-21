from typing import Dict
import torch
from torch import nn, Tensor
from .. import properties as p


class NudgedElasticBand(nn.Module):

    def __init__(self, k: float):
        super().__init__()
        self.k = torch.tensor(k)

    def forward(self, inp: Dict[str, Tensor], create_graph, retain_graph):
        pos = inp[p.pos]
        frc = inp[p.frc]
        pos_mns = pos[:-2, :, :]
        pos_pls = pos[2:, :, :]
        pos_zer = pos[1:-1, :, :]
        vec_pls = pos_pls - pos_zer
        vec_mns = pos_zer - pos_mns

        frc_neb = nudge(frc[1:-1, :, :], vec_pls, vec_mns, self.k)
        frc = frc + torch.nn.functional.pad(frc_neb, [0, 0, 0, 0, 1, 1])
        out = inp.copy()
        out[p.frc] = frc
        return out


def unit(vec: Tensor):
    return vec / dot(vec, vec).sqrt()


def dot(a: Tensor, b: Tensor):
    return (a * b).sum(dim=-2, keepdim=True).sum(dim=-1, keepdim=True)


def nudge(frc, vecp, vecm, k):
    tau = unit(unit(vecp) + unit(vecm))
    term1 = dot(frc, tau) * tau
    term2 = k * dot(vecp - vecm, tau) * tau
    return - term1 + term2
