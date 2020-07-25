from typing import Dict
import torch
from torch import nn, Tensor
from .. import properties as p


class NudgedElasticBand(nn.Module):

    def __init__(self, k: float, nudged=1.0):
        super().__init__()
        self.k = torch.tensor(k)
        self.nudged = nudged

    def forward(self, inp: Dict[str, Tensor], create_graph, retain_graph):
        pos = inp[p.pos]
        frc = inp[p.frc]
        pos_mns = pos[:-2, :, :]
        pos_pls = pos[2:, :, :]
        pos_zer = pos[1:-1, :, :]
        vec_pls = pos_pls - pos_zer
        vec_mns = pos_zer - pos_mns

        if self.nudged == 1.0:
            frc_neb = nudged_elastic_band(
                frc[1:-1, :, :], vec_pls, vec_mns, self.k)
        elif self.nudged == 0.0:
            frc_neb = elastic_band(
                frc[1:-1, :, :], vec_pls, vec_mns, self.k)
        else:
            frc_neb = (
                self.nudged * nudged_elastic_band(
                    frc[1:-1, :, :], vec_pls, vec_mns, self.k) +
                (1 - self.nudged) * elastic_band(
                    frc[1:-1, :, :], vec_pls, vec_mns, self.k))
        frc = frc + torch.nn.functional.pad(frc_neb, [0, 0, 0, 0, 1, 1])
        out = inp.copy()
        out[p.frc] = frc
        return out


def unit(vec: Tensor):
    return vec / dot(vec, vec).sqrt()


def dot(a: Tensor, b: Tensor):
    return (a * b).sum(dim=-2, keepdim=True).sum(dim=-1, keepdim=True)


def nudged_elastic_band(frc, vecp, vecm, k):
    tau = unit(unit(vecp) + unit(vecm))
    term1 = dot(frc, tau) * tau
    term2 = k * dot(vecp - vecm, tau) * tau
    return - term1 + term2


def elastic_band(frc, vecp, vecm, k):
    return (vecp - vecm) * k
