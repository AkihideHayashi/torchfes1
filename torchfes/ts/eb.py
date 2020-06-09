import warnings
from typing import Dict
import torch
from torch import nn, Tensor
from .. import properties as p


class ElasticBand(nn.Module):
    def __init__(self, k: float):
        super().__init__()
        self.k = k

    def forward(self, inp: Dict[str, Tensor]):
        warnings.warn('eb not tested yet.')
        pos = inp[p.pos]
        diff = pos[1:] - pos[:-1]
        sod = diff.pow(2).sum(-1)
        eng_bnd = (sod * self.k).sum(-1) * 0.5
        eng_mol = 0.5 * (
            torch.nn.functional.pad(eng_bnd, [0, 1]) +
            torch.nn.functional.pad(eng_bnd, [1, 0]))
        return eng_mol
