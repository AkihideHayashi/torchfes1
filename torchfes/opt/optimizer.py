from typing import Dict
import torch
from torch import nn, Tensor
from .utils import Lagrangian
from .. import properties as p


class Optimizer(nn.Module):
    def __init__(self, lag: Lagrangian, dir: nn.Module):
        super().__init__()
        self.lag = lag
        self.dir = dir

    def _direction(self, mol: Dict[str, Tensor]):
        mol = self.dir(mol)
        gsd = mol[p.gen_stp_dir]
        n_bch = gsd.size(0)
        mol[p.gen_stp_siz] = torch.ones(
            [n_bch], dtype=gsd.dtype, device=gsd.device)
        mol[p.gen_stp] = mol[p.gen_stp_dir]
        return mol

    def forward(self, mol: Dict[str, Tensor]):
        if p.gen_stp not in mol:
            mol = self.lag(mol, create_graph=True)
            mol = self._direction(mol)
        mol[p.gen_pos] = mol[p.gen_pos] + mol[p.gen_stp]
        mol = self.lag(mol, create_graph=True)
        mol = self._direction(mol)
        return mol
