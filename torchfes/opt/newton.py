from typing import Dict
from torch import nn, Tensor
from .utils import set_hessian, solve
from .. import properties as p


class NewtonDirection(nn.Module):
    def forward(self, mol: Dict[str, Tensor]):
        mol = set_hessian(mol)
        mol[p.gen_stp_dir] = -solve(mol[p.gen_grd], mol[p.gen_hes])
        return mol
