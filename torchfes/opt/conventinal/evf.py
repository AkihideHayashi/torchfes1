from typing import Dict
from torch import nn, Tensor
from ..utils import Lagrangian, set_hessian
from ..linesearch import limit_step_size
from ..functional import eigenvector_following_direction
from ... import properties as p


class EigenVectorFollowing(nn.Module):
    def __init__(self, lag: Lagrangian, siz: float):
        super().__init__()
        self.lag = lag
        self.siz = siz

    def forward(self, mol: Dict[str, Tensor]):
        if p.gen_hes not in mol:
            mol = self.lag(mol, create_graph=True)
            mol = set_hessian(mol)
        s = eigenvector_following_direction(mol[p.gen_hes], mol[p.gen_grd], 1)
        mol[p.gen_stp_dir] = s
        mol = limit_step_size(mol, self.siz)
        mol[p.gen_pos] = mol[p.gen_pos] + mol[p.gen_stp]
        mol = self.lag(mol, create_graph=True)
        mol = set_hessian(mol)
        return mol
