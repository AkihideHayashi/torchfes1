from typing import Dict
from torch import nn, Tensor
from .. import properties as p
from .utils import set_hessian
from .functional import eigenvector_following_direction


class EigenVectorFollowing(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.n = n

    def forward(self, mol: Dict[str, Tensor]):
        mol = mol.copy()
        mol = set_hessian(mol)
        mol[p.gen_stp_dir] = eigenvector_following_direction(
            mol[p.gen_hes], mol[p.gen_grd], self.n)
        return mol
