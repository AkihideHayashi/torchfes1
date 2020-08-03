from typing import Dict
from torch import Tensor, nn
from .. import properties as p
from ..utils import detach


class Optimizer(nn.Module):
    def __init__(self, gen, spc, vec):
        super().__init__()
        self.gen = gen
        self.spc = spc
        self.vec = vec

    def forward(self, mol: Dict[str, Tensor]):
        mol = self.gen(mol)
        mol = self.vec(mol)
        mol[p.gen_pos] = mol[p.gen_pos] + mol[p.gen_vec]
        mol = self.spc(mol)
        return detach(mol)
