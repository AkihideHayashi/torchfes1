from typing import Dict
import numpy as np
import torch
from torch import Tensor
from torchfes.forcefield import EvlAdjEngFrc
from .. import properties as p
from ..utils import grad


class SciPyWrapper:
    def __init__(self, evl: EvlAdjEngFrc, mol: Dict[str, Tensor], col=None):
        self.mol = mol
        self.evl = evl
        self.col = col

    def _msk(self):
        if p.fix_msk in self.mol:
            return self.mol[p.fix_msk]
        else:
            return self.mol[p.pos] != self.mol[p.pos]

    def _inner(self, x: np.ndarray, msk):
        self.mol[p.pos][~msk] = torch.from_numpy(x).to(self.mol[p.pos])
    
    def get_mol(self, x: np.ndarray):
        msk = self._msk()
        self.mol[p.pos][~msk] = torch.from_numpy(x).to(self.mol[p.pos])
        self.mol = self.evl(self.mol)
        return self.mol

    def _evl(self):
        self.mol = self.evl(self.mol)

    def fun(self, x: np.ndarray):
        msk = self._msk()
        self._inner(x, msk)
        self._evl()
        return self.mol[p.eng].sum().numpy()

    def jac(self, x: np.ndarray):
        msk = self._msk()
        self._inner(x, msk)
        self._evl()
        ret = -self.mol[p.frc][~msk]
        return ret.numpy()

    def constraints_fun(self, x: np.ndarray, i, j):
        msk = self._msk()
        self._inner(x, msk)
        col = self.col(self.mol) - self.mol[p.con_cen]
        ret = col[i, j]
        return ret.numpy()

    def constraints_jac(self, x: np.ndarray, i, j):
        msk = self._msk()
        self._inner(x, msk)
        self.mol[p.pos].requires_grad_(True)
        col = self.col(self.mol) - self.mol[p.con_cen]
        jac = grad(col[i, j], self.mol[p.pos])
        ret = jac[~msk]
        self.mol[p.pos].detach_()
        return ret.numpy()

    def x0(self):
        msk = self._msk()
        return self.mol[p.pos][~msk].numpy()
