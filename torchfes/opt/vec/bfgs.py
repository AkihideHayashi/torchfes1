from typing import Dict
import torch
from torch import nn, Tensor
from ...general import PosEngFrc, PosEngFrcStorage
from ...utils import grad


class QuasiNewtonInitWithDiagonal(nn.Module):
    def __init__(self, a: float):
        super().__init__()
        self.a = a

    def forward(self, pef: PosEngFrc, _: Dict[str, Tensor]):
        n_bch, n_dim = pef.pos.size()
        eye = torch.eye(n_dim).unsqueeze(0).expand([n_bch, n_dim, n_dim])
        return eye * self.a


class QuasiNewtonInitWithExact(nn.Module):
    def __init__(self, evl):
        super().__init__()
        self.evl = evl

    def forward(self, inp: PosEngFrc, env: Dict[str, Tensor]):
        pef: PosEngFrc = self.evl(env, inp.pos, frc_grd=True)
        frc = pef.frc
        pos = pef.pos
        hes_lst = []
        n_bch, n_dim = inp.pos.size()
        for i in range(n_dim):
            grd_out = pef.pos.new_zeros([n_bch, n_dim])
            grd_out[:, i] = 1.0
            hes_lst.append(
                grad(-frc, pos, grd_out, create_graph=False, retain_graph=True)
            )
        hes = torch.stack(hes_lst, 2)
        return hes.inverse()


def _mv(A: Tensor, x: Tensor):
    return (A @ x.unsqueeze(-1)).squeeze(-1)


def _vv(x: Tensor, y: Tensor):
    return (x * y).sum(-1)


def _uns(x: Tensor):
    return x.unsqueeze(-1).unsqueeze(-1)


def outer(x: Tensor, y: Tensor):
    return x[:, :, None] * y[:, None, :]


class BFGS(nn.Module):
    def __init__(self, evl, ini):
        super().__init__()
        self.evl = evl
        self.ini = ini
        self.hes_inv = torch.tensor([])
        self.vec = torch.tensor([])
        self.old = PosEngFrcStorage()

    def get_vec(self, frc: Tensor):
        return _mv(self.hes_inv, frc)

    def _init(self, pef: PosEngFrc, env: Dict[str, Tensor]):
        self.hes_inv = self.ini(pef, env)
        self.vec = self.get_vec(pef.frc)
        self.old(pef)
        return pef, self.vec

    def init(self, pos: Tensor, env: Dict[str, Tensor]):
        pef = self.evl(env, pos)
        return self._init(pef, env)

    def peek(self):
        return self.vec

    def forward(self, pef: PosEngFrc, env: Dict[str, Tensor], flt: Tensor,
                reset: bool = False):
        if reset:
            self._init(pef, env)
            return self.vec
        if flt.all().item():
            old: PosEngFrc = self.old()
            s = pef.pos - old.pos
            y = old.frc - pef.frc
            B = self.hes_inv
            num1 = _uns((_vv(s, y) + _vv(y, _mv(B, y)))) * outer(s, s)
            den1 = _uns(_vv(s, y).pow(2))
            num2_1 = outer(_mv(B, y), s)
            num2_2 = num2_1.transpose(1, 2)
            num2 = num2_1 + num2_2
            den2 = _uns(_vv(s, y))
            self.hes_inv = B + (num1 / den1) - (num2 / den2)
            self.vec = self.get_vec(pef.frc)
            return self.vec
        else:
            if flt.any().item():
                raise NotImplementedError(
                    'BFGS for sync=False is not implemented.'
                )
            return self.vec
