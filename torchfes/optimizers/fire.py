from itertools import count
from typing import NamedTuple, Dict, Callable, Optional
import torch
from torch import Tensor
from torch.nn import Module
from ..md import leap_frog
from ..forcefield import EvalEnergiesForces, EvlAdjEngFrc
from .. import properties as p
from ..mol import add_md


class FIRE(NamedTuple):
    dtm_ini: float
    dtm_max: float
    a0: float = 0.1
    n_min: int = 5
    f_a: float = 0.99
    f_inc: float = 1.1
    f_dec: float = 0.5


def fire_ensemble(fire: FIRE, evl: EvlAdjEngFrc,
                  mol: Dict[str, Tensor], fin: Callable,
                  con: Optional[Module] = None, con_tol: float = 1e-4):
    md = leap_frog(evl, con=con, tol=con_tol, bme=False)
    md = md.to(mol[p.pos].device)
    dtm_ini, dtm_max, a0, n_min, f_a, f_inc, f_dec = fire
    a = a0
    NP = 0
    dtm = dtm_ini
    mol = add_md(mol, dtm, 0.0)
    for i in count():
        mol = md(mol)
        if fin(i, mol):
            return
        F = mol[p.frc]
        v = mol[p.mom] / mol[p.mas].unsqueeze(-1)
        P = (F * v).sum()
        Fn = (F * F).sum().sqrt()
        vn = (v * v).sum().sqrt()
        Fh = F / Fn
        v = (1 - a) * v + a * Fh * vn
        if P > 0:
            NP += 1
            if NP > n_min:
                dtm = min(dtm * f_inc, dtm_max)
                a = a * f_a
        else:
            NP = 0
            dtm = dtm * f_dec
            a = a0
            v.fill_(0.0)
        mol[p.dtm].fill_(dtm)
        mol[p.mom] = v * mol[p.mas].unsqueeze(-1)
