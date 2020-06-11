from typing import Dict

import torch
from torch import Tensor, nn

from ..general import PosEngFrcStorage, where_pef
from ..forcefield import EvalEnergiesForcesGeneral


class LineSearchOptimizer(nn.Module):
    """All methods that use line search are performed using this class.
    Args:
        evl: energy and forces evaluator.
        vec: from torchfes.opt.vec
             determine the good direction for optimize.
        stp: from torchfes.opt.stp
             determine line search step.
        con: from torchfes.opt.con
             wolfe condition or others.
        reset: reset stp for each line search?
        sync: syncronize line search - vector detemination sycle among batch?
    """
    def __init__(self, evl: EvalEnergiesForcesGeneral,
                 vec, stp, con, reset, sync: bool):
        super().__init__()
        self.evl = evl
        self.vec = vec
        self.stp = stp
        self.con = con
        self.pef = PosEngFrcStorage()
        self.reset = reset
        self.n_vec = 0
        self.sync = sync

    def init(self, env: Dict[str, Tensor], pos: Tensor):
        pef, _ = self.vec.init(pos, env)
        self.pef(pef)
        self.stp.init(pef, pef.eng == pef.eng, self.reset)

    def get_flt_vec(self, con: Tensor):
        if self.sync:
            if (con == 0).all():
                return torch.ones_like(con, dtype=torch.bool)
            else:
                return torch.zeros_like(con, dtype=torch.bool)
        else:
            return con == 0

    def forward(self, env: Dict[str, Tensor], reset: bool = False):
        pef = self.pef()
        stp = self.stp.peek()
        vec = self.vec.peek()
        pos_tmp = pef.pos + stp * vec
        pef_tmp = self.evl(env, pos_tmp)
        con = self.con(pef, pef_tmp, stp, vec)

        flt_vec = self.get_flt_vec(con)
        flt_stp = ~flt_vec

        pef = pef_tmp
        vec = self.vec(pef, env, flt_vec, reset)
        stp = self.stp.init(pef, flt_vec, self.reset)
        self.n_vec += 1

        stp = self.stp(con, pef_tmp, flt_stp)

        self.pef(where_pef(flt_vec, pef, self.pef()))
        return pef_tmp
