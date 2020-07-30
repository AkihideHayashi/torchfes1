from typing import Dict
import torch
from torch import Tensor, nn
from .. import properties as p
from .eigenvectorfollowing import eigenvector_following
from .derivative import hessian, grad
from .utils import limit_step_size, vector_pos_lag, split_pos_frc_lag
from ..utils import detach, Sequential


class EigenVectorFollowing(nn.Module):
    def __init__(self, adj, eng, con, ord_sdp: int, max_stp: float):
        super().__init__()
        self.adj_eng = Sequential(adj, eng)
        self.con = con
        self.ord_sdp = ord_sdp  # order of saddle point
        self.max_stp = max_stp

    def forward(self, inp: Dict[str, Tensor]):
        out = inp.copy()
        if p.con_lag not in inp:
            out[p.con_lag] = torch.zeros_like(out[p.con_cen])
        _, n_con = out[p.con_cen].size()

        x, out = vector_pos_lag(out)

        out = self.adj_eng(out)
        con = self.con(out) - out[p.con_cen]

        L = out[p.eng] - (out[p.con_lag] * con).sum(1)
        g = grad(L, x, create_graph=True)
        h = hessian(L, x, g).detach()
        stp = eigenvector_following(h, g.detach(), self.ord_sdp + n_con)
        stp = limit_step_size(stp, self.max_stp)

        out = split_pos_frc_lag(x + stp, g, out)
        return detach(out)
