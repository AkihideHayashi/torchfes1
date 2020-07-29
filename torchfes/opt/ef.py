from typing import Dict
import torch
from torch import Tensor, nn
from .. import properties as p
from .eigenvectorfollowing import eigenvector_following
from .derivative import hessian, grad
from .utils import vector_pos_lag, split_pos_lag
from ..utils import detach


# cel + pos -> cel_pos_flt
# lmd_pos


class EigenVectorFollowing(nn.Module):
    def __init__(self, adj, eng, con, ord_sdp: int, max_stp: float):
        super().__init__()
        self.adj = adj
        self.eng = eng
        self.con = con
        self.ord_sdp = ord_sdp  # order of saddle point
        self.max_stp = max_stp

    def forward(self, inp: Dict[str, Tensor]):
        out = inp.copy()
        if p.con_lag not in inp:
            out[p.con_lag] = torch.zeros_like(out[p.con_cen])
        n_bch, n_con = out[p.con_cen].size()
        _, _, n_dim = out[p.pos].size()

        q, pos, lag = vector_pos_lag(out[p.pos], out[p.con_lag])
        out[p.pos] = pos
        out[p.con_lag] = lag
        out = self.adj(out)
        out = self.eng(out)
        con = self.con(out) - out[p.con_cen]
        assert con.size() == (n_bch, n_con)
        L = out[p.eng] - (lag * con).sum(1)
        g = grad(L, q, create_graph=True)
        frc, _ = split_pos_lag(g, pos, lag)
        out[p.frc] = frc.detach()
        h = hessian(L, q, g)
        stp = eigenvector_following(h.detach(), g.detach(),
                                    self.ord_sdp + n_con)
        stp_siz = stp.norm(p=2, dim=1)[:, None].expand_as(stp)
        stp = torch.where(
            stp_siz > self.max_stp,
            stp / stp_siz * self.max_stp,
            stp
        )
        q = q + stp
        pos_, lag_ = split_pos_lag(q, pos, lag)
        out[p.con_lag] = lag_
        out[p.pos] = pos_
        return detach(out)
