from typing import Dict
import torch
from torch import nn, Tensor
from .. import properties as p
from .eigenvectorfollowing import eigenvector_following


class Newton(nn.Module):
    def __init__(self, hes_inv):
        super().__init__()
        self.hes_inv = hes_inv

    def forward(self, inp: Dict[str, Tensor]):
        out = self.hes_inv(inp)
        m = torch.tensor(-1.0)
        out[p.gen_dir] = (
            out[p.gen_hes_inv] @ out[p.gen_grd][:, :, None]).squeeze(2) * m
        return out


class NewtonSolve(nn.Module):
    def __init__(self, hes):
        super().__init__()
        self.hes = hes

    def forward(self, inp: Dict[str, Tensor]):
        out = self.hes(inp)
        m = torch.tensor(-1.0)
        out[p.gen_dir] = torch.solve(
            out[p.gen_grd][:, :, None], out[p.gen_hes])[0].squeeze(2) * m
        return out


class SolveCalcFC(nn.Module):
    def __init__(self, exact, update):
        super().__init__()
        self.update = update
        self.exact = exact

    def forward(self, inp: Dict[str, Tensor]):
        if p.gen_dir in inp:
            out = self.update(inp)
        else:
            out = self.exact(inp)
        m = torch.tensor(-1.0)
        out[p.gen_dir] = torch.solve(
            out[p.gen_grd][:, :, None], out[p.gen_hes])[0].squeeze(2) * m
        return out


class CalcFC(nn.Module):
    def __init__(self, exact, update_inv):
        super().__init__()
        self.update_inv = update_inv
        self.exact = exact

    def forward(self, inp: Dict[str, Tensor]):
        if p.gen_dir in inp:
            out = self.update_inv(inp)
        else:
            out = self.exact(inp)
        m = torch.tensor(-1.0)
        out[p.gen_dir] = (
            out[p.gen_hes_inv] @ out[p.gen_grd][:, :, None]).squeeze(2) * m
        return out


class EigenVectorFollowing(nn.Module):
    def __init__(self, hes, ord_sdp: int):
        super().__init__()
        self.hes = hes
        self.ord_sdp = ord_sdp  # order of saddle point

    def forward(self, inp: Dict[str, Tensor]):
        out = inp.copy()
        n_con = inp[p.con_cen].size(1)
        out = self.hes(inp)
        out[p.gen_dir] = eigenvector_following(out[p.gen_hes].detach(),
                                               out[p.gen_grd].detach(),
                                               self.ord_sdp + n_con)
        return out
