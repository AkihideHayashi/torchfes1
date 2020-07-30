from typing import Dict
import torch
from torch import nn, Tensor
from .. import properties as p


def limit_step_size(stp: Tensor, siz: float):
    stp_siz = stp.norm(p=2, dim=1)[:, None].expand_as(stp)
    stp = torch.where(
        stp_siz > siz,
        stp / stp_siz * siz,
        stp
    )
    return stp


class LimitStepSize(nn.Module):
    def __init__(self, stp, max_stp: float):
        super().__init__()
        self.stp = stp
        self.max_stp = max_stp

    def forward(self, inp: Dict[str, Tensor]):
        out = self.stp(inp)
        stp = out[p.gen_stp]
        stp = limit_step_size(stp, self.max_stp)
        out[p.gen_stp] = stp
        return out
