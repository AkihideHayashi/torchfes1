from typing import Dict
import torch
from torch import nn, Tensor
from .. import properties as p


class NoColVar(nn.Module):
    def forward(self, inp: Dict[str, Tensor]):
        pos = inp[p.pos]
        bch = pos.size(0)
        return torch.zeros([bch, 0], dtype=pos.dtype, device=pos.device)
