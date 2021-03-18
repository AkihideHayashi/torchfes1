import math
from typing import Dict
import torch
from torch import nn, Tensor
from .. import properties as p


class Length(nn.Module):
    def __init__(self, idx, cel=False):
        super().__init__()
        self.idx = idx
        self.cel = cel
        self.pbc = torch.tensor([math.inf for _ in range(idx.size(1))])
        assert not self.cel

    def forward(self, inp: Dict[str, Tensor]):
        r1, r2 = inp[p.pos][:, self.idx, :].unbind(1)
        r = (r1 - r2).norm(dim=2)
        return r
