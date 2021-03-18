import math
from typing import Dict
import torch
from torch import nn, Tensor
from .. import properties as p


class Ratio(nn.Module):
    def __init__(self, idx, cel=False):
        super().__init__()
        self.idx = idx
        self.cel = cel
        self.pbc = torch.tensor([math.inf for _ in range(idx.size(1))])
        assert not self.cel

    def forward(self, inp: Dict[str, Tensor]):
        r1, r2, r3 = inp[p.pos][:, self.idx, :].unbind(1)
        n1 = (r1 - r2).norm(dim=2)
        n2 = (r2 - r3).norm(dim=2)
        # print(r1.tolist(), r2.tolist(), r3.tolist())
        # print('ratio', n1 / n2)
        return n1 / n2
