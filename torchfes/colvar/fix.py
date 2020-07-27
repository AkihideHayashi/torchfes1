import math
from typing import Dict
import torch
from torch import nn, Tensor
from .. import properties as p


class Fix(nn.Module):
    def __init__(self, mask):
        super().__init__()
        if mask.dim() != 3:
            raise KeyError(f'Fix mask must have dim=3 but have {mask.dim()}')
        self.mask = mask
        self.pbc = torch.tensor([math.inf for _ in range(mask.sum())])

    def forward(self, inp: Dict[str, Tensor]):
        return inp[p.pos][:, self.mask]
