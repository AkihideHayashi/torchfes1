import math
from typing import Dict, List, Union
import torch
from torch import nn, Tensor
from .. import properties as p


class Fix(nn.Module):
    idx: Tensor

    def __init__(self, idx: Union[List[int], Tensor], n_dim: int = 3):
        super().__init__()
        self.n_dim = 3
        if isinstance(idx, list):
            idx = torch.tensor(idx)
        self.register_buffer('idx', idx)
        self.pbc = torch.tensor([math.inf for _ in range(len(idx) * n_dim)])

    def forward(self, inp: Dict[str, Tensor]):
        ret = inp[p.pos][:, self.idx, :].flatten(1)
        assert ret.size(1) == self.pbc.size(0)
        return ret
