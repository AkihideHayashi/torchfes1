from typing import Dict
from torch import Tensor, nn


class ColVarSft(nn.Module):
    def __init__(self, col, val):
        super().__init__()
        self.col = col
        self.val = val

    def forward(self, inp: Dict[str, Tensor]):
        col = self.col(inp)
        return col - self.val
