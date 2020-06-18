from typing import Dict

from torch import Tensor, nn

from pointneighbor import AdjSftSpc


class ColVarSft(nn.Module):
    def __init__(self, col, val):
        super().__init__()
        self.col = col
        self.val = val

    def forward(self, inp: Dict[str, Tensor], adj: AdjSftSpc):
        col = self.col(inp, adj)
        return col - self.val
