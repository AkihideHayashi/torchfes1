from typing import Dict
from torch import nn, Tensor


class Eval(nn.Module):
    def __init__(self, adj, eng):
        super().__init__()
        self.adj = adj
        self.eng = eng

    def forward(self):
        pass
