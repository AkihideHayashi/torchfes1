from typing import List, Dict
import array
from torch import nn, Tensor
from pointneighbor import AdjSftSpc
from torchfes import properties as p

try:
    import plumed
except ModuleNotFoundError:
    pass


class Plumed(nn.Module):
    def __init__(self, commands: List[str]):
        super().__init__()
        if 'plumed' not in locals():
            raise RuntimeError('plumed not imported.')
        self.plumed = plumed.Plumed()
        self.box = array.array('d', [0] * 9)
        self.step = 0

    def forward(self, inp: Dict[str, Tensor], _: AdjSftSpc):
        pos = inp[p.pos]
        mas = inp[p.mas]
