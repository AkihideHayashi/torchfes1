from typing import Dict
from torch import Tensor, nn
from .. import properties as p


def dihedral(pos, num):
    r = pos[:, num, :]
    rr10 = r[:, :, 1, :] - r[:, :, 0, :]
    rr23 = r[:, :, 2, :] - r[:, :, 3, :]
    r10 = rr10.norm(2, -1)
    r23 = rr23.norm(2, -1)
    return ((rr10 * rr23).sum(-1) / (r10 * r23)).acos()


class Dihedral(nn.Module):
    def __init__(self, num):
        super().__init__()
        self.num = num

    def forward(self, inp: Dict[str, Tensor]):
        return dihedral(inp[p.pos], self.num)
