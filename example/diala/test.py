from typing import Dict
from math import pi, inf
from ase.build import molecule
import torch
from torch import nn, Tensor, jit
from torch.utils.data import DataLoader
from ignite.handlers import Timer
import torchfes as fes
from torchfes.colvar.dihedral import dihedral
from torchfes.fes.mtd_new import MetaDynamics, GaussianPotential, get_log_max
from torchfes import properties as p


class ColVar(nn.Module):
    def __init__(self, idx):
        super().__init__()
        self.idx = idx
        self.pbc = torch.tensor([2 * pi, 2 * pi])

    def forward(self, inp: Dict[str, Tensor]):
        return dihedral(self.idx, inp[p.pos])


def main():
    dl = DataLoader(
        dataset=[molecule('C2H6')], batch_size=1,
        collate_fn=fes.data.ToDictTensor(['H', 'C'])
    )
    inp = next(iter(dl))
    pos = inp[fes.p.pos]
    idx = torch.tensor([[2, 0, 1, 5], [2, 0, 1, 7]]).t()
    print(dihedral(idx, pos) / pi * 180)
    gp = GaussianPotential(ColVar(idx))
    inp[p.mtd_cen] = torch.tensor(
        [[0.0, 0.5], [1.0, 2.0], [1.0, 0.0], [0.0, 0.0]])
    inp[p.mtd_hgt] = torch.tensor([1.0, 2.0, 3.0, 2.0])
    inp[p.mtd_prc] = torch.tensor(
        [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0], [1.0, 1.0]])
    timer = Timer()
    gp(inp)
    print(timer.value())
    timer.reset()
    gp(inp)
    print(timer.value())
    timer.reset()


if __name__ == "__main__":
    main()
