import math
from typing import Dict
import matplotlib.pyplot as plt
import torch
from torch import nn, Tensor
import torchfes as fes


class ColVar(nn.Module):
    def __init__(self):
        super().__init__()
        self.pbc = torch.tensor([math.inf])

    def forward(self, inp: Dict[str, Tensor]):
        ret = inp[fes.p.pos][:, 0, 0][:, None]
        assert ret.size() == (1, 1)
        return ret


def main():
    hil = {}
    with fes.rec.open_torch('hil_mtd.pt', 'rb', 'hdx_mtd.pkl') as f:
        for data in f:
            hil = fes.fes.mtd.add_gaussian(hil, data)
    min_ = -0.5
    max_ = 0.5
    bins = 40

    x = fes.analyze.histc_axis(bins, min_, max_)
    y = -fes.fes.mtd.gaussian_potential(x[:, None], ColVar().pbc, hil)

    plt.plot(x, y - y.min())
    plt.plot(x, x * x)
    plt.show()


if __name__ == "__main__":
    main()
