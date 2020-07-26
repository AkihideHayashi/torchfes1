import sys
import math
from pathlib import Path
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
    n = int(sys.argv[1])
    hil = {}
    hil_path = Path('hil')
    with fes.rec.open_trj(hil_path, 'rb') as f:
        for data in f:
            hil = fes.fes.mtd.cat_gaussian(hil, data)
    min_ = -0.5
    max_ = 0.5
    bins = 40

    x = fes.analyze.histc_axis(bins, min_, max_)
    hil = {key: val[n:n+1] for key, val in hil.items()}
    y = -fes.fes.mtd.gaussian(hil, ColVar().pbc, x[:, None])

    plt.plot(x, y - y.min())
    plt.plot(x, x * x)
    plt.show()


if __name__ == "__main__":
    main()
