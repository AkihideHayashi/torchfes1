import matplotlib.pyplot as plt
import torch
from torchfes.fes.bme import bme_postprocess
from torchfes.recorder import hdf5_recorder
from torchfes import properties as p


def main():
    with hdf5_recorder('bme.hdf5', 'r') as f:
        lmd = f[p.bme_lmd]
        cor = f[p.bme_cor]
        kbt = f[p.kbt]
        fix = f[p.bme_fix]
    x = torch.linspace(-0.9, 0.9, lmd.size(1))
    y = bme_postprocess(lmd, kbt, cor, fix)
    plt.plot(x[1:-1], y[1:-1])
    plt.show()


if __name__ == "__main__":
    main()
