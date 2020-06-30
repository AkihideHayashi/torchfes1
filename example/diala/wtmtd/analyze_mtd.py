import argparse
from math import pi
import matplotlib.pyplot as plt
import torch
import torchfes as fes
from torchfes.fes.mtd import gaussian_potential, add_gaussian
from torchfes.recorder import PathPair, open_torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('exec', choices=['col', 'hgt', 'fes'])
    args = parser.parse_args()
    if args.exec == 'col':
        plot_col()
    elif args.exec == 'hgt':
        plot_hgt()
    else:
        assert args.exec == 'fes'
        plot_fes()


def plot_col():
    hil = {}
    with open_torch(PathPair('hil'), 'r') as f:
        for new in f:
            hil = add_gaussian(hil, new)
    for i in range(hil[fes.p.mtd_cen].size(1)):
        cen = hil[fes.p.mtd_cen][:, i]
        plt.plot(cen, '.', label=str(i))
    plt.legend()
    plt.show()


def plot_hgt():
    hil = {}
    with open_torch(PathPair('hil'), 'r') as f:
        for new in f:
            hil = add_gaussian(hil, new)
    hgt = hil[fes.p.mtd_hgt]
    plt.plot(hgt)
    plt.show()


def plot_fes():
    hil = {}
    with open_torch(PathPair('hil'), 'r') as f:
        for new in f:
            hil = add_gaussian(hil, new)
    idx = torch.tensor([[5, 7, 9, 15]]).t() - 1
    col = fes.colvar.Dihedral(idx)
    x = torch.linspace(-pi, pi, 50)[:, None]
    y = -gaussian_potential(x, col.pbc, hil)
    plt.plot(x.squeeze(-1).squeeze(-1), y)
    plt.show()


if __name__ == "__main__":
    main()
