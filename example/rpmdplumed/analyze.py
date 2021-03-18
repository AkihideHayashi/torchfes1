import torch
import numpy as np
import pickle
import matplotlib.pyplot as plt
import pandas as pd


def read_plumed(path):
    with open(path) as f:
        first = next(f).split()[1:]
        assert first[0] == 'FIELDS'
        column = first[1:]
        data = [[] for key in column]
        for line in f:
            if line[0] == '#':
                continue
            if len(line.split()) != len(column):
                break
            for i, w in enumerate(line.split()):
                data[i].append(float(w))
        return pd.DataFrame({key: val for key, val in zip(column, data)})


kbt = 0.1
beta = 10.0
n_ring = 16
hbar = 1.0
k = 1.0
kbt = 0.1
beta = 10.0


def gound(x, hbar, m, k):
    # print(hbar, m, k)
    omega = torch.sqrt(k / m)
    xi = torch.sqrt(m * omega / hbar) * x
    return torch.exp(-0.5 * xi * xi)


def tmp():
    with open('tmp.pkl', 'rb') as f:
        pos, pot, kin = torch.load(f)

    pot = torch.stack(pot).mean()
    kin = torch.stack(kin).mean()
    print(pot + kin)
    pos = torch.stack(pos).flatten()

    hist, bin_ = np.histogram(pos, bins=101, range=(-10, 10))
    x = 0.5 * (bin_[1:] + bin_[:-1])
    y = np.exp(- 0.5 * x * x / kbt)
    y = y / y.sum() * hist.sum()
    z = gound(x, hbar, torch.tensor(1.0), k) ** 2
    z = z / z.sum() * hist.sum()
    print('END')
    plt.plot(x, hist)
    plt.plot(x, y)
    plt.plot(x, z)
    plt.show()


def main():
    with open('tmp.pkl', 'rb') as f:
        pos, pot, kin = torch.load(f)
    x = torch.stack(pos).mean(1)
    plt.plot(x)
    plt.show()


def main():
    data = read_plumed('COLVAR')
    pos = data['a.x']
    # plt.plot(pos)
    # plt.plot(data['metad.rct'])
    # plt.plot(data['metad.bias'])
    bias = data['metad.rbias']
    weight = np.exp(beta * bias)
    hist, bin_ = np.histogram(pos, bins=101, range=(-10, 10), weights=weight)
    x = 0.5 * (bin_[1:] + bin_[:-1])
    y = np.exp(- 0.5 * x * x / kbt)
    y = y / y.sum() * hist.sum()
    z = gound(x, hbar, torch.tensor(1.0), k) ** 2
    z = z / z.sum() * hist.sum()
    plt.plot(x, hist)
    plt.plot(x, y)
    plt.plot(x, z)
    plt.show()


if __name__ == "__main__":
    main()
