import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ase.units import kB


def read_hills(path, grid):
    data = pd.read_csv(path, delim_whitespace=True)
    x = data['p.x']
    h = data['height'][None, :]
    s = data['sigma_p.x'][None, :]
    ret = -np.sum(h * np.exp(-(grid[:, None] - x[None, :])**2 / (s * s)), axis=1)
    return ret - ret.min()


kbt = 300 * kB
data = pd.read_csv("colvar.ssv", delim_whitespace=True)
x = data["p.x"]
time = data["time"]
rbias = data["metad.rbias"]
bias = data["metad.bias"]
rct = data["metad.rct"]
max_rbias = rbias.max()
# rbias = rbias - max_rbias
w = np.exp(rbias / kbt)
grid = np.linspace(-0.5, 0.5, 10)
sigma = (grid[1] - grid[0]) / 4
n = (np.exp(-0.5 * (grid[:, None] - x[None, :])**2 / sigma / sigma) * w[None, :] / sigma / np.sqrt(2 * np.pi)).sum(axis=1)
# n = np.zeros_like(grid)
# for xi, wi in zip(x, w):
#     i = np.argmin(np.abs(xi - grid))
#     n[i] += wi
# print(n)
f = -kbt * np.log(n)
plt.plot(grid, f - f.min())
plt.plot(grid, grid * grid)
plt.plot(grid, read_hills('hills.ssv', grid))
plt.show()
print(kbt)

# plt.plot(time, x)
# plt.show()

# plt.plot(time, rbias)
# plt.plot(time, bias)
# plt.plot(time, rct)
# plt.plot(time, np.exp(rbias))
# plt.show()
