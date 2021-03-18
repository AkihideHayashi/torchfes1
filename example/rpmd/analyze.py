import torch
import numpy as np
import pickle
import matplotlib.pyplot as plt

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


with open('tmp.pkl', 'rb') as f:
    pos, pot, kin = pickle.load(f)

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
