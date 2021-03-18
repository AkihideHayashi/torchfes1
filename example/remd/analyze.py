import numpy as np
import torch
import torchfes as fes
import matplotlib.pyplot as plt


n = int(sys.argv[1])
pos_lst = []
with fes.rec.TorchTrajectory('md', 'rb') as rec:
    for mol in rec:
        idx = torch.argsort(mol[fes.p.kbt])
        pos = mol[fes.p.pos][idx]
        pos_lst.append(pos)
    kbt = rec[0][fes.p.kbt]

pos = torch.stack(pos_lst)[:, 0, 0, 0]
hist, bin_edges = np.histogram(pos)
x = 0.5 * (bin_edges[1:] + bin_edges[:-1])
y = torch.exp(-0.5 * x * x / kbt[0])
y = y / y.sum() * hist.sum()
plt.plot(x, hist)
plt.plot(x, y)
plt.show()
