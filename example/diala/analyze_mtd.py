from math import pi
import matplotlib.pyplot as plt
import torch
import torchfes as fes
from torchfes import properties as p
from torchfes.fes.mtd_new import gaussian_potential


def main():
    idx = torch.tensor([[5, 7, 9, 15]]).t() - 1
    col = fes.colvar.Dihedral(idx)
    datas = []
    with fes.rec.open_torch_mp('hills.pt', 'r') as f:
        for data in f:
            datas.append(data)
    datas = {key: torch.cat([data[key] for data in datas])
             for key in (p.mtd_cen, p.mtd_hgt, p.mtd_prc)}
    x = torch.linspace(-pi, pi, 50)[:, None, None]
    y = -torch.cat([gaussian_potential(xi, col.pbc, datas) for xi in x])
    plt.plot(x.squeeze(-1).squeeze(-1), y)
    plt.show()


if __name__ == "__main__":
    main()
