import matplotlib.pyplot as plt
import torch
import torchfes as fes


def main():
    pos = []
    kbt = []
    path = fes.rec.PathPair('md')
    with fes.rec.open_torch(path, 'rb') as f:
        for data in f:
            pos.append(data[fes.p.pos].item())
            kbt.append(data[fes.p.kbt].item())
    pos = torch.tensor(pos)
    kbt = torch.tensor(kbt).mean().item()
    min_ = pos.min().item()
    max_ = pos.max().item()
    bins = 40

    x = fes.analyze.histc_axis(bins, min_, max_)
    count = pos.histc(bins, min_, max_)
    y = - kbt * count.log()

    plt.plot(x, y - y.min())
    plt.plot(x, x * x)
    plt.show()


if __name__ == "__main__":
    main()
