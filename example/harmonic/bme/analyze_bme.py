import matplotlib.pyplot as plt
import torch
import torchfes as fes


def main():
    datas = {key: [] for key in (fes.p.bme_cen, fes.p.bme_lmd, fes.p.bme_ktg,
                                 fes.p.bme_fix)}
    with fes.rec.open_torch('trj_bme.pt', 'rb', 'idx_bme.pkl') as f:
        for data in f:
            for key in datas:
                datas[key].append(data[key])
    datas = {key: torch.stack(val) for key, val in datas.items()}
    jac = fes.fes.bme.bme_postprocess(
        datas[fes.p.bme_lmd], datas[fes.p.bme_ktg], datas[fes.p.bme_fix])
    jac = jac.squeeze(0)
    x = datas[fes.p.bme_cen].mean(0).squeeze(1)
    plt.plot(x, jac)
    plt.show()
    # min_ = pos.min().item()
    # max_ = pos.max().item()
    # bins = 40

    # x = fes.analyze.histc_axis(bins, min_, max_)
    # count = pos.histc(bins, min_, max_)
    # y = - kbt * count.log()

    # plt.plot(x, y - y.min())
    # plt.plot(x, x * x)
    # plt.show()


if __name__ == "__main__":
    main()
