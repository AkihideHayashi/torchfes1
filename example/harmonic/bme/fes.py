from pathlib import Path
import matplotlib.pyplot as plt
import torch
import torchfes as fes


def main():
    keys = (fes.p.con_cen, fes.p.con_mul, fes.p.bme_ktg, fes.p.bme_fix)
    datas = {key: [] for key in keys}
    trj_path = Path('trj')
    with fes.rec.open_trj(trj_path, 'rb') as f:
        for i, data in enumerate(f):
            if i < 1:
                continue
            for key in datas:
                datas[key].append(data[key])
    datas = {key: torch.stack(val) for key, val in datas.items()}
    jac = fes.fes.bme.bme_postprocess(
        datas[fes.p.con_mul], datas[fes.p.bme_ktg], datas[fes.p.bme_fix])
    jac = jac.squeeze(0)
    x = datas[fes.p.con_cen].mean(0).squeeze(1)
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
