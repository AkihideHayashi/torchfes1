import matplotlib.pyplot as plt
import torchfes as fes


def main():
    hil = {}
    with fes.rec.open_torch('hil_wtmtd.pt', 'rb', 'hdx_wtmtd.pkl') as f:
        for data in f:
            hil = fes.fes.mtd.add_gaussian(hil, data)
    hgt = hil[fes.p.mtd_hgt]

    plt.plot(hgt)
    plt.show()


if __name__ == "__main__":
    main()
