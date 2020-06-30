import matplotlib.pyplot as plt
import torchfes as fes


def main():
    hil_path = fes.rec.PathPair('hil')
    hil = fes.rec.read_mtd(hil_path)
    hgt = hil[fes.p.mtd_hgt]

    plt.plot(hgt)
    plt.show()


if __name__ == "__main__":
    main()
