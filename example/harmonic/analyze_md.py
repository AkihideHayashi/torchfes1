import torchfes as fes


def main():
    with fes.rec.open_torch('trj_md.pt', 'rb', 'idx_md.pkl') as f:
        print(f[0][fes.p.pos])


if __name__ == "__main__":
    main()
