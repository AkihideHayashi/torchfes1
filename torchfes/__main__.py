import sys
import shutil
import argparse
from pathlib import Path
import tempfile
import torchfes as fes
from torchfes.recorder import PathPair, open_torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('exe', type=str,
                        choices=['repair', 'sanity-check'])
    parser.add_argument('-f', type=Path, help='file')
    parser.add_argument('-b', type=Path, help='backup')
    args = parser.parse_args()
    if args.exe == 'repair':
        if args.b is None:
            assert args.f is not None
            full_repair(PathPair(args.f))
        else:
            part_repair(PathPair(args.f), PathPair(args.b))
    elif args.exe == 'sanity-check':
        sanity_check(PathPair(args.f))
    else:
        raise NotImplementedError()


def sanity_check(trj: PathPair):
    with fes.rec.open_torch(trj, 'rb') as f:
        try:
            _ = f[-1]
            sys.exit(0)
        except (RuntimeError, EOFError):
            sys.exit(1)


def part_repair(trj: PathPair, bak: PathPair):
    with tempfile.TemporaryDirectory() as tmp_path:
        tmp = PathPair(tmp_path)
        shutil.copy(bak.trj, tmp.trj)
        shutil.copy(bak.idx, tmp.idx)
        with open_torch(bak, 'rb') as fb:
            n_bak = len(fb)
        with open_torch(trj, 'rb') as fi, open_torch(tmp, 'ab') as fo:
            try:
                for i in range(n_bak, len(fi)):
                    data = fi[i]
                    fo.write(data)
            except (RuntimeError, EOFError):
                pass
        shutil.move(tmp.trj, trj.trj)
        shutil.move(tmp.idx, trj.idx)


def full_repair(trj: PathPair):
    with tempfile.TemporaryDirectory() as tmp_path:
        tmp = PathPair(tmp_path)
        with open_torch(trj.trj, 'rb') as fi, open_torch(tmp, 'wb') as fo:
            for data in fi:
                fo.write(data)
        shutil.move(tmp.idx, trj.idx)
        shutil.move(tmp.trj, trj.trj)


if __name__ == "__main__":
    main()
