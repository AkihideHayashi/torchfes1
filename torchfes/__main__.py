import sys
from typing import Optional
import shutil
import argparse
from pathlib import Path
import tempfile
import torchfes as fes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('exe', type=str,
                        choices=['repair', 'sanity-check'])
    parser.add_argument('--trj', type=Path)
    parser.add_argument('--idx', type=Path)
    parser.add_argument('--trj-bk', type=Path)
    parser.add_argument('--idx-bk', type=Path)
    args = parser.parse_args()
    if args.exe == 'repair':
        if args.trj_bk is None and args.idx_bk is None:
            assert False
            full_repair(args.trj, args.idx)
        else:
            assert args.trj_bk is not None
            assert args.idx_bk is not None
            part_repair(args.trj, args.idx, args.trj_bk, args.idx_bk)
    elif args.exe == 'sanity-check':
        sanity_check(args.trj, args.idx)
    else:
        raise NotImplementedError()


def sanity_check(trj: Path, idx: Path):
    with fes.rec.open_torch(trj, 'rb', idx) as f:
        try:
            _ = f[-1]
            sys.exit(0)
        except (RuntimeError, EOFError):
            sys.exit(1)


def part_repair(trj: Path, idx: Path, trj_bak: Path, idx_bak: Path):
    with tempfile.TemporaryDirectory() as tmp:
        tmp_trj = Path(tmp) / trj.name
        tmp_idx = Path(tmp) / idx.name
        shutil.copy(trj_bak, tmp_trj)
        shutil.copy(idx_bak, tmp_idx)
        with fes.rec.open_torch(trj, 'rb', idx) as fi, \
                fes.rec.open_torch(tmp_trj, 'ab', tmp_idx) as fo, \
                fes.rec.open_torch(trj_bak, 'rb', idx_bak) as fb:
            try:
                for i in range(len(fb), len(fi)):
                    data = fi[i]
                    fo.write(data)
            except RuntimeError:
                pass
        shutil.move(tmp_idx, idx)
        shutil.move(tmp_trj, trj)


def full_repair(trj: Path, idx: Optional[Path]):
    with tempfile.TemporaryDirectory() as tmp:
        tmp_trj = Path(tmp) / trj.name
        with fes.rec.open_torch(trj, 'rb') as fi, \
                fes.rec.open_torch(tmp_trj, 'wb') as fo:
            for data in fi:
                fo.write(data)
        if idx is not None:
            tmp_idx = Path(tmp) / idx.name
            fes.rec.torch.index.make_index(tmp_trj, tmp_idx)
            shutil.move(tmp_idx, idx)
        shutil.move(tmp_trj, trj)


if __name__ == "__main__":
    main()
