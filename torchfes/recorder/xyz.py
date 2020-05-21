import io
from typing import Union, Dict, List, IO, Any
from pathlib import Path
from torch import Tensor
import numpy as np
from .. import properties as p


def _make_xyz(sym, elm, pos, ent):
    assert elm.ndim == 1
    assert pos.ndim == 2
    assert ent.ndim == 1
    so = io.StringIO()
    so.write(f'{ent.sum()}\n\n')
    for s, ps in zip(sym[elm[ent]], pos[ent, :]):
        so.write(f'{s} {ps[0]} {ps[1]} {ps[2]}\n')
    return so.getvalue()


def make_xyz(sym: np.ndarray, inp: Dict[str, Tensor]):
    return [_make_xyz(sym, elm, pos, ent) for elm, pos, ent in
            zip(inp[p.elm].cpu(), inp[p.pos].cpu(), inp[p.ent].cpu())]


def write_xyz(dir_path: Union[str, Path],
              sym: np.ndarray, inp: Dict[str, Tensor]):
    dir_path = Path(dir_path)
    if not dir_path.is_dir():
        dir_path.mkdir()
    for i, xyz in enumerate(make_xyz(np.array(sym), inp)):
        path = dir_path / (str(i) + '.xyz')
        if path.is_file():
            mode = 'a'
        else:
            mode = 'w'
        with open(path, mode) as f:
            f.write(xyz)


class XYZRecorder:
    def __init__(self, dir_path: Union[str, Path], mode: str,
                 sym: List[str], n_bch: int):
        self.dir_path = Path(dir_path)
        self.f: List[IO[Any]] = []
        self.n_bch = n_bch
        self.sym = np.array(sym)
        if not self.dir_path.is_dir():
            self.dir_path.mkdir()
        for i in range(self.n_bch):
            path = self.dir_path / f'{i}.xyz'
            self.f.append(open(path, mode=mode))

    def __exit__(self, *_):
        self.close()

    def close(self):
        for f in self.f:
            f.close()
        self.f.clear()

    def append(self, inp: Dict[str, Tensor]):
        for (f, xyz) in zip(self.f, make_xyz(np.array(self.sym), inp)):
            f.write(xyz)
