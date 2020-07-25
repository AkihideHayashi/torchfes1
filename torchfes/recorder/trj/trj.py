from typing import Dict, Union
from pathlib import Path
import torch
from torch import Tensor
from .pathpair import PathPair


def _load_all(path):
    with open(path, 'rb') as f:
        while True:
            try:
                yield torch.load(f)
            except EOFError:
                break


class TorchTrajectory:
    def __init__(self, path: Union[str, Path, PathPair], mode: str):
        self.mode = mode
        if mode not in ('rb', 'wb', 'ab'):
            raise KeyError(mode)
        if isinstance(path, (str, Path)):
            path = PathPair(path)
        if not path.is_dir():
            path.mkdir()
        if path.idx.is_file() and mode in ('ab', 'rb'):
            self.idx = list(_load_all(path.idx))
        else:
            self.idx = []
        self.f_trj = open(path.trj, mode)
        self.f_idx = open(path.idx, mode)

    def close(self):
        self.f_trj.close()
        self.f_idx.close()
        self.idx.clear()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def append(self, data: Dict[str, Tensor]):
        if self.mode not in ('ab', 'wb'):
            raise RuntimeError(self.mode)
        self.idx.append(self.f_trj.tell())
        torch.save(self.f_trj.tell(), self.f_idx)
        torch.save(data, self.f_trj)

    def __getitem__(self, i: int):
        if self.mode not in ('rb',):
            raise RuntimeError(self.mode)
        self.f_trj.seek(self.idx[i])
        return torch.load(self.f_trj)

    def __len__(self):
        return len(self.idx)
