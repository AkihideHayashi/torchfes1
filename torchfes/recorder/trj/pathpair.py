from typing import Union
from pathlib import Path


class PathPair:
    def __init__(self, parent: Union[str, Path]):
        if isinstance(parent, str):
            parent = Path(parent)
        self.trj = parent / 'trj.pt'
        self.idx = parent / 'idx.pt'
        self.parent = parent

    def _check(self):
        assert self.idx.parent == self.parent
        assert self.trj.parent == self.parent

    def mkdir(self):
        self._check()
        _mkdir(self.parent)

    def is_dir(self):
        return self.parent.is_dir()

    def is_file(self):
        if self.trj.is_file():
            if self.idx.is_file():
                return True
            else:
                raise RuntimeError()
        else:
            if self.idx.is_file():
                raise RuntimeError()
            else:
                return False


def _mkdir(path: Path):
    if not path.parent.is_dir():
        _mkdir(path.parent)
    if not path.is_dir():
        path.mkdir()
