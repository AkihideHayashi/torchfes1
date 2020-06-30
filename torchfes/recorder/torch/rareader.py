import pickle
import torch
from .pathpair import PathPair


def _read_all(path):
    with open(path, 'rb') as f:
        while True:
            try:
                yield pickle.load(f)
            except EOFError:
                break


class RandomAccessTorchReader:
    def __init__(self, path: PathPair):
        self.idx = list(_read_all(path.idx))
        self.f = open(path.trj, 'rb')

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        if isinstance(i, int):
            return self._getitem_int(i)
        else:
            raise NotImplementedError()

    def _getitem_int(self, i):
        self.f.seek(self.idx[i])
        return torch.load(self.f)

    def close(self):
        self.f.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def write(self, _):
        raise RuntimeError(f'{self}.write is not available.')
