import pickle
import torch
from .pathpair import PathPair


def make_index(path: PathPair):
    path.mkdir()
    with open(path.trj, 'rb') as ft, open(path.idx, 'wb') as fi:
        while True:
            try:
                n = ft.tell()
                torch.load(ft)
                pickle.dump(n, fi)
            except (EOFError, RuntimeError):
                break
