import pickle
import torch


def make_index(path, index):
    with open(path, 'rb') as f, open(index, 'wb') as fi:
        while True:
            try:
                n = f.tell()
                torch.load(f)
                pickle.dump(n, fi)
            except (EOFError, RuntimeError):
                break
