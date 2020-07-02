import pickle
import warnings
import multiprocessing
from multiprocessing import Queue
from typing import Dict
import torch
from torch import Tensor
from .pathpair import PathPair


class MultiProcessingTorchWriter:
    """Multiprocess torch writer."""

    def __init__(self, path: PathPair, mode: str):
        if mode in ('r', 'w', 'a'):
            mode = mode + 'b'
        self.process, self.queue = _create_writer_loop(path, mode)

    def write(self, inp: Dict[str, Tensor]):
        if self.queue.full():
            warnings.warn('Disk IO becomes peformance determinig.')
        self.queue.put({key: val.to('cpu') for key, val in inp.items()})

    def close(self):
        self.queue.put(StopIteration)
        self.process.join()
        del self.queue

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def __getitem__(self, _):
        raise RuntimeError(f'{self}.__getitem__ is not available.')

    def __len__(self):
        raise RuntimeError(f'{self}.__len__ is not available.')


def _loop_torch_writer(path: PathPair, mode: str, queue: Queue):
    path.mkdir()
    with open(path.trj, mode) as ft, open(path.idx, mode) as fi:
        while True:
            data = queue.get()
            if data is StopIteration:
                break
            pickle.dump(ft.tell(), fi)
            torch.save(data, ft)
            ft.flush()
            fi.flush()


def _create_writer_loop(path: PathPair, mode: str):
    queue: multiprocessing.Queue = multiprocessing.Queue()
    process = multiprocessing.Process(
        target=_loop_torch_writer, args=(path, mode, queue))
    process.start()
    return process, queue
