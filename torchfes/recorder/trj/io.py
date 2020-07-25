from typing import Union
from pathlib import Path
from torch.multiprocessing import Queue
from ...multiprocessing import QueueProcess
from .trj import TorchTrajectory


def _write(put: Queue, _: Queue, path: Union[str, Path]):
    with TorchTrajectory(path, 'wb') as f:
        while True:
            data = put.get()
            if not data:
                break
            f.append(data)


def _read_all(_: Queue, get: Queue, path: Union[str, Path]):
    with TorchTrajectory(path, 'rb') as f:
        for data in f:
            get.put(data)
    get.put(StopIteration)


def open_trj(path: Union[str, Path], mode: str = 'rb', mp: bool = False):
    if mode in ('wb', 'ab'):
        return QueueProcess(_write, (path,))
    if mode == 'rb':
        if not mp:
            return TorchTrajectory(path, mode)
        else:
            return QueueProcess(_read_all, (path,))
