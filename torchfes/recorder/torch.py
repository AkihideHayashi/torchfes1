import warnings
import multiprocessing
from typing import Union, Dict, List
from pathlib import Path
import torch
from torch import Tensor


def read_trj(path: Union[str, Path], properties: List[str]):
    data_lst: Dict[str, List[Tensor]] = {prp: [] for prp in properties}
    with open(path, 'rb') as f:
        while True:
            try:
                data = torch.load(f)
                for prp in properties:
                    data_lst[prp].append(data[prp])
            except (EOFError, RuntimeError):
                break
    return [torch.stack(data_lst[key]) for key in properties]


def _loop_torch(path: Path, mode: str, queue: multiprocessing.Queue):
    with open(path, mode) as f:
        while True:
            data = queue.get()
            if len(data) == 0:
                break
            torch.save(data, f)
            f.flush()


def _create_loop(path: Path, mode: str):
    queue: multiprocessing.Queue = multiprocessing.Queue()
    process = multiprocessing.Process(
        target=_loop_torch, args=(path, mode, queue))
    process.start()
    return process, queue


class TorchRecorder:
    def __init__(self, path: Union[str, Path], mode: str):
        if isinstance(path, str):
            path = Path(path)
        if mode in ('r', 'w', 'a'):
            mode = mode + 'b'
        self.process, self.queue = _create_loop(path, mode)

    def append(self, inp: Dict[str, Tensor]):
        if self.queue.full():
            warnings.warn('Disk IO becomes peformance determinig.')
        self.queue.put(inp)

    def extend(self, inps: List[Dict[str, Tensor]]):
        for inp in inps:
            self.append(inp)

    def close(self):
        self.queue.put({})
        self.process.join()
        del self.queue

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
