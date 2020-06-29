import os
import pickle
import warnings
import multiprocessing
from typing import Union, Dict, Optional
from pathlib import Path
import torch
from torch import Tensor


class MultiProcessingTorchWriter:
    """Multiprocess torch writer."""
    def __init__(self, path: Union[str, Path], mode: str,
                 index: Optional[Union[str, Path]] = None):
        if isinstance(path, str):
            path = Path(path)
        if isinstance(index, str):
            index = Path(index)
        if mode in ('r', 'w', 'a'):
            mode = mode + 'b'
        self.process, self.queue = _create_writer_loop(path, mode, index)

    def write(self, inp: Dict[str, Tensor]):
        if self.queue.full():
            warnings.warn('Disk IO becomes peformance determinig.')
        self.queue.put(inp)

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


def _loop_torch_writer(path: Path, mode: str, queue: multiprocessing.Queue,
                       index: Optional[Path]):
    if index is None:
        index = Path(os.devnull)
    with open(path, mode) as f, open(index, mode) as fi:
        while True:
            data = queue.get()
            if data is StopIteration:
                break
            pickle.dump(f.tell(), fi)
            torch.save(data, f)
            f.flush()
            fi.flush()


def _create_writer_loop(path: Path, mode: str, index: Optional[Path]):
    queue: multiprocessing.Queue = multiprocessing.Queue()
    process = multiprocessing.Process(
        target=_loop_torch_writer, args=(path, mode, queue, index))
    process.start()
    return process, queue
