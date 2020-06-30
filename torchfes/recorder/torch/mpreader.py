from typing import Dict
import multiprocessing
from pathlib import Path
import torch
from torch import Tensor


class MultiProcessingTorchReader:
    """Multiprocess torch reader."""

    def __init__(self, path: Path):
        self.process, self.queue = _create_reader_loop(path)

    def __iter__(self):
        return self

    def __next__(self):
        ret = self.queue.get()
        if ret is StopIteration:
            raise ret()
        else:
            return ret

    def close(self):
        self.process.terminate()
        self.process.join()
        del self.queue

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def __getitem__(self, _):
        raise RuntimeError(f'{self}.__getitem__ is not available.')

    def write(self, _: Dict[str, Tensor]):
        raise RuntimeError(f'{self}.write is not available.')

    def __len__(self):
        raise RuntimeError(f'{self}.__len__ is not available.')


def _loop_torch_reader(path: Path, queue: multiprocessing.Queue):
    i = 0
    with open(path, 'rb') as f:
        while True:
            try:
                data = torch.load(f, map_location=torch.device('cpu'))
                queue.put(data)
            except (EOFError, RuntimeError):
                queue.put(StopIteration)
                break
            i += 1


def _create_reader_loop(path: Path):
    queue: multiprocessing.Queue = multiprocessing.Queue()
    process = multiprocessing.Process(
        target=_loop_torch_reader, args=(path, queue))
    process.start()
    return process, queue
