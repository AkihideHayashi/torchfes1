from typing import Union, Type, Optional
from pathlib import Path
from torch.multiprocessing import Queue
from ...multiprocessing import QueueProcess
from .trj import TorchTrajectory
from .unbind import UnbindTrajectory
from ...utils import detach


def _write(put: Queue, _: Queue, path: Union[str, Path], unbind, digit, mode):
    recorder: Union[UnbindTrajectory, TorchTrajectory]
    if unbind:
        recorder = UnbindTrajectory(path, mode, digit)
    else:
        recorder = TorchTrajectory(path, mode)
    with recorder as f:
        while True:
            data = put.get()
            if data is StopIteration:
                break
            f.append(data)


def _read_all(put: Queue, get: Queue, path: Union[str, Path]):
    with TorchTrajectory(path, 'rb') as f:
        for data in f:
            get.put(detach(data))
            if not put.empty():
                x = put.get()
                if x is StopIteration:
                    break
                else:
                    raise RuntimeError(str(x))
    get.put(StopIteration)


def open_trj(path: Union[str, Path], mode: str = 'rb', unbind: bool = False,
             digit: Optional[int] = None):
    recorder: Union[Type[TorchTrajectory], Type[UnbindTrajectory]]
    if mode in ('wb', 'ab'):
        return QueueProcess(_write, (path, unbind, digit, mode))
    elif mode == 'rb':
        return QueueProcess(_read_all, (path,))
    else:
        raise KeyError(mode)
