from pathlib import Path
from typing import Dict, List, Union
import multiprocessing

import h5py
import torch
from torch import Tensor

_trj = 'trajectory'
_thr = 'throughout'


def hdf5_recorder(path: Union[str, Path], mode: str):
    return HDF5Recorder(path, mode)


def _group_to_tensor(grp: h5py.Group):
    return {k: torch.tensor(v[()]) for k, v in grp.items()}


class HDF5Recorder:
    def __init__(self, path: Union[str, Path], mode: str):
        self.path = Path(path)
        if mode == 'r+' and not self.path.is_file():
            mode = 'w'
        self.mode = mode
        self.f = h5py.File(self.path, self.mode)
        for grp in (_trj, _thr):
            if grp not in self.f:
                self.f.create_group(grp)
        self.trj = self.f[_trj]
        self.thr = self.f[_thr]

    def close(self):
        self.f.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def __len__(self):
        return len(self.f[_trj])

    def _append_trj(self, inp: Dict[str, Tensor]):
        g = self.trj.create_group(str(len(self.f[_trj])))
        for key, val in inp.items():
            g[key] = val

    def _get_trj_snapshot(self, i: int):
        n = len(self.trj)
        if i >= n:
            raise IndexError()
        grp = self.trj[str(i % n)]
        return _group_to_tensor(grp)

    def _get_trj_property(self, prp: str):
        n = len(self.trj)
        return torch.stack([
            torch.tensor(self.trj[str(i)][prp][()]) for i in range(n)
        ])

    def _get_trj_snapshot_property(self, i: int, prp: str):
        n = len(self.trj)
        if i >= n:
            raise IndexError()
        return torch.tensor(self.trj[str(i % n)][prp][()])

    def _get_thr(self, key: str):
        return torch.tensor(self.thr[key][()])

    def _set_thr(self, key: str, val: Tensor):
        self.thr[key] = val

    def _is_trj_prp(self, key: str):
        if len(self.trj) == 0:
            return False
        return key in self.trj['0']

    def _is_thr_prp(self, key: str):
        return key in self.thr

    def __getitem__(self, key):
        if isinstance(key, str):
            is_trj = self._is_trj_prp(key)
            is_thr = self._is_thr_prp(key)
            if is_trj and is_thr:
                raise RuntimeError(f'{key} exists in both trj and thr.')
            elif is_trj:
                return self._get_trj_property(key)
            elif is_thr:
                return self._get_thr(key)
            else:
                raise KeyError(f'{key} not in hdf5.')
        elif isinstance(key, int):
            return self._get_trj_snapshot(key)
        elif isinstance(key, tuple):
            i, prp = key
            assert isinstance(i, int)
            assert isinstance(prp, str)
            return self._get_trj_snapshot_property(i, prp)
        else:
            raise NotImplementedError(
                f'{type(key)} is not available as hdf key.')

    def __setitem__(self, key, val):
        if key in self.thr:
            del self.thr[key]
        self.thr[key] = val

    def append(self, inp: Dict[str, Tensor]):
        self._append_trj({key: val.cpu() for key, val in inp.items()})

    def extend(self, inps: List[Dict[str, Tensor]]):
        for inp in inps:
            self.append(inp)


def loop_hdf5(path: Path, queue: multiprocessing.Queue):
    while True:
        data = queue.get()
        if len(data) == 0:
            break
        with HDF5Recorder(path, 'a') as f:
            f.append(data)


def create_loop(path):
    queue = multiprocessing.Queue()
    process = multiprocessing.Process(target=loop_hdf5, args=(path, queue))
    process.start()
    return process, queue


def hdf5_recorder_mp(path: Union[str, Path], init=False):
    return HDF5RecorderMP(path, init)


class HDF5RecorderMP:
    def __init__(self, path: Union[str, Path], init=False):
        if isinstance(path, str):
            path = Path(path)
        if init and path.is_file():
            path.unlink()
        self.path = path
        self.process, self.queue = create_loop(path)

    def append(self, inp: Dict[str, Tensor]):
        self.queue.put(inp)

    def extend(self, inps: List[Dict[str, Tensor]]):
        for inp in inps:
            self.append(inp)

    def close(self):
        self.queue.put({})
        self.process.join()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
