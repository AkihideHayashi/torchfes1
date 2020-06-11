from typing import List, Union
import numpy as np


def pad(array: np.ndarray, shape: List[int],
        constant_values: Union[bool, int, float, str]):
    assert array.ndim == len(shape)
    pad_width = [(0, max(0, si - asi)) for si, asi in zip(shape, array.shape)]
    return np.pad(array, pad_width, mode='constant',
                  constant_values=constant_values)


def cat(arrays: List[np.ndarray],
        constant_values: Union[bool, int, float, str], axis=0):
    for arr in arrays:
        assert arr.ndim == arrays[0].ndim
    shape = []
    for i in range(arrays[0].ndim):
        shape.append(max(arr.shape[i] for arr in arrays))
    shape[axis] = 0
    padded = [pad(arr, shape, constant_values) for arr in arrays]
    return np.concatenate(padded, axis=axis)


def stack(arrays: List[np.ndarray],
          constant_values: Union[bool, int, float, str], axis=0):
    for arr in arrays:
        assert arr.ndim == arrays[0].ndim
    shape = []
    for i in range(arrays[0].ndim):
        shape.append(max(arr.shape[i] for arr in arrays))
    padded = [pad(arr, shape, constant_values) for arr in arrays]
    return np.stack(padded, axis=axis)
