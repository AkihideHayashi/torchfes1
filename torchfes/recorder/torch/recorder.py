from typing import Union
from pathlib import Path
from .mpreader import MultiProcessingTorchReader
from .mpwriter import MultiProcessingTorchWriter
from .rareader import RandomAccessTorchReader
from .index import make_index
from .pathpair import PathPair


def open_torch(path: Union[str, Path, PathPair], mode: str,
               random_access: bool = True):
    if isinstance(path, str):
        path = Path(path)
    if isinstance(path, Path):
        path = PathPair(path)
    if mode in ('r', 'rb'):
        if random_access:
            if not path.idx.is_file():
                make_index(path)
            return RandomAccessTorchReader(path)
        else:
            return MultiProcessingTorchReader(path.idx)
    else:
        return MultiProcessingTorchWriter(path, mode)
