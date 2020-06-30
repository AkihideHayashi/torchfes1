from typing import Union
from pathlib import Path
from .mpreader import MultiProcessingTorchReader
from .mpwriter import MultiProcessingTorchWriter
from .rareader import RandomAccessTorchReader
from .index import make_index
from .pathpair import PathPair


def open_torch(path: Union[str, Path, PathPair], mode: str):
    if isinstance(path, str):
        path = Path(path)
    if mode in ('r', 'rb'):
        if isinstance(path, PathPair):
            if not path.idx.is_file():
                make_index(path)
            return RandomAccessTorchReader(path)
        else:
            return MultiProcessingTorchReader(path)
    else:
        if isinstance(path, Path):
            raise ValueError('path must be PathPair for writing mode.')
        return MultiProcessingTorchWriter(path, mode)
