from typing import Union, Optional
from pathlib import Path
from .mpreader import MultiProcessingTorchReader
from .mpwriter import MultiProcessingTorchWriter
from .rareader import RandomAccessTorchReader
from .index import make_index


def open_torch(path: Union[str, Path], mode: str,
               index: Optional[Union[str, Path]] = None):
    if mode in ('r', 'rb'):
        if index is None:
            return MultiProcessingTorchReader(path)
        else:
            index = Path(index)
            if not index.is_file():
                make_index(path, index)
            return RandomAccessTorchReader(path, index)
    else:
        return MultiProcessingTorchWriter(path, mode, index=index)
