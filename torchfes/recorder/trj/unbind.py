from typing import Dict, Union
from pathlib import Path
from torch import Tensor
from ... import properties as p
from .trj import TorchTrajectory
from ...data import unbind


class UnbindTrajectory:
    def __init__(self, path: Union[str, Path], mode: str, digit: int):
        if isinstance(path, str):
            path = Path(path)
        assert mode in ('rb', 'wb', 'ab')
        self.path = path
        self.mode = mode
        self.fmt = '{{:>0{}}}'.format(digit)

    def append(self, data: Dict[str, Tensor]):
        for d in unbind(data):
            idt = d[p.idt].item()
            path = self.path / self.fmt.format(idt)
            with TorchTrajectory(path, 'ab') as f:
                f.append(d)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        pass
