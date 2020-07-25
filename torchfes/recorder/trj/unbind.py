from typing import Dict, Union
from pathlib import Path
from torch import Tensor
from ... import properties as p
from .trj import TorchTrajectory
from ...data import unbind


class UnbindTrajectory:
    def __init__(self, path: Union[str, Path], mode: str):
        if isinstance(path, str):
            path = Path(path)
        assert mode in ('rb', 'wb', 'ab')
        self.path = path
        self.mode = mode

    def append(self, data: Dict[str, Tensor]):
        for d in unbind(data):
            idt = d[p.idt].item()
            path = self.path / f'{idt:03}'
            with TorchTrajectory(path, 'ab') as f:
                f.append(d)
