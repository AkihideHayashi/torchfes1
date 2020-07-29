from typing import Dict

from torch import Tensor, nn

from ...general import PosEngFrc


class SD(nn.Module):
    def __init__(self, evl: nn.Module, lmd: Tensor):
        super().__init__()
        self.lmd = lmd
        self.evl = evl

    def forward(self, pef: PosEngFrc, env: Dict[str, Tensor], flt: Tensor,
                reset=False):
        assert isinstance(env, dict)
        assert isinstance(flt, Tensor)
        assert isinstance(reset, bool)
        return pef.frc * self.lmd
