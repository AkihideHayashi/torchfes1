"""Wolfe Condition."""
import torch
from torch import nn, Tensor
from ..general import PosEngFrc


def _dot(a, b):
    return (a * b).sum(-1).unsqueeze(-1)


class WolfeCondition(nn.Module):
    """
    Args:
        old: old pos, eng, and frc.
        new: line search new point.
        old.pos + stp * vec == new.pos
    Returns:
        ret > 0 -> stp is too big.
        ret < 0 -> stp is too small.
        ret = 0 -> stp satisfies wolfe's condition.
    """
    def __init__(self, c1: float, c2: float):
        super().__init__()
        self.c1 = c1
        self.c2 = c2
        assert 0 < c1 < c2 < 1

    def forward(self, old: PosEngFrc, new: PosEngFrc,
                stp: Tensor, vec: Tensor):
        assert torch.allclose(old.pos + stp * vec, new.pos)
        big = new.eng > old.eng - self.c1 * stp * _dot(old.frc, vec)
        sml = _dot(new.frc, vec) > self.c2 * _dot(old.frc, vec)
        assert not bool((big & sml).any())
        return (big.to(torch.long) - sml.to(torch.long)).to(new.eng)
