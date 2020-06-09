from typing import Dict, NamedTuple, Optional
import torch
from torch import nn, Tensor
from .. import properties as p


class PosEngFrc(NamedTuple):
    pos: Tensor
    eng: Tensor
    frc: Tensor


class PosEngFrcStorage(nn.Module):
    def __init__(self):
        super().__init__()
        self.pos = torch.tensor([])
        self.eng = torch.tensor([])
        self.frc = torch.tensor([])

    def forward(self, inp: Optional[PosEngFrc] = None):
        if inp is not None:
            self.pos = inp.pos
            self.eng = inp.eng
            self.frc = inp.frc
        return PosEngFrc(pos=self.pos, eng=self.eng, frc=self.frc)


def pos_cel_vec(inp: Dict[str, Tensor], pos_key: str, cel_key: str,
                use_cel: bool):
    if use_cel:
        val = torch.cat([inp[cel_key].detach(), inp[cel_key].detach()], dim=1)
    else:
        val = inp[pos_key].detach()
    n_bch, n_atm_cel, n_dim = val.size()
    return val.view([n_bch, n_atm_cel * n_dim])


class Encoder(nn.Module):
    def __init__(self, use_cel: bool = False):
        super().__init__()
        self.use_cel = use_cel

    def forward(self, inp: Dict[str, Tensor]):
        pos = pos_cel_vec(inp, p.pos, p.cel, self.use_cel)
        frc = pos_cel_vec(inp, p.frc, p.frc_cel, self.use_cel)
        eng = inp[p.eng_tot]
        return PosEngFrc(pos=pos, frc=frc, eng=eng)


class Decoder(nn.Module):
    def __init__(self, use_cel: bool = False):
        super().__init__()
        self.use_cel = use_cel

    def forward(self, inp: Dict[str, Tensor], pos_cel: Tensor):
        assert pos_cel.dim() == 2
        n_bch, n_atm, n_dim = inp[p.pos].size()
        out = inp.copy()
        if self.use_cel:
            pos_cel = pos_cel.view([n_bch, n_atm + n_dim, n_dim])
            cel = pos_cel[:, :n_dim, :]
            pos = pos_cel[:, n_dim:, :]
            out[p.pos] = pos
            out[p.cel] = cel
        else:
            out[p.pos] = pos_cel.view([n_bch, n_atm, n_dim])
        return out
