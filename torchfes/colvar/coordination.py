from math import inf
from typing import Dict, Tuple, List
import torch
from torch import nn, Tensor
import pointneighbor as pn
from ..adj import get_adj_sft_spc, vec_sod
from .. import properties as p


def ravel1(idx: List[Tensor], siz: List[int]):
    return pn.fn.ravel1(
        torch.stack(idx), torch.tensor(siz, device=idx[0].device), dim=0)


def smap(a, c, d, rd):
    return (1.0 + c * rd ** a) ** d


def smap_c(a, b):
    return 2 ** (a / b) - 1


def smap_d(a, b):
    return - b / a


class Smap(nn.Module):
    a: Tensor
    b: Tensor
    c: Tensor
    d: Tensor
    r0: Tensor
    d0: Tensor

    def __init__(self, d0, r0, a, b):
        super().__init__()
        self.register_buffer('d0', d0)
        self.register_buffer('r0', r0)
        self.register_buffer('a', a)
        self.register_buffer('b', b)
        self.register_buffer('c', smap_c(self.a, self.b))
        self.register_buffer('d', smap_d(self.a, self.b))

    def forward(self, eij, dst):
        rd = (dst - self.d0[eij]) / self.r0[eij]
        ret = smap(self.a[eij], self.c[eij], self.d[eij], rd
                   ).masked_fill(eij < 0, 0.0)
        return ret.masked_fill(rd < 0, 1.0)


def rational_almost(rd, nn, nd):
    num = 1 - rd.pow(nn)
    den = 1 - rd.pow(nd)
    return num / den


def rational_singularity(rd, nn, nd):
    return 0.5 * nn * (2 + (nn - nd) * (rd - 1)) / nd


class Rational(nn.Module):
    def __init__(self, d0, r0, nn, nd):
        super().__init__()
        self.register_buffer('d0', d0)
        self.register_buffer('r0', r0)
        self.register_buffer('nn', nn)
        self.register_buffer('nd', nd)
        self.eps = 1e-3

    def forward(self, eij, dst):
        rd = (dst - self.d0[eij]) / self.r0[eij]
        nn = self.nn[eij]
        nd = self.nd[eij]
        rat_almost = rational_almost(rd, nn, nd)
        rat_singul = rational_singularity(rd, nn, nd)
        sing = (rd < 1 + self.eps) & (rd > 1 - self.eps)
        rat = torch.where(sing, rat_singul, rat_almost)
        return rat.masked_fill(rd < 0, 1.0)


def mollifier_inner(x, a, rc):
    return (
        1
        - torch.exp(- a / (rc * rc - (rc - x) ** 2)) / torch.exp(-a / rc / rc)
    )


def mollifier_outer(x: Tensor, rc: Tensor) -> Tensor:
    return (x <= 0).to(x)


def mollifier(x: Tensor, a: Tensor, rc: Tensor):
    mask = (x > 1e-6) & (x < rc - 1e-6)
    outer = mollifier_outer(x, rc)
    inner = mollifier_inner(x[mask], a[mask], rc[mask])
    return outer.masked_scatter(mask, inner)


class Mollifier(nn.Module):
    a: Tensor
    d0: Tensor
    rc: Tensor

    def __init__(self, a, d0, rc):
        super().__init__()
        self.register_buffer('a', a)
        self.register_buffer('d0', d0)
        self.register_buffer('rc', rc)

    def forward(self, eij, dst):
        rc = self.rc[eij]
        d0 = self.d0[eij]
        a = self.a[eij]
        return mollifier(dst - d0, a, rc - d0)


class Coordination(nn.Module):
    elm: Tensor
    coef: Tensor

    def __init__(self, mod, numel: int, rc: float,
                 items: Dict[Tuple[int, int], Dict[str, float]]):
        super().__init__()
        elm = -torch.ones([numel, numel], dtype=torch.long)
        dic: Dict[str, List[float]] = {}
        for n, ((i, j), prp) in enumerate(items.items()):
            elm[i, j] = n
            elm[j, i] = n
            for key, val in prp.items():
                if key not in dic:
                    dic[key] = []
                dic[key].append(val)
        self.register_buffer('elm', elm)
        self.mod = mod(**{key: torch.tensor(val) for key, val in dic.items()})
        self.rc = rc
        self.n = n + 1
        self.pbc = torch.full([self.n], inf)

    def forward(self, inp: Dict[str, Tensor]):
        adj = get_adj_sft_spc(inp, p.coo, self.rc)
        n, i, j = pn.coo2_n_i_j(adj)
        ei = inp[p.elm][n, i]
        ej = inp[p.elm][n, j]
        eij = self.elm[ei, ej]
        adapt = eij >= 0
        vec, sod = vec_sod(inp, adj)
        dis: Tensor = sod[adapt].sqrt()
        eij = eij[adapt]
        coords = self.mod(eij, dis)
        n_bch, _ = inp[p.elm].size()
        idx = ravel1([n[adapt], eij], [n_bch, self.n])
        coord = torch.zeros([n_bch * self.n],
                            dtype=sod.dtype, device=sod.device)
        coord.index_add_(0, idx, coords)
        ret = coord.view([n_bch, self.n]) / 2
        return ret
