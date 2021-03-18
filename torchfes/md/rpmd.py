from math import sqrt, cos, sin, pi
from typing import Dict, NamedTuple
import torch
from torch import is_distributed, nn, Tensor
import torchfes as fes


def rpmd_transformation_matrix(n: int, device):
    if n % 2 == 1:
        raise NotImplementedError('n must be even now')
    C = torch.zeros((n, n), device=device)
    for j in range(n):
        for k in range(n):
            if k == 0:
                C[j, k] = sqrt(1 / n)
            elif 1 <= k <= n / 2 - 1:
                C[j, k] = sqrt(2 / n) * cos(2 * pi * j * k / n)
            elif k == n // 2:
                C[j, k] = sqrt(1 / n) * (-1) ** j
            else:
                C[j, k] = sqrt(2 / n) * sin(2 * pi * j * k / n)
    return C


def rpmd_frequency(omega_n: Tensor, n: int):
    assert omega_n.dim() == 1
    k = torch.arange(n, device=omega_n.device)
    omega_k = 2 * omega_n * torch.sin(k * pi / n)
    return omega_k


class RpmdBase(NamedTuple):
    beta: float
    beta_n: Tensor  # [bch]
    omega_k: Tensor  # [bch]
    c: Tensor  # [bch, bch]
    c_t: Tensor  # [bch, bch]


def rpmd_base(hbar: float, beta: float, n: int, device):
    o = torch.ones(n, device=device)
    beta_n = beta / n * o
    c = rpmd_transformation_matrix(n, device)
    ct = c.t()
    omega_n = 1 / (beta_n * hbar)
    omega_k = rpmd_frequency(omega_n, n)
    return RpmdBase(beta, beta_n, omega_k, c, ct)


def rpmd_evolution_matrix(mas: Tensor, omega_k: Tensor, dtm: float, num_dim: int):
    assert mas.dim() == 2
    assert omega_k.dim() == 1
    o = omega_k[:, None]
    pq = - mas * o * torch.sin(o * dtm)
    qp = torch.where(o != 0.0, 1.0 / (mas * o) * torch.sin(o * dtm), dtm / mas)
    pp = torch.cos(o * dtm).expand_as(pq)
    qq = torch.cos(o * dtm).expand_as(pq)
    v = torch.stack([pp, pq, qp, qq], dim=2)  # [bch, atm, 4]
    # v = torch.stack([pp, qp, pq, qq], dim=2)  # [bch, atm, 4]
    m = v.reshape([*mas.size(), 2, 2])  # [bch, atm, 2, 2]
    # assert torch.allclose(m[:, :, 0, 0], pp)
    # assert torch.allclose(m[:, :, 0, 1], pq)
    # assert torch.allclose(m[:, :, 1, 0], qp), (m[:, :, 1, 0])
    # assert torch.allclose(m[:, :, 1, 1], qq)
    m = m[:, :, None, :, :].expand([-1, -1, num_dim, 2, 2])
    return m  # [bch, atm, dim, 2, 2]


class RpmdPos(nn.Module):
    e: Tensor
    c: Tensor
    ct: Tensor

    def __init__(self, base: RpmdBase, mas: Tensor, dtm: float, num_dim: int) -> None:
        super().__init__()
        e = rpmd_evolution_matrix(
            mas, base.omega_k, dtm, num_dim).detach().clone()
        self.register_buffer('e', e)  # [bch, atm, dim, 2, 2]
        self.register_buffer('c', base.c)  # [bch, bch]
        self.register_buffer('ct', base.c_t)  # [bch, bch]

    def forward(self, mol: Dict[str, Tensor]):
        p = mol[fes.p.mom]
        q = mol[fes.p.pos]
        PQ = torch.stack([p, q], dim=3)  # [bch, atm, dim, 2]
        bch, atm, dim, _ = PQ.size()
        PQ = (self.ct @ PQ.view([bch, atm * dim * 2])).view([bch, atm, dim, 2])
        PQ = (
            self.e.view([bch * atm * dim, 2, 2]
                        ) @ PQ.view([bch * atm * dim, 2, 1])
        ).view([bch, atm, dim, 2])
        PQ = (self.c @ PQ.view([bch, atm * dim * 2])).view([bch, atm, dim, 2])
        p, q = PQ.unbind(3)
        ret = mol.copy()
        ret[fes.p.mom] = p
        ret[fes.p.pos] = q
        return ret


class RpmdKin(nn.Module):
    def __init__(self, base: RpmdBase) -> None:
        super().__init__()
        self.beta = base.beta

    def forward(self, mol: Dict[str, Tensor]):
        pos = mol[fes.p.pos]
        frc = mol[fes.p.frc]
        n, atm, dim = pos.size()
        N = atm * dim
        pos_mean = pos.mean(dim=0, keepdim=True)
        delta_pos = pos - pos_mean
        correction = -(delta_pos * frc).sum() / (2 * n)
        if fes.p.fix_msk in mol:
            N = N - int(mol[fes.p.fix_msk].sum(dim=2).sum(dim=1).to(pos).mean().item())
        kin = N / (2 * self.beta) + correction
        ret = mol.copy()
        ret[fes.p.rpm_kin] = kin
        return ret


class RpmdLangevin(nn.Module):
    c: Tensor
    ct: Tensor
    c1: Tensor
    c2mb: Tensor

    def __init__(self, base: RpmdBase, tau: float, dtm: float, mas: Tensor, num_dim: int) -> None:
        assert isinstance(tau, float)
        assert isinstance(dtm, float)
        super().__init__()
        self.beta_n = base.beta_n
        self.c = base.c
        self.ct = base.c_t
        gamma = 2 * base.omega_k
        gamma[0] = 1.0 / tau
        c1 = torch.exp(-dtm * gamma)
        c2 = torch.sqrt(1 - c1 * c1)
        c2mb = torch.sqrt(mas[:, :, None] / base.beta_n[:,
                                                        None, None]) * c2[:, None, None]
        c1 = c1[:, None, None].expand(
            [-1, mas.size(1), num_dim]).detach().clone()
        c2mb = c2mb.expand([-1, -1, num_dim]).detach().clone()
        self.register_buffer('c1', c1)
        self.register_buffer('c2mb', c2mb)

    def forward(self, mol: Dict[str, Tensor]):
        p = mol[fes.p.mom]
        B, A, D = p.size()
        c1 = self.c1.view([B, A * D])
        c2mb = self.c2mb.view([B, A * D])
        p = p.view([B, A * D])
        p = self.ct @ p
        p = c1 * p + c2mb * torch.randn([B, A * D], device=p.device)
        p = self.c @ p
        ret = mol.copy()
        ret[fes.p.mom] = p.view([B, A, D])
        return ret
