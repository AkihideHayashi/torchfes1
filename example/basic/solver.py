import torch

from torchfes.opt.solver import conjugate_gradient

torch.set_default_dtype(torch.float64)


def main():
    A_ = torch.rand(5, 10, 10)
    A = A_ + A_.transpose(1, 2)
    b = torch.rand(5, 10)
    x_exact = torch.solve(b[:, :, None], A).solution.squeeze(-1)
    x = conjugate_gradient(A, b.unsqueeze(-1), b.unsqueeze(-1),
                           torch.ones(5) * 1e-30).squeeze(-1)

    print(torch.allclose(A @ x_exact[:, :, None], b[:, :, None], atol=1e-5))
    print(torch.allclose(A @ x[:, :, None], b[:, :, None], atol=1e-4))
    print((A @ x[:, :, None] - b[:, :, None]).abs().max())


if __name__ == "__main__":
    main()