import torch


def histc_axis(bins, min_, max_):
    tmp = torch.linspace(min_, max_, bins + 1)
    x = (tmp[1:] + tmp[:-1]) * 0.5
    return x
