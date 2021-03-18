# -*- coding: utf-8 -*-
from numpy import exp, log, pi
from numpy.linalg import eigh, det
import numpy as np


def energy_translation(beta):
    return 1.5 / beta


def entropy_translation(m_2pihbarhbar, beta, p):
    return 2.5 + log(m_2pihbarhbar ** 1.5 / (beta ** 2.5 * p))


def gibbs_translation(m_2pihbarhbar, beta, p):
    return - (3 / 2 * log(m_2pihbarhbar) - 5 / 2 * log(beta) - log(p)) / beta


def energy_vibration(hbaromega, beta):
    t = hbaromega * beta
    return np.sum(t * (0.5 + exp(-t) / (1 - exp(-t)))) / beta


def entropy_vibration(hbaromega, beta):
    t = hbaromega * beta
    et = exp(-t)
    return np.sum((t * et) / (1 - et) - log(1 - et))


def helmholtz_vibration(hbaromega, beta):
    t = hbaromega * beta  # theta / T
    return np.sum(1 / 2 * t + log(1 - exp(-t))) / beta


def tensor_of_inertia(coordinate, mass):
    G = mass @ coordinate / np.sum(mass)
    R = coordinate - G
    return sum((r @ r * np.eye(3) - np.outer(r, r)) * m for r, m in zip(R, mass))


def energy_rotation(I2_hbar, beta):
    if det(I2_hbar / beta) < 1E-8:
        return 1 / beta
    else:
        return 1.5 / beta


def entropy_rotation(I2_hbar, beta, sigma):
    t = det(I2_hbar / beta)
    e, _ = eigh(I2_hbar / beta)
    if t > 1E-8:
        return 1 / 2 * log(pi * t) - log(sigma) + 3 / 2
    elif abs(e[1] - e[2]) < 1E-8:
        return 1 + log(e[1] / sigma)
    else:
        raise NotImplementedError()


def helmholtz_rotation(I2_hbar, beta, sigma):
    """I2_hbar: I * 2 / hbar
    sigma: normaly 1, linear 2
    """
    t = det(I2_hbar / beta)
    e, _ = eigh(I2_hbar / beta)
    if t > 1E-8:
        return (-1/2 * np.log(np.pi * np.prod(t)) + np.log(sigma)) / beta
    elif np.abs(e[1] - e[2]) < 1E-8:
        return -1 * np.log(e[1] / sigma) / beta
    else:
        raise NotImplementedError()


def sum_helmholtz(fs, beta):
    mf = (max(fs) + min(fs)) / 2
    return -1 * np.log(sum(np.exp(-(beta * (f - mf))) for f in fs)) / beta + mf[0]
