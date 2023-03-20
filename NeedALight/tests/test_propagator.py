"""Tests for propagator functions"""
# pylint: disable=redefined-outer-name
# pylint: disable=invalid-name
# pylint: disable=too-many-arguments
# pylint: disable=too-many-locals
# pylint: disable=unused-argument
# pylint: disable=unnecessary-lambda-assignment

import pytest
import numpy as np
from thewalrus.random import random_symplectic
from NeedALight.propagator import (
    Hprop,
    Total_prog,
    phases,
    JSA,
)


@pytest.mark.parametrize("N", range(50, 500, 50))
@pytest.mark.parametrize("vp", [0.1, 0.2, 0.3])
def test_phases_identity(N, vp):
    """Tests that phases outputs the identity matrix when there is no interaction."""
    l = 1.0  # amplification region length
    sig = 1  # pump wave packet spread
    a = 1.61 / 1.13  # from symmetric grp vel matching
    vi = vp / (1 - 2 * a * vp / (l * sig))
    vs = vp / (1 + 2 * a * vp / (l * sig))
    Ndomain = 1000
    dz = l / Ndomain
    wi = -4
    wf = 4
    x = np.linspace(wi, wf, N)

    domain = np.ones(Ndomain)

    pumpscaled = lambda x: np.exp(-((x) ** 2) / (2 * (sig) ** 2)) / np.power(
        np.pi * (sig), 1 / 4
    )
    prod, P, Nm = Hprop(0, vs, vi, vp, dz, x, pumpscaled, 4)
    T = Total_prog(domain, prod, P, Nm)
    U = phases(T, vs, vi, vp, l, x)

    assert np.allclose(U, np.eye(len(U)))


@pytest.mark.parametrize("N", range(50, 500, 50))
@pytest.mark.parametrize("Np", [0.0001, 0.0002, 0.0003, 0.0004])
def test_JSA_lowgain(N, Np):
    """Tests that JSA returns the well known analytical result for low gain."""
    l = 1.0  # amplification region length
    sig = 1  # pump wave packet spread
    vp = 0.1
    a = 1.61 / 1.13  # from symmetric grp vel matching
    vi = vp / (1 - 2 * a * vp / (l * sig))
    vs = vp / (1 + 2 * a * vp / (l * sig))
    Ndomain = 1000
    dz = l / Ndomain
    wi = -4
    wf = 4
    x = np.linspace(wi, wf, N)
    domain = np.ones(Ndomain)

    pumpscaled = lambda x: np.exp(-((x) ** 2) / (2 * (sig) ** 2)) / np.power(
        np.pi * (sig), 1 / 4
    )
    prod, P, Nm = Hprop(Np, vs, vi, vp, dz, x, pumpscaled, 4)
    T = Total_prog(domain, prod, P, Nm)
    J, _Ns, _K, _M, _Nums, _Numi = JSA(T, vs, vi, vp, len(domain) * dz, x)
    J_theory = np.abs(
        (np.sqrt(Np) * l / (np.sqrt(2 * np.pi * vs * vi * vp)))
        * np.exp(-((x + x[:, np.newaxis]) ** 2) / (2 * (sig) ** 2))
        / np.power(np.pi * (sig), 1 / 4)
        * (np.sinc(a * (x - x[:, np.newaxis]) / (np.pi * sig)))
    )

    Jn = J / np.linalg.norm(J)
    Jtn = J_theory / np.linalg.norm(J_theory)

    Fidelity = np.sum(Jn * np.conjugate(Jtn))

    assert np.allclose(Fidelity, 1, atol=10**-5)
