"""Tests for propagator functions"""
# pylint: disable=redefined-outer-name
# pylint: disable=invalid-name
# pylint: disable=too-many-arguments
# pylint: disable=too-many-locals
# pylint: disable=unused-argument
# pylint: disable=unnecessary-lambda-assignment

import pytest
import numpy as np
from scipy.linalg import expm
from NeedALight.propagator import Hprop, Total_prog, phases, JSA, SXPM_prop


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

    Jn = np.abs(J) / np.linalg.norm(np.abs(J))
    Jtn = J_theory / np.linalg.norm(J_theory)

    Fidelity = np.sum(Jn * np.conjugate(Jtn))

    assert np.allclose(Fidelity, 1, atol=10**-5)


@pytest.mark.parametrize("Np", [0.000002, 0.02, 0.04])
@pytest.mark.parametrize("spm", [0, 0.1, 0.5, 1])
@pytest.mark.parametrize("z2", [-0.5, 0, 0.5])
def test_pump_spm(Np, spm, z2):
    """Tests that the numerics used for pump spm match the analytics"""
    N = 301  # Number of frequency values
    vp = 0.1  # pump velocity
    l = 1.0  # amplification region length
    sig = 1  # pump wave packet spread

    wi = -40
    wf = 40
    w = np.linspace(wi, wf, N)

    # Defining gaussian pump pulse in frequency. Includes Np in definition
    def pump(x):
        return (
            np.sqrt(Np)
            * np.exp(-((x) ** 2) / (2 * (sig) ** 2))
            / np.power(np.pi * sig**2, 1 / 4)
        )

    # Defining energy density function for gaussian pump pulse. Includes Np in definition
    def density(x):
        return Np * np.exp(-((x) ** 2) / (4 * (sig) ** 2))

    # Defining gaussian pump  in real space. Note that now this includes Np in definition
    def pumpz2(x):
        return (
            np.sqrt(Np)
            * np.exp(-((x) ** 2) / (2 * (vp / sig) ** 2))
            / np.power(np.pi * (vp / sig) ** 2, 1 / 4)
        )

    z = np.linspace(-10 * l, 10 * l, 4000)
    dz = z[1] - z[0]

    # Analytic expression
    #Includes a manually implemented Fourier Transform
    beta = np.real_if_close(
        np.sum(
            np.exp(-1j * np.tensordot(w, z, axes=0) / vp)
            / np.sqrt(2 * np.pi * vp)
            * dz
            * pumpz2(z + 2 * l)
            * np.exp(
                1j
                * (2 * np.pi * spm / vp)
                * (pumpz2(z + 2 * l) ** 2)
                * (
                    (
                        (z2 + l / 2) * np.heaviside(z2 + l / 2, 0)
                        - (z2 - l / 2) * np.heaviside(z2 - l / 2, 0)
                    )
                    - (
                        (z + l / 2) * np.heaviside(z + l / 2, 0)
                        - (z - l / 2) * np.heaviside(z - l / 2, 0)
                    )
                )
            ),
            axis=1,
        )
        * np.exp(-1j * w * 2 * l / (vp))
    )
    pumpz = expm(
        1j
        * (spm * (w[1] - w[0]) / vp**2)
        * (z2 + l / 2)
        * density(-w + w[:, np.newaxis])
    ) @ pump(w)

    assert np.allclose(beta, pumpz)


@pytest.mark.parametrize("Np", [0.000002, 0.02, 0.04])
@pytest.mark.parametrize("N", [51, 101, 201])
def test_comparison(Np, N):
    """Tests that the propagators for both methods are equivalent when S-X-PM are off"""
    vp = 0.1  # pump velocity
    l = 1.0  # amplification region length
    sig = 1  # pump wave packet spread
    a = 1.61 / 1.13  # from symmetric grp vel matching

    y = 1  # Value for squeezing interaction. Vary through Np instead
    spm = 0  # Value of self-phase modulation for pump
    xpms = 0  # value of cross-phase modulation for signal
    xpmi = 0  # Value of cross-phase modulation for idler

    # Ensuring that we are in the symmetric velocity group matching regime.
    def symmetric_v(vp, sig, l, a):
        vi = vp / (1 - 2 * a * vp / (l * sig))
        vs = vp / (1 + 2 * a * vp / (l * sig))
        return vs, vi

    vs, vi = symmetric_v(vp, sig, l, a)

    # Frequency values
    wi = -30
    wf = 30
    w = np.linspace(wi, wf, N)

    # Unpoled domain
    dz = l / 1000
    domain = np.arange(-l / 2, l / 2, dz)

    # Defining gaussian pump pulse. Includes Np in definition
    def pump(x):
        return (
            np.sqrt(Np)
            * np.exp(-((x) ** 2) / (2 * (sig) ** 2))
            / np.power(np.pi * sig**2, 1 / 4)
        )

    # Defining energy density function for gaussian pump pulse. Includes Np in definition
    def density(x):
        return Np * np.exp(-((x) ** 2) / (4 * (sig) ** 2))
    
    #For the the case where we dont S-X-PM
    def pump2(x):
        return np.exp(-((x) ** 2) / (2 * (sig) ** 2)) / np.power(np.pi * sig**2, 1 / 4)

    domain2 = np.asarray([1])

    prod1, P1, Nm1 = Hprop(
        Np, vs, vi, vp, l, w, pump2, 1
    )  # This generates heisenberg propagators with different signs for length "l"
    T1 = Total_prog(
        domain2, prod1, P1, Nm1
    )  # This generates the total propagator given a specific domain.

    T2 = SXPM_prop(vs, vi, vp, y, spm, xpms, xpmi, pump, density, domain, w)
    assert np.allclose(T1, T2)


@pytest.mark.parametrize("xpms", [0, 0.5, 1])
@pytest.mark.parametrize("xpmi", [0, 0.5, 1])
@pytest.mark.parametrize("spm", [0.1, 0.5, 1])
def test_noint(xpms, xpmi, spm):
    """Tests that when the nonlinear interaction is off,
    contributions to the propagator are as expected"""
    N = 101
    vp = 0.1  # pump velocity
    l = 1.0  # amplification region length
    sig = 1  # pump wave packet spread
    a = 1.61 / 1.13  # from symmetric grp vel matching

    y = 0  # Value for squeezing interaction. Vary through Np instead
    Np = 0.02

    # Ensuring that we are in the symmetric velocity group matching regime.
    def symmetric_v(vp, sig, l, a):
        vi = vp / (1 - 2 * a * vp / (l * sig))
        vs = vp / (1 + 2 * a * vp / (l * sig))
        return vs, vi

    vs, vi = symmetric_v(vp, sig, l, a)

    # Frequency values
    wi = -30
    wf = 30
    w = np.linspace(wi, wf, N)

    # Unpoled domain
    dz = l / 1000
    domain = np.arange(-l / 2, l / 2, dz)

    # Defining gaussian pump pulse. Includes Np in definition
    def pump(x):
        return (
            np.sqrt(Np)
            * np.exp(-((x) ** 2) / (2 * (sig) ** 2))
            / np.power(np.pi * sig**2, 1 / 4)
        )

    # Defining energy density function for gaussian pump pulse. Includes Np in definition
    def density(x):
        return Np * np.exp(-((x) ** 2) / (4 * (sig) ** 2))

    Tnospm = SXPM_prop(vs, vi, vp, y, 0, xpms, xpmi, pump, density, domain, w)
    N2 = len(Tnospm)
    Uss = Tnospm[0 : N2 // 2, 0 : N2 // 2]
    Uiis = Tnospm[N2 // 2 : N2, N2 // 2 : N2]
    TSPM = SXPM_prop(vs, vi, vp, y, spm, xpms, xpmi, pump, density, domain, w)
    UssSPM = TSPM[0 : N2 // 2, 0 : N2 // 2]
    UsiSPM = TSPM[0 : N2 // 2, N2 // 2 : N2]
    UissSPM = TSPM[N2 // 2 : N2, 0 : N2 // 2]
    UiisSPM = TSPM[N2 // 2 : N2, N2 // 2 : N2]

    assert all(
        [np.allclose(Uss, UssSPM),
        np.allclose(Uiis, UiisSPM),
        np.allclose(UsiSPM, np.zeros_like(UsiSPM)),
        np.allclose(UissSPM, np.zeros_like(UissSPM))]
    )
