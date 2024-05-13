"""Tests for propagator functions"""

# pylint: disable=redefined-outer-name
# pylint: disable=invalid-name
# pylint: disable=too-many-arguments
# pylint: disable=too-many-locals
# pylint: disable=unused-argument
# pylint: disable=unnecessary-lambda-assignment
# pylint: disable=consider-using-enumerate
# pylint: disable=no-name-in-module

import pytest
import numpy as np
from scipy.linalg import expm
from scipy.special import erfi
from thewalrus.quantum.gaussian_checks import is_symplectic
from NeedALight.propagator import (
    Hprop,
    Total_prog,
    phases,
    JSA,
    SXPM_prop,
    JSAK,
    Total_propK,
    symplectic_prop,
)
from NeedALight.magnus import (
    Magnus1,
    Magnus3_Re,
    Magnus3_Im,
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
    # Includes a manually implemented Fourier Transform
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

    # For the the case where we dont S-X-PM
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
        [
            np.allclose(Uss, UssSPM),
            np.allclose(Uiis, UiisSPM),
            np.allclose(UsiSPM, np.zeros_like(UsiSPM)),
            np.allclose(UissSPM, np.zeros_like(UissSPM)),
        ]
    )


@pytest.mark.parametrize("nk", range(51, 201, 50))
@pytest.mark.parametrize("Np", [0.00001, 0.00002, 0.00003, 0.00004])
def test_JSAK_lowgain(nk, Np):
    """Tests that JSAK returns the well known analytical result for low gain."""
    l = 1.0  # amplification region length
    sig = 1  # pump wave packet spread
    vp = 0.1
    a = 1.61 / 1.13  # from symmetric grp vel matching
    vi = vp / (1 - 2 * a * vp / (l * sig))
    vs = vp / (1 + 2 * a * vp / (l * sig))
    # Crystal properties
    Ndomain = 1000  # Number of spatial points for non-linear/crystal grid
    dz = (l) / Ndomain
    domain_width = dz
    number_domains = Ndomain
    L = number_domains * domain_width

    # Momenta grid
    k_ft = 200 / L
    dk = k_ft / nk
    k = np.arange(-k_ft / 2, k_ft / 2, dk)

    # Domain for a tophat potential
    domain = np.asarray([1] * int(Ndomain) + [1])
    ws = vs * k
    wi = vi * k
    z_list = np.linspace(0.0005, 1.0005, 1001) - l / 2
    t = np.arange(0, 20 + 0.2, 0.2)

    # defining gaussian pump pulse in momentum space with z_0=l
    def pump(x, scale=1):
        return (
            np.exp(-((x) ** 2) / (2 * ((sig / vp) / scale) ** 2))
            / np.power(np.pi * ((sig / vp) / scale) ** 2, 1 / 4)
            * np.exp(1j * x * l)
        )

    sc = 1
    # Pump function in (t,z) for dispersion and momentum profile determined above.
    # To help with FT we define a different momenta grid
    nkp = 801  # Number of momentum values for pump FT
    kp_ft = 1000 / L
    dkp = kp_ft / nkp
    kp = np.arange(-kp_ft / 2, kp_ft / 2, dkp)

    # Pump dispersion relation
    wp = vp * kp

    Lambda = np.zeros([len(t), len(z_list)], dtype=np.complex128)
    for i in range(len(t)):
        Lambda[i, :] = (
            np.sqrt(Np)
            * np.sum(
                np.exp(-1j * wp[:, np.newaxis] * t[i])
                * np.exp(1j * kp[:, np.newaxis] * (z_list))
                * pump(kp[:, np.newaxis], scale=1 / sc),
                axis=0,
            )
            * dkp
            / np.sqrt(2 * np.pi)
        )

    K = Total_propK(domain, Lambda, z_list, k, t, ws, wi)
    J, _Ns, _Schmidt, _M, _Nums, _Numi = JSAK(K, dk)

    J_theory = np.abs(
        (np.sqrt(Np) / (np.sqrt(2 * np.pi * vs * vi)))
        * np.exp(-((vi * k + vs * k[:, np.newaxis]) ** 2) / (2 * (sig) ** 2))
        / np.power(np.pi * (sig / vp) ** 2, 1 / 4)
        * (
            L
            * np.sinc(
                L * (k * (1 - vi / vp) + k[:, np.newaxis] * (1 - vs / vp)) / (2 * np.pi)
            )
        )
    )

    Jn = np.abs(J) / np.linalg.norm(np.abs(J))
    Jtn = J_theory / np.linalg.norm(J_theory)

    Fidelity = np.sum(Jn * np.conjugate(Jtn))

    assert np.allclose(Fidelity, 1, atol=10**-5)


@pytest.mark.parametrize("nk", [101, 201])
@pytest.mark.parametrize("Np", [0.00001, 0.02])
@pytest.mark.parametrize("vp", [0.08, 0.1, 0.12])
def test_ktzw_comparison(nk, Np, vp):
    """Tests that JSA and JSAK functions return similar outputs for Ns,
    JSA Fidelity, and Schmidt number. We allow for 0.5% error due to boundary issues"""
    l = 1.0  # amplification region length
    sig = 1  # pump wave packet spread
    a = 1.61 / 1.13  # from symmetric grp vel matching
    vi = vp / (1 - 2 * a * vp / (l * sig))
    vs = vp / (1 + 2 * a * vp / (l * sig))
    # Crystal properties
    Ndomain = 1000  # Number of spatial points for non-linear/crystal grid
    dz = (l) / Ndomain
    domain_width = dz
    number_domains = Ndomain
    L = number_domains * domain_width

    # Momenta grid
    k_ft = 200 / L
    dk = k_ft / nk
    k = np.arange(-k_ft / 2, k_ft / 2, dk)

    # Domain for a tophat potential
    domain = np.asarray([1] * int(Ndomain) + [1])
    sc = 1
    Lambda = lambda x, t: np.sqrt(Np) * pump(x + l - vp * t, scale=1 / sc)
    ws = vs * k
    wi = vi * k
    z_list = np.linspace(0.0005, 1.0005, 1001) - l / 2
    t = np.arange(0, 20 + 0.05, 0.05)

    # defining gaussian pump pulse in momentum space with z_0=l
    def pump(x, scale=1):
        return (
            np.exp(-((x) ** 2) / (2 * ((sig / vp) / scale) ** 2))
            / np.power(np.pi * ((sig / vp) / scale) ** 2, 1 / 4)
            * np.exp(1j * x * l)
        )

    sc = 1
    # Pump function in (t,z) for momentum profile specified above and dispersion specified below.
    # To help with FT we define a different momenta grid
    nkp = 801  # Number of momentum values for pump FT
    kp_ft = 1000 / L
    dkp = kp_ft / nkp
    kp = np.arange(-kp_ft / 2, kp_ft / 2, dkp)

    # Pump dispersion relation
    wp = vp * kp

    Lambda = np.zeros([len(t), len(z_list)], dtype=np.complex128)
    for i in range(len(t)):
        Lambda[i, :] = (
            np.sqrt(Np)
            * np.sum(
                np.exp(-1j * wp[:, np.newaxis] * t[i])
                * np.exp(1j * kp[:, np.newaxis] * (z_list))
                * pump(kp[:, np.newaxis], scale=1 / sc),
                axis=0,
            )
            * dkp
            / np.sqrt(2 * np.pi)
        )

    K = Total_propK(domain, Lambda, z_list, k, t, ws, wi)
    Jk, Nsk, Schmidtk, _M, _Nums, _Numi = JSAK(K, dk)

    # With z-w functions
    wi = -8.5
    wf = 8.5
    x = np.linspace(wi, wf, nk)
    pumpscaled = lambda x: np.exp(-((x) ** 2) / (2 * (sig) ** 2)) / np.power(
        np.pi * (sig) ** 2, 1 / 4
    )
    prod, P, Nm = Hprop(Np, vs, vi, vp, dz, x, pumpscaled, 4)
    T = Total_prog(domain, prod, P, Nm)
    J, Ns, Schmidt, _M, _Nums, _Numi = JSA(T, vs, vi, vp, len(domain) * dz, x)

    Jmax = np.abs(np.amax(J))
    Jkmax = np.abs(np.amax(Jk)) / np.sqrt(vs * vi)

    ErrorJSA = np.abs(Jmax - Jkmax) / Jmax
    ErrorNs = np.abs(Ns - Nsk) / Ns
    ErrorSchmidt = np.abs(Schmidt - Schmidtk) / Schmidt

    assert all(
        [
            ErrorJSA < 1,
            ErrorNs < 0.5,
            ErrorSchmidt < 0.5,
        ]
    )


@pytest.mark.parametrize("nk", [101, 201])
@pytest.mark.parametrize("Np", [0.00001, 0.02])
@pytest.mark.parametrize("vp", [0.08, 0.1, 0.12])
def test_symplectic(nk, Np, vp):
    """Tests that the extended propagators are truly symplectic"""
    l = 1.0  # amplification region length
    sig = 1  # pump wave packet spread
    a = 1.61 / 1.13  # from symmetric grp vel matching
    vi = vp / (1 - 2 * a * vp / (l * sig))
    vs = vp / (1 + 2 * a * vp / (l * sig))
    # Crystal properties
    Ndomain = 1000  # Number of spatial points for non-linear/crystal grid
    dz = (l) / Ndomain
    domain_width = dz
    number_domains = Ndomain
    L = number_domains * domain_width

    # Momenta grid
    k_ft = 200 / L
    dk = k_ft / nk
    k = np.arange(-k_ft / 2, k_ft / 2, dk)

    # Domain for a tophat potential
    domain = np.asarray([1] * int(Ndomain) + [1])
    ws = vs * k
    wi = vi * k
    z_list = np.linspace(0.0005, 1.0005, 1001) - l / 2
    t = np.arange(0, 20 + 0.5, 0.5)

    # defining gaussian pump pulse in momentum space with z_0=l
    def pump2(x, scale=1):
        return (
            np.exp(-((x) ** 2) / (2 * ((sig / vp) / scale) ** 2))
            / np.power(np.pi * ((sig / vp) / scale) ** 2, 1 / 4)
            * np.exp(1j * x * l)
        )

    # Pump function in (t,z) for momentum profile specified above and dispersion specified below.
    # To help with FT we define a different momenta grid
    nkp = 801  # Number of momentum values for pump FT
    kp_ft = 1000 / L
    dkp = kp_ft / nkp
    kp = np.arange(-kp_ft / 2, kp_ft / 2, dkp)

    # Pump dispersion relation
    wp = vp * kp
    sc = 1
    Lambda2 = np.zeros([len(t), len(z_list)], dtype=np.complex128)
    for i in range(len(t)):
        Lambda2[i, :] = (
            np.sqrt(Np)
            * np.sum(
                np.exp(-1j * wp[:, np.newaxis] * t[i])
                * np.exp(1j * kp[:, np.newaxis] * (z_list))
                * pump2(kp[:, np.newaxis], scale=1 / sc),
                axis=0,
            )
            * dkp
            / np.sqrt(2 * np.pi)
        )

    K = Total_propK(domain, Lambda2, z_list, k, t, ws, wi)

    # With z-w functions
    wi = -8.5
    wf = 8.5
    x = np.linspace(wi, wf, nk)
    pumpscaled = lambda x: np.exp(-((x) ** 2) / (2 * (sig) ** 2)) / np.power(
        np.pi * (sig) ** 2, 1 / 4
    )
    prod, P, Nm = Hprop(Np, vs, vi, vp, dz, x, pumpscaled, 4)
    T = Total_prog(domain, prod, P, Nm)

    K_ext = symplectic_prop(K, -1 / (vs - vp), 1 / (vi + vp), 1 / vp, t[-1], k)
    T_ext = symplectic_prop(T, vs, vi, vp, len(domain) * dz, x)

    assert all(
        [
            is_symplectic(K_ext),
            is_symplectic(T_ext),
        ]
    )


@pytest.mark.parametrize("Np", [0.00001, 0.02, 0.2])
@pytest.mark.parametrize("vp", [0.08, 0.1, 0.12])
def test_ktpump(Np, vp):
    """Tests that obtaining the pump matrix from the equation
    of motion gives the proper solution for linear dispersion"""
    l = 1.0  # amplification region length
    sig = 1  # pump wave packet spread

    # Domain for a tophat potential
    z_list = np.linspace(0.0005, 1.0005, 1001) - l / 2
    t = np.arange(0, 20 + 0.5, 0.5)

    # defining gaussian pump pulse in momentum space with z_0=l
    def pump2(x, scale=1):
        return (
            np.exp(-((x) ** 2) / (2 * ((sig / vp) / scale) ** 2))
            / np.power(np.pi * ((sig / vp) / scale) ** 2, 1 / 4)
            * np.exp(1j * x * l)
        )

    # Pump function in (t,z) for momentum profile specified above and dispersion specified below.
    # To help with FT we define a different momenta grid
    nkp = 801  # Number of momentum values for pump FT
    kp_ft = 1000 / l
    dkp = kp_ft / nkp
    kp = np.arange(-kp_ft / 2, kp_ft / 2, dkp)

    # Pump dispersion relation
    wp = vp * kp
    sc = 1
    Lambda2 = np.zeros([len(t), len(z_list)], dtype=np.complex128)
    for i in range(len(t)):
        Lambda2[i, :] = (
            np.sqrt(Np)
            * np.sum(
                np.exp(-1j * wp[:, np.newaxis] * t[i])
                * np.exp(1j * kp[:, np.newaxis] * (z_list))
                * pump2(kp[:, np.newaxis], scale=1 / sc),
                axis=0,
            )
            * dkp
            / np.sqrt(2 * np.pi)
        )

    # defining gaussian pump pulse
    def pump(x, scale=1):
        return np.exp(-((x) ** 2) / (2 * ((vp / sig) * scale) ** 2)) / np.power(
            np.pi * ((vp / sig) * scale) ** 2, 1 / 4
        )

    Lambda = np.sqrt(Np) * pump(z_list + l - vp * t[:, np.newaxis], scale=1 / sc)

    assert np.allclose(Lambda2, Lambda)


@pytest.mark.parametrize("N", [101, 201, 301, 401])
@pytest.mark.parametrize("sig", [0.8, 1, 1.2, 2])
def test_magnus(N, sig):
    """Tests Magnus terms agree with analytical results for Gaussian PMF."""
    l = 1.0  # amplification region length
    a = 1.61 / 1.13  # from symmetric grp vel matching
    vp = 0.1  # pump velocity, does not change anything for magnus.

    def symmetric_v(vp, sig, l, a):
        vi = vp / (1 - 2 * a * vp / (l * sig))
        vs = vp / (1 + 2 * a * vp / (l * sig))
        return vs, vi

    vs, vi = symmetric_v(vp, sig, l, a)

    kappa = 1 / vs - 1 / vp

    # Frequency values
    wi = -4
    wf = 4
    w = np.linspace(wi, wf, N)

    # For the pump, we take it ot be L2 normalized.
    def pump(x):
        return np.exp(-((x) ** 2) / (2 * (sig) ** 2)) / np.power(np.pi * sig**2, 1 / 4)

    # Define a Gaussian phase-matching function.
    def PMF2(x, y, z):
        return np.exp(-((x / vs + y / vi - z / vp) ** 2) / (2 * (sig * kappa) ** 2))

    # Product of PMF and Pump Envelope which is to be integrated
    F = lambda x, y, z: pump(z) * PMF2(x, y, z)

    # Obtaining magnus terms from cubature functions
    J1 = Magnus1(F, w)
    J3 = Magnus3_Re(F, w)
    K3 = Magnus3_Im(F, w)

    # Next we consider analytics.
    tau = 1 / (np.sqrt(2) * sig)  # To equate the two different conventions used.

    f0 = (
        lambda x: np.sqrt(2 * tau)
        * np.exp(-2 * tau**2 * (x**2))
        / np.power(np.pi, 1 / 4)
    )
    f1 = lambda x: np.sqrt(3) * f0(x) * erfi(np.sqrt(4 / 3) * tau * x)

    J1_th = f0(w) * f0(w[:, np.newaxis])
    J3_th = (f0(w) * f0(w[:, np.newaxis]) - f1(w) * f1(w[:, np.newaxis])) / 12
    K3_th = -(f0(w) * f1(w[:, np.newaxis]) - f1(w) * f0(w[:, np.newaxis])) / (
        4 * np.sqrt(3)
    )  # Extra minus sign is due to different conventions used

    J1n = J1 / np.linalg.norm(J1)
    J1tn = J1_th / np.linalg.norm(J1_th)

    J3n = J3 / np.linalg.norm(J3)
    J3tn = J3_th / np.linalg.norm(J3_th)

    K3n = K3 / np.linalg.norm(K3)
    K3tn = K3_th / np.linalg.norm(K3_th)

    FidelityJ1 = np.sum(J1n * np.conjugate(J1tn))
    FidelityJ3 = np.sum(J3n * np.conjugate(J3tn))
    FidelityK3 = np.sum(K3n * np.conjugate(K3tn))

    assert all(
        [
            np.allclose(FidelityJ1, 1, atol=10**-4),
            np.allclose(FidelityJ3, 1, atol=10**-2),
            np.allclose(FidelityK3, 1, atol=10**-2),
        ]
    )
