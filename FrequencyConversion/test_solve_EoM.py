"""Tests for the solve_EoM functions"""

import pytest
import numpy as np
from scipy.optimize import curve_fit
from FC_solve_EoM import get_prop, get_JCA, get_schmidt, get_schmidt2, pulse_shape, get_blocks

# pylint: disable=invalid-name
# pylint: disable=too-many-arguments
# pylint: disable=too-many-locals
# pylint: disable=consider-using-enumerate


@pytest.mark.parametrize("N", [101, 301, 1001])
@pytest.mark.parametrize("amp", np.arange(0.01, 0.3, 0.02))
def test_prop_blocks_1(N, amp):
    """Tests that the propagator blocks follow the right relations"""
    vc, vp, va = 1 / 1.5, 1 / 3, 1 / 4.5  # group velocities
    dw_range = 10  # Range of frequency values
    dw = np.linspace(-dw_range, dw_range, N)  # frequency difference vector
    L = 1.35  # length of propagation in FC medium
    sig = 1.1  # width of the (Hermite-Gaussian) pump shape
    pump = amp * pulse_shape(dw[:, np.newaxis] - dw, sig)  # pump
    _, prop = get_prop(dw, vp, va, vc, pump, L)  # matrix to exponentiate and propagator

    Ua, Va, Vc, Uc = get_blocks(prop)
    Vc = -Vc

    cond1a = np.allclose(Ua @ Ua.T.conj() + Va @ Va.T.conj(), np.identity(N))
    cond1b = np.allclose(Uc @ Uc.T.conj() + Vc @ Vc.T.conj(), np.identity(N))
    assert cond1a and cond1b


@pytest.mark.parametrize("N", [101, 301, 1001])
@pytest.mark.parametrize("amp", np.linspace(0.01, 0.3, 10))
def test_prop_blocks_2(N, amp):
    """Tests that the propagator blocks follow the right relations"""
    vc, vp, va = 1 / 1.5, 1 / 3, 1 / 4.5  # group velocities
    dw_range = 10  # Range of frequency values
    dw = np.linspace(-dw_range, dw_range, N)  # frequency difference vector
    L = 1.35  # length of propagation in FC medium
    sig = 1.1  # width of the (Hermite-Gaussian) pump shape
    pump = amp * pulse_shape(dw[:, np.newaxis] - dw, sig)  # pump
    _, prop = get_prop(dw, vp, va, vc, pump, L)  # matrix to exponentiate and propagator

    Ua, Va, Vc, Uc = get_blocks(prop)
    Vc = -Vc

    assert np.allclose(Ua @ Vc.T.conj(), Va @ Uc.T.conj())


@pytest.mark.parametrize("N", [101, 301])
@pytest.mark.parametrize("amp", np.linspace(0.01, 0.3, 10))
def test_noint(N, amp):
    """Tests that when the nonlinear interaction is off, the propagator is diagonal"""
    vc, vp, va = 1 / 1.5, 1 / 3, 1 / 4.5  # group velocities
    dw_range = 10  # Range of frequency values
    dw = np.linspace(-dw_range, dw_range, N)  # frequency difference vector
    L = 1.35  # length of propagation in FC medium
    sig = 1.1  # width of the (Hermite-Gaussian) pump shape
    pump = amp * pulse_shape(dw[:, np.newaxis] - dw, sig)  # pump
    _, prop = get_prop(dw, vp, va, vc, pump, L, D=0)  # propagator

    assert np.allclose(prop - np.diag(np.diagonal(prop)), np.zeros(prop.shape))


@pytest.mark.parametrize("N", [101, 301])
@pytest.mark.parametrize("L", np.arange(0.5, 3, 0.5))
def test_JCA_low_gain(N, L):
    """Tests that the JCA at low gain matches a sinc times a gaussian"""
    vc, vp, va = 1 / 1.5, 1 / 3, 1 / 4.5  # group velocities
    dw_range = 10  # Range of frequency values
    dw = np.linspace(-dw_range, dw_range, N)  # frequency difference vector
    sig = 1.1  # width of the (Hermite-Gaussian) pump shape
    amp = 1e-2  # pump amplitude for low gain
    pump = amp * pulse_shape(dw[:, np.newaxis] - dw, sig)  # pump
    _, prop = get_prop(dw, vp, va, vc, pump, L, D=0)  # propagator

    def phase_matching(dw_a, dw_c, L):
        dk = dw_a * (1 / va - 1 / vp) + dw_c * (1 / vp - 1 / vc)
        return L / 2 * np.sinc(dk * L / 2)

    def JSA_model(X, Y, sig, L, amp):
        return np.abs(phase_matching(X, Y, L) * amp * pulse_shape(X - Y, sig))

    def _JSA_model(M, sig, L, amp):
        x, y = M
        return JSA_model(x, y, sig, L, amp)

    J = get_JCA(prop)

    X, Y = np.meshgrid(dw, dw)

    xdata = np.vstack((np.ravel(X), np.ravel(Y)))
    ydata = np.ravel(np.abs(J))
    p0 = (sig, amp, 6.28)

    popt, pcov = curve_fit(_JSA_model, xdata, ydata, p0)

    fit = JSA_model(X, Y, popt[0], popt[1], popt[2])

    assert np.allclose(J, fit)

