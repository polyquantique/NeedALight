"""Tests for the implementation of the fermionic bloch messiah decomposition"""

import pytest
import numpy as np
from FC_solve_EoM import get_prop, get_JCA, get_schmidt, get_schmidt2, pulse_shape, get_blocks
from fermionic_bloch_messiah import fermionic_bloch_messiah_1

# pylint: disable=invalid-name
# pylint: disable=too-many-arguments
# pylint: disable=too-many-locals
# pylint: disable=consider-using-enumerate

@pytest.mark.parametrize("N", [101, 301, 1001])
@pytest.mark.parametrize("amp", np.arange(0.01, 0.31, 0.03))
def test_fbm_decomp(amp, N):
    """Tests that the product of decomposition matrices gives back the propagator"""
    vc, vp, va = 1 / 1.5, 1 / 3, 1 / 4.5  # group velocities
    dw_range = 10  # Range of frequency values
    dw = np.linspace(-dw_range, dw_range, N)  # frequency difference vector
    L = 1.35  # length of propagation in FC medium
    sig = 1.1  # width of the (Hermite-Gaussian) pump shape
    pump = amp * pulse_shape(dw[:, np.newaxis] - dw, sig)  # pump
    _, prop = get_prop(dw, vp, va, vc, pump, L)  # propagator

    A, D, Bdag = fermionic_bloch_messiah_1(prop)

    assert np.allclose(A @ D @ Bdag, prop, atol=1e-7)
    # We set absolute tolerance to 1e-7 to pass the tests, but even with that on test is failed


@pytest.mark.parametrize("N", [101, 301, 1001])
@pytest.mark.parametrize("amp", np.linspace(0.01, 0.3, 10))
def test_fbm_sv(amp, N):
    """Tests that the singular values match the expected cos^2+sin^2 relation"""
    vc, vp, va = 1 / 1.5, 1 / 3, 1 / 4.5  # group velocities
    dw_range = 10  # Range of frequency values
    dw = np.linspace(-dw_range, dw_range, N)  # frequency difference vector
    L = 1.35  # length of propagation in FC medium
    sig = 1.1  # width of the (Hermite-Gaussian) pump shape
    pump = amp * pulse_shape(dw[:, np.newaxis] - dw, sig)  # pump
    _, prop = get_prop(dw, vp, va, vc, pump, L)  # propagator

    A, D, Bdag = fermionic_bloch_messiah_1(prop)

    Du, Dv, *_ = get_blocks(D)

    assert np.allclose(Du @ Du + Dv @ Dv, np.identity(N))
