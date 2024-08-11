import numpy as np

# pylint: disable=invalid-name
# pylint: disable=too-many-arguments
# pylint: disable=too-many-locals
# pylint: disable=consider-using-enumerate

def fermionic_bloch_messiah_1(M):
    """Implements the fermionic Bloch-Messiah decomposition"""

    size = M.shape[0]
    size = size // 2
    Ua = M[:size, :size]
    Va = M[:size, size:]
    Vc = M[size:, :size]
    Uc = M[size:, size:]
    Aa, C, Badag = np.linalg.svd(Ua)
    Ac, _, Bcdag = np.linalg.svd(Uc)
    isDa = np.diag(Aa.T.conj() @ Va @ Bcdag.T.conj())
    isDaabs = np.abs(isDa)
    sqrtangle = np.angle(isDa) / 2
    phase = np.exp(1j * sqrtangle)
    conjphase = phase.conj()
    zero = np.zeros_like(Ac)
    A = np.block([[Aa * phase, zero], [zero, Ac * conjphase]])
    D = np.block([[np.diag(C), np.diag(isDaabs)], [-np.diag(isDaabs), np.diag(C)]])
    Bdag = np.block([[conjphase[:, None] * Badag, zero], [zero, phase[:, None] * Bcdag]])
    return A, D.real, Bdag
