"""Functions to calculate the numerical solution to the FC equations of motion using linear algebra"""

import numpy as np
from scipy.linalg import expm
from scipy.special import hermite

# pylint: disable=invalid-name
# pylint: disable=too-many-arguments
# pylint: disable=too-many-locals
# pylint: disable=consider-using-enumerate


def dag(M):
    """
    Gives the Hermitian conjugate of input matrix.

    Args:
        M (array): input matrix

    Returns:
        array: the Hermitian conjugate matrix
    """
    return M.conj().T


def pulse_shape(dw, sig, n=0):
    """
    Gives a Hermite-Gaussian pulse shape.

    Args:
        dw (array): frequency vector or grid
        sig (float): width of the pulse
        n (int): index of Hermite polynomial used. Default is 0 (first Hermite polynomial)

    Returns:
        array: pulse shape
    """
    gaussian = np.exp(-(dw**2) / (2 * sig**2)) / np.sqrt(2 * np.pi * sig**2)
    return gaussian * hermite(n)(dw / sig)


def get_blocks(M):
    """
    Separates a matrix into 4 blocks.

    Args:
        M (array): matrix to split

    Returns:
        (array, :top left block
        array,  :top right block
        array,  :bottom left block
        array)  :bottom right block
    """
    side = M.shape[0]
    Ua = M[0 : side // 2, 0 : side // 2]
    Va = M[0 : side // 2, side // 2 : side]
    Vc = M[side // 2 : side, 0 : side // 2]
    Uc = M[side // 2 : side, side // 2 : side]
    return Ua, Va, Vc, Uc


def get_prop(dw, vp, va, vc, pump, L, D=1):
    """
    Generates the propagator given the crystal and pump parameters, and a frequency vector.

    Args:
        dw (array): frequency vector
        vp (float): group velocity for pump
        va (float): group velocity for a beam
        vc (float): group velocity for c beam
        pump (array): pump pulse shape as a matrix (already scaled)
        L (float): length of propagation in the crystal
        D (float): number depicting the strength of the nonlinear interaction

    Returns:
        (array, :logm of the propagator
        array)  :propagator
    """

    # Upper left and bottom right (diagonal) blocks
    ka = 1 / va - 1 / vp
    kc = 1 / vc - 1 / vp

    Ka = np.diag(1j * dw * ka)
    Kc = np.diag(1j * dw * kc)
    # Bottom left and upper right (non-diagonal) blocks

    J = np.block([[Ka, -1j * np.conj(D) * pump], [-1j * D * pump.conj().T, Kc]])
    prop = expm(J * L)
    return J, prop


def check_prop(prop, print_checks=False):
    """
    Verifies that the obtained propagator fulfills the 6 conditions imposed in the paper
    by Andreas Christ from 2013 (equations A3-4 and A6-7).
    Returns the maximum error for each condition.

    Args :
        prop (matrix): the propagator to perform the checks on
        print_checks (bool): if True, prints if the condition is True or False. False by default.

    Returns :
        array: maximum value of the absolute value of the condition mismatch,
               for each condition (always an array of 6 floats)
    """
    Ua, Va, Vc, Uc = get_blocks(prop)
    Vc = -Vc
    N = Ua.shape[0]
    cond1a = Ua @ dag(Ua) + Va @ dag(Va)
    cond1b = Uc @ dag(Uc) + Vc @ dag(Vc)
    cond3a = dag(Ua) @ Ua + dag(Vc) @ Vc
    cond3b = dag(Uc) @ Uc + dag(Va) @ Va
    cond2 = Ua @ dag(Vc) - Va @ dag(Uc)
    cond4 = dag(Ua) @ Va - dag(Vc) @ Uc
    I = np.identity(N)
    O = np.zeros((N, N))
    conds = np.array([cond1a, cond1b, cond2, cond3a, cond3b, cond4])
    comps = np.array([I, I, O, I, I, O])
    labels = ["1a", "1b", "2", "3a", "3b", "4"]
    to_return = np.array([])
    for i in range(len(labels)):
        cond = conds[i]
        comp = comps[i]
        txt = "Condition {name} is {TF}"
        if print_checks:
            print(txt.format(name=labels[i], TF=np.allclose(cond, comp)))
        to_return = np.append(to_return, np.max(np.abs(cond - comp)))
    return to_return


def remove_phases(prop, dw, L, vp, va, vc):
    """
    Removes the free space propagation phases from the propagator.

    Args:
        prop (array): propagator with phases
        dw (array): frequency vector
        L (float): length of propagation in crystal
        vp (float): group velocity for pump
        va (float): group velocity for a beam
        vc (float): group velocity for c beam

    Returns:
        array :the phase-free propagator as a matrix
    """
    kc = 1 / vp - 1 / vc
    ka = 1 / va - 1 / vp

    # separate into blocks
    Ua, Va, Vc, Uc = get_blocks(prop)
    # remove phases
    Uc = np.diag(np.exp(1j * dw * kc * L)) @ Uc
    Vc = np.diag(np.exp(1j * dw * kc * L)) @ Vc
    Va = np.diag(np.exp(-1j * dw * ka * L)) @ Va
    Ua = np.diag(np.exp(-1j * dw * ka * L)) @ Ua
    return np.block([[Ua, Va], [Vc, Uc]])


def get_JCA(prop):
    """
    Gives the Joint Spectral Amplitude based on the propagator.

    Args:
        prop (array): propagator with phases
        dw (array): frequency vector
        L (float) : length of propagation in crystal
        vp (float): group velocity for pump
        va (float): group velocity for a beam
        vc (float): group velocity for c beam

    Returns:
        array :JCA as a matrix
    """
    Ua, _, Vc, _ = get_blocks(prop)
    M = Ua @ np.conj(Vc).T
    U, s, V = np.linalg.svd(M)
    S = np.diag(s)
    R = np.arcsin(2 * S) / 2
    J = U @ R @ V
    return J


def get_schmidt(prop):
    """
    Gives average number of photons, eigenfunctions (Schmidt modes)
    and eigenvalues (Schmidt coefficients squared) from propagator.
    This is done using an eigendecomposition.
    For the method with straight-up svds, see get_schmidt2.

    Args:
        prop(array): propagator
        dw (array): frequency vector
        L (float): length of propagation crystal
        vp (float): group velocity for pump
        va (float): group velocity for a beam
        vc (float): group velocity for c beam

    Returns:
        (float, :avg number of photons
        array,  :Schmidt modes
        array)  :Schmidt coefficients squared
    """
    _, Va, _, _ = get_blocks(prop)
    Numa = Va @ np.conj(Va).T  # Numbers matrix for a-beam
    Na = np.real(np.trace(Numa))  # average number of photons in a-beam
    val, u = np.linalg.eigh(Numa)  # schmidt coefficients squared and modes from eigendecomposition
    val = np.flip(val)
    u = np.flip(u, axis=1)
    return Na, val, u


def get_schmidt2(prop):
    """
    Alternative method to get Schmidt modes and coefficients
    by doing svds of the propagator blocks directly.

    Args :
        prop (array): the propagator

    Returns :
        (array, :the U matrix from the svd of Va
        array,  :the array of singular values from the svd of Va
        array,  :the V matrix from the svd of Va
        array,  :the U matrix from the svd of Vc
        array,  :the array of singular values from the svd of Vc
        array)  :the V matrix from the svd of Vc
    """
    _, Va, Vc, _ = get_blocks(prop)
    varphiV, sinrk1, phiV = np.linalg.svd(Va)
    xiV, sinrk2, psiV = np.linalg.svd(Vc)
    varphiV = np.flip(varphiV, axis=0)
    phiV = np.flip(phiV, axis=0)
    xiV = np.flip(xiV, axis=0)
    psiV = np.flip(psiV, axis=0)
    return np.conj(varphiV), sinrk1, phiV.T, np.conj(xiV), sinrk2, psiV.T
