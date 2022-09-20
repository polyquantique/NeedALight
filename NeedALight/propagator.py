"""Functions to produce JSA for variable domain configs """

from itertools import product
import numpy as np
from scipy.linalg import expm, sqrtm

# pylint: disable=invalid-name
# pylint: disable=too-many-arguments
# pylint: disable=too-many-locals


def Hprop(Np, vs, vi, vp, l, x, f, n):
    """Generate Heisenberg propagator for given values and pump pulse

    Args:
        Np (int): Pump photon number
        vs (float): signal velocity
        vi (float): idler velocity
        vp (float): pump velovity
        l (float): length of crystal slice
        x (array): vector of frequency values
        f (function): fuctional form of pump pulse (normalized to 1 and not Np)
        n (int): number of segments we wish to precalculate
    Returns:
        (array, :Heisenberg propagator for all possible orientations of n-segments
        array,  :Heisenberg propagator g(z)=1
        array)  :Heisenberg propagator g(z)=-1
    """
    # Constructing matrix for EoM
    G = np.diag((1 / vs - 1 / vp) * x)
    H = np.diag((1 / vi - 1 / vp) * x)
    F = (
        np.sqrt(Np)
        * (x[len(x) - 1] - x[0])
        / (len(x) - 1)
        / (np.sqrt(2 * np.pi * vs * vi * vp))
    ) * f(x + x[:, np.newaxis])
    Q = np.block([[G, F], [-np.conj(F).T, -np.conj(H).T]])
    Q2 = np.block(
        [[G, -F], [np.conj(F).T, -np.conj(H).T]]
    )  # sgn(g(z)) only changes sgn(F)
    P = expm(1j * Q * l)
    N = expm(1j * Q2 * l)
    # Finding the products for domain lengths of n-segments
    prod = np.ones((2**n, len(P), len(P)), dtype=np.complex128)
    for i, config in enumerate(product((P, N), repeat=n)):
        T = np.eye(len(P), dtype=np.complex128)
        for j in range(n):
            T = config[j] @ T
        prod[i] = T
    return prod, P, N


def Total_prog(domain, prod, P, N):
    """Heisenberg Propagator for aperiodically polled domain

    Args:
        domain (array): vector of +/-1's representing sgn of potential
        P (array): Heisenberg propagator for segment dz with g(z)=1
        N (array): Heisenberg propagator for segment dz with g(z)=-1

    Returns:
        array: Heisenberg propagator over poled crystal
    """
    Nd = len(domain)
    n = int(np.log2(len(prod)))
    remainder = Nd % n
    T = np.eye(len(P), dtype=np.complex128)
    x = list(product((1, -1), repeat=n))

    for i in range(0, Nd - remainder, n):
        index = x.index(tuple(domain[i : i + n]))
        T = prod[index] @ T

    if remainder != 0:
        leftover = domain[-remainder::]
        for i, element in enumerate(leftover):
            if element > 0:
                T = P @ T
            else:
                T = N @ T
    return T


def phases(T, vs, vi, vp, l, x):
    """Removes free propagation phases from Heisenberg Propagator

    Args:
        T (array): Full Heisenberg Propagator
        vs (float): signal velocity
        vi (float): idler velocity
        vp (float): pump velocity
        l (float): total length of crystal
        x (array): vector of frequencies

    Returns:
        U (array): Heisenberg Propagator w/o free propagation phases

    """

    N = len(T)
    ks = 1 / vs - 1 / vp
    ki = 1 / vi - 1 / vp
    Uss = (
        np.diag(np.exp(-1j * ks * x * l / 2))
        @ T[0 : N // 2, 0 : N // 2]
        @ np.diag(np.exp(-1j * ks * x * l / 2))
    )
    Usi = (
        np.diag(np.exp(-1j * ks * x * l / 2))
        @ T[0 : N // 2, N // 2 : N]
        @ np.diag(np.exp(1j * ki * x * l / 2))
    )
    Uiss = (
        np.diag(np.exp(1j * ki * x * l / 2))
        @ T[N // 2 : N, 0 : N // 2]
        @ np.diag(np.exp(-1j * ks * x * l / 2))
    )
    Uiis = (
        np.diag(np.exp(1j * ki * x * l / 2))
        @ T[N // 2 : N, N // 2 : N]
        @ np.diag(np.exp(1j * ki * x * l / 2))
    )
    U = np.block([[Uss, Usi], [Uiss, Uiis]])
    return U


def JSA(T, vs, vi, vp, l, x):
    """Joint spectral amplitude

    Args:
        U (array): Heisenberg propagator with free propagation
        vs (float): signal velocity
        vi (float): idler velocity
        vp (float): pump velocity
        l (float): total length of crystal
        x (array): vector of frequencies

    Returns:
        (array, :Joint spectral amplitude
        float,  :Number of signal photons
        float,  :K number
        array,  :M moment matrix
        array,  :signal number matrix
        array)  :Idler number matrix
    """
    N = len(T)
    dw = (x[len(x) - 1] - x[0]) / (len(x) - 1)
    # Removing free propagation phases and breaking it into blocks
    U = phases(T, vs, vi, vp, l, x)
    Uss = U[0 : N // 2, 0 : N // 2]
    Usi = U[0 : N // 2, N // 2 : N]
    Uiss = U[N // 2 : N, 0 : N // 2]
    # Constructing the moment matrix
    M = Uss @ (np.conj(Uiss).T)
    # Using SVD of M to construct JSA
    L, s, Vh = np.linalg.svd(M)
    Sig = np.diag(s)
    D = np.arcsinh(2 * Sig) / 2
    J = np.abs(L @ D @ Vh) / dw
    # Number of signal photons
    Nums = np.conj(Usi) @ Usi.T
    Numi = Uiss @ (np.conj(Uiss).T)
    Ns = np.real(np.trace(Nums))
    # Finding K
    K = (np.trace(np.sinh(D) ** 2)) ** 2 / np.trace(np.sinh(D) ** 4)
    return J, Ns, K, M, Nums, Numi

# The next two functions (is_symplectic and blochmessiah) can be deleted once TheWalrus' official version is 0.20.0

def is_symplectic(S, rtol=1e-05, atol=1e-08):
    """Checks if the matrix is symplectic
    Args:
        S (array): a matrix
        rtol (float): the relative tolerance parameter used in `np.allclose`
        atol (float): the absolute tolerance parameter used in `np.allclose`
    Returns:
        (bool): whether the given matrix is symplectic
    """
    (n, m) = S.shape
    if n != m:
        return False
    if n % 2 != 0:
        return False

    n = n // 2
    A = S[0:n, 0:n]
    B = S[0:n, n : 2 * n]
    C = S[n : 2 * n, 0:n]
    D = S[n : 2 * n, n : 2 * n]

    if (
        np.allclose(A.T @ C, C.T @ A, rtol=rtol, atol=atol)
        and np.allclose(B.T @ D, D.T @ B, rtol=rtol, atol=atol)
        and np.allclose(A.T @ D - C.T @ B, np.eye(n), rtol=rtol, atol=atol)
    ):
        return True

    return False




def blochmessiah(S):
    """Returns the Bloch-Messiah decomposition of a symplectic matrix S = uff @ dff @ vff
       where uff and vff are orthogonal symplectic matrices and dff is a diagonal matrix
       of the form diag(d1,d2,...,dn,d1^-1, d2^-1,...,dn^-1),

    Args:
        S (array): 2N x 2N real symplectic matrix

    Returns:
        (array, : orthogonal symplectic matrix uff
        array,  : diagional matrix
        array)  : orthogonal symplectic matrix vff
    """

    N, m = S.shape

    if N != m:
        return False
    if N % 2 != 0:
        return False
    if not is_symplectic(S):
        return False
    # Changing Basis
    R = (1 / np.sqrt(2)) * np.block(
        [[np.eye(N // 2), 1j * np.eye(N // 2)], [np.eye(N // 2), -1j * np.eye(N // 2)]]
    )
    Sc = R @ S @ np.conjugate(R).T
    # Polar Decomposition
    u1, d1, v1 = np.linalg.svd(Sc)
    Sig = u1 @ np.diag(d1) @ np.conjugate(u1).T
    Unitary = u1 @ v1
    # Blocks of Unitary and Hermitian symplectics
    alpha = Unitary[0 : N // 2, 0 : N // 2]
    beta = Sig[0 : N // 2, N // 2 : N]
    # Bloch-Messiah in this Basis
    u2, d2, v2 = np.linalg.svd(beta)
    sval = np.arcsinh(d2)
    takagibeta = u2 @ sqrtm(np.conjugate(u2).T @ (v2.T))
    uf = np.block(
        [[takagibeta, 0 * takagibeta], [0 * takagibeta, np.conjugate(takagibeta)]]
    )
    vf = np.block(
        [
            [np.conjugate(takagibeta).T @ alpha, 0 * takagibeta],
            [0 * takagibeta, np.conjugate(np.conjugate(takagibeta).T @ alpha)],
        ]
    )
    df = np.block(
        [
            [np.diag(np.cosh(sval)), np.diag(np.sinh(sval))],
            [np.diag(np.sinh(sval)), np.diag(np.cosh(sval))],
        ]
    )
    # Rotating Back to Original Basis
    uff = np.conjugate(R).T @ uf @ R
    vff = np.conjugate(R).T @ vf @ R
    dff = np.conjugate(R).T @ df @ R
    dff = np.real_if_close(dff)
    return uff, dff, vff
