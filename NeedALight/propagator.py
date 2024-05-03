"""Functions to produce JSA for variable domain configs """

from itertools import product
import numpy as np
from scipy.linalg import expm

# pylint: disable=invalid-name
# pylint: disable=too-many-arguments
# pylint: disable=too-many-locals
# pylint: disable=consider-using-enumerate


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
    J = L @ D @ Vh / dw
    # Number of signal photons
    Nums = np.conj(Usi) @ Usi.T
    Numi = Uiss @ (np.conj(Uiss).T)
    Ns = np.real(np.trace(Nums))
    # Finding K
    K = (np.trace(np.sinh(D) ** 2)) ** 2 / np.trace(np.sinh(D) ** 4)
    return J, Ns, K, M, Nums, Numi


def symplectic_prop(T, vs, vi, vp, l, x):
    """Given a full Heisenberg propagator, generates a symplectic propagator in the xxpp basis

    Args:
        U (array): Heisenberg propagator with free propagation
        vs (float): signal velocity
        vi (float): idler velocity
        vp (float): pump velocity
        l (float): total length of crystal
        x (array): vector of frequencies
    Returns:
        array: Symplectic propagator in the xxpp basis.
    """
    N = len(T)
    # Removing free propagation phases and breaking it into blocks
    U = phases(T, vs, vi, vp, l, x)
    Uss = U[0 : N // 2, 0 : N // 2]
    Usi = U[0 : N // 2, N // 2 : N]
    Uiss = U[N // 2 : N, 0 : N // 2]
    Uiis = U[N // 2 : N, N // 2 : N]
    # First we rewrite propagator in extended basis: (a1,a2...an,b1,..bn,a1\dag... etc)
    diag = np.block([[Uss, 0 * Uss], [0 * Uiis, np.conj(Uiis)]])
    offdiag = np.block([[0 * Usi, Usi], [np.conj(Uiss), 0 * Uiss]])
    U2doubled = np.block([[diag, offdiag], [np.conj(offdiag), np.conj(diag)]])
    # This rotates from Xa Xb Pa Pb to ab a\dag b\dag basis. Need to apply inverse.
    R = (1 / np.sqrt(2)) * np.block(
        [
            [np.eye(len(U2doubled) // 2), 1j * np.eye(len(U2doubled) // 2)],
            [np.eye(len(U2doubled) // 2), -1j * np.eye(len(U2doubled) // 2)],
        ]
    )

    return np.conj(R).T @ U2doubled @ R


def SXPM_prop(vs, vi, vp, y, spm, xpms, xpmi, beta, density, domain, w):
    """Generate Full Heisenberg propagator including SPM and XPM for an unpoled
        crystal.

    Args:
        Np (int): Pump photon number
        vs (float): signal velocity
        vi (float): idler velocity
        vp (float): pump velovity
        y (float): squeezing interaction strength
        spm (float): pump self-phase modulation strength
        xpms (float): signal cross-phase modulation strength
        xpmi (float): idler cross-phase modulation strength
        beta (function): fuctional form of pump pulse at z0(normalized to Np)
        density (function): Fourier transform of the energy density
        w (array): vector of frequency values
    Returns:
        array: Full Heisenberg propagator

    """
    N = len(w)
    P = np.eye(2 * N)
    dz = domain[1] - domain[0]
    dw = w[1] - w[0]
    # these are all the unique values of w+w' that we need
    w2 = np.linspace(2 * w[0], 2 * w[-1], 2 * N - 1)
    # Need to generate the propagator for each z and stitch together
    for z in domain:
        # First generate the SPM matrix
        # This should evaluate beta(z,w+w') at all unique values
        betaz_vec = expm(
            1j * (spm * dw / vp**2) * (z - domain[0]) * density(-w2 + w2[:, np.newaxis])
        ) @ beta(w2)
        # This converts it into matrix form
        betaz_mat = np.zeros((N, N))
        for i in range(N - 1):
            betaz_mat = (
                betaz_mat
                + np.diag(
                    betaz_vec[len(w2) // 2 + 1 + i] * np.ones(len(w) - 1 - i), k=i + 1
                )
                + np.diag(
                    betaz_vec[len(w2) // 2 - 1 - i] * np.ones(len(w) - 1 - i), k=-i - 1
                )
            )

        betaz_mat = np.flipud(
            betaz_mat + np.diag(betaz_vec[len(w2) // 2] * np.ones(N), k=0)
        )

        # Constructing the
        G = np.diag((1 / vs - 1 / vp) * w) + (
            xpms * dw / (2 * np.pi * vs * vp)
        ) * density(-w + w[:, np.newaxis])
        H = np.diag((1 / vi - 1 / vp) * w) + (
            xpmi * dw / (2 * np.pi * vi * vp)
        ) * np.conj(density(-w + w[:, np.newaxis]))
        F = (dw * y / (np.sqrt(2 * np.pi * vs * vi * vp))) * betaz_mat
        Q = np.block([[G, F], [-np.conj(F).T, -np.conj(H).T]])
        P = expm(1j * Q * dz) @ P

    return P


def FtS2(domain, pump, z_list, k):
    """Generates the fourier transform evaluated at (k,t) of the product of the domain configuration
    and pump pulse as functions of (z,t) via looping over k. This should allow for more k-values
    when compared to FtS which uses np.tensordot which doesn't work for large sets of k.

    Args:
        domain (array): array of +/- 1's characterizing the domain configuration
        pump (function): functional form of pump pulse as a function of (t,z) normalized to Np
        z_list (array): list of position values, taken at center point, of crystal positions
        k (array): array momentum values
    Returns:
        (array): S(k+k',t) matrix for a specific time 't'
    """
    S = np.zeros((len(k), len(k)), dtype=np.complex128)
    dz = z_list[1] - z_list[0]

    for i in range(len(k)):
        for l in range(len(k)):
            S[i, l] = np.sum(
                np.exp(-1j * (k[i] + k[l]) * z_list)
                * pump
                * domain
                / np.sqrt(2 * np.pi)
                * dz
            )

    return S


def Total_propK(domain, pump, z_list, k, t, ws, wi):
    """Generates the total Heisenberg propagator

    Args:
        domain (array): array of +/- 1's characterizing the domain configuration
        pump (array): array of pump pulse as a function of (t,z) normalized to Np
        z_list (array): list of position values, taken at center point, where nonlinearity exists
        k (array): array of momentum values
        ws (array): dispersion relation for signal photons (function of momenta)
        wi (array): dispersion relation for idler photon (function of momenta)
    Returns:
        (array): K(k+k',tf), the total Heisenberg propagator at final time
    """
    # Initializing
    K = np.identity(2 * len(k), dtype=np.complex128)
    dk = k[1] - k[0]
    dt = t[1] - t[0]

    # Constructing the diagonal blocks
    Rs = np.diag(-1j * ws)
    Ri = np.diag(1j * wi)

    for i in range(len(t)):
        S = 1j * FtS2(domain, pump[i, :], z_list, k) * dk / np.sqrt(2 * np.pi)
        Q = np.block([[Rs, S], [np.conjugate(S), Ri]])
        K = expm(Q * dt) @ K

    return K


def JSAK(K, dk):
    """Given a total Heisenberg propagator in (k,t) generates the JSA as well as any relevant
       moment matrix and property

    Args:
        K (array): Total Heisenberg propagator
        dk (float): momentum stepsize

    Returns:
        (array, :Joint spectral amplitude
        float,  :Number of signal photons
        float,  :Schmidt number
        array,  :M moment matrix
        array,  :signal number matrix
        array)  :Idler number matrix
    """
    N = len(K)
    Kss = K[0 : N // 2, 0 : N // 2]
    Ksi = K[0 : N // 2, N // 2 : N]
    Kiss = K[N // 2 : N, 0 : N // 2]
    # Constructing the moment matrix
    M = Kss @ (np.conj(Kiss).T)
    # Using SVD of M to construct JSA
    L, s, Vh = np.linalg.svd(M)
    Sig = np.diag(s)
    D = np.arcsinh(2 * Sig) / 2
    J = L @ D @ Vh / dk
    # Number of signal photons
    Nums = np.conj(Ksi) @ Ksi.T
    Numi = Kiss @ (np.conj(Kiss).T)
    Ns = np.real(np.trace(Nums))
    # Finding K
    Schmidt = (np.trace(np.sinh(D) ** 2)) ** 2 / np.trace(np.sinh(D) ** 4)

    return J, Ns, Schmidt, M, Nums, Numi
