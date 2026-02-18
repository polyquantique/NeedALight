"""Functions to produce JSA/moments for SPDC with variable domain configs """

from itertools import product
import numpy as np
from scipy.linalg import expm
from NeedALight.utils import phases 

# pylint: disable=invalid-name
# pylint: disable=too-many-arguments
# pylint: disable=too-many-locals
# pylint: disable=consider-using-enumerate


def Prop_precal(vs, vi, vp, dz, w, pump, n = 4):
    """Generate Heisenberg propagator for given values and pump pulse assuming linear dispersion
    and precalculates all products of n=4 domain configurations

    Args:
        vs (float): signal velocity
        vi (float): idler velocity
        vp (float): pump velovity
        dz (float): length of crystal slice
        w (array): vector of frequency values
        pump (function): fuctional form of pump pulse (normalized to Np)
        n (int): number of segments we wish to precalculate
    Returns:
        prod (array): Heisenberg propagator for all possible orientations of n-segments
        P (array): Heisenberg propagator g(z)=1
        N (array): Heisenberg propagator g(z)=-1
    """
    # Constructing matrix for EoM
    G = np.diag((1 / vs - 1 / vp) * w)
    H = np.diag((1 / vi - 1 / vp) * w)
    dw = np.abs(w[1]-w[0])
    F = dw / (np.sqrt(2 * np.pi * np.abs( vs * vi * vp))) * pump(w + w[:, np.newaxis])
    Q = np.block([[G, F], [-np.conj(F).T, -np.conj(H).T]])
    Q2 = np.block(
        [[G, -F], [np.conj(F).T, -np.conj(H).T]]
    )  # sgn(g(z)) only changes sgn(F)
    P = expm(1j * Q * dz)
    N = expm(1j * Q2 * dz)
    # Finding the products for domain lengths of n-segments
    prod = np.ones((2**n, len(P), len(P)), dtype=np.complex128)
    for i, config in enumerate(product((P, N), repeat=n)):
        T = np.eye(len(P), dtype=np.complex128)
        for j in range(n):
            T = config[j] @ T
        prod[i] = T
    return prod, P, N

def Total_prop(domain, prod, P, N):
    """Heisenberg Propagator for aperiodically polled domain, assuming linear dispersion

    Args:
        domain (array): vector of +/-1's representing sgn of potential
        P (array): Heisenberg propagator for segment dz with g(z)=1
        N (array): Heisenberg propagator for segment dz with g(z)=-1

    Returns:
        T (array): Heisenberg propagator over poled crystal
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

def JSA(K, dk):
    """Given a total Heisenberg propagator generates the JSA as well as any relevant
       moment matrix and value

    Args:
        T (array): Total Heisenberg propagator
        dk (float): discretization stepsize

    Returns:
        J (array): Joint spectral amplitude
        Ns (float): Number of signal photons
        Schmidt (float): Schmidt number
        M (array): M moment matrix
        Nums (array): Signal number matrix
        Numi (array): Idler number matrix
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

def SPulsed_lin(vs, vi, vp, pump, domain, dz, l, w, rmv=True):
    """Heisenberg propagator, joint spectral amplitude, 2nd order moments, and other values for pulsed SPDC assuming linear dispersion

        Args:
            vs (float): signal velocity
            vi (float): idler velocity
            vp (float): pump velocity
            pump (function): fuctional form of pump pulse (normalized to Np)
            domain (array): vector of +/-1's representing sgn of potential
            dz (float): poling/size of domain slices
            l (float): length of nonlinear crystal
            w (vector): array of frequencies
            rmv (string): option to choose whether or not we remove free-propagating phases.



        Returns:
            T (array): Heisenberg propagator (free-phaseless)
            J (array): Joint spectral amplitude
            Ns (float): Number of signal photons
            Schmidt (float): Schmidt number (K)
            M (array): M moment matrix
            Nums (array): Signal number matrix
            Numi (array): Idler number matrix
    
    """
    N = len(w)
    dw = np.abs(w[1]-w[0])

    #Generating the full propagator
    if len(domain) == 1:
        #Unpoled/periodically poled crystal
        prod,P,N = Prop_precal(vs, vi, vp, l, w, pump, n = 1)
        T = P
    else:
        #apperiodically poled crystal
        prod,P,N = Prop_precal(vs, vi, vp, dz, w, pump)
        T = Total_prop(domain, prod, P, N)

    #Removing free-propagating phases
    if rmv:
        ks = (1 / vs - 1 / vp) * w
        ki = (1 / vi - 1 / vp) * w
        T = phases(T, ks, ki, l)

    J, Ns, Schmidt, M, Nums, Numi = JSA(T,dw)


    return T, J, Ns, Schmidt, M, Nums, Numi

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
        P (array): Full Heisenberg propagator

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

        # Constructing the diagonal blocks
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

def SPulsed_arb(ks, ki, kp_w, gamma, dw, z_list, domain, Lambda_w):
    """Heisenberg propagator, joint spectral amplitude, 2nd order moments, and other values for pulsed SPDC with arbitrary dispersion
    
     Args:
        ks (array): signal dispersion relation
        ki (array): idler dispersion relation
        kp_w (array): pump dispersion matrix  kp(w+w') needed in the exponential
        dw (float): frequency discretization step 
        gamma (float): interaction strength parameter
        z_list (array): discretized interaction region
        domain (array): poling configuration
        Lambda_w (array): Pump envelope in frequency space evaluated at (w+w')

    Returns:
        T (array): Heisenberg propagator (free-phaseless)
        J (array): Joint spectral amplitude
        Ns (float): Number of signal photons
        Schmidt (float): Schmidt number (K)
        M (array): M moment matrix
        Nums (array): Signal number matrix
        Numi (array): Idler number matrix
    """

    dz = z_list[1] - z_list[0]
    # Initializing
    T = np.identity(2 * len(ks), dtype=np.complex128)

    # Constructing the diagonal blocks
    Rs = np.zeros([len(ks),len(ks)])

    # Note that for the pump, we explicitely remove the linear free-propagating phases here (For plotting purposes later)
    for i in range(len(z_list)):
        F = (
             1j*gamma
            * Lambda_w
            * np.exp(1j * (kp_w - ks[:, np.newaxis] - ki) * z_list[i])   
            * domain[i]
            * dw
        )
        Q = np.block([[Rs, F], [np.conjugate(F).T, Rs]])
        T = expm(Q * dz) @ T

    J, Ns, Schmidt, M, Nums, Numi = JSA(T,dw)    

    return T, J, Ns, Schmidt, M, Nums, Numi

def FtS(domain, pump, z_list, k):
    """Generates the fourier transform evaluated at (k,t) of the product of the domain configuration
    and pump pulse as functions of (z,t) via looping over k.

    Args:
        domain (array): array of +/- 1's characterizing the domain configuration
        pump (function): functional form of pump pulse as a function of (t,z) normalized to Np
        z_list (array): list of position values, taken at center point, of crystal positions
        k (array): array momentum values
    Returns:
        S(k+k',t) (array): matrix for a specific time 't'
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
        K(k+k',tf) (array): total Heisenberg propagator at final time
    """
    # Initializing
    K = np.identity(2 * len(k), dtype=np.complex128)
    dk = k[1] - k[0]
    dt = t[1] - t[0]

    # Constructing the diagonal blocks
    Rs = np.diag(-1j * ws)
    Ri = np.diag(1j * wi)

    for i in range(len(t)):
        S = 1j * FtS(domain, pump[i, :], z_list, k) * dk / np.sqrt(2 * np.pi)
        Q = np.block([[Rs, S], [np.conjugate(S), Ri]])
        K = expm(Q * dt) @ K

    return K




