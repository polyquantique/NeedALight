"""Functions to produce JSA/moments for Frequency Conversion with variable domain configs """

from itertools import product
import numpy as np
from scipy.linalg import expm, cossin
from NeedALight.utils import blocks, phases 

# pylint: disable=invalid-name
# pylint: disable=too-many-arguments
# pylint: disable=too-many-locals
# pylint: disable=consider-using-enumerate


def Prop_precal_fc(vs, vi, vp, dz, w, pump, n = 4):
    """Generate Heisenberg propagator for given values and pump pulse assuming linear dispersion
    and precalculates all products of n=4 domain configurations

    Args
    --------
        vs (float): signal velocity
        vi (float): idler velocity
        vp (float): pump velovity
        dz (float): length of crystal slice
        w (array): vector of frequency values
        pump (function): fuctional form of pump pulse (normalized to Np)
        n (int): number of segments we wish to precalculate
    Returns
    --------
        prod (array): Heisenberg propagator for all possible orientations of n-segments
        P (array): Heisenberg propagator g(z)=1
        N (array): Heisenberg propagator g(z)=-1
    """
    # Constructing matrix for EoM
    G = np.diag((1 / vs - 1 / vp) * w)
    H = np.diag((1 / vi - 1 / vp) * w)
    dw = np.abs(w[1]-w[0])
    F = dw / (np.sqrt(2 * np.pi * np.abs( vs * vi * vp))) * pump(w - w[:, np.newaxis]).conj()
    Q = np.block([[G, F], [np.conj(F).T, np.conj(H).T]])
    Q2 = np.block(
        [[G, -F], [-np.conj(F).T, np.conj(H).T]]
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

    Args
    --------
        domain (array): vector of +/-1's representing sgn of potential
        P (array): Heisenberg propagator for segment dz with g(z)=1
        N (array): Heisenberg propagator for segment dz with g(z)=-1

    Returns
    --------
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

def JSD(K, dk):
    """Given a total Heisenberg propagator generates the JSD as well as any relevant moment matrix and value

        Args
        --------
            T (array): Total Heisenberg propagator
            dk (float): discretization stepsize

        Returns
        --------
            J (array): Joint spectral distribution
            Eff (array): vector of conversion efficiencies
            S (float): selectivity 
            Us (array): matrix of output signal mode column vectors
            Ui (array): matrix of output idler mode column vectors
            Vs (array): matrix of input signal mode column vectors
            Vi (array): matrix of input idler mode column vectors

    """
    #Obtaining blocks
    Kss, Ksi, Kiss, _Kiis = blocks(K)
    # Using SVD of Top-right block to construct JSD
    L, s, Vh = np.linalg.svd(Ksi)
    Sig = np.diag(s)
    D = np.arcsin(Sig)
    J = (L @ D @ Vh / dk).T 
    # Bloch-messiah decomp
    N = len(K)//2
    U, cs, Vd = cossin(K, p = N, q = N)
    Eff = np.diag(cs[0 : N, N : 2*N]**2)
    S =  np.sum(Eff**2)/np.sum(Eff)**2
    Us, _Temp, _Temp, Ui = blocks(U)
    Vs, _Temp, _Temp, Vi = blocks(np.conj(Vd).T)

    return J, Eff, S, Us, Ui, Vs, Vi

def FPulsed_lin(vs, vi, vp, pump, domain, dz, l, w, rmv=True):
    """Heisenberg propagator, joint spectral distribution, input-output mode, and other values for pulsed frequency conversion assuming linear dispersion

        Args
        --------
            vs (float): signal velocity
            vi (float): idler velocity
            vp (float): pump velocity
            pump (function): fuctional form of pump pulse (normalized to Np)
            domain (array): vector of +/-1's representing sgn of potential
            dz (float): poling/size of domain slices
            l (float): length of nonlinear crystal
            w (vector): array of frequencies
            rmv (string): option to choose whether or not we remove free-propagating phases.



        Returns
        --------
            T (array): Heisenberg propagator (free-phaseless)
            J (array): Joint spectral distribution
            Eff (array): vector of conversion efficiencies
            S (float): selectivity 
            Us (array): matrix of output signal mode column vectors
            Ui (array): matrix of output idler mode column vectors
            Vs (array): matrix of input signal mode column vectors
            Vi (array): matrix of input idler mode column vectors
    
    """
    N = len(w)
    dw = np.abs(w[1]-w[0])

    #Generating the full propagator
    if len(domain) == 1:
        #Unpoled/periodically poled crystal
        prod,P,N = Prop_precal_fc(vs, vi, vp, l, w, pump, n = 1)
        T = P
    else:
        #apperiodically poled crystal
        prod,P,N = Prop_precal_fc(vs, vi, vp, dz, w, pump)
        T = Total_prop(domain, prod, P, N)

    #Removing free-propagating phases
    if rmv:
        ks = (1 / vs - 1 / vp) * w
        ki = (1 / vi - 1 / vp) * w
        T = phases(T, ks, -ki, l) #Phases made for spdc, need -ki for fc

    J, Eff, S, Us, Ui, Vs, Vi = JSD(T,dw)


    return T, J, Eff, S, Us, Ui, Vs, Vi

def FPulsed_arb(ks, ki, kp_w, gamma, w, z_list, domain, Lambda_w):
    """Heisenberg propagator, joint spectral distribution, input-output mode, and other values for pulsed frequency conversion assuming linear dispersion
    
     Args
     --------
            ks (array): signal dispersion relation
            ki (array): idler dispersion relation
            kp_w (array): pump dispersion matrix  kp(w+w') needed in the exponential
            w (array): vector of frequencies 
            gamma (float): interaction strength parameter
            z_list (array): discretized interaction region
            domain (array): poling configuration
            Lambda_w (function): Pump envelope function in frequency space

            
     Returns
     --------
            T (array): Heisenberg propagator (free-phaseless)
            J (array): Joint spectral distribution
            Eff (array): vector of conversion efficiencies
            S (float): selectivity 
            Us (array): matrix of output signal mode column vectors
            Ui (array): matrix of output idler mode column vectors
            Vs (array): matrix of input signal mode column vectors
            Vi (array): matrix of input idler mode column vectors
    
    """

    dz = z_list[1] - z_list[0]
    dw = np.abs(w[1]-w[0])
    # Initializing
    T = np.identity(2 * len(ks), dtype=np.complex128)

    # Constructing the diagonal blocks
    Rs = np.zeros([len(ks),len(ks)])

    # Note that for the pump, we explicitely remove the linear free-propagating phases here (For plotting purposes later)
    for i in range(len(z_list)):
        F = (
             gamma
            * Lambda_w(w-w[:,np.newaxis]).conj()
            * np.exp(-1j * (kp_w + ks[:, np.newaxis] - ki) * z_list[i])  
            * domain[i]
            * dw
        )
        Q = np.block([[Rs, F], [np.conjugate(F).T, Rs]]) 
        T = expm(1j*Q * dz) @ T

    J, Eff, S, Us, Ui, Vs, Vi = JSD(T,dw)  

    return T, J, Eff, S, Us, Ui, Vs, Vi

def Fcw(ks, ki, gamma, L):
    """Efficiency for continuous-wave frequency conversion

    Args
    --------
        ks (array): vector of dispersion relation for signal to any order
        ki (array): vector of dispersion relation for idler to any order
        gamma (float): interaction strength
        L (float): crystal length

    Returns
    --------
         T (array): Heisenberg propagator (free-phaseless)
         J (array): Joint spectral distribution
         Eff (array): vector of conversion efficiencies
    """
    #Because cossin outputs the sin/cos values in order, we need to eval the efficiency and JSD point-by-point

    JSD = np.zeros(len(ks))
    Eff = np.zeros(len(ks))

    for i in range(len(ks)):
        # Constructing the CW propagator
        Rs = 1j * ks[i]
        Ri = 1j * ki[i]
        F = 1j * gamma 
        Q = np.asarray([[Rs, F], [F, Ri]])

        K = expm(Q * L)

        U, cs, Vd = cossin(K, p = 1, q = 1)
        Eff[i] = cs[0, 1]**2

    return Eff
