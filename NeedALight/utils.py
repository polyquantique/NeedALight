"""Utility functions"""

import numpy as np


def blocks(T):
    """ Given a (2n x 2n) matrix, outputs the four (n x n) blocks
    
    Args:
        T (array): Even square matrix

    Returns:
        Tss (array): Top left (n x n) block
        Tsi (array): Top right (n x n) block
        Tiss (array): Bottom left (n x n) block
        Tiis (array): Bottom right (n x n) block
    """
    N = len(T)
    #Obtaining blocks
    Tss = T[0 : N // 2, 0 : N // 2]
    Tsi = T[0 : N // 2, N // 2 : N]
    Tiss = T[N // 2 : N, 0 : N // 2]
    Tiis = T[N // 2 : N, N // 2 : N]

    return Tss, Tsi, Tiss, Tiis

def phases(T, ks, ki, l):
    """Removes free propagation phases from Heisenberg Propagator. Assumes nonlinear region from z=-l/2 to z=l/2

    Args:
        T (array): Full Heisenberg Propagator
        ks (float): signal dispersion
        ki (float): idler dispersion
        l (float): total length of crystal

    Returns:
        U (array): Heisenberg Propagator w/o free propagation phases

    """

    N = len(T)
    Uss = (
        np.diag(np.exp(-1j * ks  * l / 2))
        @ T[0 : N // 2, 0 : N // 2]
        @ np.diag(np.exp(-1j * ks  * l / 2))
    )
    Usi = (
        np.diag(np.exp(-1j * ks  * l / 2))
        @ T[0 : N // 2, N // 2 : N]
        @ np.diag(np.exp(1j * ki  * l / 2))
    )
    Uiss = (
        np.diag(np.exp(1j * ki * l / 2))
        @ T[N // 2 : N, 0 : N // 2]
        @ np.diag(np.exp(-1j * ks  * l / 2))
    )
    Uiis = (
        np.diag(np.exp(1j * ki  * l / 2))
        @ T[N // 2 : N, N // 2 : N]
        @ np.diag(np.exp(1j * ki  * l / 2))
    )
    U = np.block([[Uss, Usi], [Uiss, Uiis]])
    return U

def symplectic_prop(T):
    """Given a full Heisenberg propagator, generates a symplectic propagator in the xxpp basis

    Args:
        T (array): Heisenberg propagator

    Returns:
        array: Symplectic propagator in the xxpp basis.
    """
    #Obtaining blocks
    Uss, Usi, Uiss, Uiis = blocks(T)
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

def cov_mat(Nums,Numi,M):
    """Constructs the covariance matrix in the 'xxpp' basis from the second order moments
    
    Args:
        Nums (array): signal photon number matrix
        Numi (array): idler photon number matrix
        M (array): phase-sensitive moment

    Returns:
        V (array): Covariance matrix  
    """
    N_tot = np.block([[Nums, 0 * Nums],[0 * Numi, Numi]])
    M_tot = np.block([[0 * M, M],[M.T, 0 * M]])
    R =(1 / np.sqrt(2)) * np.block([[ np.eye(len(N_tot)), 1j * np.eye(len(N_tot))],[np.eye(len(N_tot)), -1j * np.eye(len(N_tot))]])
    V =2 * np.conj(R).T @ (np.block([[N_tot.T, M_tot], [np.conj(M_tot), N_tot]])+np.eye(len(R))/2) @ R


    return np.real_if_close(V)

def Update_bs(ks, ki, eta_s, eta_i, Ns, Ni, M):
    """Modifies 2nd order moment matrices to account for presence of both additional dispersion
     and absorption via a beamsplitter interaction
    Args:
        ks (array): vector of additional dispersion for signal
        ki (array): vector of additional dispersion for idler
        eta_s (array): vector of absorption/loss parameter for signal
        eta_i (array): vector of absorption/loss parameter for idler
        Ns (array): signal number matrix before dispersive/absorptive region
        Ni (array): idler number matrix before dispersive/absorptive region
        M (array):  signal/idler correlation matrix before dispersive/absorptive region
    Returns:
        Ns_f (array): signal number matrix after dispersive/absorptive region
        Ni_f (array): idler number matrix after dispersive/absorptive region
        M_f (array):  signal/idler correlation matrix after dispersive/absorptive region
    """
    Rs = np.diag(np.exp(1j * ks))
    Ri = np.diag(np.exp(1j * ki))
    e_s = np.diag(np.sqrt(eta_s))
    e_i = np.diag(np.sqrt(eta_i))

    Ns_f = np.conj(Rs) @ e_s @ Ns @ e_s @ Rs
    Ni_f = np.conj(Ri) @ e_i @ Ni @ e_i @ Ri
    M_f = Rs @ e_s @ M @ e_i @ Ri

    return Ns_f, Ni_f, M_f

def Update_spdc(U, Ns, Ni, M):
    """Modifies 2nd order moment matrices to account for a pass in a nonlinear region 
    subjected to spdc interaction

    Args:
        U (array): Heisenberg propagator
        Ns (array): signal number matrix before the nonlinear region
        Ni (array): idler number matrix before the nonlinear region
        M (array):  signal/idler correlation matrix before the nonlinear region
    Returns:
        Ns_f (array,array): signal number matrix after the nonlinear region
        Ni_f (array,array): idler number matrix after the nonlinear region
        M_f (array,array):  signal/idler correlation matrix after the nonlinear region
    """
    # Breaking the propagator into blocks

    N = len(U)
    Uss, Usi, Uiss, Uiis = blocks(U)

    Ns_f = (
        np.conj(Uss) @ Ns @ Uss.T
        + np.conj(Uss) @ np.conj(M) @ Usi.T
        + np.conj(Usi) @ M.T @ Uss.T
        + np.conj(Usi) @ Ni.T @ Usi.T
        + np.conj(Usi) @ Usi.T
    )

    Ni_f = (
        Uiss @ Ns.T @ np.conj(Uiss).T
        + Uiss @ M @ np.conj(Uiis).T
        + Uiis @ np.conj(M).T @ np.conj(Uiss).T
        + Uiis @ Ni @ np.conj(Uiis).T
        + Uiss @ np.conj(Uiss).T
    )

    M_f = (
        Uss @ Ns.T @ np.conj(Uiss).T
        + Uss @ M @ np.conj(Uiis).T
        + Usi @ np.conj(M).T @ np.conj(Uiss).T
        + Usi @ Ni @ np.conj(Uiis).T
        + Uss @ np.conj(Uiss).T
    )

    return Ns_f, Ni_f, M_f