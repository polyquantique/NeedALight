"""Utility functions"""

import numpy as np



def phases(T, ks, ki, l):
    """Removes free propagation phases from Heisenberg Propagator

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

#Do I even use this?
def symplectic_prop(T):
    """Given a full Heisenberg propagator, generates a symplectic propagator in the xxpp basis

    Args:
        T (array): Heisenberg propagator

    Returns:
        array: Symplectic propagator in the xxpp basis.
    """
    N = len(T)
    #Obtaining blocks
    Uss = T[0 : N // 2, 0 : N // 2]
    Usi = T[0 : N // 2, N // 2 : N]
    Uiss = T[N // 2 : N, 0 : N // 2]
    Uiis = T[N // 2 : N, N // 2 : N]
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