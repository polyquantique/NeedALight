"""Functions to produce Magnus corrections terms from low-gain JSA """

import numpy as np
from cubature import cubature

# pylint: disable=invalid-name
# pylint: disable=too-many-arguments
# pylint: disable=too-many-locals
# pylint: disable=unnecessary-lambda-assignment


def Magnus1(F, w):
    """Generates the first Magnus term

    Args:
        F (function): product of phase-matching function and pump pulse
        w (array): list of frequencies

    Returns:
        (array): J1(w,w'): first order Magnus term
    """

    return F(w, w[:, np.newaxis], w + w[:, np.newaxis])


def Magnus3_Re(F, w, c_factor=0.01**3):
    """Generates the real part of the third order Magnus correction J3

    Args:
        F (function): product of phase-matching function and pump pulse
        w (array): list of frequencies
        c_factor (float): convergence factor for cubature

    Returns:
        (array): J3(w,w'): real part of third order Magnus term
    """
    # This term is broken down into two different contribution.

    # First Contribution

    # We first define the integrand. The factor in front is for convergence.
    J3 = (
        lambda x, y, ws, wi: (c_factor)
        * np.conj(F(x, y, x + y))
        * F(ws, y, ws + y)
        * F(x, wi, wi + x)
        * (np.pi**2 / 3)
    )

    # We map integrands from (-inf,inf) to (-1,1)
    def func(x_array):
        x = x_array[0]
        y = x_array[1]
        return (
            J3(x / (1 - x**2), y / (1 - y**2), w, w[:, np.newaxis])
            * ((1 + x**2) / (1 - x**2) ** 2)
            * ((1 + y**2) / (1 - y**2) ** 2)
        ).flatten()

    # Note that this implicitely uses our previously defined vector of frequencies
    # Ouput must be a vector, not a higher order tensor hence why we flatten

    # Integral properties
    ndim1 = 2  # Number of variables we are integrating over
    fdim1 = len(w) ** 2  # Dimension of the output/J3
    xmin1 = np.array([-1, -1])
    xmax1 = np.array([1, 1])

    # Using cubature
    test1, _err1 = cubature(func, ndim1, fdim1, xmin1, xmax1)

    # Reshaping vector into Matrix and removing the convergence factor
    J3_1 = test1.reshape(len(w), len(w)) / (c_factor)

    # Second Contribution

    # We first define the integrand. We expressed this term differently.
    # We map integrands from (-inf,inf) to (-1,1).
    # Principal values are symmetrized and evaluated from (0,inf)->(0,1).
    def func2(x_array):
        x = x_array[0]
        y = x_array[1]
        z = x_array[2]
        v = x_array[3]

        xx = x / (1 - x**2)
        yy = y / (1 - y**2)
        zz = z / (1 - z)
        vv = v / (1 - v)

        xj = (1 + x**2) / (1 - x**2) ** 2
        yj = (1 + y**2) / (1 - y**2) ** 2
        zj = 1 / (1 - z) ** 2
        vj = 1 / (1 - v) ** 2

        return (
            0.1
            * (c_factor)
            * (
                xj
                * yj
                * zj
                * vj
                * (
                    F(w, yy, w + yy + zz)
                    * (
                        np.conj(F(xx, yy, xx + yy + zz + vv))
                        * F(xx, w[:, np.newaxis], w[:, np.newaxis] + xx + vv)
                        - np.conj(F(xx, yy, xx + yy + zz - vv))
                        * F(xx, w[:, np.newaxis], w[:, np.newaxis] + xx - vv)
                    )
                    - F(w, yy, w + yy - zz)
                    * (
                        np.conj(F(xx, yy, xx + yy - zz + vv))
                        * F(xx, w[:, np.newaxis], w[:, np.newaxis] + xx + vv)
                        - np.conj(F(xx, yy, xx + yy - zz - vv))
                        * F(xx, w[:, np.newaxis], w[:, np.newaxis] + xx - vv)
                    )
                )
                / (zz * vv)
            ).flatten()
        )

    # Note that we include an additional multiplicative factor here for convergence issues.

    # Integral properties
    ndim2 = 4  # Number of variables we are integrating over
    fdim2 = len(w) ** 2  # Dimension of the output/J3
    xmin2 = np.array([-1, -1, 0, 0])
    xmax2 = np.array([1, 1, 1, 1])

    # Using cubature
    test2, _err = cubature(func2, ndim2, fdim2, xmin2, xmax2, vectorized=False)

    # Reshaping vector into Matrix and removing the convergence factors
    J3_2 = test2.reshape(len(w), len(w)) / (c_factor) / 0.1

    return J3_1 + J3_2


def Magnus3_Im(F, w, c_factor=0.01**3):
    """Generates the imaginary part of the third order Magnus correction J3

    Args:
        F (function): product of phase-matching function and pump pulse
        w (array): list of frequencies
        c_factor (float): convergence factor for cubature

    Returns:
        (array): K3(w,w'): imagineray part of third order Magnus term
    """

    # Defining the K3 function for cubature. Extra factor for convergence.

    K3 = lambda x, y, z, ws, wi: (c_factor) * (
        F(ws, y, ws + y + z) * np.conj(F(x, y, x + y + z)) * F(x, wi, wi + x)
        + F(ws, x, ws + x) * F(y, wi, wi + y + z) * np.conj(F(y, x, x + y + z))
    )

    # We map integrands from (-inf,inf) to (-1,1).
    # Principal values are symmetrized and evaluated from (0,inf)->(0,1).
    def func3(x_array):
        x = x_array[0]
        y = x_array[1]
        z = x_array[2]
        return (
            (
                K3(x / (1 - x**2), y / (1 - y**2), z / (1 - z), w, w[:, np.newaxis])
                - K3(x / (1 - x**2), y / (1 - y**2), -z / (1 - z), w, w[:, np.newaxis])
            )
            * ((1 + x**2) / ((1 - x**2) ** 2))
            * ((1 + y**2) / ((1 - y**2) ** 2))
            * (1 / (z * (1 - z)))
        ).flatten()

    # Integral properties
    ndim3 = 3  # Number of variables we are integrating over
    fdim3 = len(w) ** 2  # Dimension of the output/J3
    xmin3 = np.array([-1, -1, 0])
    xmax3 = np.array([1, 1, 1])

    # Using cubature
    test3, _err3 = cubature(
        func3, ndim3, fdim3, xmin3, xmax3, adaptive="h", vectorized=False
    )

    # Reshaping vector into Matrix and removing the convergence factors
    K3 = test3.reshape(len(w), len(w)) / (c_factor)

    return K3*np.pi


def Magnus1CW_data(ws, wi, k_o, k_e, k_pol, p_k, L):
    """Generates first Magnus term for Continous-Wave model from given data set

    Args:
        ws (array): signal frequencies of interest, interpolated from data given a range of wavelengths
        wi (array): corresponding idler frequencies such that wi=wp_avg-ws
        k_o (function): interpolating function generating wavevector given ordinary(signal) frequency
        k_e (function): interpolating function generating wavevector given extra-ordinary(idler) frequency
        k_pol (float): poling wavevector to induce conservation of momenta
        p_k (float): wavevector of CW pump
        L (float): length of interaction region/crystal

    Returns:
        J1(w): first order magnus term.

    """
    J1 = np.sinc(L * (k_o(ws) + k_e(wi) + k_pol - p_k) / (2 * np.pi))

    return J1


def Magnus3CW_data_Re(omega, ws, wi, k_o, k_e, k_pol, p_k, L, c_factork=0.0000005):
    """Generates real part of the third order Magnus term for Continous-Wave model from given data set

    Args:
        omega_o (array): full data set of ordinary frequencies
        omega_o (array): full data set of extra-ordinary frequencies
        ws (array): signal frequencies of interest, interpolated from data given a range of wavelengths
        wi (array): corresponding idler frequencies such that wi=wp_avg-ws
        k_o (function): interpolating function generating wavevector given ordinary(signal) frequency
        k_e (function): interpolating function generating wavevector given extra-ordinary(idler) frequency
        k_pol (float): poling wavevector to induce conservation of momenta
        p_k (float): wavevector of CW pump
        L (float): length of interaction region/crystal
        c_factork (float): convergence factor for cubature functions

    Returns:
        J3(w): real contribution of third order magnus term

    """
    # First part of contribution is simple.
    J1 = np.sinc(L * (k_o(ws) + k_e(wi) + k_pol - p_k) / (2 * np.pi))

    J3_1 = np.conjugate(J1) * J1 * J1 * np.pi**2 / 3

    # Second Part of contribution
    # Integrand for J_3^2 contribution
    def J3_2_int(ws, wi, wq, wq2):
        N = len(ws)
        J3_2 = np.zeros(N)
        for i in range(N):
            if (
                omega[-1] <= ws[i] - wq <= omega[0]
                and omega[-1] <= wi[i] - wq2 <= omega[0]
            ):
                J3_2[i] = (
                    np.sinc(
                        L
                        * (k_o(ws[i] - wq) + k_e(wi[i] - wq2) + k_pol - p_k)
                        / (2 * np.pi)
                    )
                    * np.sinc(
                        L * (k_o(ws[i] - wq) + k_e(wi[i]) + k_pol - p_k) / (2 * np.pi)
                    )
                    * np.sinc(
                        L * (k_o(ws[i]) + k_e(wi[i] - wq2) + k_pol - p_k) / (2 * np.pi)
                    )
                )
            else:
                J3_2[i] = 0
        return J3_2

    # Defining function to fit cubature requirements
    def funcJ3_cuba(x_array, ws, wi):
        x = x_array[0]
        y = x_array[1]
        return (
            c_factork
            * (
                J3_2_int(ws, wi, x / (1 - x), y / (1 - y))
                - J3_2_int(ws, wi, -x / (1 - x), y / (1 - y))
                - J3_2_int(ws, wi, x / (1 - x), -y / (1 - y))
                + J3_2_int(ws, wi, -x / (1 - x), -y / (1 - y))
            )
            / (x * y * (1 - x) * (1 - y))
        )

    ndim = 2  # Number of variables we are integrating over
    fdim = len(ws)  # Dimension of the output
    xmin = np.array([0, 0])
    xmax = np.array([1, 1])
    J3t, _J3Et = cubature(
        funcJ3_cuba,
        ndim,
        fdim,
        xmin,
        xmax,
        args=(ws, wi),
        adaptive="h",
        vectorized=False,
    )
    J3_2 = J3t / c_factork

    return J3_1 + J3_2


def Magnus3CW_data_Im(omega, ws, wi, k_o, k_e, k_pol, p_k, L, c_factork=0.00001):
    """Generates imaginary part of the third order Magnus term for Continous-Wave model from given data set

    Args:
        omega_o (array): full data set of ordinary frequencies
        omega_o (array): full data set of extra-ordinary frequencies
        ws (array): signal frequencies of interest, interpolated from data given a range of wavelengths
        wi (array): corresponding idler frequencies such that wi=wp_avg-ws
        k_o (function): interpolating function generating wavevector given ordinary(signal) frequency
        k_e (function): interpolating function generating wavevector given extra-ordinary(idler) frequency
        k_pol (float): poling wavevector to induce conservation of momenta
        p_k (float): wavevector of CW pump
        L (float): length of interaction region/crystal
        c_factork (float): convergence factor for cubature functions

    Returns:
        K3(w): real contribution of third order magnus term

    """
    # Corrections depends on First order term
    J1 = np.sinc(L * (k_o(ws) + k_e(wi) + k_pol - p_k) / (2 * np.pi))

    # Two contributions from integrals
    def K1_int(ws, wi, wq):
        N = len(ws)
        K1 = np.zeros(N)
        for i in range(N):
            if omega[-1] <= ws[i] - wq <= omega[0]:
                K1[i] = (
                    np.sinc(
                        L * (k_o(ws[i] - wq) + k_e(wi[i]) + k_pol - p_k) / (2 * np.pi)
                    )
                    ** 2
                )
            else:
                K1[i] = 0
        return K1

    def K2_int(ws, wi, wq):
        N = len(ws)
        K2 = np.zeros(N)
        for i in range(N):
            if omega[-1] <= wi[i] - wq <= omega[0]:
                K2[i] = (
                    np.sinc(
                        L * (k_o(ws[i]) + k_e(wi[i] - wq) + k_pol - p_k) / (2 * np.pi)
                    )
                    ** 2
                )
            else:
                K2[i] = 0
        return K2

    # Defining function to fit cubature requirements
    def funcK_cuba(x_array, ws, wi):
        x = x_array[0]
        return (
            c_factork
            * (
                K1_int(ws, wi, x / (1 - x))
                + K2_int(ws, wi, x / (1 - x))
                - K1_int(ws, wi, -x / (1 - x))
                - K2_int(ws, wi, -x / (1 - x))
            )
            / (x * (1 - x))
        )

    ndim = 1  # Number of variables we are integrating over
    fdim = len(ws)  # Dimension of the output
    xmin = np.array([0])
    xmax = np.array([1])

    Kt, _KEt = cubature(
        funcK_cuba,
        ndim,
        fdim,
        xmin,
        xmax,
        args=(ws, wi),
        adaptive="h",
        vectorized=False,
    )

    # The full corrections has an overall factor of J1.
    K3 = np.pi * J1 * Kt / c_factork

    return K3


def Magnus1CW_fit(ws, wi, ks_poly, ki_poly, k_pol, p_k, L):
    """Generates first Magnus term for Continous-Wave model using polynomial fits for dispersion relations

    Args:
        ws (array): signal frequencies of interest, interpolated from data given a range of wavelengths
        wi (array): corresponding idler frequencies such that wi=wp_avg-ws
        ks_poly (poly1d): expansion polynomial for signal dispersion relation
        ki_poly (poly1d): expansion polynomial for idler dispersion relation
        k_pol (float): poling wavevector to induce conservation of momenta
        p_k (float): wavevector of CW pump
        L (float): length of interaction region/crystal

    Returns:
        J1(w): first order magnus term.

    """
    J1 = np.sinc(L * (ks_poly(ws) + ki_poly(wi) + k_pol - p_k) / (2 * np.pi))

    return J1


def Magnus3CW_fit_Re(ws, wi, ks_poly, ki_poly, k_pol, p_k, L, cfit=0.1):
    """Generates real part of third order Magnus term for Continous-Wave model using polynomial fits for dispersion relations

    Args:
        ws (array): signal frequencies of interest, interpolated from data given a range of wavelengths
        wi (array): corresponding idler frequencies such that wi=wp_avg-ws
        ks_poly (poly1d): expansion polynomial for signal dispersion relation
        ki_poly (poly1d): expansion polynomial for idler dispersion relation
        k_pol (float): poling wavevector to induce conservation of momenta
        p_k (float): wavevector of CW pump
        L (float): length of interaction region/crystal
        cfit (float): convergence factor for cubature functions

    Returns:
        J3(w): real part of third order magnus term.

    """

    def PMF(ws, wi):
        return cfit * np.sinc(
            L * (ks_poly(ws) + ki_poly(wi) + k_pol - p_k) / (2 * np.pi)
        )

    # First contribution
    J3_1 = (np.pi**2 / 3) * (PMF(ws, wi) / cfit) ** 3

    # Second contribution
    J3_2_int_fit = (
        lambda x, y, ws, wi: PMF(ws - y, wi - x) * PMF(ws, wi - x) * PMF(ws - y, wi)
    )

    def funcJ3_fit_cuba(x_array, ws, wi):
        x = x_array[0]
        y = x_array[1]
        return (
            cfit
            * (
                J3_2_int_fit(x / (1 - x), y / (1 - y), ws, wi)
                - J3_2_int_fit(-x / (1 - x), y / (1 - y), ws, wi)
                - J3_2_int_fit(x / (1 - x), -y / (1 - y), ws, wi)
                + J3_2_int_fit(-x / (1 - x), -y / (1 - y), ws, wi)
            )
            / (x * y * (1 - x) * (1 - y))
        )

    # Integral properties
    ndim = 2  # Number of variables we are integrating over
    fdim = len(ws)  # Dimension of the output/J3
    xmin = np.array([0, 0])
    xmax = np.array([1, 1])

    # Using cubature
    J3fitT, _J3fitET = cubature(
        funcJ3_fit_cuba,
        ndim,
        fdim,
        xmin,
        xmax,
        args=(ws, wi),
        adaptive="h",
        vectorized=False,
    )

    J3_2_fit = J3fitT / cfit**4

    return J3_1 + J3_2_fit


def Magnus3CW_fit_Im(ws, wi, ks_poly, ki_poly, k_pol, p_k, L, cfit=0.1):
    """Generates imaginary part of third order Magnus term for Continous-Wave model using polynomial fits for dispersion relations

    Args:
        ws (array): signal frequencies of interest, interpolated from data given a range of wavelengths
        wi (array): corresponding idler frequencies such that wi=wp_avg-ws
        ks_poly (poly1d): expansion polynomial for signal dispersion relation
        ki_poly (poly1d): expansion polynomial for idler dispersion relation
        k_pol (float): poling wavevector to induce conservation of momenta
        p_k (float): wavevector of CW pump
        L (float): length of interaction region/crystal
        cfit (float): convergence factor for cubature functions

    Returns:
        K3(w): imaginary part of third order magnus term.

    """

    def PMF(ws, wi):
        return cfit * np.sinc(
            L * (ks_poly(ws) + ki_poly(wi) + k_pol - p_k) / (2 * np.pi)
        )

    K1_int_fit = lambda x, ws, wi: PMF(ws, wi - x) * PMF(ws, wi - x) * PMF(ws, wi)
    K2_int_fit = lambda x, ws, wi: PMF(ws - x, wi) * PMF(ws - x, wi) * PMF(ws, wi)

    def funcK_fit_cuba(x_array, ws, wi):
        x = x_array[0]
        return (
            cfit
            * (
                K1_int_fit(x / (1 - x), ws, wi)
                - K1_int_fit(-x / (1 - x), ws, wi)
                + K2_int_fit(x / (1 - x), ws, wi)
                - K2_int_fit(-x / (1 - x), ws, wi)
            )
            / (x * (1 - x))
        )

    # Integral properties
    ndim = 1  # Number of variables we are integrating over
    fdim = len(ws)  # Dimension of the output/J3
    xmin = np.array([0])
    xmax = np.array([1])

    # Using cubature
    K_fitT, _K_fitET = cubature(
        funcK_fit_cuba,
        ndim,
        fdim,
        xmin,
        xmax,
        args=(ws, wi),
        adaptive="h",
        vectorized=False,
    )

    K_fit = np.pi * K_fitT / cfit**4

    return K_fit
