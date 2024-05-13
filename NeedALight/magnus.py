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
        (array): J1(w,w'), first order Magnus term
    """

    return F(w, w[:, np.newaxis], w + w[:, np.newaxis])


def Magnus3_Re(F, w):
    """Generates the real part of the third order Magnus correction J3

    Args:
        F (function): product of phase-matching function and pump pulse
        w (array): list of frequencies

    Returns:
        (array): J3(w,w'), first order Magnus term
    """
    # This term is broken down into two different contribution.

    # First Contribution

    # We first define the integrand. The factor in front is for convergence.
    J3 = (
        lambda x, y, ws, wi: (0.01**3)
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
    J3_1 = test1.reshape(len(w), len(w)) / (0.01**3)

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
            * (0.01**3)
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
    J3_2 = test2.reshape(len(w), len(w)) / (0.01**3) / 0.1

    return J3_1 + J3_2


def Magnus3_Im(F, w):
    """Generates the imaginary part of the third order Magnus correction J3

    Args:
        F (function): product of phase-matching function and pump pulse
        w (array): list of frequencies

    Returns:
        (array): J3(w,w'), first order Magnus term
    """

    # Defining the K3 function for cubature. Extra factor for convergence.

    K3 = lambda x, y, z, ws, wi: (0.01**3) * (
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
    K3 = test3.reshape(len(w), len(w)) / (0.01**3)

    return K3
