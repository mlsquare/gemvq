import numpy as np

"""
Closest Point Algorithm for the D_n Lattice.
The lattice D_n consists of points where the sum of components is even.
"""


def g_x(x):
    """
    Compute g(x) by rounding the vector x to the nearest integers,
    but flip the rounding for the coordinate farthest from an integer.
    """
    f_x = np.round(x)
    delta = np.abs(x - f_x)
    k = np.argmax(delta)
    g_x_ = f_x.copy()

    if f_x[k] < x[k]:
        g_x_[k] = f_x[k] + 1
    else:
        g_x_[k] = f_x[k] - 1

    return g_x_


def closest_point_Dn(x):
    """
    Find the closest point in the D_n lattice for a given vector x.
    """
    f_x = np.round(x)
    g_x_res = g_x(x)
    return f_x if np.sum(f_x) % 2 == 0 else g_x_res


"""
Closest Point Algorithm for the E_8 Lattice.
The lattice E_8 is constructed from D_8 and a coset.
"""


def closest_point_E8(x):
    """
    Find the closest point in the E_8 lattice for a given vector x.
    """
    y_0 = np.round(x) if np.sum(np.round(x)) % 2 == 0 else g_x(x)

    f_x_shifted = np.round(x - 0.5)
    g_x_shifted = g_x(x - 0.5)

    y_1 = f_x_shifted + 0.5 if np.sum(f_x_shifted) % 2 == 0 else g_x_shifted + 0.5

    if np.linalg.norm(x - y_0) < np.linalg.norm(x - y_1):
        return y_0
    else:
        return y_1
