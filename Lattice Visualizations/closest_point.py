import math

import numpy as np

def f_x(x):
    # Implementation of a custom rounding function for single values or arrays
    if isinstance(x, (int, float)):
        return _f_x(x)
    elif isinstance(x, np.ndarray):
        vectorized_round = np.vectorize(_f_x)
        return vectorized_round(x)
    else:
        raise TypeError("Input must be an int, float, or numpy array.")


def _f_x(x):
    # an implementetion of round that's more accurate to our needs
    if x == 0:
        return 0
    m = math.floor(x)
    if m <= x <= m + 0.5:
        return m
    if m + 0.5 < x < m + 1:
        return m + 1
    if -m-0.5 <= x <= -m:
        return -m
    if -m-1 < x < -m-0.5:
        return -m-1

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


def upscale(u):
    M = np.array([[1, 0, -1], [1/np.sqrt(3), -2/np.sqrt(3), 1/np.sqrt(3)]])
    return np.dot(u, M)


def downscale(x):
    M_t = np.array([[1, 0, -1], [1 / np.sqrt(3), -2 / np.sqrt(3), 1 / np.sqrt(3)]]).T
    return 0.5 * np.dot(x, M_t)


"""
Closest Point Algorithm for the A_2 Lattice.
"""
def closest_point_A2(u):
    x = upscale(u)
    s = np.sum(x)
    x_p = x - (s / len(x)) * np.array([1, 1, 1])
    f_x_p = f_x(x_p)
    delta = int(np.sum(f_x_p))

    distances = x - f_x_p
    sorted_indices = np.argsort(distances)

    if delta == 0:
        return downscale(f_x_p)
    elif delta > 0:
        for i in range(delta):
            f_x_p[sorted_indices[i]] -= 1
    elif delta < 0:
        for i in range(-delta):
            f_x_p[sorted_indices[-i-1]] += 1

    return downscale(f_x_p)



