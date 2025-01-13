
import numpy as np

def custom_round(x):
    if isinstance(x, np.ndarray):
        return np.array([custom_round(val) for val in x])
    else:
        if x > 0:
            return np.floor(x + 0.5) - 1 if (x - np.floor(x)) == 0.5 else np.floor(x + 0.5)
        else:
            return np.ceil(x - 0.5) + 1 if (np.ceil(x) - x) == 0.5 else np.ceil(x - 0.5)


"""
Closest Point Algorithm for the D_n Lattice.
The lattice D_n consists of points where the sum of components is even.
"""


def g_x(x):
    """
    Compute g(x) by rounding the vector x to the nearest integers,
    but flip the rounding for the coordinate farthest from an integer.
    """
    f_x = custom_round(x)
    delta = np.abs(x - f_x)
    k = np.argmax(delta)
    g_x_ = f_x.copy()

    if x[k] >= 0:
        g_x_[k] = f_x[k] + 1 if f_x[k] < x[k] else f_x[k] - 1
    else:
        g_x_[k] = f_x[k] + 1 if f_x[k] <= x[k] else f_x[k] - 1

    return g_x_


def closest_point_Dn(x):
    """
    Find the closest point in the D_n lattice for a given vector x.
    """
    f_x = custom_round(x)
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
    y_0 = custom_round(x) if np.sum(custom_round(x)) % 2 == 0 else g_x(x)

    f_x_shifted = custom_round(x - 0.5)
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
    f_x_p = custom_round(x_p)
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



