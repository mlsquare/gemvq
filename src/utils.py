import numpy as np
from nested_lattice_quantizer import HierarchicalNestedLatticeQuantizer as HQuantizer

SIG_D3 = 3/24
SIG_D4 = (1/8) * (929/1620)
SIG_E8 = 0

def calculate_mse(x, x_hat):
    return np.mean((x - x_hat) ** 2)


def get_z2():
    return np.array([
        [1, 0],
        [0, 1]
    ])


def get_z3():
    return np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ])


def get_a2():
    return np.array([
        [1, 0],
        [0.5, np.sqrt(3)/2]
    ]).T


def get_d2():
    return np.array([
        [1, -1],
        [2, 0]
    ]).T


def get_d3():
    return np.array([
        [1, -1, 0],
        [0, 1, -1],
        [0, 1, 1]
    ]).T

def get_d4():
    return np.array([
        [-1, -1, 0, 0],
        [1, -1, 0, 0],
        [0, 1, -1, 0],
        [0, 0, 1, -1]
    ]).T


def get_e8():
    return np.array([
        [2,  0,  0,  0,  0,  0,  0,  0],
        [-1,  1,  0,  0,  0,  0,  0,  0],
        [0,  -1,  1,  0,  0,  0,  0,  0],
        [0,  0,  -1,  1,  0,  0,  0,  0],
        [0,  0,  0,  -1,  1,  0,  0,  0],
        [0,  0,  0,  0,  -1,  1,  0,  0],
        [0,  0,  0,  0,  0,  -1,  1,  0],
        [0.5,  0.5,  0.5,  0.5,  0.5,  0.5,  0.5,  0.5]
    ]).T


def precompute_hq_lut(G, Q_nn, q, m, eps):
    """Precompute a lookup table for hierarchical quantization."""
    hq = HQuantizer(G=G, Q_nn=Q_nn, q=q, beta=1, alpha=1, M=m, d=eps)
    codebook = hq.create_q_codebook()
    lookup_table = {}
    for enc1, lattice_point1 in codebook.items():
        for enc2, lattice_point2 in codebook.items():
            inner_product = np.dot(lattice_point1, lattice_point2)
            lookup_table[(enc1, enc2)] = inner_product
    return lookup_table


def calculate_weighted_sum(a_list, b_list, lut, q):
    """Calculate the weighted sum for given encoding vectors a_list, b_list, LUT, and scalar q."""
    k = len(a_list)
    if len(b_list) != k:
        raise ValueError("a_list and b_list must have the same length.")

    total_sum = 0
    for i in range(k):
        for j in range(k):
            weight = q ** (i + j)
            total_sum += weight * lut[(tuple(a_list[i]), tuple(b_list[j]))]

    return total_sum
