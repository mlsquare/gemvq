import numpy as np

SIG_D3 = 3/24
SIG_D4 = (1/8) * (929/1620)
SIG_E8 = 0

def calculate_mse(x, x_hat):
    return np.mean((x - x_hat) ** 2)


def get_rho_correlated_vectors(rho, size):
    vectors = np.random.normal(0, 1, size=(2, size))
    u, z = vectors[0], vectors[1]
    return u, rho * u + np.sqrt(1-(rho ** 2)) * z


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
