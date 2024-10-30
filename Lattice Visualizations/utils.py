import numpy as np


def calculate_mse(x, decoded_x):
    """
    Calculate the Mean Squared Error between the original vector x and the decoded vector decoded_x.
    """
    return np.mean((x - decoded_x) ** 2)

def get_z2():
    return np.array([
        [1, 0],
        [0, 1]
    ])

def get_d3():
    return np.array([
        [1, -1, 0],
        [0, 1, -1],
        [0, 1, 1]
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
