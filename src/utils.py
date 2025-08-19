"""
Utility Functions Module

This module provides utility functions and lattice generator matrices for the
LatticeQuant library. It contains common functions used across all modules
for lattice operations, analysis, and computation.

Lattice Generators:
    - get_d4(), get_a2(), get_e8(): High-dimensional lattice matrices
    - get_z2(), get_z3(): Integer lattice matrices
    - get_d2(), get_d3(): D-series lattice matrices

Analysis Functions:
    - calculate_mse(): Mean squared error calculation
    - calculate_t_entropy(): Entropy calculation for overload handling
    - precompute_hq_lut(): Lookup table generation
    - calculate_weighted_sum(): Weighted inner product estimation

Constants:
    - SIG_D3, SIG_D4, SIG_E8: Lattice-specific parameters

This module serves as a central repository for common functionality used
by the quantizers, applications, and adaptive modules.
"""

import numpy as np
from .quantizers.hierarchical_nested_lattice_quantizer import HierarchicalNestedLatticeQuantizer as HQuantizer

# Constants for lattice parameters
SIG_D3 = 3/24
SIG_D4 = np.sqrt(2) * 0.076602
SIG_E8 = (1/8) * (929/1620)


def calculate_mse(x, x_hat):
    """
    Calculate the Mean Squared Error (MSE) between two vectors.
    
    Parameters:
    -----------
    x : numpy.ndarray
        Original vector.
    x_hat : numpy.ndarray
        Reconstructed/estimated vector.
        
    Returns:
    --------
    float
        The mean squared error between x and x_hat.
    """
    return np.mean((x - x_hat) ** 2)


def get_z2():
    """
    Get the generator matrix for the Z^2 lattice.
    
    The Z^2 lattice is the standard integer lattice in 2D.
    
    Returns:
    --------
    numpy.ndarray
        2x2 identity matrix representing the Z^2 lattice.
    """
    return np.array([
        [1, 0],
        [0, 1]
    ])


def get_z3():
    """
    Get the generator matrix for the Z^3 lattice.
    
    The Z^3 lattice is the standard integer lattice in 3D.
    
    Returns:
    --------
    numpy.ndarray
        3x3 identity matrix representing the Z^3 lattice.
    """
    return np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1]
    ])


def get_a2():
    """
    Get the generator matrix for the A_2 lattice.
    
    The A_2 lattice is a 2D hexagonal lattice with excellent
    packing properties.
    
    Returns:
    --------
    numpy.ndarray
        2x2 generator matrix for the A_2 lattice.
        
    Notes:
    ------
    The A_2 lattice has a hexagonal structure and is optimal
    for 2D sphere packing.
    """
    return np.array([
        [1, 0],
        [0.5, np.sqrt(3)/2]
    ]).T


def get_d2():
    """
    Get the generator matrix for the D_2 lattice.
    
    The D_2 lattice is a 2D lattice where the sum of coordinates
    must be even.
    
    Returns:
    --------
    numpy.ndarray
        2x2 generator matrix for the D_2 lattice.
    """
    return np.array([
        [1, -1],
        [2, 0]
    ]).T


def get_d3():
    """
    Get the generator matrix for the D_3 lattice.
    
    The D_3 lattice is a 3D lattice where the sum of coordinates
    must be even.
    
    Returns:
    --------
    numpy.ndarray
        3x3 generator matrix for the D_3 lattice.
    """
    return np.array([
        [1, -1, 0],
        [0, 1, -1],
        [0, 1, 1]
    ]).T

def get_d4():
    """
    Get the generator matrix for the D_4 lattice.
    
    The D_4 lattice is a 4D lattice where the sum of coordinates
    must be even. It has excellent packing properties in 4D.
    
    Returns:
    --------
    numpy.ndarray
        4x4 generator matrix for the D_4 lattice.
        
    Notes:
    ------
    The D_4 lattice is optimal for 4D sphere packing and is
    commonly used in lattice quantization applications.
    """
    return np.array([
        [-1, -1, 0, 0],
        [1, -1, 0, 0],
        [0, 1, -1, 0],
        [0, 0, 1, -1]
    ]).T


def get_e8():
    """
    Get the generator matrix for the E_8 lattice.
    
    The E_8 lattice is an 8D lattice with exceptional properties.
    It is optimal for 8D sphere packing and has many applications
    in coding theory and quantization.
    
    Returns:
    --------
    numpy.ndarray
        8x8 generator matrix for the E_8 lattice.
        
    Notes:
    ------
    The E_8 lattice is constructed from the D_8 lattice and a coset.
    It has the highest known packing density in 8D.
    """
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


def calculate_t_entropy(T_values, q):
    """
    Calculate the entropy of the T values (overload counts) for rate calculation.
    
    Parameters:
    -----------
    T_values : list or numpy.ndarray
        List of T values (overload counts) from quantization.
    q : int
        The quantization parameter.
        
    Returns:
    --------
    tuple
        (H_T, T_counts) where H_T is the entropy in bits and T_counts
        is the count of each T value.
    """
    T_counts = np.bincount(T_values, minlength=q ** 2)
    T_probs = T_counts / np.sum(T_counts)
    H_T = -sum(p * np.log2(p) for p in T_probs if p > 0)
    return H_T, T_counts


def precompute_hq_lut(G, Q_nn, q, m, eps):
    """
    Precompute a lookup table for hierarchical quantization.
    
    This function creates a lookup table that maps pairs of encoding vectors
    to their inner products, which can be used for efficient inner product
    estimation in hierarchical quantization.
    
    Parameters:
    -----------
    G : numpy.ndarray
        Generator matrix for the lattice.
    Q_nn : function
        Closest point function for the lattice.
    q : int
        Quantization parameter.
    m : int
        Number of hierarchical levels.
    eps : float
        Small perturbation parameter.
        
    Returns:
    --------
    dict
        Lookup table mapping (enc1, enc2) tuples to inner product values.
        
    Notes:
    ------
    The lookup table is used to accelerate inner product calculations
    in hierarchical quantization by precomputing all possible inner
    products between quantized vectors.
    """
    hq = HQuantizer(G=G, Q_nn=Q_nn, q=q, beta=1, alpha=1, M=m, eps=eps, dither=np.zeros(len(G)))
    codebook = hq.create_q_codebook(with_dither=False)
    lookup_table = {}
    for enc1, lattice_point1 in codebook.items():
        for enc2, lattice_point2 in codebook.items():
            inner_product = np.dot(lattice_point1, lattice_point2)
            lookup_table[(enc1, enc2)] = inner_product
    return lookup_table


def calculate_weighted_sum(a_list, b_list, lut, q):
    """
    Calculate the weighted sum for given encoding vectors using a lookup table.
    
    This function computes the weighted sum of inner products between
    pairs of encoding vectors, which is used in hierarchical quantization
    for inner product estimation.
    
    Parameters:
    -----------
    a_list : list
        List of encoding vectors for the first vector.
    b_list : list
        List of encoding vectors for the second vector.
    lut : dict
        Lookup table mapping (enc1, enc2) to inner product values.
    q : int
        Quantization parameter.
        
    Returns:
    --------
    float
        The weighted sum of inner products.
        
    Raises:
    -------
    ValueError
        If a_list and b_list have different lengths.
        
    Notes:
    ------
    The weighted sum is computed as:
    sum_{i,j} q^(i+j) * lut[(a_list[i], b_list[j])]
    where the weights q^(i+j) correspond to the hierarchical levels.
    """
    k = len(a_list)
    if len(b_list) != k:
        raise ValueError("a_list and b_list must have the same length.")

    total_sum = 0
    for i in range(k):
        for j in range(k):
            weight = q ** (i + j)
            total_sum += weight * lut[(tuple(a_list[i]), tuple(b_list[j]))]

    return total_sum
