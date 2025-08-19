import numpy as np
import matplotlib.pyplot as plt
from .nested_lattice_quantizer import NestedLatticeQuantizer as NQuantizer
from .hierarchical_nested_lattice_quantizer import HierarchicalNestedLatticeQuantizer as HQuantizer
from .closest_point import custom_round
from .utils import *
from .closest_point import closest_point_A2, closest_point_Dn


def pad_vector(vec, lattice_dim):
    """
    Pad the vector with zeros to make its length a multiple of lattice_dim.
    
    Parameters:
    -----------
    vec : numpy.ndarray
        Input vector to be padded.
    lattice_dim : int
        The lattice dimension to pad to.
        
    Returns:
    --------
    numpy.ndarray
        Padded vector with length divisible by lattice_dim.
    """
    remainder = len(vec) % lattice_dim
    if remainder != 0:
        padding = lattice_dim - remainder
        vec = np.concatenate([vec, np.zeros(padding)])
    return vec

def calculate_mse_and_overload_for_samples(samples, quantizer):
    """
    Calculate the MSE and overload values for quantized samples.
    
    Parameters:
    -----------
    samples : list
        List of input vectors to be quantized.
    quantizer : object
        Quantizer object (NestedLatticeQuantizer or HierarchicalNestedLatticeQuantizer).
        
    Returns:
    --------
    tuple
        (mse, overload_counts) where mse is the mean squared error and
        overload_counts is a list of T values (overload counts).
    """
    mse = 0
    overload_counts = []
    for sample in samples:
        encoded, T_value = quantizer.encode(sample, with_dither=False)
        decoded = quantizer.decode(encoded, T_value, with_dither=False)
        mse += np.mean((sample - decoded) ** 2)
        overload_counts.append(T_value)
    mse /= len(samples)
    return mse, overload_counts


def find_best_beta(G, Q_nn, q, m, alpha, sig_l, eps):
    """
    Find the best beta parameter for a quantizer by minimizing MSE.
    
    This function performs a grid search over beta values to find the
    optimal scaling parameter that minimizes the rate-distortion function.
    
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
    alpha : float
        Scaling parameter for overload handling.
    sig_l : float
        Lattice parameter.
    eps : float
        Small perturbation parameter.
        
    Returns:
    --------
    tuple
        (optimal_R, optimal_beta) where optimal_R is the optimal rate
        and optimal_beta is the optimal beta parameter.
        
    Notes:
    ------
    The function evaluates the rate-distortion performance for different
    beta values and selects the one that minimizes the distortion-rate
    function f(beta) = MSE / 2^(-2*R).
    """
    d = len(G)
    beta_min  = (1 / q ** m) * np.sqrt(1 / sig_l) * np.sqrt(d / (d + 2))
    betas = beta_min + 0.05 * beta_min * np.arange(0, 40)

    min_f_beta = float("inf")
    optimal_beta = beta_min
    optimal_R = -1
    optimal_dx = -1
    vector_dim = len(G)

    overload_percentage = None

    samples = [np.random.normal(0, 1, size=vector_dim) for _ in range(2000)]
    for idx, beta in enumerate(betas):
        quantizer = HQuantizer(G=G, Q_nn=Q_nn, q=q, beta=beta, alpha=alpha, M=m, eps=eps, dither= np.zeros(d))
        mse, T_values = calculate_mse_and_overload_for_samples(samples, quantizer)

        H_T, T_counts = calculate_t_entropy(T_values, q)
        R = m * np.log2(q) + (H_T / len(G))

        f_beta = mse / (2 ** (-2 * R))
        if f_beta < min_f_beta:
            min_f_beta = f_beta
            optimal_beta = beta
            optimal_dx = idx
            optimal_R = R
            overload_percentage = (1 - (T_counts[0] / sum(T_counts))) * 100

    print(f"optimal beta index is: {optimal_dx}, overload percentage:{overload_percentage:.3f}")
    return optimal_R, optimal_beta


def calculate_inner_product_distortion(G, Q_nn, q, m, beta, alpha, samples, eps, lut=None):
    """
    Calculate the distortion for inner products between different pairs of samples.
    
    This function compares the distortion of inner product estimation between
    nested lattice quantization and hierarchical quantization methods.
    
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
    beta : float
        Scaling parameter for quantization.
    alpha : float
        Scaling parameter for overload handling.
    samples : list
        List of input vectors.
    eps : float
        Small perturbation parameter.
    lut : dict, optional
        Lookup table for hierarchical quantization.
        
    Returns:
    --------
    tuple
        (voronoi_mse, hierarchical_mse) where voronoi_mse is the MSE for
        nested lattice quantization and hierarchical_mse is the MSE for
        hierarchical quantization.
        
    Notes:
    ------
    The function computes inner products between all pairs of samples and
    compares the distortion between the true inner products and the estimated
    inner products using both quantization methods.
    """
    voronoi_errors = []
    hierarchical_errors = []
    lattice_dim = len(G)
    n = len(samples[0])
    nested_quantizer = NQuantizer(G, Q_nn, q=q **m, beta=beta, alpha=alpha, eps=eps, dither=np.zeros(lattice_dim))
    hierarchical_quantizer = HQuantizer(G, Q_nn=Q_nn, q=q, beta=beta, alpha=alpha, M=m, eps=eps, dither=np.zeros(lattice_dim))

    for i in range(len(samples)):
        for j in range(i + 1, len(samples)):
            vec1 = pad_vector(samples[i], lattice_dim)
            vec2 = pad_vector(samples[j], lattice_dim)

            true_inner_product = np.dot(vec1, vec2)

            voronoi_inner_product = 0
            hierarchical_inner_product = 0

            for k in range(0, len(vec1), lattice_dim):
                block1 = vec1[k:k + lattice_dim]
                block2 = vec2[k:k + lattice_dim]

                voronoi_res = calculate_inner_product_for_chunks(block1, block2, nested_quantizer)
                voronoi_inner_product += voronoi_res

                enc_hierarchical_1, t_h1 = hierarchical_quantizer.encode(block1, with_dither=False)
                enc_hierarchical_2, t_h2 = hierarchical_quantizer.encode(block2, with_dither=False)
                c = (2 ** (hierarchical_quantizer.alpha * (t_h1 + t_h2))) * (hierarchical_quantizer.beta ** 2)
                hierarchical_res = calculate_weighted_sum(enc_hierarchical_1, enc_hierarchical_2, lut,
                                                          hierarchical_quantizer.q)
                hierarchical_inner_product += hierarchical_res * c

            voronoi_errors.append((voronoi_inner_product - true_inner_product) ** 2)
            hierarchical_errors.append((hierarchical_inner_product - true_inner_product) ** 2)

    voronoi_mse = np.mean(voronoi_errors) / n
    hierarchical_mse = np.mean(hierarchical_errors) / n
    return voronoi_mse, hierarchical_mse


def calculate_inner_product_for_chunks(block1, block2, quantizer):
    """
    Calculate inner product between two blocks using a quantizer.
    
    Parameters:
    -----------
    block1 : numpy.ndarray
        First block vector.
    block2 : numpy.ndarray
        Second block vector.
    quantizer : object
        Quantizer object.
        
    Returns:
    --------
    float
        Inner product between the quantized blocks.
    """
    enc1, t1 = quantizer.encode(block1, with_dither=False)
    enc2, t2 = quantizer.encode(block2, with_dither=False)
    decoded1 = quantizer.decode(enc1, t1, with_dither=False)
    decoded2 = quantizer.decode(enc2, t2, with_dither=False)
    return np.dot(decoded1, decoded2)


def distortion_rate_theoretical(R):
    """
    Calculate the theoretical distortion-rate function.
    
    Parameters:
    -----------
    R : float
        Rate in bits per dimension.
        
    Returns:
    --------
    float
        Theoretical distortion for the given rate.
        
    Notes:
    ------
    This implements the theoretical distortion-rate function for
    inner product estimation: D(R) = 2 * 2^(-2R) - 2^(-4R)
    """
    return 2 * 2 ** (-2 * R) - 2 ** (-4 * R)


def plot_distortion_rate():
    """
    Plot the distortion-rate function comparing different quantization methods.
    
    This function generates a comprehensive comparison of the distortion-rate
    performance between theoretical bounds, nested lattice quantization,
    and hierarchical quantization for inner product estimation.
    
    Notes:
    ------
    The function:
    1. Sets up parameters for D4 lattice quantization
    2. Generates random test samples
    3. Computes optimal beta for each hierarchical level
    4. Calculates distortion for both quantization methods
    5. Plots the results comparing against theoretical bounds
    """
    q = 4
    m_values = np.arange(1, 5)
    G = get_d4()
    eps = 1e-8 * np.random.normal(0, 1, size=len(G))
    Q_nn = closest_point_Dn
    alpha = 1 / 3
    sig_l = np.sqrt(2) * 0.076602

    vector_dim = 512
    sample_size = 100
    variance = 1
    samples = [np.random.normal(0, variance, size=vector_dim) for _ in range(sample_size)]
    nested_distortions = []
    hierarchical_distortions = []
    theoretical_distortions = []
    R_values = []

    for m in m_values:
        rate, beta = find_best_beta(G, Q_nn, q, m, alpha, sig_l, eps)
        R_values.append(rate)
        gamma = (beta ** 2) * q**(2*m) * sig_l
        print(f"For q=4, M={m}, beta={beta:.4f}, R={rate:.4f}, gamma_0={gamma:.4f}")

        lookup_table = precompute_hq_lut(G, Q_nn, q, m, eps)

        nested_distortion, hierarchical_distortion = calculate_inner_product_distortion(G, Q_nn, q, m, beta, alpha,
                                                                                        samples, eps=eps, lut=lookup_table)
        theoretical_distortion = distortion_rate_theoretical(rate)

        nested_distortions.append(nested_distortion)
        hierarchical_distortions.append(hierarchical_distortion)
        theoretical_distortions.append(theoretical_distortion)

        print(f"Theoretical Distortion: {theoretical_distortion:.6f}")
        print(f"Nested Quantizer Distortion: {nested_distortion:.6f}")
        print(f"Hierarchical Quantizer Distortion: {hierarchical_distortion:.6f}")
        print("-" * 40)

    plt.figure(figsize=(10, 6))
    plt.plot(R_values, theoretical_distortions, label=f"Theoretical $\\Gamma (R)$", linestyle="--", color="black")
    plt.plot(R_values, nested_distortions, label=f"$q^M$ Voronoi Code", marker="o", color="blue")
    plt.plot(R_values, hierarchical_distortions, label="Hierarchical Quantizer", marker="s", color="red")

    plt.yscale("log", base=2)

    plt.xlabel("R = M log2 (q) + H(T)/d")
    plt.ylabel("D (logarithmic scale)")
    plt.title("Distortion-Rate Function for Inner Products")
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == "__main__":
    plot_distortion_rate()