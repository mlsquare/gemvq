import matplotlib.pyplot as plt
from src.closest_point import custom_round
from utils import *
from closest_point import closest_point_A2, closest_point_Dn
from hierarchical_nested_lattice_quantizer import HierarchicalNestedLatticeQuantizer as HQ
import numpy as np


def generate_rho_correlated_samples(rho, num_samples, vector_dim):
    """Generate pairs of rho-correlated Gaussian samples."""
    cov_matrix = np.array([[1, rho], [rho, 1]])
    L = np.linalg.cholesky(cov_matrix)
    uncorrelated_samples = np.random.normal(size=(num_samples * vector_dim, 2))
    correlated_samples = np.dot(uncorrelated_samples, L.T)
    x_samples = correlated_samples[:, 0].reshape(num_samples, vector_dim)
    y_samples = correlated_samples[:, 1].reshape(num_samples, vector_dim)
    return x_samples, y_samples


def calculate_distortion(x_samples, y_samples, quantizer, lut=None, use_dithers=False):
    """
    Calculate the distortion of the inner product estimation for given samples.
    """
    d = len(quantizer.G)
    n = len(x_samples[0])
    distortions = []
    for x, y in zip(x_samples, y_samples):
        true_inner_product = np.dot(x, y)
        quantized_inner_product = 0
        if use_dithers:
            dither_x = np.random.uniform(-0.5, 0.5, size=x.shape)
            dither_y = np.random.uniform(-0.5, 0.5, size=y.shape)
            x = x + dither_x
            y = y + dither_y

        for k in range(0, len(x), d):
            block1 = x[k:k + d]
            block2 = y[k:k + d]

            enc_x, t_x = quantizer.encode(block1)
            enc_y, t_y = quantizer.encode(block2)
            c = (2 ** (quantizer.alpha * (t_x + t_y))) * (quantizer.beta ** 2)
            res = calculate_weighted_sum(enc_x, enc_y, lut, q=4)
            quantized_inner_product += res * c

        distortion = (quantized_inner_product - true_inner_product) ** 2
        distortions.append(distortion)
    return np.mean(distortions) / n


def plot_distortion_rho():
    G = get_d4()
    d = len(G)
    eps = 1e-8 * np.random.normal(0, 1, size=d)
    Q_nn = closest_point_Dn
    alpha = 1 / 3
    beta = 0.2093
    # sig_l = np.sqrt(2) * 0.076602

    q, m = 4, 2
    rho_values = np.linspace(-0.99, 0.99, 50)
    num_samples = 2000

    vector_dim = 80

    # nested_quantizer = NQuantizer(G, Q_nn, q=q**m, beta=1, alpha=alpha, d=eps)
    hierarchical_quantizer = HQuantizer(G, Q_nn=Q_nn, q=q, beta=beta, alpha=alpha, M=m, d=eps)

    lookup_table = precompute_hq_lut(G, Q_nn, q, m, eps)

    distortions_no_dithers = []
    # distortions_with_dithers = []

    for rho in rho_values:
        x_samples, y_samples = generate_rho_correlated_samples(rho, num_samples, vector_dim)

        distortion_no_dither = calculate_distortion(
            x_samples, y_samples, hierarchical_quantizer, lut=lookup_table, use_dithers=False
        )
        distortions_no_dithers.append(distortion_no_dither)

        # Compute distortion with dithers
        # distortion_with_dither = calculate_distortion(
        #     x_samples, y_samples, nested_quantizer, lut=lookup_table, use_dithers=True
        # )
        # distortions_with_dithers.append(distortion_with_dither)

    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(rho_values, distortions_no_dithers, label="Without Dithers", color="blue", marker="o")
    # plt.plot(rho_values, distortions_with_dithers, label="With Dithers", color="red", marker="s")

    plt.xlabel("$\\rho$")
    plt.ylabel("Distortion (MSE)")
    plt.title("Distortion vs Correlation Coefficient ($\\rho$)")
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == "__main__":
    plot_distortion_rho()

