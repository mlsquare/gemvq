import numpy as np
import matplotlib.pyplot as plt
from nested_lattice_quantizer import (NestedLatticeQuantizer as NQuantizer,
                                      HierarchicalNestedLatticeQuantizer as HQuantizer)
from utils import get_a2, get_d4
from closest_point import closest_point_A2, closest_point_Dn


def pad_vector(vec, lattice_dim):
    """Pad the vector with zeros to make its length a multiple of lattice_dim."""
    remainder = len(vec) % lattice_dim
    if remainder != 0:
        padding = lattice_dim - remainder
        vec = np.concatenate([vec, np.zeros(padding)])
    return vec


def precompute_hq_lut(G, Q_nn, q):
    """Precompute a lookup table for hierarchical quantization."""
    hq = HQuantizer(G=G, Q_nn=Q_nn, q=q, beta=1, alpha=1/3)
    codebook = hq.create_q_codebook()
    lookup_table = {}
    for enc1, lattice_point1 in codebook.items():
        for enc2, lattice_point2 in codebook.items():
            inner_product = np.dot(lattice_point1, lattice_point2)
            lookup_table[(enc1, enc2)] = inner_product
    return lookup_table


def calculate_inner_product_distortion(quantizer, samples, method="voronoi code", lut=None):
    """Calculate the distortion for inner products between different pairs of samples."""
    errors = []
    lattice_dim = quantizer.G.shape[0]
    n = len(samples[0])

    for i in range(len(samples)):
        for j in range(i + 1, len(samples)):
            vec1 = pad_vector(samples[i], lattice_dim)
            vec2 = pad_vector(samples[j], lattice_dim)

            true_inner_product = np.dot(vec1, vec2)

            quantized_inner_product = 0
            for k in range(0, len(vec1), lattice_dim):
                block1 = vec1[k:k + lattice_dim]
                block2 = vec2[k:k + lattice_dim]

                if method == "voronoi code":
                    enc1, t1 = quantizer.encode(block1)
                    enc2, t2 = quantizer.encode(block2)
                    decoded1 = quantizer.decode(enc1, t1)
                    decoded2 = quantizer.decode(enc2, t2)
                    res = np.dot(decoded1, decoded2)
                    quantized_inner_product += res
                elif method == "hierarchical":
                    enc1, t1 = quantizer.encode(block1)
                    enc2, t2 = quantizer.encode(block2)

                    a1, a2 = tuple(enc1[0].astype(int)), tuple(enc1[1].astype(int))
                    b1, b2 = tuple(enc2[0].astype(int)), tuple(enc2[1].astype(int))

                    c = (2 ** (t1 + t2))
                    res = lut[(a1, b1)] + quantizer.q * (lut[(a1, b2)] + lut[(a2, b1)]) + quantizer.q**2 * lut[(a2, b2)]

                    assert np.dot(quantizer.decode(enc1, t1), quantizer.decode(enc2, t2)) == res * c * (quantizer.beta ** 2)
                    quantized_inner_product += res * c * (quantizer.beta ** 2)
                else:
                    raise ValueError("Unsupported method. Use 'nested' or 'hierarchical'.")

            errors.append((quantized_inner_product - true_inner_product) ** 2)

    return np.mean(errors) / n


def distortion_rate_theoretical(R):
    return 2 * 2 ** (-2 * R) - 2 ** (-4 * R)


def calculate_best_beta(quantizer, samples):
    """Find the best beta minimizing the MSE for the quantizer."""
    beta_values = np.linspace(0.05, 1.0, 20)
    best_beta = None
    min_mse = float("inf")

    for beta in beta_values:
        quantizer.beta = beta
        mse = np.mean([calculate_inner_product_distortion(quantizer, samples, method="voronoi code")])
        if mse < min_mse:
            min_mse = mse
            best_beta = beta

    return best_beta


def main():
    q = 4
    m_values = np.arange(2, 6)
    G = get_d4()
    vector_dim = 128
    sample_size = 20
    variance = 1
    samples = [np.random.normal(0, variance, size=vector_dim) for _ in range(sample_size)]

    nested_distortions = []
    hierarchical_distortions = []
    theoretical_distortions = []
    R_values = []

    for m in m_values:
        print(f"Calculating q={q}, m={m}...")
        R = m * np.log2(q)
        R_values.append(R)

        nested_quantizer = NQuantizer(G, closest_point_Dn, q=q ** m, beta=0.1, alpha=1/3)
        hierarchical_quantizer = HQuantizer(G, Q_nn=closest_point_Dn, q=q, beta=0.1, alpha=1/3, M=m)

        lookup_table = precompute_hq_lut(G, closest_point_Dn, q)

        nested_distortion = calculate_inner_product_distortion(nested_quantizer, samples, method="voronoi code")
        hierarchical_distortion = calculate_inner_product_distortion(hierarchical_quantizer, samples,
                                                                     method="hierarchical", lut=lookup_table)
        theoretical_distortion = distortion_rate_theoretical(R)

        nested_distortions.append(nested_distortion)
        hierarchical_distortions.append(hierarchical_distortion)
        theoretical_distortions.append(theoretical_distortion)

        print(f"q={q}, R={R:.2f}")
        print(f"Theoretical Distortion: {theoretical_distortion:.6f}")
        print(f"Nested Quantizer Distortion: {nested_distortion:.6f}")
        print(f"Hierarchical Quantizer Distortion: {hierarchical_distortion:.6f}")
        print("-" * 40)

    plt.figure(figsize=(10, 6))
    plt.plot(R_values, theoretical_distortions, label="Theoretical D(R)", linestyle="--", color="black")
    plt.plot(R_values, nested_distortions, label=f"$q^M$ Voronoi Code", marker="o", color="blue")
    plt.plot(R_values, hierarchical_distortions, label="Hierarchical Quantizer", marker="s", color="red")
    plt.xlabel("Rate (R)")
    plt.ylabel("Distortion (D)")
    plt.title("Distortion-Rate Comparison for Inner Products")
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == "__main__":
    main()