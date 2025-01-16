import numpy as np
import matplotlib.pyplot as plt
from nested_lattice_quantizer import (NestedLatticeQuantizer as NQuantizer,
                                      HierarchicalNestedLatticeQuantizer as HQuantizer)
from src.closest_point import custom_round
from utils import get_a2, get_d4
from closest_point import closest_point_A2, closest_point_Dn


def pad_vector(vec, lattice_dim):
    """Pad the vector with zeros to make its length a multiple of lattice_dim."""
    remainder = len(vec) % lattice_dim
    if remainder != 0:
        padding = lattice_dim - remainder
        vec = np.concatenate([vec, np.zeros(padding)])
    return vec

def calculate_mse_and_overload_for_samples(samples, quantizer):
    """Calculate the MSE and overload values for quantized samples."""
    mse = 0
    overload_counts = []
    for sample in samples:
        encoded, T_value = quantizer.encode(sample)
        decoded = quantizer.decode(encoded, T_value)
        mse += np.mean((sample - decoded) ** 2)
        overload_counts.append(T_value)
    mse /= len(samples)
    return mse, overload_counts


def find_best_beta(G, Q_nn, q, m, alpha, sig_l):
    """Find the best beta for a quantizer by minimizing MSE."""
    d = len(G)
    beta_min  = (1 / q ** m) * np.sqrt(1 / sig_l) * np.sqrt(d / (d + 2))
    betas = beta_min + 0.05 * beta_min * np.arange(0, 40)
    min_f_beta = float("inf")
    optimal_beta = beta_min
    vector_dim = len(G)
    samples = [np.random.normal(0, 1, size=vector_dim) for _ in range(1000)]
    for beta in betas:
        quantizer = HQuantizer(G=G, Q_nn=Q_nn, q=q, beta=beta, alpha=alpha, M=m)
        mse, T_values = calculate_mse_and_overload_for_samples(samples, quantizer)

        T_counts = np.bincount(T_values, minlength=q ** 2)
        T_probs = T_counts / np.sum(T_counts)
        H_T = -sum(p * np.log2(p) for p in T_probs if p > 0)
        R = m * np.log2(q) + (H_T / len(G))

        f_beta = mse / (2 ** (-2 * R))
        if f_beta < min_f_beta:
            min_f_beta = f_beta
            optimal_beta = beta

    return optimal_beta


def precompute_hq_lut(G, Q_nn, q, m):
    """Precompute a lookup table for hierarchical quantization."""
    hq = HQuantizer(G=G, Q_nn=Q_nn, q=q, beta=1, alpha=1, M=m)
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


def calculate_inner_product_distortion(G, Q_nn, q, m, beta, alpha, samples, lut=None):
    """Calculate the distortion for inner products between different pairs of samples."""
    voronoi_errors = []
    hierarchical_errors = []
    small_errors = []
    lattice_dim = len(G)
    n = len(samples[0])
    r_q = (1 - float(q) ** (1 - m)) / (q - 1)
    small_quantizer = NQuantizer(G, Q_nn, q=q **m * (1 - r_q), beta=beta, alpha=alpha)
    nested_quantizer = NQuantizer(G, Q_nn, q=q **m, beta=beta, alpha=alpha)
    hierarchical_quantizer = HQuantizer(G, Q_nn=Q_nn, q=q, beta=beta, alpha=alpha, M=m)

    for i in range(len(samples)):
        for j in range(i + 1, len(samples)):
            vec1 = pad_vector(samples[i], lattice_dim)
            vec2 = pad_vector(samples[j], lattice_dim)

            true_inner_product = np.dot(vec1, vec2)

            voronoi_inner_product = 0
            small_inner_product = 0
            hierarchical_inner_product = 0

            for k in range(0, len(vec1), lattice_dim):
                block1 = vec1[k:k + lattice_dim]
                block2 = vec2[k:k + lattice_dim]

                small_inner_product += calculate_inner_product_for_chunks(block1, block2, small_quantizer)
                voronoi_res = calculate_inner_product_for_chunks(block1, block2, nested_quantizer)
                voronoi_inner_product += voronoi_res

                # Hierarchical Block
                enc_hierarchical_1, t_h1 = hierarchical_quantizer.encode(block1)
                enc_hierarchical_2, t_h2 = hierarchical_quantizer.encode(block2)
                print(f"encoded {block1} to {enc_hierarchical_1} with t={t_h1}")
                print(f"decoded {block1} to {hierarchical_quantizer.decode(enc_hierarchical_1, t_h1)}")
                # print(f"VC decoded {block1} to {small_quantizer.quantize(block1)}")
                # print(f"decoded {block2} to {hierarchical_quantizer.decode(enc_hierarchical_2, t_h2)}")
                # print(f"VC decoded {block2} to {small_quantizer.quantize(block2)}")
                c = (2 ** (hierarchical_quantizer.alpha * (t_h1 + t_h2)))
                hierarchical_res = calculate_weighted_sum(enc_hierarchical_1, enc_hierarchical_2, lut,
                                                          hierarchical_quantizer.q)
                hierarchical_inner_product_without_lut = (np.dot(hierarchical_quantizer.decode(enc_hierarchical_1, t_h1),
                           hierarchical_quantizer.decode(enc_hierarchical_2, t_h2)))

                assert np.isclose(hierarchical_inner_product_without_lut,
                                  hierarchical_res * c * (hierarchical_quantizer.beta ** 2),atol=1e-7)
                hierarchical_inner_product += hierarchical_res * c * (hierarchical_quantizer.beta ** 2)

            # print(f"True Inner Product: {true_inner_product:.6f}")
            # print(f"Voronoi Inner Product: {voronoi_inner_product:.6f}")
            # print(f"Hierarchical Inner Product: {hierarchical_inner_product:.6f}")
            # print(f"Hierarchical Inner Product wo LUT: {hierarchical_inner_product_without_lut:.6f}")
            # print("-" * 40)

            voronoi_errors.append((voronoi_inner_product - true_inner_product) ** 2)
            hierarchical_errors.append((hierarchical_inner_product - true_inner_product) ** 2)
            small_errors.append((small_inner_product - true_inner_product) ** 2)

    voronoi_mse = np.mean(voronoi_errors) / n
    hierarchical_mse = np.mean(hierarchical_errors) / n
    small_mse = np.mean(small_errors) / n
    return voronoi_mse, hierarchical_mse, small_mse


def calculate_inner_product_for_chunks(block1, block2, nested_quantizer):
    small_enc1, s_tv1 = nested_quantizer.encode(block1)
    small_enc2, s_tv2 = nested_quantizer.encode(block2)
    s_decoded1 = nested_quantizer.decode(small_enc1, s_tv1)
    s_decoded2 = nested_quantizer.decode(small_enc2, s_tv2)
    return np.dot(s_decoded1, s_decoded2)


def distortion_rate_theoretical(R):
    return 2 * 2 ** (-2 * R) - 2 ** (-4 * R)


def main():
    q = 4
    m_values = np.arange(3, 7)
    G = get_d4()
    Q_nn = closest_point_Dn
    beta = 1
    alpha = 1/3
    sig_l = np.sqrt(2) * 0.076602

    vector_dim = 128
    sample_size = 30
    variance = 1
    samples = [np.random.normal(0, variance, size=vector_dim) for _ in range(sample_size)]
    # samples = [[1.42122865 , 1.35529614, -0.68013817, -1.70222617], [-1.34455197 ,-0.63796664 ,-1.74199093 , 1.92720608]]
    nested_distortions = []
    hierarchical_distortions = []
    theoretical_distortions = []
    small_distortions = []
    R_values = []

    for m in m_values:
        print(f"Calculating q={q}, m={m}...")
        R = m * np.log2(q)
        R_values.append(R)

        beta = find_best_beta(G, Q_nn, q, m, alpha, sig_l)
        print(f"For q=4 and M={m} the best beta is: {beta:.4f}")

        lookup_table = precompute_hq_lut(G, Q_nn, q, m)

        nested_distortion, hierarchical_distortion, small_distortion = (
            calculate_inner_product_distortion(G, Q_nn, q, m, beta, alpha, samples, lut=lookup_table))
        theoretical_distortion = distortion_rate_theoretical(R)

        small_distortions.append(small_distortion)
        nested_distortions.append(nested_distortion)
        hierarchical_distortions.append(hierarchical_distortion)
        theoretical_distortions.append(theoretical_distortion)

        print(f"q={q}, R={R:.2f}")
        print(f"Theoretical Distortion: {theoretical_distortion:.6f}")
        print(f"Nested Quantizer Distortion: {nested_distortion:.6f}")
        print(f"Small Nested Quantizer Distortion: {small_distortion:.6f}")
        print(f"Hierarchical Quantizer Distortion: {hierarchical_distortion:.6f}")
        print("-" * 40)

    plt.figure(figsize=(10, 6))
    plt.plot(R_values, theoretical_distortions, label="Theoretical D(R)", linestyle="--", color="black")
    plt.plot(R_values, nested_distortions, label=f"$q^M$ Voronoi Code", marker="o", color="blue")
    plt.plot(R_values, small_distortions, label=f"Small Voronoi Code", marker="o", color="pink")
    plt.plot(R_values, hierarchical_distortions, label="Tiered Quantizer", marker="s", color="red")
    plt.xlabel("Rate (R)")
    plt.ylabel("Distortion (D)")
    plt.title("Distortion-Rate Comparison for Inner Products")
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == "__main__":
    main()
