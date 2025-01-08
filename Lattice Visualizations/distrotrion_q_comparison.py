import time
from nested_lattice_quantizer import (NestedLatticeQuantizer as NQuantizer,
                                      HierarchicalNestedLatticeQuantizer as HQuantizer)
from utils import get_a2, get_d3, get_e8, calculate_mse
from closest_point import closest_point_A2, closest_point_Dn, closest_point_E8, custom_round
import numpy as np
import matplotlib.pyplot as plt


def calculate_mse_and_overload_for_samples(samples, quantizer):
    """Calculate MSE for a given quantizer and set of samples."""
    mse = 0
    i_0_count = []
    for x in samples:
        encoding, i_0 = quantizer.encode_with_overload_handling(x)
        x_hat = quantizer.decode_with_overload_handling(encoding, i_0)
        mse += calculate_mse(x, x_hat)
        i_0_count.append(i_0)
    return mse / len(samples), i_0_count


def calculate_slope(log_R, min_errors):
    """Calculate the slope of the log-log plot."""
    log_min_errors = np.log2(min_errors)
    slope = np.polyfit(log_R, log_min_errors, 1)[0]
    return slope


def calculate_rate_and_distortion(samples, quantizer, q, beta_min):
    """Calculate rate-distortion for a given quantizer."""
    betas = beta_min + 0.02 * beta_min * np.arange(0, 40)

    min_error = float("inf")
    optimal_beta = beta_min
    optimal_i_0_values = []
    opt_idx = 0

    for beta_idx, beta in enumerate(betas):
        quantizer.beta = beta

        mse, i_0_values = calculate_mse_and_overload_for_samples(samples, quantizer)

        # print(f"for beta: {beta:.4f} the overload percent is: {overload_percent * 100 :.3f}% MSE:{mse}")

        if mse < min_error:
            min_error = mse
            optimal_beta = beta
            opt_idx = beta_idx
            optimal_i_0_values = i_0_values

    print(f"Optimal beta for q={q}: {optimal_beta:.3f}, beta_idx = {opt_idx}, Minimum MSE={min_error:.3f}")

    i_0_counts = np.bincount(optimal_i_0_values)
    i_0_probs = i_0_counts / np.sum(i_0_counts)
    H_i_0 = -sum(p * np.log2(p) for p in i_0_probs if p > 0)

    total_samples = len(optimal_i_0_values)
    print(f"\nOverload Statistics for q={q}:")
    print("i_0 value | Count | Percentage")
    print("-" * 30)
    for i_0, count in enumerate(i_0_counts):
        if count > 0:
            percentage = (count / total_samples) * 100
            print(f"{i_0:^9d} | {count:^5d} | {percentage:^6.2f}%")
    print(f"Average i_0: {np.mean(optimal_i_0_values):.2f}")
    print("-" * 30)
    print(f"H(i_0) ={H_i_0}")


    R = 4 * np.log2(q) + H_i_0
    return R, min_error, H_i_0


def run_comparison_experiment(G, q_nn, q_values, n_samples, d, sigma_squared, M, sig_l, schemes):
    x_std = np.sqrt(sigma_squared)
    samples = np.random.normal(0, x_std, size=(n_samples, d))

    results = {scheme["name"]: {"R": [], "min_errors": [], "H_i_0": []} for scheme in schemes}

    markers = ['o', 's', 'x']
    colors = ['blue', 'green', 'orange']

    for q_idx, q in enumerate(q_values):
        print(f"Processing q={q} ({q_idx + 1}/{len(q_values)})...")
        for idx, scheme in enumerate(schemes):
            beta_min = (1 / q**M) * np.sqrt(1 / sig_l) * np.sqrt(d/(d+2))
            name, quantizer_class, nesting = scheme["name"], scheme["quantizer"], scheme["nesting"]
            quantizer = quantizer_class(G, q_nn, q=nesting(q), beta=beta_min)

            R, min_error, H_i_0 = calculate_rate_and_distortion(samples, quantizer, q, beta_min)
            results[name]["R"].append(R)
            results[name]["min_errors"].append(min_error)
            results[name]["H_i_0"].append(H_i_0)

    plt.figure(figsize=(10, 6))

    for idx, (name, scheme_results) in enumerate(results.items()):
        R = scheme_results["R"]
        min_errors = scheme_results["min_errors"]
        H_i_0 = scheme_results["H_i_0"]
        R_plus_H = [r + h for r, h in zip(R, H_i_0)]
        plt.plot(R_plus_H, min_errors, label=name, marker=markers[idx], color=colors[idx])


    q_2_rates = np.add(results[schemes[2]["name"]]["R"], results[schemes[2]["name"]]["H_i_0"])
    benchmark_distortions = [2 ** (-2 * k) for k in q_2_rates]
    plt.plot(q_2_rates, benchmark_distortions, label=f"Error benchmark for $q^2$ quantizer", color='red', linestyle="--")

    plt.xlabel(r"$2 \log_2 (q) + H(i_0)$")
    plt.ylabel("Distortion (log scale)")
    plt.title("Distortion-Rate Function with $D_3$ Lattice and Overload Mechanism")
    plt.yscale("log")
    plt.legend()
    plt.grid()
    plt.show()

    return results


def main():
    num_samples = 5000
    q_values = np.arange(3, 9)
    sigma_squared = 1
    G = get_d3()
    q_nn = closest_point_Dn
    sig_l = 3/24
    M = 2

    schemes = [
        {"name": r"$q(q-1)$ Nested Quantizer", "quantizer": NQuantizer, "nesting": lambda q: int(np.ceil(q))*(int(np.ceil(q)) - 1)},
        {"name": "Hierarchical Quantizer",  "quantizer": HQuantizer, "nesting": lambda q: int(np.ceil(q))},
        {"name": r"$q^2$ Nested Quantizer", "quantizer": NQuantizer, "nesting": lambda q: int(np.ceil(q) ** 2)},
    ]

    results = run_comparison_experiment(G, q_nn, q_values, num_samples, len(G), sigma_squared, M, sig_l, schemes)

    print("Comparison complete. Results:")
    print(results)


main()
