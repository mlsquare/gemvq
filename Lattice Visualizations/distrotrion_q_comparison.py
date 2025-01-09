import time
from nested_lattice_quantizer import (NestedLatticeQuantizer as NQuantizer,
                                      HierarchicalNestedLatticeQuantizer as HQuantizer)
from utils import get_a2, get_d3, get_d4, get_e8, calculate_mse
from closest_point import closest_point_A2, closest_point_Dn, closest_point_E8, custom_round
import numpy as np
import matplotlib.pyplot as plt


def calculate_mse_and_overload_for_samples(samples, quantizer):
    """Calculate MSE for a given quantizer and set of samples."""
    mse = 0
    i_0_count = []
    for x in samples:
        encoding, i_0 = quantizer.encode(x)
        x_hat = quantizer.decode(encoding, i_0)
        mse += calculate_mse(x, x_hat)
        i_0_count.append(i_0)
    return mse / len(samples), i_0_count


def calculate_slope(log_R, min_errors):
    """Calculate the slope of the log-log plot."""
    log_min_errors = np.log2(min_errors)
    slope = np.polyfit(log_R, log_min_errors, 1)[0]
    return slope


def calculate_rate_and_distortion(name, samples, quantizer, q, beta_min):
    """Calculate rate-distortion for a given quantizer with updated optimization criterion."""
    betas = beta_min + 0.05 * beta_min * np.arange(0, 40)
    d = len(quantizer.G)

    min_f_beta = float("inf")
    optimal_beta = beta_min
    optimal_mse = None
    optimal_H_i_0 = None
    optimal_R = None
    best_beta_idx = -1
    for beta_idx, beta in enumerate(betas):
        quantizer.beta = beta
        mse, i_0_values = calculate_mse_and_overload_for_samples(samples, quantizer)

        i_0_counts = np.bincount(i_0_values)
        i_0_probs = i_0_counts / np.sum(i_0_counts)
        H_i_0 = -sum(p * np.log2(p) for p in i_0_probs if p > 0)
        R = 2 * np.log2(q) + (H_i_0 / d)

        f_beta = mse / (2 ** (-2 * R))

        if f_beta < min_f_beta:
            min_f_beta = f_beta
            optimal_beta = beta
            optimal_mse = mse
            optimal_H_i_0 = H_i_0
            optimal_R = R
            best_beta_idx = beta_idx

    print(f"For q={q} and scheme {name}: Optimal beta: {optimal_beta:.3f}, beta_idx:{best_beta_idx}, Minimum MSE: {optimal_mse:.3f}, "
          f"Minimum f(beta)={min_f_beta:.3f}, optimal_H_i_0: {optimal_H_i_0}")
    return optimal_R, optimal_mse, optimal_beta


def run_comparison_experiment(G, q_nn, q_values, n_samples, d, sigma_squared, M, sig_l, schemes):
    x_std = np.sqrt(sigma_squared)
    samples = np.random.normal(0, x_std, size=(n_samples, d))

    results = {scheme["name"]: {"R": [], "min_errors": []} for scheme in schemes}

    markers = ['o', 's', 'x']
    colors = ['blue', 'green', 'orange']

    for q_idx, q in enumerate(q_values):
        print(f"Processing q={q} ({q_idx + 1}/{len(q_values)})...")
        beta_min = (1 / q ** M) * np.sqrt(1 / sig_l) * np.sqrt(d / (d + 2))
        print(f"For q={q} the initial beta min: {beta_min}")
        for idx, scheme in enumerate(schemes):
            name, quantizer_class, nesting = scheme["name"], scheme["quantizer"], scheme["nesting"]
            quantizer = quantizer_class(G, q_nn, q=nesting(q), beta=beta_min)

            R, min_error, optimal_beta = calculate_rate_and_distortion(name, samples, quantizer, q, beta_min)
            results[name]["R"].append(R)
            results[name]["min_errors"].append(min_error)

    plt.figure(figsize=(10, 6))
    for idx, (name, scheme_results) in enumerate(results.items()):
        R = scheme_results["R"]
        min_errors = scheme_results["min_errors"]
        plt.plot(R, min_errors, label=name, marker=markers[idx], color=colors[idx])

    q_2_rates = results[schemes[2]["name"]]["R"]
    benchmark_distortions = [2 ** (-2 * k) for k in q_2_rates]
    plt.plot(q_2_rates, benchmark_distortions, label=f"Error benchmark for $q^2$ quantizer", color='red',
             linestyle="--")

    plt.xlabel(r"$2 \log_2 (q) + H(i_0)/d$")
    plt.ylabel("Distortion (log scale)")
    plt.title("Distortion-Rate Function with $D_4$ Lattice and Overload Mechanism")
    plt.yscale("log")
    plt.legend()
    plt.grid()
    plt.show()

    return results


def main():
    num_samples = 5000
    q_values = np.arange(3, 9)
    sigma_squared = 1
    G = get_d4()
    q_nn = closest_point_Dn
    sig_l = np.sqrt(2) * 0.076602
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
