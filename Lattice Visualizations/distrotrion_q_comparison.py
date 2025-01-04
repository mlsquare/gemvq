import time
from nested_lattice_quantizer import (NestedLatticeQuantizer as NQuantizer,
                                      HierarchicalNestedLatticeQuantizer as HQuantizer)
from utils import get_a2, get_d3, get_e8, calculate_mse
from closest_point import closest_point_A2, closest_point_Dn, closest_point_E8, custom_round
import numpy as np
import matplotlib.pyplot as plt


def calculate_mse_for_samples(samples, quantizer):
    """Calculate MSE for a given quantizer and set of samples."""
    mse = 0
    for x in samples:
        x_hat = quantizer.quantize(x)
        mse += calculate_mse(x, x_hat)
    return mse / len(samples)


def calculate_slope(log_R, min_errors):
    """Calculate the slope of the log-log plot."""
    log_min_errors = np.log2(min_errors)
    slope = np.polyfit(log_R, log_min_errors, 1)[0]
    return slope


def run_comparison_experiment(G, q_nn, q_values, n_samples, d, sigma_squared, sig_l, schemes):
    x_std = np.sqrt(sigma_squared)
    # samples = np.random.uniform(-0.5, 0.5, size=(n_samples, d))
    samples = np.random.normal(0, x_std, size=(n_samples, d))

    results = {scheme["name"]: {"R": [], "min_errors": []} for scheme in schemes}

    for q in q_values:
        R = 2 * np.log2(q)
        for scheme in schemes:
            name, quantizer_class, nesting = (scheme["name"], scheme["quantizer"], scheme["nesting"])

            beta_min = (1 /q**2) * np.sqrt(sigma_squared / sig_l)
            betas = beta_min + 0.25 * beta_min * np.arange(0, 20)

            min_error = float("inf")
            optimal_beta = beta_min
            for i, beta in enumerate(betas):
                quantizer = quantizer_class(G, q_nn, beta=beta, q=nesting(q))
                mse = calculate_mse_for_samples(samples, quantizer)

                if mse < min_error:
                    min_error = mse
                    optimal_beta = beta

            results[name]["R"].append(R)
            results[name]["min_errors"].append(min_error)

            print(f"Scheme: {name}, q = {q}, Optimal beta: {optimal_beta:.3f}, Minimum error: {min_error:.3f}")

    plt.figure(figsize=(10, 6))
    for scheme in schemes:
        name = scheme["name"]
        R = results[name]["R"]
        min_errors = results[name]["min_errors"]
        plt.plot(R, min_errors, label=name, marker='o')


        slope = calculate_slope(R, min_errors)
        print(f"Slope for {name}: {slope:.3f}")

    R_values = np.linspace(min(R), max(R), 100)
    pareto_log_distortions = - 2 * (R_values - min(R))
    pareto_distortions = 2 ** pareto_log_distortions
    plt.plot(R_values, pareto_distortions, label="Pareto Line (slope = -2)", color="red", linestyle="--")

    plt.xlabel(r"$R = 2 \log_2 q$")
    plt.ylabel("MSE (log scale)")
    plt.yscale("log")
    plt.title("Comparison of Quantization Schemes using $D_3$ Lattice")
    plt.legend()
    plt.grid(True)
    plt.show()

    return results


def main():
    num_samples = 1000
    q_values = np.array(np.linspace(2, 12, 10)).astype(int)
    sigma_squared = 1
    G = get_d3()
    q_nn = closest_point_Dn
    sig_l = 3/24

    schemes = [
        {"name": r"$q(q-1)$ Nested Quantizer", "quantizer": NQuantizer, "nesting": lambda q: int(np.ceil(q))*(int(np.ceil(q)) - 1)},
        {"name": "Hierarchical Quantizer",  "quantizer": HQuantizer, "nesting": lambda q: int(np.ceil(q))},
        {"name": r"$q^2$ Nested Quantizer", "quantizer": NQuantizer, "nesting": lambda q: int(np.ceil(q) ** 2)},
    ]

    results = run_comparison_experiment(G, q_nn, q_values, num_samples, len(G), sigma_squared, sig_l, schemes)

    print("Comparison complete. Results:")
    print(results)


main()
