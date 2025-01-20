from nested_lattice_quantizer import NestedLatticeQuantizer as NQuantizer
from hierarchical_nested_lattice_quantizer import HierarchicalNestedLatticeQuantizer as HQuantizer
from utils import get_a2, get_d3, get_d4, get_e8, calculate_mse, calculate_t_entropy
from closest_point import closest_point_A2, closest_point_Dn, closest_point_E8, custom_round
import numpy as np
import matplotlib.pyplot as plt


def calculate_mse_and_overload_for_samples(samples, quantizer):
    mse = 0
    T_count = []
    for x in samples:
        encoding, t = quantizer.encode(x, with_dither=False)
        x_hat = quantizer.decode(encoding, t, with_dither=False)
        mse += calculate_mse(x, x_hat)
        T_count.append(t)
    return mse / len(samples), T_count


def calculate_slope(log_R, min_errors):
    log_min_errors = np.log2(min_errors)
    slope = np.polyfit(log_R, log_min_errors, 1)[0]
    return slope


def calculate_rate_and_distortion(name, samples, quantizer, q, beta_min):
    betas = beta_min + 0.05 * beta_min * np.arange(0, 40)
    d = len(quantizer.G)

    min_f_beta = float("inf")
    optimal_beta = beta_min
    optimal_mse = None
    optimal_H_T = None
    optimal_R = None
    overload_percentage = None

    for beta_idx, beta in enumerate(betas):
        quantizer.alpha = 1/3
        quantizer.beta = beta
        mse, T_values = calculate_mse_and_overload_for_samples(samples, quantizer)

        H_T, T_counts = calculate_t_entropy(T_values, q)
        R = 2 * np.log2(q) + (H_T / d)

        f_beta = mse / (2 ** (-2 * R))

        if f_beta < min_f_beta:
            min_f_beta = f_beta
            optimal_beta = beta
            optimal_mse = mse
            optimal_H_T = H_T
            optimal_R = R
            overload_percentage = (1 - (T_counts[0] / sum(T_counts))) * 100


    print(f"For q={q} and scheme {name}: Optimal beta: {optimal_beta:.3f}, "
          f"Minimum MSE: {optimal_mse:.6f}, Minimum f(beta, alpha): {min_f_beta:.3f}, "
          f"optimal_H_T: {optimal_H_T:.4f}, overload percent: {overload_percentage:.2f}%")
    return optimal_R, optimal_mse, optimal_beta


def run_comparison_experiment(G, q_nn, q_values, n_samples, d, sigma_squared, M, sig_l, schemes, eps):
    x_std = np.sqrt(sigma_squared)
    samples = np.random.normal(0, x_std, size=(n_samples, d))

    results = {scheme["name"]: {"R": [], "min_errors": []} for scheme in schemes}

    markers = ['o', 's', 'x']
    colors = ['blue', 'green', 'orange']

    for q_idx, q in enumerate(q_values):
        print(f"Processing q={q} ({q_idx + 1}/{len(q_values)})...")
        beta_min = (1 / q ** M) * np.sqrt(1 / sig_l) * np.sqrt(d / (d + 2))

        for idx, scheme in enumerate(schemes):
            name, quantizer_class, nesting = scheme["name"], scheme["quantizer"], scheme["nesting"]
            quantizer = quantizer_class(G, q_nn, q=nesting(q), beta=beta_min, alpha=1, eps=eps, M=2, dither=np.zeros(d))

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
    plt.plot(q_2_rates, benchmark_distortions, label=f"Theoretical benchmark", color='red',
             linestyle="--")

    plt.xlabel(r"$R = 2 \log_2 (q) + H(T)/d$")
    plt.ylabel("D (logarithmic scale)")
    plt.title("Distortion-Rate Function with $D_4$ Lattice")
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
    eps = 1e-8 * np.random.normal(0, 1, size=len(G))
    q_nn = closest_point_Dn
    sig_l = np.sqrt(2) * 0.076602
    M = 2

    schemes = [
        {"name": r"$q(q-1)$ Voronoi Code", "quantizer": NQuantizer, "nesting": lambda q: int(q * (q-1))},
        {"name": "Hierarchical Quantizer",  "quantizer": HQuantizer, "nesting": lambda q: int(q)},
        {"name": r"$q^2$ Voronoi Code", "quantizer": NQuantizer, "nesting": lambda q: int(q ** 2)},
    ]

    results = run_comparison_experiment(G, q_nn, q_values, num_samples, len(G), sigma_squared, M, sig_l, schemes, eps)

    print("Comparison complete. Results:")
    print(results)

main()
