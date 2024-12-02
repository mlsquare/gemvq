from nested_lattice_quantizer import LatticeQuantizer
from utils import *
from closest_point import closest_point_Dn, closest_point_E8
from plot_stats import *
import numpy as np


def run_experiment(lattice_name, q_values, d, G, Q_nn, sig_l, n_samples):
    x_std = 10
    optimal_betas = []
    min_errors = []
    samples = np.random.normal(0, x_std, size=(n_samples, d))

    for q in q_values:
        beta_min = (1 / q) * np.sqrt((x_std ** 2) / sig_l)
        betas = np.maximum(0.1, beta_min + 0.5 * beta_min * np.arange(0, 20))
        errors = []
        min_error = float("inf")
        optimal_beta = beta_min

        for beta in betas:
            quantizer = LatticeQuantizer(G, Q_nn, beta=beta, q=q)
            avg_err = 0

            for x in samples:
                y = quantizer.encode(x)
                x_hat = quantizer.decode(y)
                avg_err += calculate_mse(x, x_hat)

            avg_err /= n_samples
            errors.append(avg_err)

            if avg_err < min_error:
                min_error = avg_err
                optimal_beta = beta

        min_errors.append(min_error)
        optimal_betas.append(optimal_beta)

        print(f"q = {q}, Optimal beta: {optimal_beta:.3f}, Minimum error: {min_error:.3f}")

    plt.figure(figsize=(10, 6))
    plt.plot(q_values, min_errors, marker='o', linestyle='-')
    plt.xlabel("q values")
    plt.ylabel("Minimum error")
    plt.title(f"Minimum error vs. q for {lattice_name}")
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(8, 6))
    plt.plot(q_values, min_errors, marker='o', linestyle='-', color='b')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('q (log scale)')
    plt.ylabel('Minimum Error (log scale)')
    plt.title('Minimum Error vs q on Log-Log Scale')
    plt.grid(True, which="both", ls="--", linewidth=0.5)

    log_q_values = np.log(q_values)
    log_min_errors = np.log(min_errors)
    slope, intercept = np.polyfit(log_q_values, log_min_errors, 1)  # Linear fit in log-log scale

    plt.text(0.1, 0.9, f'Slope: {slope:.3f}', transform=plt.gca().transAxes, color='red')

    plt.show()
    print(f"Slope of the log-log plot: {slope:.3f}")

    return optimal_betas, min_errors


def main():
    num_samples = 10000
    q_values = np.array(np.linspace(4, 100, 25)).astype(int)

    # print("running z3...")
    # z2 = get_z2()
    # run_experiment("z2", q_values, 2, z2, np.round, 1/12, num_samples)

    print("running d3...")
    d3 = get_d3()
    run_experiment("D3", q_values, 3, d3, closest_point_Dn, 3/24, num_samples)

    print("running e8...")
    e8 = get_e8()
    run_experiment("E8", q_values, 8, e8, closest_point_E8, (1/8) * (929/1620), num_samples)

    print("Done")


main()
