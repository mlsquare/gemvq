from nested_lattice_quantizer import Quantizer
from utils import *
from closest_point import closest_point_Dn, closest_point_E8
from plot_stats import *
import numpy as np


def run_experiment(lattice_name, q_values, d, G, Q_nn, n_samples):
    errors = []
    for q in q_values:
        n_betas = int(5 * (q - 0.1))
        betas = np.linspace(0.1, q, n_betas)
        errors_for_beta = []
        for beta in betas:
            avg_err = 0
            quantizer = Quantizer(G, Q_nn, beta=beta, q=q)
            samples = np.random.normal(0, 10, size=(n_samples, d))
            for x in samples:
                y = quantizer.encode(x)
                x_hat = quantizer.decode(y)
                avg_err += calculate_mse(x, x_hat)

            avg_err /= n_samples
            errors_for_beta.append(avg_err)
        errors_for_beta = np.array(errors_for_beta)
        smallest_error, best_beta = np.min(errors_for_beta), betas[np.argmin(errors_for_beta)]
        errors.append(smallest_error)
        print(f"Best beta for q={q}: {best_beta:.2f} with average error: {smallest_error:.4f}. product= {best_beta*q}")

    plot_q_results(lattice_name, q_values, errors)


def main():
    num_samples = 1000
    q_values = np.array(np.linspace(2, 100, 25)).astype(int)

    # print("running z2...")
    # run_experiment("z2", q_values, 2, get_z2(), np.round, num_samples)

    print("running D3...")
    run_experiment("D3", q_values, 3, get_d3(), closest_point_Dn, num_samples)

    print("running e8...")
    run_experiment("E8", q_values, 8, get_e8(), closest_point_E8, num_samples)

    print("Done")


main()
