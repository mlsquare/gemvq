from nested_lattice_quantizer import Quantizer
from utils import *
from closest_point import closest_point_Dn, closest_point_E8
from plot_stats import *
import numpy as np


def calculate_avg_error(d, q, G, Q_nn, num_samples=50):
    errors = []
    betas = np.linspace(0.01, q, 5)

    for beta in betas:
        quantizer = Quantizer(G, Q_nn, beta=1, q=q)
        avg_error = 0

        for _ in range(num_samples):
            x = np.random.normal(0, 10, d)
            encoded_x = quantizer.encode(x)
            decoded_x = quantizer.decode(encoded_x)
            error = calculate_mse(x, decoded_x)
            avg_error += error

        avg_error /= num_samples
        errors.append(avg_error)
    best_beta_index = np.argmin(errors)
    best_beta = betas[best_beta_index]
    # plot_beta_results(betas, errors, best_beta)
    return errors, best_beta, errors[best_beta_index]


def run_experiment(lattice_name, q_values, d, G, Q_nn, num_samples=50):
    results = {}
    for q in q_values:
        errors, best_beta, smallest_error = calculate_avg_error(d, q, G, Q_nn, num_samples)
        results[q] = (best_beta, smallest_error)

    q_list = list(results.keys())
    average_errors = [results[q][1] for q in q_list]

    plot_q_results(lattice_name, q_list, average_errors)

    for q in q_list:
        print(f"Best beta for q={q}: {results[q][0]:.2f} with average error: {results[q][1]:.4f}")


def main():
    num_samples = 100
    q_values = np.linspace(2, 100, 10).astype(int)

    G_E_8 = get_e8()
    print("running e8...")
    run_experiment("E8", q_values, 8, G_E_8, closest_point_E8, num_samples)

    G_D_3 = get_d3()
    print("running d3...")
    run_experiment("D3", q_values, 3, G_D_3, closest_point_Dn, num_samples)
    print("Done")


main()
