from nested_lattice_quantizer import Quantizer
from utils import *
from closest_point import closest_point_Dn, closest_point_E8
from plot_stats import *
import numpy as np


def run_experiment(lattice_name, q_values, d, G, Q_nn, n_samples):
    errors = []
    for q in q_values:
        avg_err = 0

        quantizer = Quantizer(G, Q_nn, beta=1, q=q)
        samples = np.random.normal(0, 10, size=(n_samples, d))

        for x in samples:
            y = quantizer.encode(x)
            x_hat = quantizer.decode(y)
            avg_err += calculate_mse(x, x_hat)

        avg_err /= n_samples
        errors.append(avg_err)

    plot_q_results(lattice_name, q_values, errors)
    print(errors)


def main():
    num_samples = 10000
    q_values = np.array(np.linspace(2, 100, 25)).astype(int)

    # print("running z3...")
    # run_experiment("z3", q_values, 3, get_z3(), np.round, num_samples)

    # print("running D3...")
    # run_experiment("D3", q_values, 3, get_d3(), closest_point_Dn, num_samples)

    print("running e8...")
    run_experiment("E8", q_values, 8, get_e8(), closest_point_E8, num_samples)

    print("Done")


main()
