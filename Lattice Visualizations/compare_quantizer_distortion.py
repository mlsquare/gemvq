import numpy as np
import matplotlib.pyplot as plt

from closest_point import closest_point_Dn, closest_point_A2
from nested_lattice_quantizer import (NestedLatticeQuantizer as NQuantizer,
                                      HierarchicalNestedLatticeQuantizer as HQuantizer)
from utils import get_d2, get_a2


def generate_gaussian_points(mean, variance, num_points):
    """Generate points from a Gaussian distribution."""
    return np.random.normal(mean, np.sqrt(variance), size=(num_points, 2))


def compute_distortion(points, quantizer):
    """Compute distortion as the mean squared error."""
    distortions = []
    for point in points:
        quantized = quantizer.quantize(point)
        distortion = np.sum((point - quantized) ** 2)
        distortions.append(distortion)
    return np.mean(distortions)


def compare_quantizers(G, Q, q, variances, num_points):
    """Compare distortions for the nested and regular quantizers."""
    nested_quantizer = HQuantizer(G, Q, q, beta=1, M=2)
    big_lattice_quantizer = NQuantizer(G, Q, q**2, beta=1)
    small_lattice_quantizer = NQuantizer(G, Q, q*(q-1), beta=1)

    nested_distortions = []
    big_lattice_distortions = []
    small_lattice_distortions = []

    for variance in variances:
        points = generate_gaussian_points(mean=0, variance=variance, num_points=num_points)

        nested_distortion = compute_distortion(points, nested_quantizer)
        big_distortion = compute_distortion(points, big_lattice_quantizer)
        small_distortion = compute_distortion(points, small_lattice_quantizer)

        nested_distortions.append(nested_distortion)
        big_lattice_distortions.append(big_distortion)
        small_lattice_distortions.append(small_distortion)

    plt.figure(figsize=(10, 6))
    plt.plot(variances, np.array(nested_distortions),
             label="Nested Lattice Quantizer", marker='o', linestyle='-', color='blue')
    plt.plot(variances, np.array(big_lattice_distortions),
             label="q^2 Lattice Quantizer", marker='x', linestyle='--', color='orange')
    plt.plot(variances, np.array(small_lattice_distortions),
             label="q(q-1) Lattice Quantizer", marker='+', linestyle='-.', color='pink')

    plt.xlabel("Variance of Gaussian Distribution")
    plt.ylabel("Distortion (Mean Squared Error)")
    plt.title(f"Quantizer Distortion Comparison (q={q})")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    G = get_a2()
    q = 6
    variances = np.linspace(1, 120.0, 15)
    num_points = 500
    compare_quantizers(G, closest_point_A2, q, variances, num_points)
