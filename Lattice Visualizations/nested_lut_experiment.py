import numpy as np
from matplotlib import pyplot as plt

from closest_point import closest_point_A2, closest_point_Dn
from nested_lattice_quantizer import NestedQuantizer, Quantizer
from utils import get_a2, get_d2


def run_experiment():
    points = np.random.uniform(-30, 30, (1000, 2))
    # points = np.array([[0, 90], [50, 10 * np.sqrt(3)/2]])
    G = get_d2()

    q = 10
    quantizer = Quantizer(G, closest_point_Dn, 1, q)
    n_quantizer = NestedQuantizer(G, closest_point_Dn, q)

    print(closest_point_A2(points[0]))

    quantized_points = []
    n_quantized_points = []
    for point in points:
        enc = quantizer.encode(point)
        dec = quantizer.decode(enc)
        # print(f"decoded {point} original quantizer", dec)
        quantized_points.append(dec)

        bl, bm = n_quantizer.encode(point)
        n_dec = n_quantizer.decode(bl, bm)
        # print("decoded point nested quantizer", n_dec)
        n_quantized_points.append(n_dec)

    quantized_points = np.array(quantized_points)
    n_quantized_points = np.array(n_quantized_points)

    # fine_points = generate_a2_points(G, closest_point_A2)


    plt.figure(figsize=(10, 10))
    # plt.scatter(fine_points[:, 0], fine_points[:, 1], color='green', s=10, label='Fine Lattice Points', alpha=0.5)
    plt.scatter(points[:, 0], points[:, 1], color='blue', s=10, label='Original Points', alpha=0.5)
    plt.scatter(quantized_points[:, 0], quantized_points[:, 1], color='red', s=10, label='Quantized Points', alpha=0.5)
    # plt.scatter(n_quantized_points[:, 0], n_quantized_points[:, 1], color='green', s=10, label='Quantized Points', alpha=0.5)
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.legend()
    plt.title("A2 Lattice Quantization in $\mathbb{R}^2$")
    plt.grid()
    plt.show()


def plot_a2_points(G, Q):
    original_range = np.arange(0, 10)
    fine_points = []
    coarse_points = []
    voronoi_points = []
    # fine[5.5 0.8660254], coarse[5. 8.66025404], vor[0.5 - 7.79422863]

    for i in original_range:
        for j in original_range:
            fine_point = np.dot(G, np.array([i, j]))
            coarse_point = 10 * Q(fine_point / 10)
            coarse_points.append(coarse_point)
            fine_points.append(fine_point)
            voronoi_points.append(fine_point-coarse_point)
            print(f"fine {i, j, fine_point}, coarse {coarse_point}, vor {fine_point-coarse_point}")

    # Convert lists to arrays for indexing
    fine_points = np.array(fine_points)
    coarse_points = np.array(coarse_points)
    voronoi_points = np.array(voronoi_points)

    plt.figure(figsize=(10, 10))
    plt.scatter(fine_points[:, 0], fine_points[:, 1], color='green', s=10, label='Cubic Cell', alpha=0.5)
    plt.scatter(coarse_points[:, 0], coarse_points[:, 1], color='red', s=10, label='Scaled Cell', alpha=0.5)
    plt.scatter(voronoi_points[:, 0], voronoi_points[:, 1], color='blue', s=10, label='Voronoi Cell', alpha=0.5)
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.legend()
    plt.title("A2 Lattice Cells in $\mathbb{R}^2$")
    plt.grid()
    plt.show()


def main():
    run_experiment()
    print("Done")


# plot_a2_points(get_a2(), closest_point_A2)
main()
