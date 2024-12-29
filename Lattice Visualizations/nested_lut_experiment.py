import numpy as np
from matplotlib import pyplot as plt
from scipy.spatial import Voronoi, voronoi_plot_2d

from closest_point import closest_point_Dn, closest_point_A2
from nested_lattice_quantizer import NestedQuantizer, Quantizer
from utils import get_d2, get_a2


def generate_codebook(G, closest_point, q, with_plot=True):
    points = []
    quantizer = Quantizer(G, closest_point, q=q, beta=1)

    for i in range(2*q):
        for j in range(q +10):
            l_p = np.dot(G, np.array([i, j]))
            enc = quantizer.encode(l_p)
            dec = quantizer.decode(enc)
            points.append(dec)

    points = np.array(points)

    if with_plot:
        plot_lattice_points(points, q)

    return points


def plot_lattice_points(points, q):
    plt.figure(figsize=(8, 8))
    plt.scatter(points[:, 0], points[:, 1], color='blue', s=10, label='Lattice Points')
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.legend()
    plt.title(f"Lattice Codebook with Shaping Region, $q$ = {q}")
    plt.grid(True)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()


def compare_codebooks(G, closest_point, l_points, q, M, with_plot=True):
    lattice_points = []
    labels = []
    mismatches = []
    matches = []
    alpha = int((q**M - q) / (q-1))

    n_quantizer = NestedQuantizer(G, closest_point, q, beta=1, M=M)
    l_quantizer = Quantizer(G, closest_point, q**M - alpha, beta=1)

    point_map = {}
    duplicates = []

    d = 1e-9 * np.random.normal(0, 1, size=2)
    for i, j in l_points:
        l_p = np.array([i, j]) + d

        b_list = n_quantizer.encode(l_p)
        dec = n_quantizer.decode(b_list)

        enc = l_quantizer.encode(l_p)
        l_dec = l_quantizer.decode(enc)

        if np.allclose(l_dec, l_p, atol=1e-7):
            if not np.allclose(dec, l_dec, atol=1e-7):
                mismatches.append((l_p, dec, l_dec))
            else:
                matches.append((l_p, dec))

        dec_tuple = tuple(np.round(dec, decimals=9))
        if dec_tuple in point_map:
            duplicates.append((dec_tuple, point_map[dec_tuple], (i, j)))
        else:
            point_map[dec_tuple] = (i, j)

        lattice_points.append(dec)
        labels.append((i, j))

    lattice_points = np.array(lattice_points)

    if duplicates:
        print("Duplicate points found:")
        for dup_point, original_values, duplicate_values in duplicates:
            print(f"Point: {dup_point}")
            print(f"Original values: {original_values}")
            print(f"Duplicate values: {duplicate_values}")
    else:
        print("All points are unique.")

    if mismatches:
        print("Mismatched points found:")
        for l_p, dec, l_dec in mismatches:
            print(f"Original point: {l_p}")
            print(f"Nested quantizer decoding: {dec}")
            print(f"Regular quantizer decoding: {l_dec}")
    else:
        print("All points matched correctly.")

    if with_plot:
        plot_with_voronoi(lattice_points, q, alpha, M)


def plot_with_voronoi(lattice_points, q, alpha, M=2):
    plt.figure(figsize=(8, 8))
    plt.scatter(lattice_points[:, 0], lattice_points[:, 1], c='blue', s=1, label='Nested Quantizer Points')

    vor = Voronoi(lattice_points)

    origin_idx = np.where(np.isclose(lattice_points, np.array([0, 0]), atol=1e-2).all(axis=1))[0][0]
    region_idx = vor.point_region[origin_idx]
    vertices = vor.vertices[vor.regions[region_idx]]

    scaled_vertices_q_qm1 = (q ** M - alpha) * vertices
    scaled_vertices_q_qp1 = (q ** M + alpha) * vertices
    scaled_vertices_q2 = q ** M * vertices

    scaled_vertices_q_qm1 = np.vstack([scaled_vertices_q_qm1, scaled_vertices_q_qm1[0]])
    scaled_vertices_q_qp1 = np.vstack([scaled_vertices_q_qp1, scaled_vertices_q_qp1[0]])
    scaled_vertices_q2 = np.vstack([scaled_vertices_q2, scaled_vertices_q2[0]])

    plt.plot(scaled_vertices_q_qm1[:, 0], scaled_vertices_q_qm1[:, 1],
             color='green', linewidth=2, label=fr'Scaled Voronoi Cell $(q^{M} - \alpha)\mathcal{{V}}$')
    plt.plot(scaled_vertices_q_qp1[:, 0], scaled_vertices_q_qp1[:, 1],
             color='pink', linewidth=2, label=fr'Scaled Voronoi Cell $(q^{M} + \alpha)\mathcal{{V}}$')
    plt.plot(scaled_vertices_q2[:, 0], scaled_vertices_q2[:, 1],
             color='orange', linewidth=2, label=rf'Scaled Voronoi Cell $q^{M} \mathcal{{V}}$')

    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.legend()
    plt.title(rf'Hierarchical Nested $A_2$ Lattice Codebook for $q^{M}$ = {q ** M}')
    plt.grid(True)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()


def main():
    G = get_a2()
    closest_point = closest_point_A2
    q = 4
    M = 2

    points1 = generate_codebook(G, closest_point, q ** M, with_plot=False)
    compare_codebooks(G, closest_point, q=q, l_points=points1, M=M, with_plot=True)

    M = 3
    points1 = generate_codebook(G, closest_point, q ** M, with_plot=False)
    compare_codebooks(G, closest_point, q=q, l_points=points1, M=M, with_plot=True)


if __name__ == "__main__":
    main()
