import numpy as np
from matplotlib import pyplot as plt

from closest_point import closest_point_Dn, closest_point_A2
from nested_lattice_quantizer import NestedLatticeQuantizer, LatticeQuantizer
from utils import get_d2, get_a2

from scipy.spatial import Voronoi, voronoi_plot_2d


def plot_codebook(G, closest_point, q):
    lattice_points = []
    quantizer = LatticeQuantizer(G, closest_point, 1, q**2)  # q^4 close points

    for i in range(2*(q**2) + 10):
        for j in range(2*(q**2) + 10):
            point = np.dot(G, np.array([i, j]))
            enc = quantizer.encode(point)
            dec = quantizer.decode(enc)
            lattice_points.append(dec)
    lattice_points = np.array(lattice_points)

    plot_scaled_voronoi_cells(lattice_points, q)


def plot_scaled_voronoi_cells(lattice_points, q, color='blue'):
    vor = Voronoi(lattice_points)

    origin_idx = np.where((lattice_points == [0, 0]).all(axis=1))[0][0]
    region_idx = vor.point_region[origin_idx]
    vertices = vor.vertices[vor.regions[region_idx]]

    scaled_vertices_q = q * (q-1) * vertices          # q *(q-1) * V_0
    scaled_vertices_q2 = q**2 * vertices      # q^2 * V_0

    scaled_vertices_q = np.vstack([scaled_vertices_q, scaled_vertices_q[0]])
    scaled_vertices_q2 = np.vstack([scaled_vertices_q2, scaled_vertices_q2[0]])

    plt.figure(figsize=(8, 8))
    plt.scatter(lattice_points[:, 0], lattice_points[:, 1], color=color, s=2, label='Lattice Points')

    plt.plot(scaled_vertices_q[:, 0], scaled_vertices_q[:, 1],
             color='green', linewidth=2, label=f'Scaled Voronoi Cell ($q(q-1) \\mathcal{{V}}_0$)')
    plt.plot(scaled_vertices_q2[:, 0], scaled_vertices_q2[:, 1],
             color='orange', linewidth=2, label=f'Scaled Voronoi Cell ($q^2 \\mathcal{{V}}_0$)')

    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.legend()
    plt.title(f"Lattice with Scaled Voronoi Cells, $q = {q}$")
    plt.grid(True)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()


def plot_nested_codebook(G, closest_point, q):
    lattice_points = []
    colors = []
    quantizer = NestedLatticeQuantizer(G, closest_point, q)

    distinct_colors = [
        "red", "orange", "yellow", "green", "blue", "purple", "cyan", "magenta",
        "brown", "pink", "lime", "navy", "teal", "coral", "gold", "maroon"
    ]

    for i in range(2 * (q ** 2)):
        for j in range(2*(q ** 2)):
            l_p = np.dot(G, np.array([i, j]))
            bl, bm = quantizer.encode(l_p)
            dec = quantizer.decode(bl, bm)
            lattice_points.append(dec)

            color_index = int(bm[0] * q + bm[1])
            colors.append(distinct_colors[color_index % len(distinct_colors)])

    lattice_points = np.array(lattice_points)
    plot_scaled_voronoi_cells(lattice_points, q, color=colors)


def generate_nested_codebook(G, closest_point, q):
    quantizer = NestedLatticeQuantizer(G, closest_point, q)
    codebook = set()
    for i in range(q):
        for j in range(q):
            point = tuple(np.dot(G, np.array([i, j])))
            bl, bm = quantizer.encode(point)
            dec = quantizer.decode(bl, bm)
            codebook.add(tuple(dec))
    return codebook


def is_codebook_contained(codebook_q1, codebook_q2, tolerance=1e-9):
    for point1 in codebook_q1:
        if not any(np.allclose(point1, point2, atol=tolerance) for point2 in codebook_q2):
            return False
    return True


# Example Usage
if __name__ == "__main__":
    D2 = get_d2()
    A2 = get_a2()
    q1 = 10

    plot_codebook(D2, closest_point_Dn, q1)
    plot_nested_codebook(D2, closest_point_Dn, q1)
    plot_codebook(A2, closest_point_A2, q1)
    plot_nested_codebook(A2, closest_point_A2, q1)
