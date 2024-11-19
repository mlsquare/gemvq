import numpy as np
from matplotlib import pyplot as plt

from closest_point import closest_point_Dn, closest_point_A2
from nested_lattice_quantizer import NestedQuantizer, Quantizer
from utils import get_d2, get_a2


def plot_codebook(G):
    q = 10
    points = []
    quantizer = Quantizer(G, closest_point_A2, 1,   q**2)  # q^4 close points

    for i in range(q**2 + 10):
        for j in range(q**2 + 10):
            l_p = np.dot(G, np.array([i, j]))
            enc = quantizer.encode(l_p)
            dec = quantizer.decode(enc)
            # print(f"i:{i}, j: {j}, lattice point {l_p}, decoded: {dec}")
            points.append(dec)

    points = np.array(points)
    print("u p", np.unique(points))

    plt.figure(figsize=(8, 8))
    plt.scatter(points[:, 0], points[:, 1], color='blue', s=10, label='Lattice Points')
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.legend()
    plt.title(f"Lattice Codebook with Shaping Region, $q^2$ = {q**2}")
    plt.grid(True)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()


def plot_nested_codebook(G):
    q = 10
    points = []
    colors = []
    quantizer = NestedQuantizer(G, closest_point_A2, q)

    distinct_colors = [
        "red", "orange", "yellow", "green", "blue", "purple", "cyan", "magenta",
        "brown", "pink", "lime", "navy", "teal", "coral", "gold", "maroon"
    ]

    for i in range(2 * (q ** 2)):
        for j in range(2*(q ** 2)):
            l_p = np.dot(G, np.array([i, j]))
            bl, bm = quantizer.encode(l_p)
            dec = quantizer.decode(bl, bm)
            points.append(dec)

            color_index = int(bm[0] * q + bm[1])
            colors.append(distinct_colors[color_index % len(distinct_colors)])

    points = np.array(points)

    plt.figure(figsize=(8, 8))
    plt.scatter(points[:, 0], points[:, 1], color=colors, s=10, label='Lattice Points')
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.title(f"Nested Lattice Codebook with Shaping Region, $q^2$ = {q ** 2}")
    plt.grid(True)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()


def is_codebook_contained(codebook_q1, codebook_q2):
    set_q1 = set(codebook_q1)
    set_q2 = set(codebook_q2)

    # Check if set_q1 is a subset of set_q2
    return set_q1.issubset(set_q2)


import numpy as np


def generate_nested_codebook(G, q):
    quantizer = NestedQuantizer(G, closest_point_A2, q)
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
    # Define a generator matrix G
    G = get_a2()

    # Generate codebooks for q1 and q2
    q1 = 9
    q2 = 10
    codebook_q1 = generate_nested_codebook(G, q1)
    codebook_q2 = generate_nested_codebook(G, q2)

    # Check if codebook_q1 is contained in codebook_q2
    result = is_codebook_contained(codebook_q1, codebook_q2)
    print(f"Is the codebook for q1={q1}^2 contained in the codebook for q2={q2}^2? {result}")

    # plot_codebook(get_a2())
    # plot_nested_codebook(get_a2())
