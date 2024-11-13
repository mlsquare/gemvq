import numpy as np
from matplotlib import pyplot as plt

from closest_point import closest_point_Dn
from nested_lattice_quantizer import NestedQuantizer, Quantizer
from utils import get_d2


def plot_codebook(G):
    q = 10
    points = []
    quantizer = Quantizer(G, closest_point_Dn, 1, q**2)

    for i in range(q**2 + 1):
        for j in range(q**2 + 1):
            l_p = np.dot(G, np.array([i, j]))
            enc = quantizer.encode(l_p)
            dec = quantizer.decode(enc)
            points.append(dec)

    points = np.array(points)

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
    quantizer = NestedQuantizer(G, closest_point_Dn, q)

    for i in range(q**2 + 1):
        for j in range(q**2 + 1):
            l_p = np.dot(G, np.array([i, j]))
            bl, bm = quantizer.encode(l_p)
            dec = quantizer.decode(bl, bm)
            points.append(dec)

    points = np.array(points)

    plt.figure(figsize=(8, 8))
    plt.scatter(points[:, 0], points[:, 1], color='blue', s=10, label='Lattice Points')
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.legend()
    plt.title(f"Nested Lattice Codebook with Shaping Region, $q^2$ = {q**2}")
    plt.grid(True)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()


plot_codebook(get_d2())
plot_nested_codebook(get_d2())
