import numpy as np
import matplotlib.pyplot as plt
import math


def plot_lattice(ax, basis, shift=np.array([0, 0]), label=None, color='blue'):
    x_range = np.arange(-5, 5, 1)
    y_range = np.arange(-5, 5, 1)
    for i in x_range:
        for j in y_range:
            point = i * basis[0] + j * basis[1] + shift
            ax.plot(point[0], point[1], 'o', color=color)
            if label:
                ax.text(point[0] + 0.1, point[1] + 0.1, label, fontsize=12)


def plot_cubic():
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect('equal')

    cubic_basis = np.array([[2, 0], [0, 2]])

    plot_lattice(ax, cubic_basis, label='L', color='blue')
    shifts = [np.array([0, 1]), np.array([1, 0]), np.array([1, 1])]
    colors = ['red', 'green', 'purple']
    labels = ['(0,1) + L', '(1,0) + L', '(1,1) + L']

    for shift, color, label in zip(shifts, colors, labels):
        plot_lattice(ax, cubic_basis, shift=shift, label=label, color=color)

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('Cubic Lattice and its Cosets')

    ax.grid(True)
    plt.xlim(-5, 5)
    plt.ylim(-5, 5)

    # Show the plot
    plt.show()


def plot_a2():
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_aspect('equal')

    a2_basis = np.array([[math.sqrt(3), 1], [0, 2]])

    plot_lattice(ax, a2_basis, label='L', color='blue')

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title('A2 Lattice')

    ax.grid(True)
    plt.xlim(-5, 5)
    plt.ylim(-5, 5)

    plt.show()


plot_cubic()
plot_a2()
