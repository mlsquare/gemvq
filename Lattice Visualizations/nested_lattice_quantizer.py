import numpy as np


class Quantizer:
    def __init__(self, G, Q_nn, beta, q):
        self.beta = beta
        self.q = q
        self.G = G
        self.Q_nn = Q_nn

    def encode(self, x):
        t = self.Q_nn(x / self.beta)
        y = np.dot(np.linalg.inv(self.G), t)
        enc = np.mod(y, self.q)
        return enc

    def decode(self, y):
        x_p = np.dot(self.G, y)
        x_pp = self.q * self.Q_nn(x_p / self.q)
        return self.beta * (x_p - x_pp)
