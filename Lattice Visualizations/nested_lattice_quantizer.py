import numpy as np


class Quantizer:
    def __init__(self, G, Q_nn, q, beta):
        self.beta = beta
        self.q = q
        self.G = G
        self.Q_nn = Q_nn

    def encode(self, x):
        t = self.Q_nn(x / self.beta)
        y = np.dot(np.linalg.inv(self.G), t)
        enc = np.mod(np.round(y), self.q).astype(int)
        return enc

    def decode(self, y):
        x_p = np.dot(self.G, y)
        x_pp = self.q * self.Q_nn(x_p / self.q)
        return self.beta * (x_p - x_pp)

    def quantize(self, x):
        return self.decode(self.encode(x))


class NestedQuantizer:
    def __init__(self, G, Q_nn, q):
        self.q = q
        self.G = G
        self.Q_nn = lambda x: Q_nn(x + np.array([1e-7, 1e-6]))

    def encode(self, x):
        G_inv = np.linalg.inv(self.G)
        x_l = self.Q_nn(x)
        b_l = np.mod(np.dot(G_inv, x_l), self.q)
        b_m = np.mod(np.dot(G_inv, self.Q_nn(x_l / self.q)), self.q)
        return b_l, b_m

    def q_Q(self, x):
        return self.q * self.Q_nn(x / self.q)

    def decode(self, b_l, b_m):
        x_l = np.dot(self.G, b_l) - self.q_Q(np.dot(self.G, b_l))
        x_m = np.dot(self.G, b_m) - self.q_Q(np.dot(self.G, b_m))
        return x_l + self.q*x_m

    def quantize(self, x):
        b_l, b_m = self.encode(x)
        x = self.decode(b_l, b_m)
        return x
