import numpy as np


class LatticeQuantizer:
    def __init__(self, G, Q_nn, q, beta=1):
        self.G = G
        self.Q_nn = Q_nn
        self.q = q
        self.beta = beta

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


class NestedLatticeQuantizer:
    def __init__(self, G, Q_nn, q):
        self.G = G
        self.Q_nn = Q_nn
        self.q = q

    def encode(self, x):
        b = np.mod(np.dot(np.linalg.inv(self.G), self.Q_nn(x)), self.q**2)
        b_l = np.mod(b, self.q)
        b_m = np.mod((b-b_l)/self.q, self.q)
        return b_l, b_m

    def q_Q(self, x):
        return self.q * self.Q_nn(x / self.q)

    def decode(self, b_l, b_m):
        x_l_hat = np.dot(self.G, b_l) - self.q_Q(np.dot(self.G, b_l))
        x_m_hat = np.dot(self.G, b_m) - self.q_Q(np.dot(self.G, b_m))
        return x_l_hat + self.q*x_m_hat

    def quantize(self, x):
        bl, bm = self.encode(x)
        return self.decode(bl, bm)



