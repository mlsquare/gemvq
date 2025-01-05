import numpy as np


class NestedLatticeQuantizer:
    def __init__(self, G, Q_nn, q, beta):
        self.G = G
        self.Q_nn = Q_nn
        self.q = q
        self.beta = beta
        self.codebook = self.create_codebook()

    def encode(self, x):
        t = self.Q_nn(x / self.beta)
        y = np.dot(np.linalg.inv(self.G), t)
        enc = np.mod(np.round(y), self.q).astype(int)
        overload_error = True in (self.Q_nn(t/ self.q) != 0 * len(self.G))
        return enc, overload_error

    def decode(self, y):
        x_p = np.dot(self.G, y)
        x_pp = self.q * self.Q_nn(x_p / self.q)
        return self.beta * (x_p - x_pp)

    def quantize(self, x):
        enc, _ = self.encode(x)
        return self.decode(enc)

    def create_codebook(self):
        d = self.G.shape[0]
        indices = np.array(np.meshgrid(*[range(self.q)] * d)).T.reshape(-1, d)
        points = []
        for idx in indices:
            l_p = np.dot(self.G, idx)
            points.append(self.quantize(l_p))
        return np.array(points)


class HierarchicalNestedLatticeQuantizer:
    def __init__(self, G, Q_nn, q, beta, M):
        self.q = q
        self.G = G
        self.M = M
        self.beta = beta
        d = 1e-8 * np.random.normal(0, 1, size=len(G))
        self.Q_nn = lambda x: Q_nn(x + d)


    def encode(self, x):
        x = x / self.beta
        G_inv = np.linalg.inv(self.G)
        x_l = x
        encoding_vectors = []

        for _ in range(self.M):
            x_l = self.Q_nn(x_l)
            b_i = np.mod(np.dot(G_inv, x_l), self.q)
            encoding_vectors.append(b_i)
            x_l = x_l / self.q

        alpha = self.q** self.M - ((self.q ** self.M - self.q)/(self.q - 1))
        overload_error = True in (self.Q_nn(x/ alpha) != 0 * len(self.G))
        return tuple(encoding_vectors), overload_error

    def q_Q(self, x):
        return self.q * self.Q_nn(x / self.q)

    def decode(self, b_list):
        x_hat_list = []
        for b in b_list:
            x_i_hat = np.dot(self.G, b) - self.q_Q(np.dot(self.G, b))
            x_hat_list.append(x_i_hat)
        x_hat = sum([np.power(self.q, i) * x_i for i, x_i in enumerate(x_hat_list)])
        return self.beta * x_hat

    def quantize(self, x):
        b_list, _ = self.encode(x)
        x_hat = self.decode(b_list)
        return x_hat
