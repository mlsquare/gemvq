import numpy as np
from closest_point import custom_round

# todo: add noise (dither)
class NestedLatticeQuantizer:
    def __init__(self, G, Q_nn, q, beta, alpha, d, M=None):
        self.G = G
        self.Q_nn = lambda x: Q_nn(x+d)
        self.q = q
        self.beta = beta
        self.alpha = alpha
        self.G_inv = np.linalg.inv(G)

    def _encode(self, x):
        t = self.Q_nn(x / self.beta)
        y = np.dot(self.G_inv, t)
        enc = np.mod(np.round(y), self.q).astype(int)

        overload_error = not np.allclose(self.Q_nn(t / self.q), 0, atol=1e-8)
        return enc, overload_error

    def encode(self, x):
        enc, did_overload = self._encode(x)
        T = 0
        while did_overload:
            T += 1
            x = x / (2 ** self.alpha)
            enc, did_overload = self._encode(x)
        return enc, T

    def _decode(self, y):
        x_p = np.dot(self.G, y)
        x_pp = self.q * self.Q_nn(x_p / self.q)
        return self.beta * (x_p - x_pp)

    def decode(self, enc, T):
        return self._decode(enc) * (2 ** (self.alpha * T))

    def quantize(self, x):
        enc, T = self.encode(x)
        return self.decode(enc, T)

    def create_codebook(self):
        d = self.G.shape[0]
        codebook = {}
        encoding_vectors = np.array(np.meshgrid(*[range(self.q)] * d)).T.reshape(-1, d)
        for enc in encoding_vectors:
            lattice_point = self.decode(enc, 0)
            codebook[tuple(enc)] = lattice_point
        return codebook


class HierarchicalNestedLatticeQuantizer:
    def __init__(self, G, Q_nn, q, beta, alpha, M, d):
        self.q = q
        self.G = G
        self.M = M
        self.beta = beta
        self.alpha = alpha
        self.eps = d
        self.G_inv = np.linalg.inv(G)
        self.Q_nn = lambda x: Q_nn(x + d)

    def _encode(self, x):
        x = x / self.beta
        x_l = x
        encoding_vectors = []

        for _ in range(self.M):
            x_l = self.Q_nn(x_l)
            b_i = custom_round(np.mod(np.dot(self.G_inv, x_l), self.q)).astype(int)
            encoding_vectors.append(b_i)
            x_l = x_l / self.q

        overload_error = not np.allclose(self.Q_nn(x_l), 0, atol=1e-8)
        return tuple(encoding_vectors), overload_error

    def encode(self, x):
        b_list, did_overload = self._encode(x)
        T = 0
        while did_overload:
            T += 1
            x = x / (2 ** self.alpha)
            b_list, did_overload = self._encode(x)
        return b_list, T

    def q_Q(self, x):
        return self.q * self.Q_nn(x / self.q)

    def _decode(self, b_list):
        x_hat_list = []
        for b in b_list:
            x_i_hat = np.dot(self.G, b) - self.q_Q(np.dot(self.G, b))
            x_hat_list.append(x_i_hat)
        x_hat = sum([np.power(self.q, i) * x_i for i, x_i in enumerate(x_hat_list)])
        return self.beta * x_hat

    def decode(self, b_list, T):
        decoded = self._decode(b_list) * (2 ** (self.alpha * T))
        return decoded

    def quantize(self, x):
        b_list, T = self.encode(x)
        return self.decode(b_list, T)

    def create_q_codebook(self):
        nq = NestedLatticeQuantizer(self.G, self.Q_nn, self.q, self.beta, self.alpha, d=self.eps)
        return nq.create_codebook()
