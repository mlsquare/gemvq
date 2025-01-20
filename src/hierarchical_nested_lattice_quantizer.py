import numpy as np
from closest_point import custom_round
from src.nested_lattice_quantizer import NestedLatticeQuantizer as NQ


class HierarchicalNestedLatticeQuantizer:
    def __init__(self, G, Q_nn, q, beta, alpha, eps, dither, M):
        self.q = q
        self.G = G
        self.M = M
        self.beta = beta
        self.alpha = alpha
        self.eps = eps
        self.G_inv = np.linalg.inv(G)
        self.Q_nn = lambda x: Q_nn(x + eps)
        self.dither = dither

    def _encode(self, x):
        x = (x / self.beta) + self.dither
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
        return self.beta * (x_hat - self.dither)

    def decode(self, b_list, T):
        decoded = self._decode(b_list) * (2 ** (self.alpha * T))
        return decoded

    def quantize(self, x):
        b_list, T = self.encode(x)
        return self.decode(b_list, T)

    def create_q_codebook(self, with_dither):
        nq = None
        if with_dither:
            nq = NQ(self.G, self.Q_nn, self.q, self.beta, self.alpha, eps=self.eps, dither=self.dither)
        return nq.create_codebook(with_dither)
