import numpy as np


class NestedLatticeQuantizer:
    def __init__(self, G, Q_nn, q, beta, alpha, eps, dither, M=None):
        self.G = G
        self.Q_nn = lambda x: Q_nn(x+eps)
        self.q = q
        self.beta = beta
        self.alpha = alpha
        self.dither = dither
        self.G_inv = np.linalg.inv(G)

    def _encode(self, x, with_dither):
        x_tag = (x / self.beta)
        if with_dither:
            x_tag = x_tag + self.dither
        t = self.Q_nn(x_tag)
        t = self.Q_nn(x_tag)
        y = np.dot(self.G_inv, t)
        enc = np.mod(np.round(y), self.q).astype(int)

        overload_error = not np.allclose(self.Q_nn(t / self.q), 0, atol=1e-8)
        return enc, overload_error

    def encode(self, x, with_dither):
        enc, did_overload = self._encode(x, with_dither)
        t = 0
        while did_overload:
            t += 1
            x = x / (2 ** self.alpha)
            enc, did_overload = self._encode(x, with_dither)
        return enc, t

    def _decode(self, y, with_dither):
        x_p = np.dot(self.G, y)
        if with_dither:
            x_p = x_p - self.dither
        x_pp = self.q * self.Q_nn(x_p / self.q)
        return self.beta * (x_p - x_pp)

    def decode(self, enc, T, with_dither):
        return self._decode(enc, with_dither) * (2 ** (self.alpha * T))

    def quantize(self, x):
        enc, T = self.encode(x)
        return self.decode(enc, T)

    def create_codebook(self, with_dither):
        d = self.G.shape[0]
        codebook = {}
        encoding_vectors = np.array(np.meshgrid(*[range(self.q)] * d)).T.reshape(-1, d)
        for enc in encoding_vectors:
            lattice_point = self.decode(enc, 0, with_dither)
            codebook[tuple(enc)] = lattice_point
        return codebook
