import numpy as np

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
