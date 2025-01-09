import numpy as np


class NestedLatticeQuantizer:
    def __init__(self, G, Q_nn, q, beta):
        self.G = G
        d = 1e-8 * np.random.normal(0, 1, size=len(G))
        self.Q_nn = lambda x: Q_nn(x + d)
        self.q = q
        self.beta = beta
        self.G_inv = np.linalg.inv(G)

    def _encode(self, x):
        t = self.Q_nn(x / self.beta)
        y = np.dot(self.G_inv, t)
        enc = np.mod(np.round(y), self.q).astype(int)

        overload_error = not np.allclose(self.Q_nn(t / self.q), 0, atol=1e-8)
        return enc, overload_error

    def encode(self, x):
        enc, did_overload = self._encode(x)
        i_0 = 0
        while did_overload:
            i_0 += 1
            x = x / 2
            enc, did_overload = self._encode(x)
        return enc, i_0

    def _decode(self, y):
        x_p = np.dot(self.G, y)
        x_pp = self.q * self.Q_nn(x_p / self.q)
        return self.beta * (x_p - x_pp)

    def decode(self, enc, i_0):
        return self._decode(enc) * (2 ** i_0)

    def quantize(self, x):
        enc, i_0 = self.encode(x)
        return self.decode(enc, i_0)

    def create_codebook(self):
        d = self.G.shape[0]
        indices = np.array(np.meshgrid(*[range(self.q)] * d)).T.reshape(-1, d)
        points = {}
        for idx in indices:
            l_p = np.dot(self.G, idx)
            points[tuple(idx)] = self.quantize(l_p)
        return points


class HierarchicalNestedLatticeQuantizer:
    def __init__(self, G, Q_nn, q, beta, M=2):
        self.q = q
        self.G = G
        self.M = M
        self.beta = beta
        self.G_inv = np.linalg.inv(G)

        d = 1e-8 * np.random.normal(0, 1, size=len(G))
        self.Q_nn = lambda x: Q_nn(x + d)

    def _encode(self, x):
        x = x / self.beta
        x_l = x
        encoding_vectors = []

        for _ in range(self.M):
            x_l = self.Q_nn(x_l)
            b_i = np.mod(np.dot(self.G_inv, x_l), self.q)
            encoding_vectors.append(b_i)
            x_l = x_l / self.q

        overload_error = not np.allclose(self.Q_nn(x_l), 0, atol=1e-8)
        return tuple(encoding_vectors), overload_error

    def encode(self, x):
        b_list, did_overload = self._encode(x)
        i_0 = 0
        while did_overload:
            i_0 += 1
            x = x / 2
            b_list, did_overload = self._encode(x)
        return b_list, i_0

    def q_Q(self, x):
        return self.q * self.Q_nn(x / self.q)

    def _decode(self, b_list):
        x_hat_list = []
        for b in b_list:
            x_i_hat = np.dot(self.G, b) - self.q_Q(np.dot(self.G, b))
            x_hat_list.append(x_i_hat)
        x_hat = sum([np.power(self.q, i) * x_i for i, x_i in enumerate(x_hat_list)])
        return self.beta * x_hat

    def decode(self, b_list, i_0):
        return self._decode(b_list) * (2 ** i_0)

    def quantize(self, x):

    def create_codebook(self):
        d = self.G.shape[0]
        indices = np.array(np.meshgrid(*[range(self.q)] * d)).T.reshape(-1, d)
        all_combinations = np.array(np.meshgrid(*[indices] * self.M)).T.reshape(-1, self.M, d)

        codebook = {}
        for combination in all_combinations:
            encoded_vectors = tuple(tuple(b) for b in combination)
            quantized_point = self.decode(list(combination))
            codebook[encoded_vectors] = quantized_point

        return codebook        b_list, i_0 = self.encode(x)
        b_list, i_0 = self.encode(x)
        return self.decode(b_list, i_0)
