import numpy as np


class Quantizer:
    def __init__(self, G, Q_nn, beta=1, q=4):
        self.beta = beta
        self.q = q
        self.G = G
        self.Q_nn = Q_nn

    def encode(self, x):
        fine_x = self.Q_nn(x / self.beta) * self.beta
        coarse_x = self.Q_nn(x / self.q * self.beta) * self.q * self.beta
        x_e = fine_x - coarse_x
        encoded = np.dot(x_e, np.linalg.inv(self.G).T)
        encoded_mod_q = np.mod(encoded, self.q)
        return encoded_mod_q

    def decode(self, y):
        x_hat = np.dot(self.G, y / self.beta) * self.beta
        closest_coarse = (self.q * self.beta) * self.Q_nn(x_hat / self.q * self.beta)
        return x_hat - closest_coarse
