import unittest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from nested_lattice_quantizer import Quantizer
from closest_point import closest_point_Dn, closest_point_E8
from utils import get_d3, get_e8

def closest_point_Zn(x):
    return np.round(x)


class TestQuantizer(unittest.TestCase):
    def test_Z2_lattice(self):
        """Test the Quantizer on Z² lattice (identity matrix)."""
        G = np.eye(2)
        quantizer = Quantizer(G, closest_point_Zn, beta=1, q=4)

        x = np.array([3.6, 3.3])
        encoded = quantizer.encode(x)
        decoded = quantizer.decode(encoded)

        np.testing.assert_almost_equal(decoded, [0, -1], decimal=5, err_msg="Z² lattice decode failed")

    def test_D3_lattice(self):
        """Test the Quantizer on D₃ lattice."""
        G = get_d3()
        quantizer = Quantizer(G, closest_point_Dn, beta=1, q=4)

        x = np.array([1.5, 2.3, -1.8])
        encoded = quantizer.encode(x)
        np.testing.assert_almost_equal(encoded, [2, 3, 1], decimal=5, err_msg="D₃ lattice encode failed")
        decoded = quantizer.decode(encoded)

        np.testing.assert_almost_equal(decoded, [2, 2, -2], decimal=5, err_msg="D₃ lattice decode failed")

    def test_E8_lattice(self):
        """Test the Quantizer on E₈ lattice."""
        # Generator matrix for E₈ lattice (this is a common representation)
        G = get_e8()
        quantizer = Quantizer(G, closest_point_E8, beta=1, q=4)

        x = np.array([1.5, 2.3, -1.8, 1.1, 0.9, -0.5, 1.2, -0.7])
        encoded = quantizer.encode(x)
        decoded = quantizer.decode(encoded)
        expected = np.array([-0.5, 0.5, 0.5, -0.5, -0.5, -2.5, -0.5, 1.5])
        np.testing.assert_almost_equal(decoded, expected, decimal=5, err_msg="E₈ lattice decode failed")


if __name__ == '__main__':
    unittest.main()
