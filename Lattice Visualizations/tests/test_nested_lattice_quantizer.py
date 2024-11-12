import unittest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from nested_lattice_quantizer import Quantizer, NestedQuantizer
from closest_point import closest_point_Dn, closest_point_E8, closest_point_A2
from utils import *

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


    def test_Z2_with_beta(self):
        G = get_z2()
        quantizer = Quantizer(G, closest_point_Zn, beta=0.5, q=20)

        x = np.array([3.6, 3.2])
        encoded = quantizer.encode(x)
        decoded = quantizer.decode(encoded)

        np.testing.assert_almost_equal(decoded, [3.5, 3], decimal=5, err_msg="Z² lattice decode failed")

    def test_z2_fundamental_cell(self):
        G = get_z2()
        quantizer = Quantizer(G, closest_point_Zn, beta=1, q=3)

        x = np.array([2,0])
        encoded = quantizer.encode(x)
        decoded = quantizer.decode(encoded)

        np.testing.assert_almost_equal(decoded, [-1, 0], decimal=5, err_msg="Z² lattice decode failed")

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
        G = get_e8()
        quantizer = Quantizer(G, closest_point_E8, beta=1, q=4)

        x = np.array([1.5, 2.3, -1.8, 1.1, 0.9, -0.5, 1.2, -0.7])
        encoded = quantizer.encode(x)
        decoded = quantizer.decode(encoded)
        expected = np.array([-0.5, 0.5, 0.5, -0.5, -0.5, -2.5, -0.5, 1.5])
        np.testing.assert_almost_equal(decoded, expected, decimal=5, err_msg="E₈ lattice decode failed")

class TestNestedQuantizer(unittest.TestCase):
    def test_A2_lattice(self):
        """Test the Quantizer on A2 lattice."""
        G = get_a2()
        quantizer = NestedQuantizer(G, closest_point_A2, q=4)

        x = np.array([9, 0])
        bl, bm = quantizer.encode(x)
        decoded = quantizer.decode(bl, bm)
        expected = np.array([-7, 0])
        np.testing.assert_almost_equal(decoded, expected, decimal=5, err_msg="A_2 lattice decode failed")

if __name__ == '__main__':
    unittest.main()
