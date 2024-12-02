import unittest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from nested_lattice_quantizer import LatticeQuantizer, NestedLatticeQuantizer
from closest_point import closest_point_Dn, closest_point_E8, closest_point_A2
from utils import *

def closest_point_Zn(x):
    return np.round(x)


class TestQuantizer(unittest.TestCase):
    def test_Z2_lattice(self):
        G = np.eye(2)
        quantizer = LatticeQuantizer(G, closest_point_Zn, beta=1, q=4)

        x = np.array([3.6, 3.3])
        encoded = quantizer.encode(x)
        decoded = quantizer.decode(encoded)

        np.testing.assert_almost_equal(decoded, [0, -1], decimal=5, err_msg="Z² lattice decode failed")


    def test_Z2_with_beta(self):
        G = get_z2()
        quantizer = LatticeQuantizer(G, closest_point_Zn, beta=0.5, q=20)

        x = np.array([3.6, 3.2])
        encoded = quantizer.encode(x)
        decoded = quantizer.decode(encoded)

        np.testing.assert_almost_equal(decoded, [3.5, 3], decimal=5, err_msg="Z² lattice decode failed")

    def test_z2_fundamental_cell(self):
        G = get_z2()
        quantizer = LatticeQuantizer(G, closest_point_Zn, beta=1, q=3)

        x = np.array([2,0])
        encoded = quantizer.encode(x)
        decoded = quantizer.decode(encoded)

        np.testing.assert_almost_equal(decoded, [-1, 0], decimal=5, err_msg="Z² lattice decode failed")

    def test_d2_quantizer(self):
        G = get_d2()
        quantizer = LatticeQuantizer(G, closest_point_Dn, beta=1, q=25)

        x = np.array([-0.66, -1.58])
        encoded = quantizer.encode(x)
        decoded = quantizer.decode(encoded)
        expected = np.array([-1, -1])
        np.testing.assert_array_equal(decoded, expected, "D2 lattice encode/decode failed")

    def test_D3_lattice(self):
        G = get_d3()
        quantizer = LatticeQuantizer(G, closest_point_Dn, beta=1, q=4)

        x = np.array([1.5, 2.3, -1.8])
        encoded = quantizer.encode(x)
        np.testing.assert_almost_equal(encoded, [2, 3, 1], decimal=5, err_msg="D₃ lattice encode failed")
        decoded = quantizer.decode(encoded)

        np.testing.assert_almost_equal(decoded, [2, 2, -2], decimal=5, err_msg="D₃ lattice decode failed")

    def test_E8_lattice(self):
        G = get_e8()
        quantizer = LatticeQuantizer(G, closest_point_E8, beta=1, q=4)

        x = np.array([1.5, 2.3, -1.8, 1.1, 0.9, -0.5, 1.2, -0.7])
        encoded = quantizer.encode(x)
        decoded = quantizer.decode(encoded)
        expected = np.array([-0.5, 0.5, 0.5, -0.5, -0.5, -2.5, -0.5, 1.5])
        np.testing.assert_almost_equal(decoded, expected, decimal=5, err_msg="E₈ lattice decode failed")

    def test_A2_lattice(self):
        G = get_a2()
        quantizer = LatticeQuantizer(G, closest_point_A2, beta=1, q=4)

        x = np.array([0.5, np.sqrt(3)/2])
        encoded = quantizer.encode(x)
        decoded = quantizer.decode(encoded)
        expected = np.array([0.5, np.sqrt(3)/2])
        np.testing.assert_almost_equal(decoded, expected, decimal=5, err_msg="A_2 lattice decode failed")

class TestNestedQuantizer(unittest.TestCase):
    def test_A2_lattice(self):
        G = get_a2()
        quantizer = NestedLatticeQuantizer(G, closest_point_A2, q=4)

        x = np.array([0.5, np.sqrt(3)/2])
        bl, bm = quantizer.encode(x)
        decoded = quantizer.decode(bl, bm)
        expected = np.array([-7, 0])
        np.testing.assert_almost_equal(decoded, expected, decimal=5, err_msg="A_2 lattice decode failed")

    def test_D2_nested_lattice(self):
        G = get_d2()
        quantizer = NestedLatticeQuantizer(G, closest_point_Dn, q=5)

        x = np.array([-1, -1])
        bl, bm = quantizer.encode(x)
        decoded = quantizer.decode(bl, bm)
        expected = np.array([-1, -1])
        np.testing.assert_almost_equal(decoded, expected, decimal=5, err_msg="D_2 lattice decode failed")

    def test_D2_scaled_nested_lattice(self):
        G = get_d2()
        quantizer = NestedLatticeQuantizer(G, closest_point_Dn, q=5)

        x = np.array([5.99, 8.32])
        bl, bm = quantizer.encode(x)
        decoded = quantizer.decode(bl, bm)
        expected = np.array([6, 8])
        np.testing.assert_almost_equal(decoded, expected, decimal=5, err_msg="D_2 lattice decode failed")


if __name__ == '__main__':
    unittest.main()
