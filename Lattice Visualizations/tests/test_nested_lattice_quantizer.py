import unittest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from nested_lattice_quantizer import NestedLatticeQuantizer as NQ, HierarchicalNestedLatticeQuantizer as HQ
from closest_point import closest_point_Dn, closest_point_E8, closest_point_A2
from utils import *

def closest_point_Zn(x):
    return np.floor(x + 0.5)


class TestQuantizer(unittest.TestCase):
    def test_Z2_lattice(self):
        """Test the Quantizer on Z² lattice (identity matrix)."""
        G = np.eye(2)
        quantizer = NQ(G, closest_point_Zn, beta=1, q=4)

        x = np.array([3.6, 3.3])
        encoded = quantizer.encode(x)
        decoded = quantizer.decode(encoded)

        np.testing.assert_almost_equal(decoded, [0, -1], decimal=5, err_msg="Z² lattice decode failed")


    def test_Z2_with_beta(self):
        G = get_z2()
        quantizer = NQ(G, closest_point_Zn, beta=0.5, q=20)

        x = np.array([3.6, 3.2])
        encoded = quantizer.encode(x)
        decoded = quantizer.decode(encoded)

        np.testing.assert_almost_equal(decoded, [3.5, 3], decimal=5, err_msg="Z² lattice decode failed")

    def test_z2_fundamental_cell(self):
        G = get_z2()
        quantizer = NQ(G, closest_point_Zn, beta=1, q=3)

        x = np.array([2,0])
        encoded = quantizer.encode(x)
        decoded = quantizer.decode(encoded)

        np.testing.assert_almost_equal(decoded, [-1, 0], decimal=5, err_msg="Z² lattice decode failed")

    def test_D3_lattice(self):
        """Test the Quantizer on D₃ lattice."""
        G = get_d3()
        quantizer = NQ(G, closest_point_Dn, beta=1, q=4)

        x = np.array([1.5, 2.3, -1.8])
        np.testing.assert_almost_equal(closest_point_Dn(x), [2, 2, -2], decimal=5, err_msg="D₃ closest point failed")
        encoded = quantizer.encode(x)
        np.testing.assert_almost_equal(encoded, [2, 3, 1], decimal=5, err_msg="D₃ lattice encode failed")
        decoded = quantizer.decode(encoded)

        np.testing.assert_almost_equal(decoded, [2, 2, -2], decimal=5, err_msg="D₃ lattice decode failed")

    def test_E8_lattice(self):
        """Test the Quantizer on E₈ lattice."""
        G = get_e8()
        quantizer = NQ(G, closest_point_E8, beta=1, q=4)

        x = np.array([1.5, 2.3, -1.8, 1.1, 0.9, -0.5, 1.2, -0.7])
        encoded = quantizer.encode(x)
        decoded = quantizer.decode(encoded)
        expected = np.array([-0.5, 0.5, 0.5, -0.5, -0.5, -2.5, -0.5, 1.5])
        np.testing.assert_almost_equal(decoded, expected, decimal=5, err_msg="E₈ lattice decode failed")

    def test_has_all_cosets(self):
        G = get_d2()
        q = 4
        quantizer = NQ(G, closest_point_Dn, q=q,  beta=1)
        points = []
        for i in range(q):
            for j in range(q):
                p = np.dot(G, np.array([i, j])) + np.array([1e-8, 1e-9])
                p_l = quantizer.quantize(p)
                points.append(p_l)
        points = np.array(points)
        np.testing.assert_equal(len(np.unique(points, axis=0)), 16, err_msg="Wrong number of cosets")

    def test_codebook(self):
        G = get_d2()
        q = 4
        quantizer = NQ(G, closest_point_Dn, q=q,  beta=1)
        np.testing.assert_equal(len(np.unique(quantizer.codebook, axis=0)), 16, err_msg="Wrong codebook size.")

        quantizer = NQ(get_d3(), closest_point_Dn, q=q,  beta=1)
        np.testing.assert_equal(len(np.unique(quantizer.codebook, axis=0)), 64, err_msg="Wrong codebook size.")


class TestNestedQuantizer(unittest.TestCase):
    def test_A2_lattice(self):
        """Test the Quantizer on A2 lattice."""
        G = get_a2()
        quantizer = HQ(G, closest_point_A2, q=3, beta=1, M=2)

        x = np.array([-0.5, -np.sqrt(3)/2])
        b_list = quantizer.encode(x)
        decoded = quantizer.decode(b_list)
        expected = np.array([-0.5, -np.sqrt(3)/2])
        np.testing.assert_almost_equal(decoded, expected, decimal=5, err_msg="A_2 lattice decode failed")

    def test_hole_in_A2_lattice(self):
        """Test the Quantizer on A2 lattice."""
        G = get_a2()
        quantizer = HQ(G, closest_point_A2, q=10, beta=1, M=2)

        x = closest_point_A2(np.array([10.5, 0]))
        b_list = quantizer.encode(x)
        decoded = quantizer.decode(b_list)
        expected = np.array(x)
        np.testing.assert_almost_equal(decoded, expected, decimal=5, err_msg="A_2 lattice decode failed")

    def test_d2_lattice(self):
        """Test the Quantizer on D2 lattice."""
        G = get_d2()
        quantizer = HQ(G, closest_point_Dn, q=6, beta=1, M=2)

        x = np.array([-1, -1])
        b_list = quantizer.encode(x)
        decoded = quantizer.decode(b_list)
        expected = np.array([-1, -1])
        np.testing.assert_almost_equal(decoded, expected, decimal=5, err_msg="D_2 lattice decode failed")

    def test_D2_scaled_nested_lattice(self):
        G = get_d2()
        quantizer = HQ(G, closest_point_Dn, q=5, beta=1, M=2)

        x = np.array([5.99, 8.32])
        b_list = quantizer.encode(x)
        decoded = quantizer.decode(b_list)
        expected = np.array([6, 8])
        np.testing.assert_almost_equal(decoded, expected, decimal=5, err_msg="D_2 lattice decode failed")


if __name__ == '__main__':
    unittest.main()
