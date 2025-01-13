import unittest
import numpy as np
import sys
import os
from itertools import product

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from nested_lattice_quantizer import NestedLatticeQuantizer as NQ, HierarchicalNestedLatticeQuantizer as HQ
from closest_point import closest_point_Dn, closest_point_E8, closest_point_A2, custom_round
from utils import *

def closest_point_Zn(x):
    return np.floor(x + 0.5)


class TestQuantizer(unittest.TestCase):
    def test_Z2_lattice(self):
        """Test the Quantizer on Z² lattice (identity matrix)."""
        G = np.eye(2)
        quantizer = NQ(G, closest_point_Zn, beta=1, q=4)

        x = np.array([3.6, 3.3])
        encoded, _ = quantizer.encode(x)
        decoded = quantizer.decode(encoded)

        np.testing.assert_almost_equal(decoded, [0, -1], decimal=5, err_msg="Z² lattice decode failed")


    def test_Z2_with_beta(self):
        G = get_z2()
        quantizer = NQ(G, closest_point_Zn, beta=0.5, q=20)

        x = np.array([3.6, 3.2])
        encoded, _ = quantizer.encode(x)
        decoded = quantizer.decode(encoded)

        np.testing.assert_almost_equal(decoded, [3.5, 3], decimal=5, err_msg="Z² lattice decode failed")

    def test_z2_fundamental_cell(self):
        G = get_z2()
        quantizer = NQ(G, closest_point_Zn, beta=1, q=3)

        x = np.array([2,0])
        encoded, _ = quantizer.encode(x)
        decoded = quantizer.decode(encoded)

        np.testing.assert_almost_equal(decoded, [-1, 0], decimal=5, err_msg="Z² lattice decode failed")

    def test_D3_lattice(self):
        """Test the Quantizer on D₃ lattice."""
        G = get_d3()
        quantizer = NQ(G, closest_point_Dn, beta=1, q=4)

        x = np.array([1.5, 2.3, -1.8])
        np.testing.assert_almost_equal(closest_point_Dn(x), [2, 2, -2], decimal=5, err_msg="D₃ closest point failed")
        encoded, _ = quantizer.encode(x)
        np.testing.assert_almost_equal(encoded, [2, 3, 1], decimal=5, err_msg="D₃ lattice encode failed")
        decoded = quantizer.decode(encoded)

        np.testing.assert_almost_equal(decoded, [2, 2, -2], decimal=5, err_msg="D₃ lattice decode failed")

    def test_D4_lattice(self):
        G = get_d4()
        quantizer = NQ(G, closest_point_Dn, beta=1, q=4)
        x = np.array([0.6, -1.1, 1.7, 0.1])
        expected = np.array([1, -1, 2, 0])

        encoded, _ = quantizer.encode(x)
        decoded = quantizer.decode(encoded)
        np.testing.assert_almost_equal(decoded, expected, decimal=5, err_msg="D4 lattice decode failed")

    def test_E8_lattice(self):
        """Test the Quantizer on E₈ lattice."""
        G = get_e8()
        quantizer = NQ(G, closest_point_E8, beta=1, q=4)

        x = np.array([1.5, 2.3, -1.8, 1.1, 0.9, -0.5, 1.2, -0.7])
        encoded, _ = quantizer.encode(x)
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
        q = 4
        quantizer = NQ(get_d2(), closest_point_Dn, q=q, beta=1)
        codebook = quantizer.create_codebook()

        assert len(codebook) == q ** 2, "Wrong codebook size for 2D lattice."

        lattice_points = np.array(list(codebook.values()))
        assert len(np.unique(lattice_points, axis=0)) == q ** 2, "Lattice points are not unique for 2D lattice."

        quantizer = NQ(get_d3(), closest_point_Dn, q=q, beta=1)
        codebook = quantizer.create_codebook()

        assert len(codebook) == q ** 3, "Wrong codebook size for 3D lattice."

        lattice_points = np.array(list(codebook.values()))
        assert len(np.unique(lattice_points, axis=0)) == q ** 3, "Lattice points are not unique for 3D lattice."

    def test_overload_handling_encode(self):
        q = 3
        d = 2
        G = np.eye(d)
        beta = d / (d + 2)
        x = np.array([20.3, 20.4])
        quantizer = NQ(G, custom_round, q, beta)
        enc, i_0 = quantizer.encode_with_overload_handling(x)
        np.testing.assert_equal(i_0, 5, err_msg="Overload mechanism did not calculate i_0 correctly")

        decoded = quantizer.decode_with_overload_handling(enc, i_0)
        expected = np.array([16, 16])
        np.testing.assert_equal(decoded, expected, err_msg="Overload mechanism did not decoded correctly")


class TestHQuantizer(unittest.TestCase):
    def test_A2_lattice(self):
        """Test the Quantizer on A2 lattice."""
        G = get_a2()
        quantizer = HQ(G, closest_point_A2, q=3, beta=1, M=2)

        x = np.array([-0.5, -np.sqrt(3)/2])
        b_list, _ = quantizer.encode(x)
        decoded = quantizer.decode(b_list)
        expected = np.array([-0.5, -np.sqrt(3)/2])
        np.testing.assert_almost_equal(decoded, expected, decimal=5, err_msg="A_2 lattice decode failed")

    def test_hole_in_A2_lattice(self):
        """Test the Quantizer on A2 lattice."""
        G = get_a2()
        quantizer = HQ(G, closest_point_A2, q=10, beta=1, M=2)

        x = closest_point_A2(np.array([10.5, 0]))
        b_list, _ = quantizer.encode(x)
        decoded = quantizer.decode(b_list)
        expected = np.array(x)
        np.testing.assert_almost_equal(decoded, expected, decimal=5, err_msg="A_2 lattice decode failed")

    def test_d2_lattice(self):
        """Test the Quantizer on D2 lattice."""
        G = get_d2()
        quantizer = HQ(G, closest_point_Dn, q=6, beta=1, M=2)

        x = np.array([-1, -1])
        b_list, _ = quantizer.encode(x)
        decoded = quantizer.decode(b_list)
        expected = np.array([-1, -1])
        np.testing.assert_almost_equal(decoded, expected, decimal=5, err_msg="D_2 lattice decode failed")

    def test_D2_scaled_nested_lattice(self):
        G = get_d2()
        quantizer = HQ(G, closest_point_Dn, q=5, beta=1, M=2)

        x = np.array([5.99, 8.32])
        b_list, _ = quantizer.encode(x)
        decoded = quantizer.decode(b_list)
        expected = np.array([6, 8])
        np.testing.assert_almost_equal(decoded, expected, decimal=5, err_msg="D_2 lattice decode failed")

    def test_decoder_uniqueness(self):
        q = 3
        M = 2
        d = 3
        G = np.eye(d)
        beta = 1.0

        quantizer = HQ(G, custom_round, q, beta, M)

        all_encodings = list(product(range(q), repeat=d * M))

        decoded_outputs = set()

        for encoding in all_encodings:
            b_list = [np.array(encoding[i * d:(i + 1) * d]) for i in range(M)]

            decoded = tuple(quantizer.decode(b_list))
            decoded_outputs.add(decoded)

        assert len(decoded_outputs) == len(all_encodings), \
            f"Decoder is not unique: {len(decoded_outputs)} unique outputs for {len(all_encodings)} encodings."

    def test_overload_mechanism(self):
        q = 2
        M = 2
        d = 2
        G = np.eye(d)
        beta = 1.0
        x = np.array([20.3, 20.4])
        quantizer = HQ(G, custom_round, q, beta, M)
        _, did_overload = quantizer.encode(x)
        np.testing.assert_equal(did_overload, True, err_msg="Overload mechanism did not detect overload")

    def test_overload_mechanism_fp(self):
        q = 4
        M = 2
        d = 2
        G = np.eye(d)
        beta = 1.0
        x = np.array([1.2, 1.4])
        quantizer = HQ(G, np.round, q, beta, M)
        _, did_overload = quantizer.encode(x)

        expected = False
        np.testing.assert_equal(did_overload, expected, err_msg="Overload mechanism falsely detected overload")

    def test_overload_handling_encode(self):
        q = 3
        G = np.eye(2)
        beta = 0.5
        x = np.array([20.3, 20.4])
        quantizer = HQ(G, custom_round, q, beta, M=2)
        b_list, i_0 = quantizer.encode_with_overload_handling(x)
        # np.testing.assert_equal(i_0, 1, err_msg="Overload mechanism did not calculate i_0 correctly")

        decoded = quantizer.decode_with_overload_handling(b_list, i_0)
        expected = np.array([16, 16])
        np.testing.assert_equal(decoded, expected, err_msg="Overload mechanism did not decoded correctly")


if __name__ == '__main__':
    unittest.main()
