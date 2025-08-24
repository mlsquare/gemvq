import unittest
import numpy as np

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.lattices.utils import closest_point_Dn, closest_point_E8, upscale, downscale, closest_point_A2


class TestClosestPointAlgorithms(unittest.TestCase):

    def test_closest_point_d2(self):
        point = np.array([-0.66, -1.58])
        result = closest_point_Dn(point)
        expected = np.array([-1, -1])
        np.testing.assert_array_equal(result, expected, "Failed to return the correct lattice point in D2")


    def test_closest_point_d3_on_lattice(self):
        point = np.array([1, 0, 0])
        result = closest_point_Dn(point)
        expected = np.array([0, 0, 0])
        np.testing.assert_array_equal(result, expected, "Failed to return the correct lattice point in D3")

    def test_closest_point_d3_on_lattice_negative(self):
        point = np.array([-1, 0, 0])
        result = closest_point_Dn(point)
        expected = np.array([0, 0, 0])
        np.testing.assert_array_equal(result, expected, "Failed to return the correct lattice point in D3")

    def test_closest_point_d3_off_lattice(self):
        point = np.array([0.5, 0.3, -0.2])
        expected = np.array([0, 0, 0])
        result = closest_point_Dn(point)
        np.testing.assert_array_equal(result, expected, "Failed to find the correct closest point in D3")

    def test_closest_point_d3_boundary_case(self):
        point = np.array([0.500001, 0.500001, 0.5])
        expected = np.array([1, 1, 0])
        result = closest_point_Dn(point)
        np.testing.assert_array_equal(result, expected, "Failed at a boundary case in D3")

    def test_closest_point_d3_another_boundary_case(self):
        point = np.array([0.5, 0.5, 0.5])
        expected = np.array([0, 0, 0])
        result = closest_point_Dn(point)
        np.testing.assert_array_equal(result, expected, "Failed at a boundary case in D3")

    def test_closest_point_d4(self):
        point = np.array([0.6, -1.1, 1.7, 0.1])
        expected = np.array([1, -1, 2, 0])
        result = closest_point_Dn(point)
        np.testing.assert_array_equal(result, expected, "Failed to find the correct closest point in D4")

    def test_closest_point_e8_on_lattice(self):
        point = np.array([0, 1, 1, 0, 0, 0, 0, 0])
        result = closest_point_E8(point)
        np.testing.assert_array_equal(result, point, "Failed to return the correct lattice point in E8")

    def test_closest_point_e8_off_lattice(self):
        point = np.array([0.1, 0.1, 0.8, 1.3, 2.2, -0.6, -0.7, 0.9])
        expected = np.array([0, 0, 1, 1, 2, 0, -1, 1])
        result = closest_point_E8(point)
        np.testing.assert_array_equal(result, expected, "Failed to find the correct closest point in E8")

    def test_closest_point_e8_boundary_case(self):
        point = np.array([0.49, 1.01, 1.01, 0.49, 0, 0, 0, 0])
        expected = np.array([0, 1, 1, 0, 0, 0, 0, 0])
        result = closest_point_E8(point)
        np.testing.assert_array_equal(result, expected, "Failed at a boundary case in E8")

    def test_closest_point_A2(self):
        point = np.array([0.4, -0.4])
        expected = np.array([0.5, -np.sqrt(3)/2])
        result = closest_point_A2(point)
        np.testing.assert_allclose(result, expected, atol=1e-9, rtol=1e-7, err_msg="Failed to find the correct closest point in A2")

    def test_closest_point_A2_on_lattice(self):
        point = np.array([1, 0])
        expected = np.array([1, 0])
        result = closest_point_A2(point)
        np.testing.assert_allclose(result, expected, atol=1e-9, rtol=1e-7, err_msg="Failed to find the correct closest point in A2")

    def test_closest_point_A2_on_lattice_edge(self):
        point = np.array([5.5, np.sqrt(3)/2])
        expected = np.array([5.5, np.sqrt(3)/2])
        result = closest_point_A2(point)
        np.testing.assert_allclose(result, expected, atol=1e-9, rtol=1e-7, err_msg="Failed to find the correct closest point in A2")

    def test_scaled_closest_point_A2(self):
        point = np.array([3, np.sqrt(3)]) + 1e-8
        expected = np.array([2.5, 2.5 * np.sqrt(3)])
        result = 5 * closest_point_A2(point/5)
        np.testing.assert_allclose(result, expected, atol=1e-9, rtol=1e-7,
                                   err_msg="Failed to find the correct closest point in scaled A2")


class TestHelpersForA2(unittest.TestCase):
    def test_upscale_for_A2(self):
        u1 = np.array([0, 0])
        expected_u1 = np.array([0, 0, 0])
        result1 = upscale(u1)
        np.testing.assert_array_equal(result1, expected_u1, f"Failed to upscale {u1}")

        u2 = np.array([1, 0])
        expected_u2 = np.array([1, 0, -1])
        result2 = upscale(u2)
        np.testing.assert_array_equal(result2, expected_u2, f"Failed to upscale {u2}")

        u3 = np.array([0.5, np.sqrt(3) / 2])
        expected_u3 = np.array([1, -1, 0])
        result3 = upscale(u3)
        np.testing.assert_array_equal(result3, expected_u3, f"Failed to upscale {u3}")

        u4 = np.array([-0.5, np.sqrt(3) / 2])
        expected_u4 = np.array([0, -1, 1])
        result4 = upscale(u4)
        np.testing.assert_array_equal(result4, expected_u4, f"Failed to upscale {u4}")

    def test_downscale_for_A2(self):
        u1 = np.array([0, 0, 0])
        result1 = downscale(u1)
        expected_u1 = np.array([0, 0])
        np.testing.assert_array_equal(result1, expected_u1, f"Failed to downscale {u1}")

        u2 = np.array([1, 0, -1])
        result2 = downscale(u2)
        expected_u2 = np.array([1, 0])
        np.testing.assert_array_equal(result2, expected_u2, f"Failed to downscale {u2}")

        u3 = np.array([1, -1, 0])
        result3 = downscale(u3)
        expected_u3 = np.array([0.5, np.sqrt(3) / 2])
        np.testing.assert_allclose(result3, expected_u3, atol=1e-9, rtol=1e-7, err_msg=f"Failed to downscale {u3}")

        u4 = np.array([0, -1, 1])
        result4 = downscale(u4)
        expected_u4 = np.array([-0.5, np.sqrt(3) / 2])
        np.testing.assert_allclose(result4, expected_u4, atol=1e-9, rtol=1e-7, err_msg=f"Failed to downscale {u4}")


if __name__ == '__main__':
    unittest.main()
