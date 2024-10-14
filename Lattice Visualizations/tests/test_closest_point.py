import unittest
import numpy as np

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from closest_point import closest_point_Dn, closest_point_E8


class TestClosestPointAlgorithms(unittest.TestCase):

    def test_closest_point_d3_on_lattice(self):
        point = np.array([1, 0, 0])
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


if __name__ == '__main__':
    unittest.main()
