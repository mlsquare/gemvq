from closest_point import closest_point_Dn, closest_point_A2
from nested_lattice_quantizer import NestedQuantizer, Quantizer
from utils import get_d2, get_a2
import numpy as np


G = get_d2()

v_range = [0, 1, 2, -1, -2]
test_vectors = [[0,0], [1,1], [2,0], [1, -1], [0, -2],
                [2,0], [-1, -1], [-2, 0], [-1, 1], [0,2]]

# Analyze quantizer behavior
print(f"{'Vector':<20} {'2Q_L(x/2)':<30}")
print("-" * 80)
for x in test_vectors:
    y = [x[0] - 1e-8, x[1] + 1e-9]
    q_scaled = 2 * closest_point_Dn(np.array(y)/2)
    print(f"{x} {q_scaled}")