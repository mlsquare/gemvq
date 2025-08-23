#!/usr/bin/env python3
"""
Test to verify that final error with all levels is lower than initial coarse error.
"""

import numpy as np
from src.quantizers.hierarchical_nested_lattice_quantizer import HierarchicalNestedLatticeQuantizer
from src.utils import get_d4
from src.quantizers.closest_point import closest_point_Dn


def test_final_error_improvement():
    """Test that final error is lower than initial coarse error."""
    
    print("=== Testing Final Error Improvement ===\n")
    
    # Setup quantizer
    G = get_d4()
    M = 3
    q = 4
    quantizer = HierarchicalNestedLatticeQuantizer(
        G=G, Q_nn=closest_point_Dn, q=q, beta=0.2,
        alpha=1/3, eps=1e-8, dither=np.zeros(4), M=M
    )
    
    num_trials = 50
    improvement_count = 0
    
    for trial in range(num_trials):
        # Test vector
        x = np.random.randn(4)
        
        # Encode and decode
        b_list, T = quantizer.encode(x, with_dither=False)
        
        # Get error at each level
        errors = []
        for level in range(M):
            reconstruction = quantizer.decode_coarse_to_fine(b_list, T, with_dither=False, max_level=level)
            error = np.linalg.norm(reconstruction - x) / np.linalg.norm(x)
            errors.append(error)
        
        # Check if final error is lower than initial error
        if errors[-1] < errors[0]:
            improvement_count += 1
        
        if trial < 5:  # Show first 5 trials
            print(f"Trial {trial+1}: Initial error = {errors[0]:.6f}, Final error = {errors[-1]:.6f}, Improvement = {errors[0]/errors[-1]:.3f}x")
    
    print(f"\nFinal error improvement rate: {improvement_count}/{num_trials} = {improvement_count/num_trials*100:.1f}%")
    
    if improvement_count == num_trials:
        print("✅ SUCCESS: Final error is always lower than initial error!")
    else:
        print(f"⚠️  WARNING: {num_trials - improvement_count} cases where final error is not lower")


def test_large_matrix_with_fixed_quantizer():
    """Test the fixed quantizer on large matrices."""
    
    print("\n=== Testing Large Matrix with Fixed Quantizer ===\n")
    
    from src.gemv.lattice_quantized_gemv import LatticeQuantizedGEMV
    
    # Create large matrix and vector
    m, n = 500, 250
    matrix = np.random.randn(m, n)
    vector = np.random.randn(n)
    
    # Make vector sparse
    num_nonzero = max(1, int(0.2 * n))
    sparsity_pattern = np.random.choice(n, num_nonzero, replace=False)
    sparse_vector = np.zeros(n)
    sparse_vector[sparsity_pattern] = vector[sparsity_pattern]
    
    # Compute exact result
    exact_result = matrix @ sparse_vector
    exact_norm = np.linalg.norm(exact_result)
    
    # Test with M=3
    M = 3
    processor = LatticeQuantizedGEMV(matrix, 'auto', 'D4', M)
    
    print(f"Matrix size: {m} x {n}")
    print(f"Number of levels: {M}")
    print(f"Selected approach: {processor.approach}")
    
    # Test each level
    errors = []
    for level in range(M):
        result = processor.multiply_coarse_to_fine(sparse_vector, max_level=level)
        error = np.linalg.norm(result - exact_result) / exact_norm
        errors.append(error)
        print(f"Level {level}: Error = {error:.6f}")
    
    # Check monotonicity
    is_monotonic = all(errors[i] >= errors[i+1] for i in range(len(errors)-1))
    print(f"Error decreases monotonically: {is_monotonic}")
    
    # Check final improvement
    final_improvement = errors[0] / errors[-1]
    print(f"Final error improvement: {final_improvement:.3f}x")
    
    if final_improvement > 1:
        print("✅ SUCCESS: Final error is lower than initial error!")
    else:
        print("❌ FAILURE: Final error is not lower than initial error")


if __name__ == "__main__":
    test_final_error_improvement()
    test_large_matrix_with_fixed_quantizer()
