#!/usr/bin/env python3
"""
Test script for the new adaptive matrix-vector multiplication approach.

This script tests the implementation where:
1. Matrix W is encoded once with maximum bit rate
2. Columns are decoded adaptively based on bit budget for each x
3. Hierarchical levels M are exploited for variable precision decoding
"""

import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.adaptive.adaptive_matvec import (
    AdaptiveMatVecProcessor,
    create_adaptive_matvec_processor,
    adaptive_matvec_multiply,
    adaptive_matvec_multiply_sparse
)


def test_basic_functionality():
    """Test basic functionality of the adaptive approach."""
    print("🧪 Testing basic functionality...")
    
    # Setup parameters - use sizes that work well with D4 lattice (4-dimensional)
    m, n = 8, 4  # 8 rows, 4 columns - will be processed in 2 blocks of 4
    max_rate = 6.0
    M = 3
    
    # Create test matrix and vector
    np.random.seed(42)
    matrix = np.random.randn(m, n)
    vector = np.random.randn(n)
    
    # Create processor
    processor = create_adaptive_matvec_processor(
        matrix, 'D4', max_rate, M
    )
    
    # Test with different column rates
    column_rates_max = [max_rate] * n
    column_rates_var = np.random.uniform(2.0, max_rate, n)
    column_rates_low = np.random.uniform(1.0, 3.0, n)
    
    # Perform computations
    result_max = processor.compute_matvec(vector, column_rates_max, use_lookup=False)
    result_var = processor.compute_matvec(vector, column_rates_var.tolist(), use_lookup=False)
    result_low = processor.compute_matvec(vector, column_rates_low.tolist(), use_lookup=False)
    
    # Compare with exact
    exact_result = matrix @ vector
    error_max = np.linalg.norm(result_max - exact_result) / np.linalg.norm(exact_result)
    error_var = np.linalg.norm(result_var - exact_result) / np.linalg.norm(exact_result)
    error_low = np.linalg.norm(result_low - exact_result) / np.linalg.norm(exact_result)
    
    print(f"   Max rate error: {error_max:.6f}")
    print(f"   Variable rate error: {error_var:.6f}")
    print(f"   Low rate error: {error_low:.6f}")
    
    # Check that errors are reasonable
    assert error_max < 0.1, f"Max rate error too high: {error_max}"
    assert error_var < 0.2, f"Variable rate error too high: {error_var}"
    assert error_low < 0.5, f"Low rate error too high: {error_low}"
    
    print("   ✅ Basic functionality test passed!")
    return True


def test_sparsity_handling():
    """Test handling of sparse vectors."""
    print("🧪 Testing sparsity handling...")
    
    # Setup parameters - use sizes that work well with D4 lattice (4-dimensional)
    m, n = 8, 4  # 8 rows, 4 columns - will be processed in 2 blocks of 4
    max_rate = 6.0
    M = 3
    
    # Create test matrix
    np.random.seed(42)
    matrix = np.random.randn(m, n)
    
    # Create sparse vector (only 2 non-zero elements)
    sparsity = 2
    non_zero_indices = np.random.choice(n, sparsity, replace=False)
    sparse_vector = np.zeros(n)
    sparse_vector[non_zero_indices] = np.random.randn(sparsity)
    
    # Define column rates
    column_rates = np.random.uniform(2.0, max_rate, n)
    
    # Perform computation
    result = adaptive_matvec_multiply_sparse(
        matrix, sparse_vector, non_zero_indices.tolist(),
        column_rates.tolist(), 'D4', max_rate, M, use_lookup=False
    )
    
    # Compare with exact
    exact_result = matrix @ sparse_vector
    error = np.linalg.norm(result - exact_result) / np.linalg.norm(exact_result)
    
    print(f"   Sparsity: {sparsity}/{n} = {sparsity/n:.2f}")
    print(f"   Error: {error:.6f}")
    
    # Check that error is reasonable
    assert error < 0.3, f"Sparsity handling error too high: {error}"
    
    print("   ✅ Sparsity handling test passed!")
    return True


def test_hierarchical_levels():
    """Test different hierarchical levels M."""
    print("🧪 Testing hierarchical levels...")
    
    # Setup parameters - use sizes that work well with D4 lattice (4-dimensional)
    m, n = 8, 4  # 8 rows, 4 columns - will be processed in 2 blocks of 4
    max_rate = 6.0
    
    # Create test matrix and vector
    np.random.seed(42)
    matrix = np.random.randn(m, n)
    vector = np.random.randn(n)
    column_rates = np.random.uniform(2.0, max_rate, n)
    
    # Test different M values
    M_values = [1, 2, 3, 4]
    errors = []
    
    for M in M_values:
        processor = create_adaptive_matvec_processor(
            matrix, 'D4', max_rate, M
        )
        
        result = processor.compute_matvec(vector, column_rates.tolist(), use_lookup=False)
        exact_result = matrix @ vector
        error = np.linalg.norm(result - exact_result) / np.linalg.norm(exact_result)
        errors.append(error)
        
        print(f"   M={M}: error={error:.6f}")
    
    # Check that errors decrease with increasing M (generally)
    assert errors[0] > errors[1] or errors[1] > errors[2] or errors[2] > errors[3], \
        "Error should generally decrease with increasing M"
    
    print("   ✅ Hierarchical levels test passed!")
    return True


def test_lookup_table_efficiency():
    """Test lookup table efficiency."""
    print("🧪 Testing lookup table efficiency...")
    
    # Setup parameters - use sizes that work well with D4 lattice (4-dimensional)
    m, n = 12, 6  # 12 rows, 6 columns - will be processed in 3 blocks of 4
    max_rate = 6.0
    M = 3
    
    # Create test matrix and vector
    np.random.seed(42)
    matrix = np.random.randn(m, n)
    vector = np.random.randn(n)
    column_rates = np.random.uniform(2.0, max_rate, n)
    
    # Create processor
    processor = create_adaptive_matvec_processor(
        matrix, 'D4', max_rate, M
    )
    
    # Test both approaches
    result_direct = processor.compute_matvec(vector, column_rates.tolist(), use_lookup=False)
    result_lookup = processor.compute_matvec(vector, column_rates.tolist(), use_lookup=True)
    
    # Compare results
    exact_result = matrix @ vector
    error_direct = np.linalg.norm(result_direct - exact_result) / np.linalg.norm(exact_result)
    error_lookup = np.linalg.norm(result_lookup - exact_result) / np.linalg.norm(exact_result)
    
    print(f"   Direct decoding error: {error_direct:.6f}")
    print(f"   Lookup table error: {error_lookup:.6f}")
    
    # Check that both approaches give reasonable results
    assert error_direct < 0.2, f"Direct decoding error too high: {error_direct}"
    assert error_lookup < 0.3, f"Lookup table error too high: {error_lookup}"
    
    print("   ✅ Lookup table efficiency test passed!")
    return True


def test_memory_usage():
    """Test memory usage characteristics."""
    print("🧪 Testing memory usage...")
    
    # Setup parameters - use sizes that work well with D4 lattice (4-dimensional)
    m, n = 12, 6  # 12 rows, 6 columns - will be processed in 3 blocks of 4
    max_rate = 6.0
    M = 3
    
    # Create test matrix
    np.random.seed(42)
    matrix = np.random.randn(m, n)
    
    # Create processor
    processor = create_adaptive_matvec_processor(
        matrix, 'D4', max_rate, M
    )
    
    # Get memory usage
    memory_usage = processor.get_memory_usage()
    compression_ratio = processor.get_compression_ratio()
    
    print(f"   Encoded matrix: {memory_usage['encoded_matrix_mb']:.2f} MB")
    print(f"   Lookup tables: {memory_usage['lookup_tables_mb']:.2f} MB")
    print(f"   Total: {memory_usage['total_mb']:.2f} MB")
    print(f"   Compression ratio: {compression_ratio:.2f}x")
    
    # Check that memory usage is reasonable
    original_mb = m * n * 4 / (1024 * 1024)  # 4 bytes per float
    assert memory_usage['total_mb'] < original_mb * 2, \
        f"Memory usage too high: {memory_usage['total_mb']:.2f} MB vs {original_mb:.2f} MB original"
    
    print("   ✅ Memory usage test passed!")
    return True


def test_edge_cases():
    """Test edge cases and error handling."""
    print("🧪 Testing edge cases...")
    
    # Test with very small matrix
    m, n = 4, 2
    max_rate = 4.0
    M = 2
    
    matrix = np.random.randn(m, n)
    vector = np.random.randn(n)
    column_rates = [max_rate] * n
    
    processor = create_adaptive_matvec_processor(matrix, 'D4', max_rate, M)
    result = processor.compute_matvec(vector, column_rates, use_lookup=False)
    exact_result = matrix @ vector
    error = np.linalg.norm(result - exact_result) / np.linalg.norm(exact_result)
    
    print(f"   Small matrix error: {error:.6f}")
    assert error < 0.5, f"Small matrix error too high: {error}"
    
    # Test with very sparse vector
    sparse_vector = np.zeros(n)
    sparse_vector[0] = 1.0  # Only one non-zero element
    non_zero_indices = [0]
    
    result_sparse = adaptive_matvec_multiply_sparse(
        matrix, sparse_vector, non_zero_indices,
        column_rates, 'D4', max_rate, M, use_lookup=False
    )
    exact_sparse = matrix @ sparse_vector
    error_sparse = np.linalg.norm(result_sparse - exact_sparse) / np.linalg.norm(exact_sparse)
    
    print(f"   Very sparse vector error: {error_sparse:.6f}")
    assert error_sparse < 0.5, f"Very sparse vector error too high: {error_sparse}"
    
    print("   ✅ Edge cases test passed!")
    return True


def main():
    """Run all tests."""
    print("🚀 Testing Adaptive Matrix-Vector Multiplication")
    print("=" * 60)
    print("Testing the new approach where:")
    print("1. Matrix W is encoded once with maximum bit rate")
    print("2. Columns are decoded adaptively based on bit budget for each x")
    print("3. Hierarchical levels M are exploited for variable precision")
    print("4. Sparsity of x is handled efficiently")
    print("=" * 60)
    
    tests = [
        test_basic_functionality,
        test_sparsity_handling,
        test_hierarchical_levels,
        test_lookup_table_efficiency,
        test_memory_usage,
        test_edge_cases
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"   ❌ Test failed: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The implementation is working correctly.")
        return True
    else:
        print("❌ Some tests failed. Please check the implementation.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 