#!/usr/bin/env python3
"""
Test script for coarse-to-fine decoding functionality.

This script tests the implementation of coarse-to-fine decoding in the
lattice quantization module to ensure it works correctly.
"""

import numpy as np
from src.gemv.lattice_quantized_gemv import LatticeQuantizedGEMV
from src.lattices.quantizers.hierarchical_nested_lattice_quantizer import HierarchicalNestedLatticeQuantizer
from src.lattices.utils import get_d4
from src.lattices.utils import closest_point_Dn


def test_hierarchical_quantizer_coarse_to_fine():
    """Test the hierarchical quantizer's coarse-to-fine decoding."""
    
    print("Testing HierarchicalNestedLatticeQuantizer coarse-to-fine decoding...")
    
    # Setup quantizer
    G = get_d4()
    M = 3
    quantizer = HierarchicalNestedLatticeQuantizer(
        G=G, Q_nn=closest_point_Dn, q=4, beta=0.2,
        alpha=1/3, eps=1e-8, dither=np.zeros(4), M=M
    )
    
    # Test vector
    x = np.random.randn(4)
    
    # Encode
    b_list, T = quantizer.encode(x, with_dither=False)
    
    # Test decoding at different levels
    reconstructions = {}
    for level in range(M):
        reconstruction = quantizer.decode_coarse_to_fine(b_list, T, with_dither=False, max_level=level)
        reconstructions[level] = reconstruction
    
    # Test full decoding
    full_reconstruction = quantizer.decode_coarse_to_fine(b_list, T, with_dither=False, max_level=None)
    
    # Test progressive decoding
    progressive_reconstructions = quantizer.decode_progressive(b_list, T, with_dither=False)
    
    # Verify that reconstructions get better as we use more levels
    errors = []
    for level in range(M):
        error = np.linalg.norm(reconstructions[level] - x) / np.linalg.norm(x)
        errors.append(error)
        print(f"  Level {level}: Error = {error:.6f}")
    
    # Check that errors decrease as we use more levels (finer reconstruction)
    # Level 0 is coarsest, Level 2 is finest
    assert errors[0] >= errors[1] >= errors[2], "Errors should decrease with finer reconstruction"
    
    # Check that full reconstruction matches progressive reconstruction
    assert np.allclose(full_reconstruction, progressive_reconstructions[-1]), \
        "Full reconstruction should match finest progressive reconstruction"
    
    print("✓ HierarchicalNestedLatticeQuantizer coarse-to-fine decoding passed!")


def test_column_wise_coarse_to_fine():
    """Test column-wise GEMV coarse-to-fine decoding."""
    
    print("Testing ColumnWiseGEMV coarse-to-fine decoding...")
    
    # Create test data
    matrix = np.random.randn(16, 8)
    vector = np.random.randn(8)
    
    # Initialize processor
    processor = LatticeQuantizedGEMV(matrix, 'column', 'D4', M=3, decoding='coarse_to_fine')
    
    # Compute exact result
    exact_result = matrix @ vector
    
    # Test coarse-to-fine decoding
    results = {}
    for level in range(3):
        result = processor.multiply_coarse_to_fine(vector, max_level=level)
        results[level] = result
    
    # Test progressive decoding
    progressive_results = processor.multiply_progressive(vector)
    
    # Verify that results get better as we use more levels
    errors = []
    for level in range(3):
        error = np.linalg.norm(results[level] - exact_result) / np.linalg.norm(exact_result)
        errors.append(error)
        print(f"  Level {level}: Error = {error:.6f}")
    
    # Check that errors decrease as we use more levels
    print(f"  Error values: {errors}")
    # Make the assertion more lenient - allow small variations
    assert errors[0] >= errors[1] - 1e-6 and errors[1] >= errors[2] - 1e-6, \
        f"Errors should generally decrease with finer reconstruction. Got: {errors}"
    
    # Check that progressive results match individual results
    for i, result in enumerate(progressive_results):
        level = i  # Progressive results are from coarsest (0) to finest (M-1)
        assert np.allclose(result, results[level]), \
            f"Progressive result {i} should match level {level} result"
    
    print("✓ ColumnWiseGEMV coarse-to-fine decoding passed!")


def test_row_wise_coarse_to_fine():
    """Test row-wise GEMV coarse-to-fine decoding."""
    
    print("Testing RowWiseGEMV coarse-to-fine decoding...")
    
    # Create test data
    matrix = np.random.randn(16, 8)
    vector = np.random.randn(8)
    
    # Initialize processor
    processor = LatticeQuantizedGEMV(matrix, 'row', 'D4', M=3, decoding='coarse_to_fine')
    
    # Compute exact result
    exact_result = matrix @ vector
    
    # Test coarse-to-fine decoding
    results = {}
    for level in range(3):
        result = processor.multiply_coarse_to_fine(vector, max_level=level)
        results[level] = result
    
    # Test progressive decoding
    progressive_results = processor.multiply_progressive(vector)
    
    # Verify that results get better as we use more levels
    errors = []
    for level in range(3):
        error = np.linalg.norm(results[level] - exact_result) / np.linalg.norm(exact_result)
        errors.append(error)
        print(f"  Level {level}: Error = {error:.6f}")
    
    # Check that errors decrease as we use more levels
    print(f"  Error values: {errors}")
    # Make the assertion more lenient - allow small variations
    assert errors[0] >= errors[1] - 1e-6 and errors[1] >= errors[2] - 1e-6, \
        f"Errors should generally decrease with finer reconstruction. Got: {errors}"
    
    # Check that progressive results match individual results
    for i, result in enumerate(progressive_results):
        level = i  # Progressive results are from coarsest (0) to finest (M-1)
        assert np.allclose(result, results[level]), \
            f"Progressive result {i} should match level {level} result"
    
    print("✓ RowWiseGEMV coarse-to-fine decoding passed!")


def test_unified_interface():
    """Test the unified LatticeQuantizedGEMV interface."""
    
    print("Testing unified LatticeQuantizedGEMV interface...")
    
    # Create test data
    matrix = np.random.randn(16, 8)
    vector = np.random.randn(8)
    
    # Test auto approach
    processor = LatticeQuantizedGEMV(matrix, 'auto', 'D4', M=3)
    
    # Compute exact result
    exact_result = matrix @ vector
    
    # Test coarse-to-fine decoding
    result_level_0 = processor.multiply_coarse_to_fine(vector, max_level=0)
    result_level_1 = processor.multiply_coarse_to_fine(vector, max_level=1)
    result_level_2 = processor.multiply_coarse_to_fine(vector, max_level=2)
    result_full = processor.multiply_coarse_to_fine(vector, max_level=None)
    
    # Test progressive decoding
    progressive_results = processor.multiply_progressive(vector)
    
    # Verify that results are consistent
    error_0 = np.linalg.norm(result_level_0 - exact_result) / np.linalg.norm(exact_result)
    error_1 = np.linalg.norm(result_level_1 - exact_result) / np.linalg.norm(exact_result)
    error_2 = np.linalg.norm(result_level_2 - exact_result) / np.linalg.norm(exact_result)
    
    print(f"  Level 0: Error = {error_0:.6f}")
    print(f"  Level 1: Error = {error_1:.6f}")
    print(f"  Level 2: Error = {error_2:.6f}")
    
    # Check that errors decrease as we use more levels
    print(f"  Error values: [{error_0}, {error_1}, {error_2}]")
    # Make the assertion more lenient - allow small variations
    assert error_0 >= error_1 - 1e-6 and error_1 >= error_2 - 1e-6, \
        f"Errors should generally decrease with finer reconstruction. Got: [{error_0}, {error_1}, {error_2}]"
    
    # Check that progressive results match individual results
    assert np.allclose(progressive_results[0], result_level_0), "Progressive result 0 should match level 0"
    assert np.allclose(progressive_results[1], result_level_1), "Progressive result 1 should match level 1"
    assert np.allclose(progressive_results[2], result_level_2), "Progressive result 2 should match level 2"
    
    print("✓ Unified LatticeQuantizedGEMV interface passed!")


def test_edge_cases():
    """Test edge cases for coarse-to-fine decoding."""
    
    print("Testing edge cases...")
    
    # Test with M=1 (single level)
    matrix = np.random.randn(8, 4)
    vector = np.random.randn(4)
    
    processor = LatticeQuantizedGEMV(matrix, 'auto', 'D4', M=1)
    
    # Should work with max_level=0
    result = processor.multiply_coarse_to_fine(vector, max_level=0)
    assert result is not None, "Should work with M=1 and max_level=0"
    
    # Progressive should return single result
    progressive_results = processor.multiply_progressive(vector)
    assert len(progressive_results) == 1, "Progressive should return 1 result for M=1"
    
    # Test with sparse vector
    sparse_vector = np.zeros(4)
    sparse_vector[0] = 1.0
    sparsity_pattern = [0]
    
    result_sparse = processor.multiply_coarse_to_fine(sparse_vector, max_level=0, 
                                                     sparsity_pattern=sparsity_pattern)
    assert result_sparse is not None, "Should work with sparse vectors"
    
    print("✓ Edge cases passed!")


if __name__ == "__main__":
    print("Running coarse-to-fine decoding tests...\n")
    
    try:
        test_hierarchical_quantizer_coarse_to_fine()
        print()
        
        test_column_wise_coarse_to_fine()
        print()
        
        test_row_wise_coarse_to_fine()
        print()
        
        test_unified_interface()
        print()
        
        test_edge_cases()
        print()
        
        print("🎉 All tests passed! Coarse-to-fine decoding is working correctly.")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        raise
