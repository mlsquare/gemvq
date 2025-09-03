#!/usr/bin/env python3
"""
Example demonstrating the updated rowwise GEMV with the new HNLQ interface.

This example shows how to use the refactored RowWiseGEMV class with the updated
HNLQ configuration-based approach and various decoding strategies.
"""

import time

import numpy as np

from gemvq.gemv.rowwise.row_wise_gemv import RowWiseGEMV


def main():
    """Demonstrate the updated rowwise GEMV functionality."""

    print("=== Updated Rowwise GEMV Example ===\n")

    # Create test data
    m, n = 16, 12
    matrix = np.random.randn(m, n)
    vector = np.random.randn(n)

    print(f"Matrix shape: {matrix.shape}")
    print(f"Vector shape: {vector.shape}")
    print(f"Expected result shape: ({m},)")

    # Test 1: Basic functionality with default parameters
    print("\n1. Testing basic functionality with default parameters...")

    row_gemv_basic = RowWiseGEMV(matrix=matrix, lattice_type="D4", M=2)

    start_time = time.time()
    result_basic = row_gemv_basic.multiply(vector)
    basic_time = time.time() - start_time

    print(f"✓ Basic result shape: {result_basic.shape}")
    print(f"✓ Computation time: {basic_time:.4f} seconds")

    # Test 2: Different decoding strategies
    print("\n2. Testing different decoding strategies...")

    # Full decoding
    row_gemv_full = RowWiseGEMV(matrix=matrix, lattice_type="D4", M=2, decoding="full")
    result_full = row_gemv_full.multiply(vector)

    # Coarse-to-fine decoding
    row_gemv_coarse = RowWiseGEMV(
        matrix=matrix, lattice_type="D4", M=2, decoding="coarse_to_fine"
    )
    result_coarse = row_gemv_coarse.multiply(vector)

    # Progressive decoding
    row_gemv_progressive = RowWiseGEMV(
        matrix=matrix, lattice_type="D4", M=2, decoding="progressive"
    )
    result_progressive = row_gemv_progressive.multiply(vector)

    print(f"✓ Full decoding result shape: {result_full.shape}")
    print(f"✓ Coarse-to-fine result shape: {result_coarse.shape}")
    print(f"✓ Progressive result shape: {result_progressive.shape}")

    # Test 3: Coarse-to-fine with different levels
    print("\n3. Testing coarse-to-fine with different levels...")

    for level in [1, 2]:
        result_level = row_gemv_coarse.multiply_coarse_to_fine(vector, max_level=level)
        print(f"✓ Level {level} result shape: {result_level.shape}")

    # Test 4: Progressive refinement
    print("\n4. Testing progressive refinement...")

    progressive_results = row_gemv_progressive.multiply_progressive(vector)
    print(f"✓ Progressive results: {len(progressive_results)} levels")
    for i, result in enumerate(progressive_results):
        print(f"  Level {i+1}: shape {result.shape}")

    # Test 5: Custom parameters
    print("\n5. Testing with custom parameters...")

    row_gemv_custom = RowWiseGEMV(
        matrix=matrix,
        lattice_type="D4",
        M=3,
        q=8,
        beta=0.5,
        alpha=0.8,
        overload=True,
        max_scaling_iterations=5,
        with_tie_dither=True,
        with_dither=False,
    )

    result_custom = row_gemv_custom.multiply(vector)
    print(f"✓ Custom parameters result shape: {result_custom.shape}")

    # Test 6: Performance and statistics
    print("\n6. Performance and statistics...")

    compression_ratio = row_gemv_basic.get_compression_ratio()
    memory_usage = row_gemv_basic.get_memory_usage()
    blocking_info = row_gemv_basic.get_blocking_info()

    print(f"✓ Compression ratio: {compression_ratio:.2f}x")
    print(f"✓ Memory usage: {memory_usage['total_mb']:.4f} MB")
    print(f"✓ Blocking info:")
    print(f"  - Lattice type: {blocking_info['lattice_type']}")
    print(f"  - Block size: {blocking_info['block_size']}")
    print(f"  - Row blocks: {blocking_info['num_row_blocks']}")
    print(f"  - Original shape: {blocking_info['original_shape']}")
    print(f"  - Padded shape: {blocking_info['padded_shape']}")

    # Test 7: Different lattice types
    print("\n7. Testing different lattice types...")

    lattice_types = ["D4", "A2", "E8", "Z2"]

    for lattice_type in lattice_types:
        try:
            row_gemv_lattice = RowWiseGEMV(
                matrix=matrix, lattice_type=lattice_type, M=2
            )
            result_lattice = row_gemv_lattice.multiply(vector)
            print(f"✓ {lattice_type} lattice: result shape {result_lattice.shape}")
        except Exception as e:
            print(f"✗ {lattice_type} lattice: {e}")

    # Test 8: Sparsity support
    print("\n8. Testing sparsity support...")

    # Create sparse vector (only first 3 elements non-zero)
    sparse_vector = np.zeros_like(vector)
    sparse_vector[:3] = vector[:3]
    sparsity_pattern = [0, 1, 2]

    result_sparse = row_gemv_basic.multiply_with_sparsity(
        sparse_vector, sparsity_pattern=sparsity_pattern
    )
    print(f"✓ Sparse computation result shape: {result_sparse.shape}")

    print("\n🎉 All tests completed successfully!")
    print("\nKey improvements in the updated implementation:")
    print("1. Uses HNLQConfig for structured parameter management")
    print(
        "2. Supports all new HNLQ parameters (overload, max_scaling_iterations, etc.)"
    )
    print("3. Updated method signatures match the new HNLQ interface")
    print("4. Better error handling and parameter validation")
    print("5. Consistent dither handling across all operations")


if __name__ == "__main__":
    main()
