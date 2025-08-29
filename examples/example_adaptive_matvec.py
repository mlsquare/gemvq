#!/usr/bin/env python3
"""
Example: Adaptive Matrix-Vector Multiplication

This script demonstrates the adaptive matrix-vector multiplication functionality
using hierarchical nested quantizers with column-wise encoding.
"""

import numpy as np

from gemvq.gemv.adaptive_processor import (adaptive_matvec_multiply,
                                          create_adaptive_matvec_processor)


def main():
    """Main demonstration function."""
    print("Adaptive Matrix-Vector Multiplication Example")
    print("=" * 50)

    # Create test data - use dimensions compatible with D4 lattice (4D)
    m, n = 4, 8  # 4 rows (D4 lattice dimension), 8 columns
    print(f"Creating {m}x{n} matrix...")

    # Random matrix
    matrix = np.random.randn(m, n)

    # Create sparse vector (only 3 non-zero elements)
    sparsity_pattern = [0, 4, 7]  # Indices of non-zero elements
    sparse_vector = np.zeros(n)
    sparse_vector[sparsity_pattern] = [1.5, 2.0, -1.0]

    # Define target bit rates for each column (different rates)
    target_rates = [2.5, 3.0, 3.5, 2.0, 4.0, 3.2, 2.8, 3.6]

    print(f"Matrix shape: {matrix.shape}")
    print(f"Sparsity pattern: {sparsity_pattern}")
    print(f"Non-zero elements: {sparse_vector[sparsity_pattern]}")
    print(f"Target rates range: {min(target_rates):.1f} - {max(target_rates):.1f} bits/dimension")

    # Perform exact computation for comparison
    print("\nPerforming exact matrix-vector multiplication...")
    exact_result = matrix @ sparse_vector

    # Perform adaptive computation
    print("Performing adaptive matrix-vector multiplication...")
    adaptive_result = adaptive_matvec_multiply(
        matrix, sparse_vector, target_rates, sparsity_pattern, "D4", 2
    )

    # Compare results
    error = np.linalg.norm(adaptive_result - exact_result) / np.linalg.norm(exact_result)

    print(f"\nResults:")
    print(f"Exact result norm: {np.linalg.norm(exact_result):.6f}")
    print(f"Adaptive result norm: {np.linalg.norm(adaptive_result):.6f}")
    print(f"Relative error: {error:.6f}")

    # Create processor for additional analysis
    print("\nCreating processor for detailed analysis...")
    processor = create_adaptive_matvec_processor(matrix, target_rates, sparsity_pattern, "D4", 2)

    # Get compression statistics
    compression_ratio = processor.get_compression_ratio()
    memory_usage = processor.get_memory_usage()

    print(f"Compression ratio: {compression_ratio:.2f}x")
    print(f"Memory usage: {memory_usage['total_mb']:.2f} MB")
    print(f"Encoded columns: {memory_usage['encoded_columns_mb']:.2f} MB")
    print(f"Lookup tables: {memory_usage['lookup_tables_mb']:.2f} MB")

    # Test with different lattice types (only compatible ones)
    print("\nTesting different lattice types...")
    # Only test D4 since our matrix is 4D
    lattice_types = ["D4"]

    for lattice_type in lattice_types:
        result = adaptive_matvec_multiply(
            matrix, sparse_vector, target_rates, sparsity_pattern, lattice_type, 2
        )
        error = np.linalg.norm(result - exact_result) / np.linalg.norm(exact_result)
        print(f"{lattice_type} lattice - Error: {error:.6f}")

    # Test with different matrix dimensions
    print("\nTesting with different matrix dimensions...")

    # Test 2D matrix with A2 lattice
    matrix_2d = np.random.randn(2, 4)
    target_rates_2d = [2.5, 3.0, 3.5, 2.8]
    sparse_vector_2d = np.zeros(4)
    sparse_vector_2d[[0, 2]] = [1.0, -1.5]
    sparsity_pattern_2d = [0, 2]

    result_2d = adaptive_matvec_multiply(
        matrix_2d, sparse_vector_2d, target_rates_2d, sparsity_pattern_2d, "A2", 2
    )
    exact_result_2d = matrix_2d @ sparse_vector_2d
    error_2d = np.linalg.norm(result_2d - exact_result_2d) / np.linalg.norm(exact_result_2d)
    print(f"A2 lattice (2D matrix) - Error: {error_2d:.6f}")

    print("\nExample completed successfully!")


if __name__ == "__main__":
    main()
