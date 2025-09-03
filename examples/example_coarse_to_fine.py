#!/usr/bin/env python3
"""
Simple example demonstrating coarse-to-fine decoding in lattice quantization.

This script shows the basic usage of the coarse-to-fine decoding functionality.
"""

import numpy as np

from gemvq.gemv.columnwise.column_wise_gemv import ColumnWiseGEMV


def main():
    """Demonstrate basic coarse-to-fine decoding."""

    print("=== Simple Coarse-to-Fine Decoding Example ===\n")

    # Create a simple test case
    matrix = np.random.randn(20, 10)
    vector = np.random.randn(10)

    print(f"Matrix shape: {matrix.shape}")
    print(f"Vector dimension: {vector.shape}")

    # Initialize processor with M=3 levels
    processor = ColumnWiseGEMV(matrix, "D4", M=3)
    print(f"Number of hierarchical levels (M): {processor.M}")

    # Compute exact result for comparison
    exact_result = matrix @ vector
    print(f"\nExact result norm: {np.linalg.norm(exact_result):.6f}")

    # Test coarse-to-fine decoding at different levels
    print("\n--- Coarse-to-Fine Decoding Results ---")

    for level in range(processor.M):
        result = processor.multiply_coarse_to_fine(vector, max_level=level)
        error = np.linalg.norm(result - exact_result) / np.linalg.norm(exact_result)
        print(f"Level {level}: Relative error = {error:.6f}")

    # Test full decoding
    full_result = processor.multiply_coarse_to_fine(vector, max_level=None)
    full_error = np.linalg.norm(full_result - exact_result) / np.linalg.norm(
        exact_result
    )
    print(f"Full decoding: Relative error = {full_error:.6f}")

    # Test progressive decoding
    print("\n--- Progressive Decoding Results ---")
    progressive_results = processor.multiply_progressive(vector)

    for i, result in enumerate(progressive_results):
        error = np.linalg.norm(result - exact_result) / np.linalg.norm(exact_result)
        print(f"Progressive step {i}: Relative error = {error:.6f}")

    print("\n=== Example Completed Successfully! ===")


if __name__ == "__main__":
    main()
