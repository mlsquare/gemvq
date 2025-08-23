#!/usr/bin/env python3
"""
Demo script for the GEMV (General Matrix-Vector Multiplication) module.

This script demonstrates both column-wise and row-wise approaches for
matrix-vector multiplication using lattice quantization with blocking strategies.
"""

import time
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

from .column_wise_gemv import ColumnWiseGEMV, column_wise_gemv
from .lattice_quantized_gemv import (LatticeQuantizedGEMV,
                                     compare_gemv_approaches)
from .row_wise_gemv import RowWiseGEMV, row_wise_gemv


def demo_basic_functionality():
    """Demonstrate basic functionality of both approaches."""
    print("=" * 60)
    print("Basic GEMV Functionality Demo")
    print("=" * 60)

    # Create test data
    m, n = 64, 32
    matrix = np.random.randn(m, n)
    vector = np.random.randn(n)

    print(f"Matrix shape: {matrix.shape}")
    print(f"Vector length: {len(vector)}")

    # Test column-wise approach
    print("\n1. Column-wise GEMV:")
    col_processor = ColumnWiseGEMV(matrix, "D4", 2)
    col_result = col_processor.multiply(vector)

    # Test row-wise approach
    print("2. Row-wise GEMV:")
    row_processor = RowWiseGEMV(matrix, "D4", 2)
    row_result = row_processor.multiply(vector)

    # Compare with exact computation
    exact_result = matrix @ vector

    col_error = np.linalg.norm(col_result - exact_result) / np.linalg.norm(exact_result)
    row_error = np.linalg.norm(row_result - exact_result) / np.linalg.norm(exact_result)

    print(f"Column-wise relative error: {col_error:.6f}")
    print(f"Row-wise relative error: {row_error:.6f}")
    print(
        f"Approach difference: {np.linalg.norm(col_result - row_result) / np.linalg.norm(row_result):.6f}"
    )

    return col_result, row_result, exact_result


def demo_sparsity_support():
    """Demonstrate sparsity support in both approaches."""
    print("\n" + "=" * 60)
    print("Sparsity Support Demo")
    print("=" * 60)

    # Create test data with sparse vector
    m, n = 128, 64
    matrix = np.random.randn(m, n)
    vector = np.random.randn(n)

    # Make vector sparse (only 20% non-zero elements)
    sparsity_ratio = 0.2
    num_nonzero = int(n * sparsity_ratio)
    sparsity_pattern = np.random.choice(n, num_nonzero, replace=False)
    sparse_vector = np.zeros(n)
    sparse_vector[sparsity_pattern] = vector[sparsity_pattern]

    print(f"Matrix shape: {matrix.shape}")
    print(f"Sparsity ratio: {sparsity_ratio:.1%}")
    print(f"Number of non-zero elements: {num_nonzero}")

    # Test with sparsity support
    col_processor = ColumnWiseGEMV(matrix, "D4", 2)
    row_processor = RowWiseGEMV(matrix, "D4", 2)

    col_result = col_processor.multiply_with_sparsity(sparse_vector, sparsity_pattern)
    row_result = row_processor.multiply_with_sparsity(sparse_vector, sparsity_pattern)
    exact_result = matrix @ sparse_vector

    col_error = np.linalg.norm(col_result - exact_result) / np.linalg.norm(exact_result)
    row_error = np.linalg.norm(row_result - exact_result) / np.linalg.norm(exact_result)

    print(f"Column-wise sparsity error: {col_error:.6f}")
    print(f"Row-wise sparsity error: {row_error:.6f}")

    return col_result, row_result, exact_result


def demo_blocking_strategies():
    """Demonstrate blocking strategies with different lattice types."""
    print("\n" + "=" * 60)
    print("Blocking Strategies Demo")
    print("=" * 60)

    # Test different lattice types
    lattice_types = ["D4", "A2", "E8", "Z2", "Z3"]
    matrix_sizes = [(100, 50), (200, 100), (400, 200)]

    results = {}

    for lattice_type in lattice_types:
        print(f"\nLattice type: {lattice_type}")
        results[lattice_type] = {}

        for m, n in matrix_sizes:
            matrix = np.random.randn(m, n)
            vector = np.random.randn(n)

            # Test unified interface
            processor = LatticeQuantizedGEMV(matrix, "auto", lattice_type, 2)
            result = processor.multiply(vector)

            # Get blocking info
            blocking_info = processor.get_blocking_info()
            approach_info = processor.get_approach_info()

            exact_result = matrix @ vector
            error = np.linalg.norm(result - exact_result) / np.linalg.norm(exact_result)

            results[lattice_type][(m, n)] = {
                "error": error,
                "approach": approach_info["selected_approach"],
                "block_size": blocking_info["block_size"],
                "num_blocks": blocking_info["total_blocks"],
                "compression_ratio": processor.get_compression_ratio(),
            }

            print(
                f"  Matrix {m}x{n}: {approach_info['selected_approach']} approach, "
                f"block size {blocking_info['block_size']}, "
                f"error {error:.6f}"
            )

    return results


def demo_performance_comparison():
    """Compare performance of different approaches."""
    print("\n" + "=" * 60)
    print("Performance Comparison Demo")
    print("=" * 60)

    # Test different matrix sizes
    sizes = [(64, 32), (128, 64), (256, 128), (512, 256)]
    results = {}

    for m, n in sizes:
        print(f"\nMatrix size: {m}x{n}")
        matrix = np.random.randn(m, n)
        vector = np.random.randn(n)

        # Create sparse vector
        sparsity_pattern = np.random.choice(n, max(1, n // 4), replace=False)
        sparse_vector = np.zeros(n)
        sparse_vector[sparsity_pattern] = vector[sparsity_pattern]

        # Compare approaches
        comparison = compare_gemv_approaches(matrix, sparse_vector, "D4", 2, sparsity_pattern)

        results[(m, n)] = comparison

        print(f"  Column-wise time: {comparison['column_wise']['time']:.4f}s")
        print(f"  Row-wise time: {comparison['row_wise']['time']:.4f}s")
        print(f"  Recommended: {comparison['recommended_approach']}")
        print(f"  Result difference: {comparison['result_difference']:.6f}")

    return results


def demo_unified_interface():
    """Demonstrate the unified interface with automatic approach selection."""
    print("\n" + "=" * 60)
    print("Unified Interface Demo")
    print("=" * 60)

    # Test different matrix shapes
    test_cases = [
        ((100, 50), "Tall matrix (column-wise preferred)"),
        ((50, 100), "Wide matrix (row-wise preferred)"),
        ((80, 80), "Square matrix (auto-selected)"),
    ]

    for (m, n), description in test_cases:
        print(f"\n{description} ({m}x{n}):")
        matrix = np.random.randn(m, n)
        vector = np.random.randn(n)

        # Test automatic selection
        processor = LatticeQuantizedGEMV(matrix, "auto", "D4", 2)
        result = processor.multiply(vector)

        approach_info = processor.get_approach_info()
        blocking_info = processor.get_blocking_info()

        exact_result = matrix @ vector
        error = np.linalg.norm(result - exact_result) / np.linalg.norm(exact_result)

        print(f"  Selected approach: {approach_info['selected_approach']}")
        print(f"  Aspect ratio: {approach_info['aspect_ratio']:.2f}")
        print(f"  Block size: {blocking_info['block_size']}")
        print(f"  Number of blocks: {blocking_info['total_blocks']}")
        print(f"  Relative error: {error:.6f}")
        print(f"  Compression ratio: {processor.get_compression_ratio():.2f}")


def plot_performance_results(results: Dict):
    """Plot performance comparison results."""
    sizes = list(results.keys())
    col_times = [results[size]["column_wise"]["time"] for size in sizes]
    row_times = [results[size]["row_wise"]["time"] for size in sizes]

    # Create size labels
    size_labels = [f"{m}x{n}" for m, n in sizes]

    x = np.arange(len(sizes))
    width = 0.35

    plt.figure(figsize=(10, 6))
    plt.bar(x - width / 2, col_times, width, label="Column-wise", alpha=0.8)
    plt.bar(x + width / 2, row_times, width, label="Row-wise", alpha=0.8)

    plt.xlabel("Matrix Size")
    plt.ylabel("Execution Time (seconds)")
    plt.title("Performance Comparison: Column-wise vs Row-wise GEMV")
    plt.xticks(x, size_labels)
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("gemv_performance_comparison.png", dpi=300, bbox_inches="tight")
    plt.show()


def run_comprehensive_demo():
    """Run a comprehensive demonstration of the GEMV module."""
    print("LatticeQuant GEMV Module Demo")
    print("=" * 60)

    # Run all demos
    basic_results = demo_basic_functionality()
    sparsity_results = demo_sparsity_support()
    blocking_results = demo_blocking_strategies()
    performance_results = demo_performance_comparison()
    demo_unified_interface()

    # Plot results
    try:
        plot_performance_results(performance_results)
        print("\nPerformance plot saved as 'gemv_performance_comparison.png'")
    except Exception as e:
        print(f"\nCould not create plot: {e}")

    print("\n" + "=" * 60)
    print("Demo completed successfully!")
    print("=" * 60)

    return {
        "basic": basic_results,
        "sparsity": sparsity_results,
        "blocking": blocking_results,
        "performance": performance_results,
    }


if __name__ == "__main__":
    # Run the comprehensive demo
    results = run_comprehensive_demo()
