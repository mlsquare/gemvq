#!/usr/bin/env python3
"""
Demo script for coarse-to-fine decoding in lattice quantization.

This script demonstrates how the lattice quantization module supports
decoding from coarse to fine levels, where higher M means coarser quantization.
The reconstruction can be stopped at any level from M-1 down to 0.
"""

import matplotlib.pyplot as plt
import numpy as np

from src.gemv.column_wise_gemv import ColumnWiseGEMV


def demo_coarse_to_fine_decoding():
    """Demonstrate coarse-to-fine decoding functionality."""

    print("=== Coarse-to-Fine Decoding Demo ===\n")

    # Create test matrix and vector
    m, n = 64, 32
    matrix = np.random.randn(m, n)
    vector = np.random.randn(n)

    # Make vector sparse (only 8 non-zero elements)
    sparsity_pattern = np.random.choice(n, 8, replace=False)
    sparse_vector = np.zeros(n)
    sparse_vector[sparsity_pattern] = vector[sparsity_pattern]

    print(f"Matrix shape: {matrix.shape}")
    print(f"Vector dimension: {vector.shape}")
    print(f"Sparsity pattern: {sparsity_pattern}")
    print(f"Number of non-zero elements: {len(sparsity_pattern)}")

    # Initialize processor with M=3 levels
    M = 3
    processor = ColumnWiseGEMV(matrix, "D4", M)

    print(f"\nNumber of hierarchical levels (M): {M}")
    print(f"Lattice type: {processor.lattice_type}")

    # Compute exact result for comparison
    exact_result = matrix @ sparse_vector

    # Test coarse-to-fine decoding at different levels
    print("\n--- Testing Coarse-to-Fine Decoding ---")

    results = {}
    errors = {}

    # Test each level from coarsest (M-1) to finest (0)
    for level in range(M - 1, -1, -1):
        result = processor.multiply_coarse_to_fine(sparse_vector, max_level=level)
        error = np.linalg.norm(result - exact_result) / np.linalg.norm(exact_result)

        results[level] = result
        errors[level] = error

        print(f"Level {level} (coarsest to finest): Relative error = {error:.6f}")

    # Test full decoding (all levels)
    full_result = processor.multiply_coarse_to_fine(sparse_vector, max_level=None)
    full_error = np.linalg.norm(full_result - exact_result) / np.linalg.norm(exact_result)
    print(f"Full decoding (all levels): Relative error = {full_error:.6f}")

    # Test progressive decoding
    print("\n--- Testing Progressive Decoding ---")
    progressive_results = processor.multiply_progressive(sparse_vector)

    print("Progressive reconstruction errors:")
    for i, result in enumerate(progressive_results):
        level = M - 1 - i  # Convert from list index to level
        error = np.linalg.norm(result - exact_result) / np.linalg.norm(exact_result)
        print(f"  Level {level}: Relative error = {error:.6f}")

    # Visualize results
    visualize_results(exact_result, results, progressive_results, M)

    return results, errors, progressive_results


def visualize_results(exact_result, results, progressive_results, M):
    """Visualize the coarse-to-fine decoding results."""

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Coarse-to-Fine Decoding Results", fontsize=16)

    # Plot 1: Individual level results vs exact
    ax1 = axes[0, 0]
    ax1.plot(exact_result, "k-", linewidth=2, label="Exact")
    colors = ["red", "blue", "green"]
    for i, level in enumerate(range(M - 1, -1, -1)):
        ax1.plot(results[level], color=colors[i], alpha=0.7, label=f"Level {level}")
    ax1.set_title("Reconstruction at Different Levels")
    ax1.set_xlabel("Output Index")
    ax1.set_ylabel("Value")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Progressive refinement
    ax2 = axes[0, 1]
    ax2.plot(exact_result, "k-", linewidth=2, label="Exact")
    for i, result in enumerate(progressive_results):
        level = M - 1 - i
        ax2.plot(result, color=colors[i], alpha=0.7, label=f"Level {level}")
    ax2.set_title("Progressive Refinement")
    ax2.set_xlabel("Output Index")
    ax2.set_ylabel("Value")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Error comparison
    ax3 = axes[1, 0]
    levels = list(range(M - 1, -1, -1))
    errors = []
    for level in levels:
        error = np.linalg.norm(results[level] - exact_result) / np.linalg.norm(exact_result)
        errors.append(error)

    ax3.bar(levels, errors, color=colors[: len(levels)])
    ax3.set_title("Relative Error by Level")
    ax3.set_xlabel("Decoding Level")
    ax3.set_ylabel("Relative Error")
    ax3.grid(True, alpha=0.3)

    # Plot 4: Progressive error
    ax4 = axes[1, 1]
    progressive_errors = []
    for result in progressive_results:
        error = np.linalg.norm(result - exact_result) / np.linalg.norm(exact_result)
        progressive_errors.append(error)

    ax4.plot(
        range(len(progressive_errors)),
        progressive_errors,
        "o-",
        color="purple",
        linewidth=2,
        markersize=8,
    )
    ax4.set_title("Progressive Error Reduction")
    ax4.set_xlabel("Refinement Step")
    ax4.set_ylabel("Relative Error")
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("coarse_to_fine_demo.png", dpi=300, bbox_inches="tight")
    plt.show()

    print(f"\nVisualization saved as 'coarse_to_fine_demo.png'")


def demo_compression_vs_quality():
    """Demonstrate the trade-off between compression and quality."""

    print("\n=== Compression vs Quality Trade-off ===\n")

    # Create test data
    m, n = 128, 64
    matrix = np.random.randn(m, n)
    vector = np.random.randn(n)

    # Test different M values
    M_values = [2, 3, 4]
    results = {}

    for M in M_values:
        print(f"Testing M = {M}")
        processor = ColumnWiseGEMV(matrix, "D4", M)

        # Test at different levels
        level_results = {}
        for level in range(M):
            result = processor.multiply_coarse_to_fine(vector, max_level=level)
            exact_result = matrix @ vector
            error = np.linalg.norm(result - exact_result) / np.linalg.norm(exact_result)
            compression_ratio = processor.get_compression_ratio()

            level_results[level] = {"error": error, "compression_ratio": compression_ratio}

            print(f"  Level {level}: Error = {error:.6f}, Compression = {compression_ratio:.2f}x")

        results[M] = level_results

    # Visualize trade-off
    visualize_compression_tradeoff(results)


def visualize_compression_tradeoff(results):
    """Visualize the compression vs quality trade-off."""

    plt.figure(figsize=(10, 6))

    colors = ["red", "blue", "green"]
    markers = ["o", "s", "^"]

    for i, (M, level_results) in enumerate(results.items()):
        errors = [level_results[level]["error"] for level in range(M)]
        compression_ratios = [level_results[level]["compression_ratio"] for level in range(M)]

        plt.plot(
            compression_ratios,
            errors,
            marker=markers[i],
            color=colors[i],
            linewidth=2,
            markersize=8,
            label=f"M = {M}",
        )

        # Annotate points
        for j, (comp, err) in enumerate(zip(compression_ratios, errors)):
            plt.annotate(
                f"L{j}", (comp, err), xytext=(5, 5), textcoords="offset points", fontsize=8
            )

    plt.xlabel("Compression Ratio")
    plt.ylabel("Relative Error")
    plt.title("Compression vs Quality Trade-off")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xscale("log")
    plt.yscale("log")

    plt.tight_layout()
    plt.savefig("compression_quality_tradeoff.png", dpi=300, bbox_inches="tight")
    plt.show()

    print(f"Trade-off visualization saved as 'compression_quality_tradeoff.png'")


if __name__ == "__main__":
    # Run the main demo
    results, errors, progressive_results = demo_coarse_to_fine_decoding()

    # Run compression vs quality demo
    demo_compression_vs_quality()

    print("\n=== Demo Completed Successfully! ===")
