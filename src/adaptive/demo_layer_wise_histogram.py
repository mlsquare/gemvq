"""
Demo script for Layer-Wise Histogram Matrix-Vector Multiplication.

This script demonstrates the layer-wise histogram technique for efficient
matrix-vector multiplication when matrix columns are stored using hierarchical
nested-lattice quantization. The demo avoids any parameter optimization to
run quickly.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
import time

from .layer_wise_histogram_matvec import LayerWiseHistogramMatVec


def plot_histogram_comparison(
    results: List[Tuple[str, float, float, float]],
    title: str = "Layer-wise Histogram Results",
    save_path: str = None,
):
    """Plot layer-wise histogram results."""
    methods = [r[0] for r in results]
    compression_ratios = [r[1] for r in results]
    errors = [r[2] for r in results]
    times = [r[3] for r in results]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

    # Compression ratio comparison
    ax1.bar(methods, compression_ratios, color="skyblue")
    ax1.set_title("Compression Ratio")
    ax1.set_ylabel("Compression Ratio")
    ax1.tick_params(axis="x", rotation=45)

    # Error comparison
    ax2.bar(methods, errors, color="lightcoral")
    ax2.set_title("Normalized Error")
    ax2.set_ylabel("Error")
    ax2.tick_params(axis="x", rotation=45)

    # Time comparison
    ax3.bar(methods, times, color="lightgreen")
    ax3.set_title("Computation Time")
    ax3.set_ylabel("Time (seconds)")
    ax3.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.suptitle(title, y=1.02)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def demo_basic_usage():
    """Demonstrate basic usage of the layer-wise histogram matvec."""
    print("Layer-Wise Histogram MatVec Demo")
    print("=" * 50)

    # Parameters
    n, d = 4, 6  # output dim, input dim
    M = 2  # hierarchical depth
    q = 2  # base

    print(f"Matrix dimensions: {n} x {d}")
    print(f"Hierarchical depth: {M}")
    print(f"Base: {q}")

    # Create simple quantizer with fixed parameters
    G = np.eye(n)

    def Q_nn(x):
        return np.round(x)

    quantizer = LayerWiseHistogramMatVec.create_quantizer(
        G=G, Q_nn=Q_nn, q=q, beta=1.0, alpha=1.0, eps=1e-8, dither=np.zeros(n), M=M
    )

    # Create matvec object
    matvec_obj = LayerWiseHistogramMatVec(quantizer)

    # Pre-defined code indices (simulating encoded matrix)
    b_matrix = [
        [0, 1],  # column 1
        [1, 0],  # column 2
        [0, 0],  # column 3
        [1, 1],  # column 4
        [0, 1],  # column 5
        [1, 0],  # column 6
    ]

    # Layer counts (how many layers each column uses)
    M_j = [2, 2, 1, 2, 2, 1]  # columns 3 and 6 use only 1 layer

    # Input vector
    x = np.array([1.0, -1.0, 0.0, 0.5, 2.0, -0.5])

    print("\nInput vector x:", x)
    print("Layer counts M_j:", M_j)
    print("Code indices:")
    for j, b_list in enumerate(b_matrix):
        print(f"  Column {j+1}: {b_list}")

    # Compute using layer-wise histogram method
    y_histogram = matvec_obj.matvec(x, b_matrix, M_j)

    print("\nResult using layer-wise histogram method:")
    print("y =", y_histogram)

    # Show layer histograms
    s = matvec_obj.compute_layer_histograms(x, b_matrix, M_j)
    print("\nLayer-wise histograms s[m,k]:")
    for m in range(s.shape[0]):
        print(f"  Layer {m}: {s[m, :]}")

    return y_histogram


def demo_paper_example():
    """Demonstrate the example from the paper."""
    print("\n" + "=" * 50)
    print("Paper Example Demo")
    print("=" * 50)

    from .layer_wise_histogram_matvec import run_paper_example

    # Run the paper example
    y_histogram, y_direct = run_paper_example()

    print(f"Paper example completed successfully!")
    print(f"Both methods agree: {np.allclose(y_histogram, y_direct, atol=1e-10)}")

    return y_histogram, y_direct


def demo_efficiency_comparison():
    """Demonstrate the efficiency of the layer-wise histogram approach."""
    print("\n" + "=" * 50)
    print("Efficiency Comparison Demo")
    print("=" * 50)

    # Parameters
    n, d = 8, 10
    M = 2
    q = 2

    # Create quantizer
    G = np.eye(n)

    def Q_nn(x):
        return np.round(x)

    quantizer = LayerWiseHistogramMatVec.create_quantizer(
        G=G, Q_nn=Q_nn, q=q, beta=1.0, alpha=1.0, eps=1e-8, dither=np.zeros(n), M=M
    )

    matvec_obj = LayerWiseHistogramMatVec(quantizer)

    # Create test data with some shared code indices
    b_matrix = [
        [0, 1],
        [0, 1],
        [1, 0],
        [1, 0],
        [0, 1],  # columns 1-5 share indices
        [1, 0],
        [0, 1],
        [1, 0],
        [0, 1],
        [1, 0],  # columns 6-10 share indices
    ]
    M_j = [2] * d  # all columns use 2 layers

    # Input vector with some zeros
    x = np.array([1.0, 0.5, -1.0, 0.0, 2.0, 0.0, 1.5, -0.5, 0.0, 1.0])

    print(f"Matrix dimensions: {n} x {d}")
    print(f"Input vector (non-zero components): {np.sum(x != 0)}/{len(x)}")

    # Count unique code indices per layer
    unique_indices_layer0 = len(set(b_matrix[j][0] for j in range(d) if M_j[j] > 0))
    unique_indices_layer1 = len(set(b_matrix[j][1] for j in range(d) if M_j[j] > 1))

    print(
        f"Unique code indices - Layer 0: {unique_indices_layer0}, Layer 1: {unique_indices_layer1}"
    )

    # Compute result
    y = matvec_obj.matvec(x, b_matrix, M_j)

    print(f"Result: {y}")
    total_unique = unique_indices_layer0 + unique_indices_layer1
    print(
        f"Efficiency gain: The histogram method pools {d} columns into "
        f"{total_unique} unique codewords"
    )

    return y


def demo_layer_wise_histogram():
    """Demonstrate layer-wise histogram matrix-vector multiplication."""
    print("Layer-wise Histogram Matrix-Vector Multiplication Demo")
    print("=" * 60)

    # Create test data
    m, n = 50, 20
    matrix = np.random.uniform(0, 1, (m, n)) * 64  # Scale by q^M
    vector = np.random.uniform(0, 1, n) * 64

    print(f"Matrix shape: {matrix.shape}")
    print(f"Vector shape: {vector.shape}")

    # Initialize layer-wise histogram processor
    processor = LayerWiseHistogramMatVec(
        matrix, lattice_type="D4", M=3, num_bins=10
    )

    # Perform multiplication
    start_time = time.time()
    result = processor.multiply(vector)
    computation_time = time.time() - start_time

    # Compare with exact computation
    exact_result = matrix @ vector
    error = np.linalg.norm(result - exact_result) / np.linalg.norm(exact_result)

    print(f"Result shape: {result.shape}")
    print(f"Relative error: {error:.6f}")
    print(f"Computation time: {computation_time:.6f} seconds")
    print(f"Compression ratio: {processor.get_compression_ratio():.2f}x")

    return matrix, vector, result, exact_result, processor


def demo_histogram_analysis():
    """Analyze histogram characteristics."""
    print("\n=== Histogram Analysis ===")

    # Create test data
    m, n = 40, 15
    matrix = np.random.uniform(0, 1, (m, n)) * 64
    vector = np.random.uniform(0, 1, n) * 64

    # Test different bin counts
    bin_counts = [5, 10, 15, 20]
    results = {}

    for num_bins in bin_counts:
        print(f"\nTesting with {num_bins} bins...")

        processor = LayerWiseHistogramMatVec(
            matrix, lattice_type="D4", M=3, num_bins=num_bins
        )

        start_time = time.time()
        result = processor.multiply(vector)
        computation_time = time.time() - start_time

        exact_result = matrix @ vector
        error = np.linalg.norm(result - exact_result) / np.linalg.norm(exact_result)

        results[num_bins] = {
            "error": error,
            "time": computation_time,
            "compression_ratio": processor.get_compression_ratio(),
        }

        print(f"  Error: {error:.6f}")
        print(f"  Time: {computation_time:.6f}s")
        print(f"  Compression: {processor.get_compression_ratio():.2f}x")

    return results


def run_demo():
    """Run the complete demo."""
    print("Layer-Wise Histogram MatVec Implementation Demo")
    print("=" * 80)

    try:
        # Basic usage demo
        demo_basic_usage()

        # Paper example demo
        demo_paper_example()

        # Efficiency comparison demo
        demo_efficiency_comparison()

        # Layer-wise histogram matrix-vector multiplication demo
        demo_layer_wise_histogram()

        # Histogram analysis demo
        demo_histogram_analysis()

        print("\n" + "=" * 80)
        print("Demo completed successfully!")
        print("The layer-wise histogram method efficiently computes matrix-vector")
        print("multiplication by pooling identical codewords at each layer.")

    except Exception as e:
        print(f"Demo failed with error: {e}")
        return False

    return True


if __name__ == "__main__":
    run_demo()
