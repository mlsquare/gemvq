"""
Demo script for Layer-Wise Histogram Matrix-Vector Multiplication.

This script demonstrates the layer-wise histogram technique for efficient
matrix-vector multiplication when matrix columns are stored using hierarchical
nested-lattice quantization. The demo avoids any parameter optimization to
run quickly.
"""

import numpy as np
from .layer_wise_histogram_matvec import LayerWiseHistogramMatVec
from ..quantizers.hierarchical_nested_lattice_quantizer import HierarchicalNestedLatticeQuantizer


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
    
    quantizer = HierarchicalNestedLatticeQuantizer(
        G=G, Q_nn=Q_nn, q=q, beta=1.0, alpha=1.0, 
        eps=1e-8, dither=np.zeros(n), M=M
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
        [1, 0]   # column 6
    ]
    
    # Layer counts (how many layers each column uses)
    M_j = [2, 2, 1, 2, 2, 1]  # columns 3 and 6 use only 1 layer
    
    # Input vector
    x = np.array([1.0, -1.0, 0.0, 0.5, 2.0, -0.5])
    
    print(f"\nInput vector x: {x}")
    print(f"Layer counts M_j: {M_j}")
    print(f"Code indices:")
    for j, b_list in enumerate(b_matrix):
        print(f"  Column {j+1}: {b_list}")
    
    # Compute using layer-wise histogram method
    y_histogram = matvec_obj.matvec(x, b_matrix, M_j)
    
    print(f"\nResult using layer-wise histogram method:")
    print(f"y = {y_histogram}")
    
    # Show layer histograms
    s = matvec_obj.compute_layer_histograms(x, b_matrix, M_j)
    print(f"\nLayer-wise histograms s[m,k]:")
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
    
    quantizer = HierarchicalNestedLatticeQuantizer(
        G=G, Q_nn=Q_nn, q=q, beta=1.0, alpha=1.0, 
        eps=1e-8, dither=np.zeros(n), M=M
    )
    
    matvec_obj = LayerWiseHistogramMatVec(quantizer)
    
    # Create test data with some shared code indices
    b_matrix = [
        [0, 1], [0, 1], [1, 0], [1, 0], [0, 1],  # columns 1-5 share indices
        [1, 0], [0, 1], [1, 0], [0, 1], [1, 0]   # columns 6-10 share indices
    ]
    M_j = [2] * d  # all columns use 2 layers
    
    # Input vector with some zeros
    x = np.array([1.0, 0.5, -1.0, 0.0, 2.0, 0.0, 1.5, -0.5, 0.0, 1.0])
    
    print(f"Matrix dimensions: {n} x {d}")
    print(f"Input vector (non-zero components): {np.sum(x != 0)}/{len(x)}")
    
    # Count unique code indices per layer
    unique_indices_layer0 = len(set(b[j][0] for j in range(d) if M_j[j] > 0))
    unique_indices_layer1 = len(set(b[j][1] for j in range(d) if M_j[j] > 1))
    
    print(f"Unique code indices - Layer 0: {unique_indices_layer0}, Layer 1: {unique_indices_layer1}")
    
    # Compute result
    y = matvec_obj.matvec(x, b_matrix, M_j)
    
    print(f"Result: {y}")
    print(f"Efficiency gain: The histogram method pools {d} columns into {unique_indices_layer0 + unique_indices_layer1} unique codewords")
    
    return y


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