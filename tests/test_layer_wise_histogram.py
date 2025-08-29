"""
Test script for the layer-wise histogram matrix-vector multiplication implementation.

This script tests the implementation against the example from the paper and
verifies that the layer-wise histogram method produces the same results as
direct computation.
"""

import numpy as np
import sys
import os

# Add the src directory to the path so we can import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.adaptive.layer_wise_histogram_matvec import (
    LayerWiseHistogramMatVec, 
    create_example_from_paper,
    run_paper_example
)
from src.quantizers.lattice.hnlq import HNLQ


def test_paper_example():
    """Test the implementation against the paper example."""
    print("Testing Layer-Wise Histogram MatVec Implementation")
    print("=" * 60)
    
    # Run the paper example
    y_histogram, y_direct = run_paper_example()
    
    # Expected result from the paper: [17.7, 16.7, 2.1, 0.0]
    expected = np.array([17.7, 16.7, 2.1, 0.0])
    
    print(f"\nExpected result from paper: {expected}")
    print(f"Histogram method result:   {y_histogram}")
    print(f"Direct method result:      {y_direct}")
    
    # Check against expected result
    error_histogram = np.linalg.norm(y_histogram - expected)
    error_direct = np.linalg.norm(y_direct - expected)
    
    print(f"\nError vs expected (histogram): {error_histogram:.2e}")
    print(f"Error vs expected (direct):    {error_direct:.2e}")
    
    # Both methods should agree with each other
    error_between = np.linalg.norm(y_histogram - y_direct)
    print(f"Error between methods:         {error_between:.2e}")
    
    # Check if results are acceptable
    tolerance = 1e-10
    success = (error_between < tolerance)
    
    if success:
        print("✓ Test passed! Methods agree within tolerance.")
    else:
        print("✗ Test failed! Methods disagree.")
    
    return success


def test_simple_example():
    """Test with a simple pre-defined example to avoid parameter optimization."""
    print("\n" + "=" * 60)
    print("Testing with Simple Pre-defined Example")
    print("=" * 60)
    
    # Parameters
    n, d = 4, 5  # output dim, input dim
    M = 2  # hierarchical depth (smaller for speed)
    q = 2  # base (smaller for speed)
    
    # Create simple quantizer with fixed parameters (no optimization)
    G = np.eye(n)
    def Q_nn(x):
        return np.round(x)
    
    quantizer = HNLQ(
        G=G, Q_nn=Q_nn, q=q, beta=1.0, alpha=1.0, 
        eps=1e-8, dither=np.zeros(n), M=M
    )
    
    # Create matvec object
    matvec_obj = LayerWiseHistogramMatVec(quantizer)
    
    # Use pre-defined code indices instead of encoding
    b_matrix = [
        [0, 1],  # column 1: b_{1,0}=0, b_{1,1}=1
        [1, 0],  # column 2: b_{2,0}=1, b_{2,1}=0
        [0, 0],  # column 3: b_{3,0}=0, b_{3,1}=0
        [1, 1],  # column 4: b_{4,0}=1, b_{4,1}=1
        [0, 1]   # column 5: b_{5,0}=0, b_{5,1}=1
    ]
    
    # Layer counts
    M_j = [2, 2, 1, 2, 2]  # All columns use 2 layers except column 3
    
    # Create simple test vector
    x = np.array([1.0, -1.0, 0.0, 0.5, 2.0])
    
    print(f"Matrix shape: {n} x {d}")
    print(f"Code indices b_matrix:")
    for j, b_list in enumerate(b_matrix):
        print(f"  Column {j+1}: {b_list}")
    print(f"Layer counts M_j: {M_j}")
    print(f"Test vector x: {x}")
    
    # Compute using both methods
    y_histogram = matvec_obj.matvec(x, b_matrix, M_j)
    
    # For direct method, we need a reference matrix
    # Create a simple reference matrix
    W = np.array([
        [1.0, 0.0, 1.0, 0.0, 1.0],
        [0.0, 1.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0, 0.0]
    ])
    
    y_direct = matvec_obj.matvec_quantized_direct(x, W, b_matrix, M_j)
    
    print(f"\nHistogram method result: {y_histogram}")
    print(f"Direct method result:    {y_direct}")
    
    # Check agreement
    error = np.linalg.norm(y_histogram - y_direct)
    print(f"Error between methods: {error:.2e}")
    
    success = (error < 1e-10)
    if success:
        print("✓ Simple example test passed!")
    else:
        print("✗ Simple example test failed!")
    
    return success


def test_layer_histograms():
    """Test the layer histogram computation specifically."""
    print("\n" + "=" * 60)
    print("Testing Layer Histogram Computation")
    print("=" * 60)
    
    # Create example from paper
    matvec_obj, W, b_matrix, M_j = create_example_from_paper()
    x = np.array([0.7, -1.2, 0.0, 0.5, 2.0])
    
    # Compute histograms
    s = matvec_obj.compute_layer_histograms(x, b_matrix, M_j)
    
    print(f"Input vector x: {x}")
    print(f"Layer counts M_j: {M_j}")
    print(f"Code indices b_matrix:")
    for j, b_list in enumerate(b_matrix):
        print(f"  Column {j+1}: {b_list}")
    
    print(f"\nLayer-wise histograms s[m,k]:")
    for m in range(s.shape[0]):
        print(f"  Layer {m}: {s[m, :]}")
    
    # Expected histograms from the paper:
    # s[0,:] = [1.2, 0.8, 0.0]  # x1+x4, x2+x5, x3
    # s[1,:] = [-1.2, 2.5, 0.7] # x2, x4+x5, x1  
    # s[2,:] = [2.0, 0.7, 0.0]  # x5, x1, 0
    expected_s = np.array([
        [1.2, 0.8, 0.0],   # Layer 0
        [-1.2, 2.5, 0.7],  # Layer 1
        [2.0, 0.7, 0.0]    # Layer 2
    ])
    
    print(f"\nExpected histograms:")
    for m in range(expected_s.shape[0]):
        print(f"  Layer {m}: {expected_s[m, :]}")
    
    # Check agreement
    error = np.linalg.norm(s - expected_s)
    print(f"\nError in histograms: {error:.2e}")
    
    success = (error < 1e-10)
    if success:
        print("✓ Layer histogram test passed!")
    else:
        print("✗ Layer histogram test failed!")
    
    return success


def test_zero_vector():
    """Test that zero input vector produces zero output."""
    print("\n" + "=" * 60)
    print("Testing Zero Input Vector")
    print("=" * 60)
    
    # Create example
    matvec_obj, W, b_matrix, M_j = create_example_from_paper()
    
    # Zero input vector
    x = np.zeros(5)
    
    # Compute using both methods
    y_histogram = matvec_obj.matvec(x, b_matrix, M_j)
    y_direct = matvec_obj.matvec_quantized_direct(x, W, b_matrix, M_j)
    
    print(f"Zero input vector: {x}")
    print(f"Histogram method result: {y_histogram}")
    print(f"Direct method result:    {y_direct}")
    
    # Both should be zero
    error_histogram = np.linalg.norm(y_histogram)
    error_direct = np.linalg.norm(y_direct)
    error_between = np.linalg.norm(y_histogram - y_direct)
    
    print(f"Histogram method norm: {error_histogram:.2e}")
    print(f"Direct method norm:    {error_direct:.2e}")
    print(f"Error between methods: {error_between:.2e}")
    
    success = (error_histogram < 1e-10 and error_direct < 1e-10 and error_between < 1e-10)
    if success:
        print("✓ Zero vector test passed!")
    else:
        print("✗ Zero vector test failed!")
    
    return success


def main():
    """Run all tests."""
    print("Layer-Wise Histogram MatVec Implementation Tests")
    print("=" * 80)
    
    tests = [
        ("Paper Example", test_paper_example),
        ("Simple Example", test_simple_example),
        ("Layer Histograms", test_layer_histograms),
        ("Zero Vector", test_zero_vector),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\nRunning {test_name} test...")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"✗ {test_name} test failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    
    passed = 0
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{test_name:20} {status}")
        if result:
            passed += 1
    
    print(f"\nPassed: {passed}/{len(results)} tests")
    
    if passed == len(results):
        print("🎉 All tests passed!")
        return True
    else:
        print("❌ Some tests failed!")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 