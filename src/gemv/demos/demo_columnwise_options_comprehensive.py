"""
Comprehensive Demo for Columnwise Matrix-Vector Multiplication Options

This demo showcases all the different columnwise matvec options implemented:
1. Standard Dot Processor (Option 4.1)
2. Adaptive Depth Processor (Option 4.1c)
3. Lookup Table Processor (Option 4.2)

The demo demonstrates:
- Basic functionality of each approach
- Sparsity handling capabilities
- Performance comparisons
- Accuracy analysis
- Different matrix sizes and sparsity levels
"""

import time
import numpy as np
from typing import Dict, List, Tuple

from .simple_columnwise_matvec import create_simple_processor, SimpleColumnwiseMatVecProcessor


def create_test_matrix(m: int, n: int) -> np.ndarray:
    """Create a test matrix with random values."""
    return np.random.randn(m, n)


def create_sparse_vector(n: int, sparsity_ratio: float) -> np.ndarray:
    """Create a sparse vector with specified sparsity ratio."""
    x = np.random.randn(n)
    
    # Make some elements zero based on sparsity ratio
    num_zeros = int(n * sparsity_ratio)
    zero_indices = np.random.choice(n, num_zeros, replace=False)
    x[zero_indices] = 0.0
    
    return x


def benchmark_strategies(
    matrix: np.ndarray,
    x: np.ndarray,
    strategies: List[str],
    num_runs: int = 5
) -> Dict[str, Dict[str, float]]:
    """Benchmark different computation strategies."""
    results = {}
    
    for strategy in strategies:
        processor = create_simple_processor(matrix, strategy=strategy)
        
        times = []
        errors = []
        
        for _ in range(num_runs):
            start_time = time.time()
            result = processor.compute_matvec(x)
            times.append(time.time() - start_time)
            
            # Compute error (compare with direct multiplication)
            direct_result = matrix @ x
            error = np.linalg.norm(result - direct_result) / np.linalg.norm(direct_result)
            errors.append(error)
        
        results[strategy] = {
            'mean_time': np.mean(times),
            'std_time': np.std(times),
            'mean_error': np.mean(errors),
            'std_error': np.std(errors)
        }
    
    return results


def demo_basic_functionality():
    """Demonstrate basic functionality of each approach."""
    print("=" * 80)
    print("DEMO 1: Basic Functionality of Columnwise MatVec Options")
    print("=" * 80)
    
    # Create test data
    m, n = 20, 16
    matrix = create_test_matrix(m, n)
    x = np.random.randn(n)
    
    print(f"Matrix dimensions: {m} x {n}")
    print(f"Vector length: {len(x)}")
    print()
    
    # Test all strategies
    strategies = ["standard_dot", "adaptive_depth", "lookup_table"]
    
    for strategy in strategies:
        print(f"Strategy: {strategy.upper()}")
        print("-" * 40)
        
        # Create processor
        processor = create_simple_processor(matrix, strategy=strategy)
        
        # Get matrix info
        info = processor.get_matrix_info()
        print(f"  Lattice type: {info['lattice_type']}")
        print(f"  Hierarchical levels (M): {info['M']}")
        print(f"  Quantization parameter (q): {info['q']}")
        
        # Compute matvec
        start_time = time.time()
        result = processor.compute_matvec(x)
        computation_time = time.time() - start_time
        
        # Compare with direct computation
        direct_result = matrix @ x
        error = np.linalg.norm(result - direct_result) / np.linalg.norm(direct_result)
        
        print(f"  Computation time: {computation_time:.6f} seconds")
        print(f"  Relative error: {error:.2e}")
        print(f"  Result shape: {result.shape}")
        print()


def demo_sparsity_handling():
    """Demonstrate sparsity handling capabilities."""
    print("=" * 80)
    print("DEMO 2: Sparsity Handling Capabilities")
    print("=" * 80)
    
    # Create test data
    m, n = 24, 20
    matrix = create_test_matrix(m, n)
    
    # Test different sparsity levels
    sparsity_levels = [0.0, 0.25, 0.5, 0.75, 0.9]
    
    print(f"Matrix dimensions: {m} x {n}")
    print()
    
    for sparsity in sparsity_levels:
        print(f"Sparsity level: {sparsity:.1%}")
        print("-" * 30)
        
        # Create sparse vector
        x = create_sparse_vector(n, sparsity)
        non_zero_count = np.sum(np.abs(x) > 1e-10)
        print(f"  Non-zero elements: {non_zero_count}/{len(x)}")
        
        # Test with adaptive depth strategy
        processor = create_simple_processor(matrix, strategy="adaptive_depth")
        
        # Compute with sparse vector
        start_time = time.time()
        result_sparse = processor.compute_matvec(x)
        time_sparse = time.time() - start_time
        
        # Create dense vector for comparison
        x_dense = x.copy()
        x_dense[x_dense == 0] = np.random.randn(np.sum(x_dense == 0)) * 0.1
        
        start_time = time.time()
        result_dense = processor.compute_matvec(x_dense)
        time_dense = time.time() - start_time
        
        # Check accuracy
        direct_sparse = matrix @ x
        error_sparse = np.linalg.norm(result_sparse - direct_sparse) / np.linalg.norm(direct_sparse)
        
        print(f"  Sparse computation time: {time_sparse:.6f}s")
        print(f"  Dense computation time: {time_dense:.6f}s")
        print(f"  Speedup: {time_dense/time_sparse:.2f}x")
        print(f"  Relative error: {error_sparse:.2e}")
        print()


def demo_performance_comparison():
    """Demonstrate performance comparison between strategies."""
    print("=" * 80)
    print("DEMO 3: Performance Comparison Between Strategies")
    print("=" * 80)
    
    # Test different matrix sizes
    test_cases = [
        (16, 12, "Small"),
        (32, 24, "Medium"),
        (64, 48, "Large")
    ]
    
    strategies = ["standard_dot", "adaptive_depth", "lookup_table"]
    
    for m, n, size_name in test_cases:
        print(f"Matrix size: {size_name} ({m} x {n})")
        print("-" * 50)
        
        # Create test data
        matrix = create_test_matrix(m, n)
        x = np.random.randn(n)
        
        # Benchmark strategies
        results = benchmark_strategies(matrix, x, strategies, num_runs=3)
        
        # Print results
        for strategy, stats in results.items():
            print(f"  {strategy}:")
            print(f"    Time: {stats['mean_time']:.6f} ± {stats['std_time']:.6f}s")
            print(f"    Error: {stats['mean_error']:.2e} ± {stats['std_error']:.2e}")
        
        print()


def demo_adaptive_depth_analysis():
    """Demonstrate adaptive depth analysis."""
    print("=" * 80)
    print("DEMO 4: Adaptive Depth Analysis")
    print("=" * 80)
    
    # Create test data
    m, n = 32, 24
    matrix = create_test_matrix(m, n)
    
    # Test different input characteristics
    test_cases = [
        ("Dense vector", np.random.randn(n)),
        ("Sparse vector (25%)", create_sparse_vector(n, 0.25)),
        ("Sparse vector (50%)", create_sparse_vector(n, 0.5)),
        ("Sparse vector (75%)", create_sparse_vector(n, 0.75)),
        ("Very sparse vector (90%)", create_sparse_vector(n, 0.9)),
    ]
    
    print(f"Matrix dimensions: {m} x {n}")
    print()
    
    for name, x in test_cases:
        print(f"Input: {name}")
        print("-" * 30)
        
        # Count non-zero elements
        non_zero_count = np.sum(np.abs(x) > 1e-10)
        sparsity_ratio = 1.0 - non_zero_count / len(x)
        print(f"  Non-zero elements: {non_zero_count}/{len(x)}")
        print(f"  Sparsity ratio: {sparsity_ratio:.3f}")
        
        # Test with adaptive depth strategy
        processor = create_simple_processor(matrix, strategy="adaptive_depth")
        
        start_time = time.time()
        result = processor.compute_matvec(x)
        computation_time = time.time() - start_time
        
        # Check accuracy
        direct_result = matrix @ x
        error = np.linalg.norm(result - direct_result) / np.linalg.norm(direct_result)
        
        print(f"  Computation time: {computation_time:.6f}s")
        print(f"  Relative error: {error:.2e}")
        print()


def demo_strategy_recommendations():
    """Provide strategy recommendations based on input characteristics."""
    print("=" * 80)
    print("DEMO 5: Strategy Recommendations")
    print("=" * 80)
    
    print("Strategy Selection Guidelines:")
    print()
    print("1. STANDARD_DOT Strategy:")
    print("   - Best for: Simple cases, high sparsity, small matrices")
    print("   - Pros: Low memory usage, simple implementation")
    print("   - Cons: May be slower for large matrices")
    print()
    
    print("2. ADAPTIVE_DEPTH Strategy:")
    print("   - Best for: Variable sparsity patterns, optimal performance")
    print("   - Pros: Automatically adapts to input characteristics")
    print("   - Cons: More complex implementation")
    print()
    
    print("3. LOOKUP_TABLE Strategy:")
    print("   - Best for: Large matrices, low sparsity, repeated computations")
    print("   - Pros: Fast computation, efficient for repeated operations")
    print("   - Cons: Higher memory usage, requires precomputation")
    print()
    
    # Demonstrate with examples
    print("Example Recommendations:")
    print("-" * 30)
    
    examples = [
        ("Small sparse matrix (16x12, 80% sparsity)", "standard_dot"),
        ("Large dense matrix (128x96, 10% sparsity)", "lookup_table"),
        ("Medium variable matrix (64x48, 50% sparsity)", "adaptive_depth"),
        ("Repeated computations", "lookup_table"),
        ("Memory-constrained environment", "standard_dot"),
    ]
    
    for description, recommendation in examples:
        print(f"  {description}: {recommendation}")


def demo_accuracy_analysis():
    """Demonstrate accuracy analysis of different strategies."""
    print("=" * 80)
    print("DEMO 6: Accuracy Analysis")
    print("=" * 80)
    
    # Create test data
    m, n = 40, 32
    matrix = create_test_matrix(m, n)
    x = np.random.randn(n)
    
    print(f"Matrix dimensions: {m} x {n}")
    print()
    
    # Test all strategies
    strategies = ["standard_dot", "adaptive_depth", "lookup_table"]
    
    print("Accuracy Comparison:")
    print("-" * 25)
    
    for strategy in strategies:
        processor = create_simple_processor(matrix, strategy=strategy)
        
        # Compute result
        result = processor.compute_matvec(x)
        
        # Compare with direct computation
        direct_result = matrix @ x
        
        # Compute various error metrics
        relative_error = np.linalg.norm(result - direct_result) / np.linalg.norm(direct_result)
        max_error = np.max(np.abs(result - direct_result))
        mean_error = np.mean(np.abs(result - direct_result))
        
        print(f"  {strategy}:")
        print(f"    Relative error: {relative_error:.2e}")
        print(f"    Max absolute error: {max_error:.2e}")
        print(f"    Mean absolute error: {mean_error:.2e}")
        print()


def main():
    """Run all demos."""
    print("Comprehensive Columnwise Matrix-Vector Multiplication Options Demo")
    print("=" * 100)
    print()
    print("This demo showcases the different columnwise matvec options:")
    print("1. Standard Dot Processor (Option 4.1)")
    print("2. Adaptive Depth Processor (Option 4.1c)")
    print("3. Lookup Table Processor (Option 4.2)")
    print()
    
    # Run all demos
    demo_basic_functionality()
    demo_sparsity_handling()
    demo_performance_comparison()
    demo_adaptive_depth_analysis()
    demo_strategy_recommendations()
    demo_accuracy_analysis()
    
    print("=" * 100)
    print("Demo completed successfully!")
    print()
    print("Summary of Key Features Demonstrated:")
    print("✓ Different computation strategies for columnwise matvec")
    print("✓ Sparsity handling and optimization")
    print("✓ Performance comparison between approaches")
    print("✓ Adaptive depth selection based on input characteristics")
    print("✓ Accuracy analysis and error metrics")
    print("✓ Strategy recommendations for different use cases")
    print()
    print("The implementation provides a complete framework for efficient")
    print("columnwise matrix-vector multiplication with lattice quantization.")


if __name__ == "__main__":
    main()
