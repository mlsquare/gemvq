"""
Demo Script for Columnwise Matrix-Vector Multiplication Options

This script demonstrates the different columnwise matvec options implemented:
1. Standard Dot Processor (Option 4.1)
2. Lookup Table Processor (Option 4.2) 
3. Adaptive Processor (Dynamic Strategy Selection)

The demo shows:
- Basic usage of each processor type
- Performance comparisons
- Sparsity handling
- Padding scenarios
- Different computation strategies
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple

from .columnwise_matvec_factory import (
    create_processor,
    create_standard_dot_processor,
    create_lookup_table_processor,
    create_adaptive_processor,
    get_available_processors,
    get_processor_info
)


def create_test_matrix(m: int, n: int, lattice_dim: int = 4) -> np.ndarray:
    """Create a test matrix with random values."""
    # Ensure dimensions are compatible with lattice
    m_pad = (lattice_dim - (m % lattice_dim)) % lattice_dim
    n_pad = (lattice_dim - (n % lattice_dim)) % lattice_dim
    
    actual_m = m + m_pad
    actual_n = n + n_pad
    
    matrix = np.random.randn(actual_m, actual_n)
    return matrix[:m, :n]  # Return original dimensions


def create_sparse_vector(n: int, sparsity_ratio: float) -> np.ndarray:
    """Create a sparse vector with specified sparsity ratio."""
    x = np.random.randn(n)
    
    # Make some elements zero based on sparsity ratio
    num_zeros = int(n * sparsity_ratio)
    zero_indices = np.random.choice(n, num_zeros, replace=False)
    x[zero_indices] = 0.0
    
    return x


def benchmark_processors(
    matrix: np.ndarray,
    x: np.ndarray,
    processors: Dict[str, any],
    num_runs: int = 5
) -> Dict[str, Dict[str, float]]:
    """Benchmark different processors."""
    results = {}
    
    for name, processor in processors.items():
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
        
        results[name] = {
            'mean_time': np.mean(times),
            'std_time': np.std(times),
            'mean_error': np.mean(errors),
            'std_error': np.std(errors)
        }
    
    return results


def demo_basic_usage():
    """Demonstrate basic usage of different processors."""
    print("=" * 60)
    print("DEMO: Basic Usage of Columnwise MatVec Processors")
    print("=" * 60)
    
    # Create test data
    m, n = 16, 12
    matrix = create_test_matrix(m, n)
    x = np.random.randn(n)
    
    print(f"Matrix shape: {matrix.shape}")
    print(f"Vector length: {len(x)}")
    print()
    
    # Create different processors
    processors = {
        'Standard Dot': create_processor(matrix, 'standard_dot'),
        'Lookup Table': create_processor(matrix, 'lookup_table'),
        'Adaptive': create_processor(matrix, 'adaptive')
    }
    
    # Test each processor
    for name, processor in processors.items():
        print(f"Testing {name} Processor:")
        
        # Get matrix info
        info = processor.get_matrix_info()
        print(f"  Original shape: {info['original_shape']}")
        print(f"  Padded shape: {info['padded_shape']}")
        print(f"  Lattice dimension: {info['lattice_dimension']}")
        print(f"  Number of blocks: {info['num_blocks']}")
        
        # Compute matvec
        start_time = time.time()
        result = processor.compute_matvec(x)
        computation_time = time.time() - start_time
        
        # Compare with direct computation
        direct_result = matrix @ x
        error = np.linalg.norm(result - direct_result) / np.linalg.norm(direct_result)
        
        print(f"  Computation time: {computation_time:.6f} seconds")
        print(f"  Relative error: {error:.2e}")
        
        # Get compression stats
        stats = processor.get_compression_stats()
        print(f"  Compression ratio: {stats['compression_ratio']:.2f}x")
        print(f"  Memory usage: {stats['memory_usage_mb']:.2f} MB")
        print()


def demo_sparsity_handling():
    """Demonstrate sparsity handling capabilities."""
    print("=" * 60)
    print("DEMO: Sparsity Handling")
    print("=" * 60)
    
    # Create test data
    m, n = 20, 16
    matrix = create_test_matrix(m, n)
    
    # Test different sparsity levels
    sparsity_levels = [0.0, 0.25, 0.5, 0.75, 0.9]
    
    for sparsity in sparsity_levels:
        print(f"Sparsity level: {sparsity:.1%}")
        x = create_sparse_vector(n, sparsity)
        
        # Count non-zero elements
        non_zero_count = np.sum(np.abs(x) > 1e-10)
        print(f"  Non-zero elements: {non_zero_count}/{len(x)}")
        
        # Test with adaptive processor
        processor = create_adaptive_processor(matrix)
        
        # Get sparsity pattern
        sparsity_pattern, sparsity_ratio = processor._detect_sparsity(x)
        print(f"  Detected sparsity ratio: {sparsity_ratio:.3f}")
        
        # Compute with different strategies
        strategies = ['standard_dot', 'layer_wise_histogram', 'inner_product']
        
        for strategy in strategies:
            start_time = time.time()
            result = processor.compute_matvec_with_strategy(x, strategy)
            time_taken = time.time() - start_time
            
            # Check accuracy
            direct_result = matrix @ x
            error = np.linalg.norm(result - direct_result) / np.linalg.norm(direct_result)
            
            print(f"    {strategy}: {time_taken:.6f}s, error: {error:.2e}")
        
        print()


def demo_padding_scenarios():
    """Demonstrate padding handling for non-multiple dimensions."""
    print("=" * 60)
    print("DEMO: Padding Scenarios")
    print("=" * 60)
    
    # Test different matrix dimensions
    test_cases = [
        (15, 13),  # Both dimensions not multiples of 4
        (16, 12),  # Both dimensions multiples of 4
        (17, 16),  # Only one dimension multiple of 4
        (20, 18),  # Both dimensions not multiples of 4
    ]
    
    for m, n in test_cases:
        print(f"Matrix dimensions: {m} x {n}")
        
        # Create matrix and vector
        matrix = create_test_matrix(m, n)
        x = np.random.randn(n)
        
        # Test with standard dot processor
        processor = create_standard_dot_processor(matrix)
        
        # Get padding info
        info = processor.get_matrix_info()
        print(f"  Original shape: {info['original_shape']}")
        print(f"  Padded shape: {info['padded_shape']}")
        print(f"  Padding rows: {info['padding_rows']}")
        print(f"  Padding cols: {info['padding_cols']}")
        
        # Test computation
        result = processor.compute_matvec(x)
        direct_result = matrix @ x
        error = np.linalg.norm(result - direct_result) / np.linalg.norm(direct_result)
        
        print(f"  Relative error: {error:.2e}")
        print()


def demo_performance_comparison():
    """Demonstrate performance comparison between different approaches."""
    print("=" * 60)
    print("DEMO: Performance Comparison")
    print("=" * 60)
    
    # Test different matrix sizes
    sizes = [(32, 24), (64, 48), (128, 96)]
    
    for m, n in sizes:
        print(f"Matrix size: {m} x {n}")
        
        # Create test data
        matrix = create_test_matrix(m, n)
        x = np.random.randn(n)
        
        # Create processors
        processors = {
            'Standard Dot': create_processor(matrix, 'standard_dot'),
            'Lookup Table': create_processor(matrix, 'lookup_table'),
            'Adaptive': create_processor(matrix, 'adaptive')
        }
        
        # Benchmark
        results = benchmark_processors(matrix, x, processors, num_runs=3)
        
        # Print results
        for name, stats in results.items():
            print(f"  {name}:")
            print(f"    Time: {stats['mean_time']:.6f} ± {stats['std_time']:.6f}s")
            print(f"    Error: {stats['mean_error']:.2e} ± {stats['std_error']:.2e}")
        
        print()


def demo_adaptive_strategy_selection():
    """Demonstrate adaptive strategy selection."""
    print("=" * 60)
    print("DEMO: Adaptive Strategy Selection")
    print("=" * 60)
    
    # Create test data
    m, n = 64, 48
    matrix = create_test_matrix(m, n)
    
    # Test different input characteristics
    test_cases = [
        ("Dense vector", np.random.randn(n)),
        ("Sparse vector (25%)", create_sparse_vector(n, 0.25)),
        ("Sparse vector (50%)", create_sparse_vector(n, 0.5)),
        ("Sparse vector (75%)", create_sparse_vector(n, 0.75)),
        ("Very sparse vector (90%)", create_sparse_vector(n, 0.9)),
    ]
    
    processor = create_adaptive_processor(matrix)
    
    for name, x in test_cases:
        print(f"Input: {name}")
        
        # Get sparsity info
        sparsity_pattern, sparsity_ratio = processor._detect_sparsity(x)
        print(f"  Sparsity ratio: {sparsity_ratio:.3f}")
        
        # Let adaptive processor choose strategy
        start_time = time.time()
        result = processor.compute_matvec(x)
        time_taken = time.time() - start_time
        
        # Get selected strategy
        stats = processor.get_performance_stats()
        selected_strategy = stats.get('selected_strategy', 'unknown')
        
        print(f"  Selected strategy: {selected_strategy}")
        print(f"  Computation time: {time_taken:.6f}s")
        
        # Check accuracy
        direct_result = matrix @ x
        error = np.linalg.norm(result - direct_result) / np.linalg.norm(direct_result)
        print(f"  Relative error: {error:.2e}")
        print()


def demo_processor_info():
    """Show information about available processors."""
    print("=" * 60)
    print("DEMO: Processor Information")
    print("=" * 60)
    
    available_processors = get_available_processors()
    print(f"Available processors: {available_processors}")
    print()
    
    for processor_type in available_processors:
        info = get_processor_info(processor_type)
        print(f"{info['name']}:")
        print(f"  Description: {info['description']}")
        print(f"  Best for: {info['best_for']}")
        print(f"  Memory usage: {info['memory_usage']}")
        print(f"  Computation speed: {info['computation_speed']}")
        print()


def main():
    """Run all demos."""
    print("Columnwise Matrix-Vector Multiplication Options Demo")
    print("=" * 80)
    print()
    
    # Run all demos
    demo_processor_info()
    demo_basic_usage()
    demo_sparsity_handling()
    demo_padding_scenarios()
    demo_performance_comparison()
    demo_adaptive_strategy_selection()
    
    print("=" * 80)
    print("Demo completed successfully!")
    print()
    print("Key Features Demonstrated:")
    print("✓ Different computation strategies (standard dot, lookup tables, adaptive)")
    print("✓ Sparsity handling and optimization")
    print("✓ Padding for non-multiple dimensions")
    print("✓ Performance comparison and benchmarking")
    print("✓ Adaptive strategy selection based on input characteristics")
    print("✓ Factory functions for easy processor creation")


if __name__ == "__main__":
    main()
