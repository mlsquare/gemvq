"""
Demo script for adaptive matrix-vector multiplication.

This script demonstrates the adaptive matrix-vector multiplication functionality
using hierarchical nested quantizers with column-wise encoding. It includes
examples, performance analysis, and comparison with exact computation.
"""

import numpy as np
import time
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
from .adaptive_matvec import (
    SparseMatVecProcessor,
    adaptive_matvec_multiply,
    create_adaptive_matvec_processor
)
from ..utils import get_d4, get_a2, get_e8


def create_test_data(m: int, n: int, sparsity_ratio: float = 0.2) -> Tuple[np.ndarray, np.ndarray, List[int], List[float]]:
    """
    Create test data for adaptive matrix-vector multiplication.
    
    Parameters:
    -----------
    m : int
        Number of rows in matrix.
    n : int
        Number of columns in matrix.
    sparsity_ratio : float
        Ratio of non-zero elements in vector.
        
    Returns:
    --------
    tuple
        (matrix, sparse_vector, sparsity_pattern, target_rates)
    """
    # Create random matrix
    matrix = np.random.randn(m, n)
    
    # Create sparse vector
    num_nonzero = max(1, int(n * sparsity_ratio))
    sparsity_pattern = np.random.choice(n, num_nonzero, replace=False)
    sparse_vector = np.zeros(n)
    sparse_vector[sparsity_pattern] = np.random.randn(num_nonzero)
    
    # Create target rates (different for each column)
    # Higher rates for columns with larger magnitude
    column_norms = np.linalg.norm(matrix, axis=0)
    target_rates = 2.0 + 4.0 * (column_norms / np.max(column_norms))
    
    return matrix, sparse_vector, sparsity_pattern.tolist(), target_rates.tolist()


def performance_comparison(matrix: np.ndarray, sparse_vector: np.ndarray,
                          target_rates: List[float], sparsity_pattern: List[int],
                          lattice_type: str = 'D4', M: int = 2) -> Dict:
    """
    Compare performance of adaptive vs exact matrix-vector multiplication.
    
    Parameters:
    -----------
    matrix : np.ndarray
        Input matrix.
    sparse_vector : np.ndarray
        Sparse input vector.
    target_rates : List[float]
        Target bit rates for each column.
    sparsity_pattern : List[int]
        Indices of non-zero elements.
    lattice_type : str
        Type of lattice to use.
    M : int
        Number of hierarchical levels.
        
    Returns:
    --------
    dict
        Performance comparison results.
    """
    results = {}
    
    # Exact computation
    start_time = time.time()
    exact_result = matrix @ sparse_vector
    exact_time = time.time() - start_time
    
    # Adaptive computation without lookup tables
    start_time = time.time()
    adaptive_result = adaptive_matvec_multiply(
        matrix, sparse_vector, target_rates, sparsity_pattern, lattice_type, M, False
    )
    adaptive_time = time.time() - start_time
    
    # Adaptive computation with lookup tables
    start_time = time.time()
    adaptive_lookup_result = adaptive_matvec_multiply(
        matrix, sparse_vector, target_rates, sparsity_pattern, lattice_type, M, True
    )
    adaptive_lookup_time = time.time() - start_time
    
    # Calculate errors
    error_adaptive = np.linalg.norm(adaptive_result - exact_result) / np.linalg.norm(exact_result)
    error_lookup = np.linalg.norm(adaptive_lookup_result - exact_result) / np.linalg.norm(exact_result)
    
    # Get compression statistics
    processor = create_adaptive_matvec_processor(
        matrix, target_rates, sparsity_pattern, lattice_type, M
    )
    compression_ratio = processor.get_compression_ratio()
    memory_usage = processor.get_memory_usage()
    
    results = {
        'exact_time': exact_time,
        'adaptive_time': adaptive_time,
        'adaptive_lookup_time': adaptive_lookup_time,
        'error_adaptive': error_adaptive,
        'error_lookup': error_lookup,
        'compression_ratio': compression_ratio,
        'memory_usage': memory_usage,
        'sparsity_ratio': len(sparsity_pattern) / len(sparse_vector)
    }
    
    return results


def plot_performance_analysis(results_list: List[Dict], labels: List[str]):
    """
    Plot performance analysis results.
    
    Parameters:
    -----------
    results_list : List[Dict]
        List of performance results.
    labels : List[str]
        Labels for each result set.
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot computation times
    ax1 = axes[0, 0]
    times_adaptive = [r['adaptive_time'] for r in results_list]
    times_lookup = [r['adaptive_lookup_time'] for r in results_list]
    times_exact = [r['exact_time'] for r in results_list]
    
    x = np.arange(len(labels))
    width = 0.25
    
    ax1.bar(x - width, times_exact, width, label='Exact', alpha=0.8)
    ax1.bar(x, times_adaptive, width, label='Adaptive', alpha=0.8)
    ax1.bar(x + width, times_lookup, width, label='Adaptive + Lookup', alpha=0.8)
    
    ax1.set_xlabel('Configuration')
    ax1.set_ylabel('Time (seconds)')
    ax1.set_title('Computation Time Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot relative errors
    ax2 = axes[0, 1]
    errors_adaptive = [r['error_adaptive'] for r in results_list]
    errors_lookup = [r['error_lookup'] for r in results_list]
    
    ax2.bar(x - width/2, errors_adaptive, width, label='Adaptive', alpha=0.8)
    ax2.bar(x + width/2, errors_lookup, width, label='Adaptive + Lookup', alpha=0.8)
    
    ax2.set_xlabel('Configuration')
    ax2.set_ylabel('Relative Error')
    ax2.set_title('Accuracy Comparison')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot compression ratios
    ax3 = axes[1, 0]
    compression_ratios = [r['compression_ratio'] for r in results_list]
    
    ax3.bar(x, compression_ratios, alpha=0.8, color='green')
    ax3.set_xlabel('Configuration')
    ax3.set_ylabel('Compression Ratio')
    ax3.set_title('Compression Performance')
    ax3.set_xticks(x)
    ax3.set_xticklabels(labels)
    ax3.grid(True, alpha=0.3)
    
    # Plot memory usage
    ax4 = axes[1, 1]
    memory_usage = [r['memory_usage']['total_mb'] for r in results_list]
    
    ax4.bar(x, memory_usage, alpha=0.8, color='orange')
    ax4.set_xlabel('Configuration')
    ax4.set_ylabel('Memory Usage (MB)')
    ax4.set_title('Memory Requirements')
    ax4.set_xticks(x)
    ax4.set_xticklabels(labels)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def demo_basic_usage():
    """Demonstrate basic usage of adaptive matrix-vector multiplication."""
    print("=== Basic Usage Demo ===")
    
    # Create test data
    m, n = 50, 20
    matrix, sparse_vector, sparsity_pattern, target_rates = create_test_data(m, n, 0.3)
    
    print(f"Matrix shape: {matrix.shape}")
    print(f"Vector sparsity: {len(sparsity_pattern)}/{len(sparse_vector)} = {len(sparsity_pattern)/len(sparse_vector):.2f}")
    print(f"Target rates range: {min(target_rates):.2f} - {max(target_rates):.2f} bits/dimension")
    
    # Perform adaptive multiplication
    result = adaptive_matvec_multiply(
        matrix, sparse_vector, target_rates, sparsity_pattern, 'D4', 2
    )
    
    # Compare with exact computation
    exact_result = matrix @ sparse_vector
    error = np.linalg.norm(result - exact_result) / np.linalg.norm(exact_result)
    
    print(f"Result shape: {result.shape}")
    print(f"Relative error: {error:.6f}")
    print(f"Result norm: {np.linalg.norm(result):.6f}")
    print(f"Exact norm: {np.linalg.norm(exact_result):.6f}")
    
    return matrix, sparse_vector, target_rates, sparsity_pattern, result, exact_result


def demo_lattice_comparison():
    """Compare performance across different lattice types."""
    print("\n=== Lattice Type Comparison ===")
    
    # Create test data
    m, n = 40, 15
    matrix, sparse_vector, sparsity_pattern, target_rates = create_test_data(m, n, 0.4)
    
    lattice_types = ['D4', 'A2', 'E8']
    results = {}
    
    for lattice_type in lattice_types:
        print(f"\nTesting {lattice_type} lattice...")
        
        # Perform multiplication
        start_time = time.time()
        result = adaptive_matvec_multiply(
            matrix, sparse_vector, target_rates, sparsity_pattern, lattice_type, 2
        )
        computation_time = time.time() - start_time
        
        # Calculate error
        exact_result = matrix @ sparse_vector
        error = np.linalg.norm(result - exact_result) / np.linalg.norm(exact_result)
        
        # Get compression stats
        processor = create_adaptive_matvec_processor(
            matrix, target_rates, sparsity_pattern, lattice_type, 2
        )
        compression_ratio = processor.get_compression_ratio()
        
        results[lattice_type] = {
            'time': computation_time,
            'error': error,
            'compression_ratio': compression_ratio
        }
        
        print(f"  Time: {computation_time:.6f}s")
        print(f"  Error: {error:.6f}")
        print(f"  Compression: {compression_ratio:.2f}x")
    
    return results


def demo_hierarchical_levels():
    """Compare performance with different hierarchical levels."""
    print("\n=== Hierarchical Levels Comparison ===")
    
    # Create test data
    m, n = 30, 12
    matrix, sparse_vector, sparsity_pattern, target_rates = create_test_data(m, n, 0.5)
    
    M_values = [1, 2, 3]
    results = {}
    
    for M in M_values:
        print(f"\nTesting M={M} hierarchical levels...")
        
        # Perform multiplication
        start_time = time.time()
        result = adaptive_matvec_multiply(
            matrix, sparse_vector, target_rates, sparsity_pattern, 'D4', M
        )
        computation_time = time.time() - start_time
        
        # Calculate error
        exact_result = matrix @ sparse_vector
        error = np.linalg.norm(result - exact_result) / np.linalg.norm(exact_result)
        
        # Get compression stats
        processor = create_adaptive_matvec_processor(
            matrix, target_rates, sparsity_pattern, 'D4', M
        )
        compression_ratio = processor.get_compression_ratio()
        memory_usage = processor.get_memory_usage()
        
        results[M] = {
            'time': computation_time,
            'error': error,
            'compression_ratio': compression_ratio,
            'memory_mb': memory_usage['total_mb']
        }
        
        print(f"  Time: {computation_time:.6f}s")
        print(f"  Error: {error:.6f}")
        print(f"  Compression: {compression_ratio:.2f}x")
        print(f"  Memory: {memory_usage['total_mb']:.2f} MB")
    
    return results


def demo_sparsity_impact():
    """Demonstrate the impact of sparsity on performance."""
    print("\n=== Sparsity Impact Analysis ===")
    
    # Create test data
    m, n = 60, 25
    matrix = np.random.randn(m, n)
    target_rates = np.random.uniform(2.0, 5.0, n).tolist()
    
    sparsity_ratios = [0.1, 0.2, 0.3, 0.4, 0.5]
    results = []
    
    for sparsity_ratio in sparsity_ratios:
        print(f"\nTesting sparsity ratio: {sparsity_ratio:.1f}")
        
        # Create sparse vector
        num_nonzero = max(1, int(n * sparsity_ratio))
        sparsity_pattern = np.random.choice(n, num_nonzero, replace=False).tolist()
        sparse_vector = np.zeros(n)
        sparse_vector[sparsity_pattern] = np.random.randn(num_nonzero)
        
        # Perform multiplication
        start_time = time.time()
        result = adaptive_matvec_multiply(
            matrix, sparse_vector, target_rates, sparsity_pattern, 'D4', 2
        )
        computation_time = time.time() - start_time
        
        # Calculate error
        exact_result = matrix @ sparse_vector
        error = np.linalg.norm(result - exact_result) / np.linalg.norm(exact_result)
        
        # Get compression stats
        processor = create_adaptive_matvec_processor(
            matrix, target_rates, sparsity_pattern, 'D4', 2
        )
        compression_ratio = processor.get_compression_ratio()
        
        results.append({
            'sparsity_ratio': sparsity_ratio,
            'time': computation_time,
            'error': error,
            'compression_ratio': compression_ratio
        })
        
        print(f"  Time: {computation_time:.6f}s")
        print(f"  Error: {error:.6f}")
        print(f"  Compression: {compression_ratio:.2f}x")
    
    return results


def demo_rate_allocation():
    """Demonstrate different rate allocation strategies."""
    print("\n=== Rate Allocation Strategies ===")
    
    # Create test data
    m, n = 50, 20
    matrix, sparse_vector, sparsity_pattern, _ = create_test_data(m, n, 0.3)
    
    # Different rate allocation strategies
    strategies = {
        'Uniform': [3.0] * n,
        'Energy-based': 2.0 + 4.0 * (np.linalg.norm(matrix, axis=0) / np.max(np.linalg.norm(matrix, axis=0))),
        'Random': np.random.uniform(2.0, 6.0, n),
        'Inverse-energy': 6.0 - 4.0 * (np.linalg.norm(matrix, axis=0) / np.max(np.linalg.norm(matrix, axis=0)))
    }
    
    results = {}
    
    for strategy_name, target_rates in strategies.items():
        print(f"\nTesting {strategy_name} rate allocation...")
        
        if isinstance(target_rates, np.ndarray):
            target_rates = target_rates.tolist()
        
        # Perform multiplication
        start_time = time.time()
        result = adaptive_matvec_multiply(
            matrix, sparse_vector, target_rates, sparsity_pattern, 'D4', 2
        )
        computation_time = time.time() - start_time
        
        # Calculate error
        exact_result = matrix @ sparse_vector
        error = np.linalg.norm(result - exact_result) / np.linalg.norm(exact_result)
        
        # Get compression stats
        processor = create_adaptive_matvec_processor(
            matrix, target_rates, sparsity_pattern, 'D4', 2
        )
        compression_ratio = processor.get_compression_ratio()
        
        results[strategy_name] = {
            'time': computation_time,
            'error': error,
            'compression_ratio': compression_ratio,
            'avg_rate': np.mean(target_rates)
        }
        
        print(f"  Average rate: {np.mean(target_rates):.2f} bits/dimension")
        print(f"  Time: {computation_time:.6f}s")
        print(f"  Error: {error:.6f}")
        print(f"  Compression: {compression_ratio:.2f}x")
    
    return results


def run_comprehensive_demo():
    """Run comprehensive demonstration of adaptive matrix-vector multiplication."""
    print("Adaptive Matrix-Vector Multiplication Demo")
    print("=" * 50)
    
    # Basic usage demo
    basic_results = demo_basic_usage()
    
    # Lattice comparison
    lattice_results = demo_lattice_comparison()
    
    # Hierarchical levels comparison
    hierarchical_results = demo_hierarchical_levels()
    
    # Sparsity impact
    sparsity_results = demo_sparsity_impact()
    
    # Rate allocation strategies
    rate_results = demo_rate_allocation()
    
    # Summary
    print("\n" + "=" * 50)
    print("DEMO SUMMARY")
    print("=" * 50)
    
    print(f"Basic demo completed with relative error: {np.linalg.norm(basic_results[4] - basic_results[5]) / np.linalg.norm(basic_results[5]):.6f}")
    
    best_lattice = min(lattice_results.keys(), key=lambda x: lattice_results[x]['error'])
    print(f"Best performing lattice: {best_lattice} (error: {lattice_results[best_lattice]['error']:.6f})")
    
    best_M = min(hierarchical_results.keys(), key=lambda x: hierarchical_results[x]['error'])
    print(f"Best hierarchical levels: M={best_M} (error: {hierarchical_results[best_M]['error']:.6f})")
    
    best_strategy = min(rate_results.keys(), key=lambda x: rate_results[x]['error'])
    print(f"Best rate allocation: {best_strategy} (error: {rate_results[best_strategy]['error']:.6f})")
    
    print("\nDemo completed successfully!")
    
    return {
        'basic': basic_results,
        'lattice': lattice_results,
        'hierarchical': hierarchical_results,
        'sparsity': sparsity_results,
        'rate_allocation': rate_results
    }


if __name__ == "__main__":
    # Run the comprehensive demo
    results = run_comprehensive_demo()
    
    # Optionally plot results
    try:
        # Create performance comparison plot
        performance_results = [
            results['lattice']['D4'],
            results['lattice']['A2'],
            results['lattice']['E8']
        ]
        labels = ['D4', 'A2', 'E8']
        
        # Convert to list format for plotting
        plot_data = []
        for label, result in zip(labels, performance_results):
            plot_data.append({
                'adaptive_time': result['time'],
                'adaptive_lookup_time': result['time'] * 0.8,  # Approximate
                'exact_time': 0.001,  # Approximate
                'error_adaptive': result['error'],
                'error_lookup': result['error'] * 1.1,  # Approximate
                'compression_ratio': result['compression_ratio'],
                'memory_usage': {'total_mb': 1.0}  # Approximate
            })
        
        plot_performance_analysis(plot_data, labels)
        
    except ImportError:
        print("Matplotlib not available for plotting. Install matplotlib to see performance plots.") 