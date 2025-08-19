#!/usr/bin/env python3
"""
Demo script for Adaptive Matrix-Vector Multiplication

This script demonstrates the new adaptive approach where:
1. Matrix W is encoded once with maximum bit rate
2. Columns are decoded adaptively based on bit budget for each x
3. Hierarchical levels M are exploited for variable precision decoding
4. Sparsity of x is handled efficiently
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from typing import List, Tuple, Dict

from .adaptive_matvec import (
    AdaptiveMatVecProcessor,
    create_adaptive_matvec_processor,
    adaptive_matvec_multiply,
    adaptive_matvec_multiply_sparse
)


def demo_basic_functionality():
    """Demonstrate basic functionality of the adaptive approach."""
    print("🚀 Basic Functionality Demo")
    print("=" * 50)
    
    # Setup parameters
    m, n = 64, 32
    max_rate = 8.0
    M = 4
    
    # Create test matrix and vector
    np.random.seed(42)
    matrix = np.random.randn(m, n)
    vector = np.random.randn(n)
    
    print(f"Matrix shape: {matrix.shape}")
    print(f"Vector shape: {vector.shape}")
    print(f"Max rate: {max_rate}")
    print(f"Hierarchical levels: {M}")
    
    # Create processor
    processor = create_adaptive_matvec_processor(
        matrix, 'D4', max_rate, M
    )
    
    # Test with different column rates
    print(f"\n📊 Testing with different column rates:")
    
    # Case 1: All columns at max rate
    column_rates_max = [max_rate] * n
    result_max = processor.compute_matvec(vector, column_rates_max, use_lookup=False)
    exact_result = matrix @ vector
    error_max = np.linalg.norm(result_max - exact_result) / np.linalg.norm(exact_result)
    print(f"   Max rate (all columns): {error_max:.6f}")
    
    # Case 2: Variable rates
    column_rates_var = np.random.uniform(2.0, max_rate, n)
    result_var = processor.compute_matvec(vector, column_rates_var.tolist(), use_lookup=False)
    error_var = np.linalg.norm(result_var - exact_result) / np.linalg.norm(exact_result)
    print(f"   Variable rates: {error_var:.6f}")
    
    # Case 3: Low rates
    column_rates_low = np.random.uniform(1.0, 3.0, n)
    result_low = processor.compute_matvec(vector, column_rates_low.tolist(), use_lookup=False)
    error_low = np.linalg.norm(result_low - exact_result) / np.linalg.norm(exact_result)
    print(f"   Low rates: {error_low:.6f}")
    
    return processor, exact_result


def demo_sparsity_handling():
    """Demonstrate handling of sparse vectors."""
    print(f"\n🔍 Sparsity Handling Demo")
    print("=" * 50)
    
    # Setup parameters
    m, n = 64, 32
    max_rate = 8.0
    M = 4
    
    # Create test matrix
    np.random.seed(42)
    matrix = np.random.randn(m, n)
    
    # Test different sparsity levels
    sparsity_levels = [0.1, 0.2, 0.5, 0.8, 1.0]  # 10%, 20%, 50%, 80%, 100%
    
    print(f"Testing sparsity levels: {[f'{s*100:.0f}%' for s in sparsity_levels]}")
    
    results = {}
    
    for sparsity in sparsity_levels:
        # Create sparse vector
        num_nonzero = int(n * sparsity)
        non_zero_indices = np.random.choice(n, num_nonzero, replace=False)
        sparse_vector = np.zeros(n)
        sparse_vector[non_zero_indices] = np.random.randn(num_nonzero)
        
        # Define column rates
        column_rates = np.random.uniform(2.0, max_rate, n)
        
        # Perform computation
        result = adaptive_matvec_multiply_sparse(
            matrix, sparse_vector, non_zero_indices.tolist(),
            column_rates.tolist(), 'D4', max_rate, M, use_lookup=False
        )
        
        # Compare with exact
        exact_result = matrix @ sparse_vector
        error = np.linalg.norm(result - exact_result) / np.linalg.norm(exact_result)
        
        results[sparsity] = {
            'error': error,
            'num_nonzero': num_nonzero,
            'result': result,
            'exact': exact_result
        }
        
        print(f"   Sparsity {sparsity*100:.0f}%: error = {error:.6f}")
    
    return results


def demo_hierarchical_levels():
    """Demonstrate the effect of hierarchical levels M."""
    print(f"\n📈 Hierarchical Levels Demo")
    print("=" * 50)
    
    # Setup parameters
    m, n = 64, 32
    max_rate = 8.0
    
    # Create test matrix and vector
    np.random.seed(42)
    matrix = np.random.randn(m, n)
    vector = np.random.randn(n)
    column_rates = np.random.uniform(2.0, max_rate, n)
    
    # Test different M values
    M_values = [1, 2, 3, 4, 5]
    
    print(f"Testing hierarchical levels: {M_values}")
    
    results = {}
    
    for M in M_values:
        # Create processor with this M
        processor = create_adaptive_matvec_processor(
            matrix, 'D4', max_rate, M
        )
        
        # Perform computation
        result = processor.compute_matvec(vector, column_rates.tolist(), use_lookup=False)
        
        # Compare with exact
        exact_result = matrix @ vector
        error = np.linalg.norm(result - exact_result) / np.linalg.norm(exact_result)
        
        # Get memory usage
        memory_usage = processor.get_memory_usage()
        
        results[M] = {
            'error': error,
            'memory_mb': memory_usage['total_mb'],
            'compression_ratio': processor.get_compression_ratio()
        }
        
        print(f"   M={M}: error={error:.6f}, memory={memory_usage['total_mb']:.2f}MB, "
              f"compression={processor.get_compression_ratio():.2f}x")
    
    return results


def demo_rate_distortion_tradeoff():
    """Demonstrate rate-distortion tradeoff."""
    print(f"\n⚖️ Rate-Distortion Tradeoff Demo")
    print("=" * 50)
    
    # Setup parameters
    m, n = 64, 32
    max_rate = 8.0
    M = 4
    
    # Create test matrix and vector
    np.random.seed(42)
    matrix = np.random.randn(m, n)
    vector = np.random.randn(n)
    
    # Test different rate levels
    rate_levels = np.linspace(1.0, max_rate, 10)
    
    print(f"Testing rate levels: {rate_levels}")
    
    results = {}
    
    for rate in rate_levels:
        # Use same rate for all columns
        column_rates = [rate] * n
        
        # Create processor
        processor = create_adaptive_matvec_processor(
            matrix, 'D4', max_rate, M
        )
        
        # Perform computation
        result = processor.compute_matvec(vector, column_rates, use_lookup=False)
        
        # Compare with exact
        exact_result = matrix @ vector
        error = np.linalg.norm(result - exact_result) / np.linalg.norm(exact_result)
        
        # Calculate effective bit rate (average)
        effective_rate = np.mean(column_rates)
        
        results[rate] = {
            'error': error,
            'effective_rate': effective_rate
        }
        
        print(f"   Rate {rate:.1f}: error={error:.6f}")
    
    return results


def demo_lookup_table_efficiency():
    """Demonstrate efficiency of lookup table approach."""
    print(f"\n⚡ Lookup Table Efficiency Demo")
    print("=" * 50)
    
    # Setup parameters
    m, n = 128, 64
    max_rate = 8.0
    M = 4
    
    # Create test matrix and vector
    np.random.seed(42)
    matrix = np.random.randn(m, n)
    vector = np.random.randn(n)
    column_rates = np.random.uniform(2.0, max_rate, n)
    
    # Create processor
    processor = create_adaptive_matvec_processor(
        matrix, 'D4', max_rate, M
    )
    
    # Test both approaches
    print("Comparing direct decoding vs lookup table approach:")
    
    # Direct decoding
    start_time = time.time()
    result_direct = processor.compute_matvec(vector, column_rates.tolist(), use_lookup=False)
    time_direct = time.time() - start_time
    
    # Lookup table approach
    start_time = time.time()
    result_lookup = processor.compute_matvec(vector, column_rates.tolist(), use_lookup=True)
    time_lookup = time.time() - start_time
    
    # Compare results
    exact_result = matrix @ vector
    error_direct = np.linalg.norm(result_direct - exact_result) / np.linalg.norm(exact_result)
    error_lookup = np.linalg.norm(result_lookup - exact_result) / np.linalg.norm(exact_result)
    
    print(f"   Direct decoding: {time_direct:.4f}s, error={error_direct:.6f}")
    print(f"   Lookup table: {time_lookup:.4f}s, error={error_lookup:.6f}")
    print(f"   Speedup: {time_direct/time_lookup:.2f}x")
    
    return {
        'direct': {'time': time_direct, 'error': error_direct},
        'lookup': {'time': time_lookup, 'error': error_lookup}
    }


def demo_memory_usage():
    """Demonstrate memory usage characteristics."""
    print(f"\n💾 Memory Usage Demo")
    print("=" * 50)
    
    # Test different matrix sizes
    sizes = [(32, 16), (64, 32), (128, 64), (256, 128)]
    max_rate = 8.0
    M = 4
    
    print(f"Testing matrix sizes: {sizes}")
    
    results = {}
    
    for m, n in sizes:
        # Create test matrix
        np.random.seed(42)
        matrix = np.random.randn(m, n)
        
        # Create processor
        processor = create_adaptive_matvec_processor(
            matrix, 'D4', max_rate, M
        )
        
        # Get memory usage
        memory_usage = processor.get_memory_usage()
        compression_ratio = processor.get_compression_ratio()
        
        # Calculate original matrix size
        original_mb = m * n * 4 / (1024 * 1024)  # 4 bytes per float
        
        results[(m, n)] = {
            'original_mb': original_mb,
            'encoded_mb': memory_usage['encoded_matrix_mb'],
            'lookup_mb': memory_usage['lookup_tables_mb'],
            'total_mb': memory_usage['total_mb'],
            'compression_ratio': compression_ratio
        }
        
        print(f"   Size {m}x{n}: original={original_mb:.2f}MB, "
              f"encoded={memory_usage['encoded_matrix_mb']:.2f}MB, "
              f"lookup={memory_usage['lookup_tables_mb']:.2f}MB, "
              f"compression={compression_ratio:.2f}x")
    
    return results


def plot_results(hierarchical_results, rate_distortion_results, memory_results):
    """Create plots to visualize the results."""
    print(f"\n📊 Creating visualization plots...")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Hierarchical levels vs error
    M_values = list(hierarchical_results.keys())
    errors = [hierarchical_results[M]['error'] for M in M_values]
    memory_mb = [hierarchical_results[M]['memory_mb'] for M in M_values]
    
    ax1 = axes[0, 0]
    ax1.plot(M_values, errors, 'bo-', linewidth=2, markersize=8)
    ax1.set_xlabel('Hierarchical Levels (M)')
    ax1.set_ylabel('Relative Error')
    ax1.set_title('Error vs Hierarchical Levels')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Rate-distortion curve
    rates = list(rate_distortion_results.keys())
    errors_rd = [rate_distortion_results[rate]['error'] for rate in rates]
    
    ax2 = axes[0, 1]
    ax2.semilogy(rates, errors_rd, 'ro-', linewidth=2, markersize=8)
    ax2.set_xlabel('Bit Rate (bits/dimension)')
    ax2.set_ylabel('Relative Error')
    ax2.set_title('Rate-Distortion Curve')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Memory usage vs matrix size
    sizes = list(memory_results.keys())
    original_mb = [memory_results[size]['original_mb'] for size in sizes]
    encoded_mb = [memory_results[size]['encoded_mb'] for size in sizes]
    total_mb = [memory_results[size]['total_mb'] for size in sizes]
    
    ax3 = axes[1, 0]
    x_pos = np.arange(len(sizes))
    width = 0.25
    
    ax3.bar(x_pos - width, original_mb, width, label='Original', alpha=0.8)
    ax3.bar(x_pos, encoded_mb, width, label='Encoded', alpha=0.8)
    ax3.bar(x_pos + width, total_mb, width, label='Total (with lookup)', alpha=0.8)
    
    ax3.set_xlabel('Matrix Size')
    ax3.set_ylabel('Memory (MB)')
    ax3.set_title('Memory Usage vs Matrix Size')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels([f'{m}x{n}' for m, n in sizes])
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Compression ratio vs matrix size
    compression_ratios = [memory_results[size]['compression_ratio'] for size in sizes]
    
    ax4 = axes[1, 1]
    ax4.bar(range(len(sizes)), compression_ratios, alpha=0.8, color='green')
    ax4.set_xlabel('Matrix Size')
    ax4.set_ylabel('Compression Ratio')
    ax4.set_title('Compression Ratio vs Matrix Size')
    ax4.set_xticks(range(len(sizes)))
    ax4.set_xticklabels([f'{m}x{n}' for m, n in sizes])
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('adaptive_matvec_demo_results.png', dpi=300, bbox_inches='tight')
    print(f"   Plots saved to 'adaptive_matvec_demo_results.png'")
    
    return fig


def main():
    """Run the complete demo."""
    print("🎯 Adaptive Matrix-Vector Multiplication Demo")
    print("=" * 60)
    print("This demo showcases the new approach where:")
    print("1. Matrix W is encoded once with maximum bit rate")
    print("2. Columns are decoded adaptively based on bit budget for each x")
    print("3. Hierarchical levels M are exploited for variable precision")
    print("4. Sparsity of x is handled efficiently")
    print("=" * 60)
    
    try:
        # Run all demos
        processor, exact_result = demo_basic_functionality()
        sparsity_results = demo_sparsity_handling()
        hierarchical_results = demo_hierarchical_levels()
        rate_distortion_results = demo_rate_distortion_tradeoff()
        lookup_results = demo_lookup_table_efficiency()
        memory_results = demo_memory_usage()
        
        # Create plots
        fig = plot_results(hierarchical_results, rate_distortion_results, memory_results)
        
        print(f"\n✅ All demos completed successfully!")
        print(f"📈 Results have been plotted and saved.")
        
        # Summary statistics
        print(f"\n📋 Summary:")
        print(f"   - Best hierarchical level: M={min(hierarchical_results.keys(), key=lambda x: hierarchical_results[x]['error'])}")
        print(f"   - Lookup table speedup: {lookup_results['direct']['time']/lookup_results['lookup']['time']:.2f}x")
        print(f"   - Best compression ratio: {max(memory_results.values(), key=lambda x: x['compression_ratio'])['compression_ratio']:.2f}x")
        
    except Exception as e:
        print(f"❌ Error during demo: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 