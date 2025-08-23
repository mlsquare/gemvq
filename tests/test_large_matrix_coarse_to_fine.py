#!/usr/bin/env python3
"""
Test coarse-to-fine decoding on large matrices.

This script tests the implementation on large matrices to verify that
relative error decreases as the number of levels increases.
"""

import numpy as np
import time
import matplotlib.pyplot as plt
from src.gemv.lattice_quantized_gemv import LatticeQuantizedGEMV


def test_large_matrix_error_progression():
    """Test error progression on large matrices."""
    
    print("=== Large Matrix Coarse-to-Fine Decoding Test ===\n")
    
    # Test different matrix sizes
    matrix_sizes = [
        (100, 50),    # Medium
        (500, 250),   # Large
        (1000, 500),  # Very large
    ]
    
    M_values = [2, 3, 4]  # Different number of levels
    
    results = {}
    
    for m, n in matrix_sizes:
        print(f"Testing matrix size: {m} x {n}")
        
        # Create large matrix and vector
        matrix = np.random.randn(m, n)
        vector = np.random.randn(n)
        
        # Make vector sparse (20% non-zero elements)
        num_nonzero = max(1, int(0.2 * n))
        sparsity_pattern = np.random.choice(n, num_nonzero, replace=False)
        sparse_vector = np.zeros(n)
        sparse_vector[sparsity_pattern] = vector[sparsity_pattern]
        
        # Compute exact result
        exact_result = matrix @ sparse_vector
        exact_norm = np.linalg.norm(exact_result)
        
        results[(m, n)] = {}
        
        for M in M_values:
            print(f"  Testing M = {M}")
            
            # Initialize processor
            start_time = time.time()
            processor = LatticeQuantizedGEMV(matrix, 'auto', 'D4', M)
            init_time = time.time() - start_time
            
            # Test each level
            level_errors = []
            level_times = []
            
            for level in range(M):
                start_time = time.time()
                result = processor.multiply_coarse_to_fine(sparse_vector, max_level=level)
                compute_time = time.time() - start_time
                
                error = np.linalg.norm(result - exact_result) / exact_norm
                level_errors.append(error)
                level_times.append(compute_time)
                
                print(f"    Level {level}: Error = {error:.6f}, Time = {compute_time:.4f}s")
            
            # Test progressive decoding
            start_time = time.time()
            progressive_results = processor.multiply_progressive(sparse_vector)
            progressive_time = time.time() - start_time
            
            # Verify progressive results match individual results
            progressive_errors = []
            for i, result in enumerate(progressive_results):
                error = np.linalg.norm(result - exact_result) / exact_norm
                progressive_errors.append(error)
                assert np.allclose(result, processor.multiply_coarse_to_fine(sparse_vector, max_level=i)), \
                    f"Progressive result {i} doesn't match level {i} result"
            
            results[(m, n)][M] = {
                'level_errors': level_errors,
                'level_times': level_times,
                'progressive_errors': progressive_errors,
                'progressive_time': progressive_time,
                'init_time': init_time,
                'compression_ratio': processor.get_compression_ratio()
            }
            
            # Verify error decreases as levels increase
            for i in range(1, len(level_errors)):
                if level_errors[i] > level_errors[i-1]:
                    print(f"    ⚠️  Warning: Error increased from level {i-1} to {i}")
                else:
                    print(f"    ✅ Error decreased from level {i-1} to {i}")
            
            print(f"    Compression ratio: {processor.get_compression_ratio():.2f}x")
            print()
    
    return results


def test_error_monotonicity():
    """Test that error decreases monotonically with more levels."""
    
    print("=== Error Monotonicity Test ===\n")
    
    # Use a large matrix for this test
    m, n = 800, 400
    matrix = np.random.randn(m, n)
    vector = np.random.randn(n)
    
    # Make vector sparse
    num_nonzero = max(1, int(0.15 * n))
    sparsity_pattern = np.random.choice(n, num_nonzero, replace=False)
    sparse_vector = np.zeros(n)
    sparse_vector[sparsity_pattern] = vector[sparsity_pattern]
    
    exact_result = matrix @ sparse_vector
    exact_norm = np.linalg.norm(exact_result)
    
    # Test different M values
    M_values = [2, 3, 4, 5]
    
    monotonicity_results = {}
    
    for M in M_values:
        print(f"Testing M = {M}")
        
        processor = LatticeQuantizedGEMV(matrix, 'auto', 'D4', M)
        
        # Test each level
        errors = []
        for level in range(M):
            result = processor.multiply_coarse_to_fine(sparse_vector, max_level=level)
            error = np.linalg.norm(result - exact_result) / exact_norm
            errors.append(error)
            print(f"  Level {level}: Error = {error:.6f}")
        
        # Check monotonicity
        is_monotonic = all(errors[i] >= errors[i+1] for i in range(len(errors)-1))
        monotonicity_results[M] = {
            'errors': errors,
            'is_monotonic': is_monotonic
        }
        
        if is_monotonic:
            print(f"  ✅ Error decreases monotonically with M = {M}")
        else:
            print(f"  ❌ Error does not decrease monotonically with M = {M}")
        
        print()
    
    return monotonicity_results


def test_compression_vs_quality_tradeoff():
    """Test the trade-off between compression and quality."""
    
    print("=== Compression vs Quality Trade-off Test ===\n")
    
    # Large matrix
    m, n = 1000, 500
    matrix = np.random.randn(m, n)
    vector = np.random.randn(n)
    
    # Make vector sparse
    num_nonzero = max(1, int(0.1 * n))
    sparsity_pattern = np.random.choice(n, num_nonzero, replace=False)
    sparse_vector = np.zeros(n)
    sparse_vector[sparsity_pattern] = vector[sparsity_pattern]
    
    exact_result = matrix @ sparse_vector
    exact_norm = np.linalg.norm(exact_result)
    
    # Test different M values
    M_values = [2, 3, 4, 5]
    
    tradeoff_results = {}
    
    for M in M_values:
        print(f"Testing M = {M}")
        
        processor = LatticeQuantizedGEMV(matrix, 'auto', 'D4', M)
        compression_ratio = processor.get_compression_ratio()
        
        # Test each level
        level_data = []
        for level in range(M):
            result = processor.multiply_coarse_to_fine(sparse_vector, max_level=level)
            error = np.linalg.norm(result - exact_result) / exact_norm
            
            # Effective compression ratio (bits used for this level)
            effective_compression = compression_ratio * (level + 1) / M
            
            level_data.append({
                'level': level,
                'error': error,
                'effective_compression': effective_compression
            })
            
            print(f"  Level {level}: Error = {error:.6f}, Effective Compression = {effective_compression:.2f}x")
        
        tradeoff_results[M] = {
            'compression_ratio': compression_ratio,
            'level_data': level_data
        }
        
        print(f"  Full compression ratio: {compression_ratio:.2f}x")
        print()
    
    return tradeoff_results


def visualize_results(results, monotonicity_results, tradeoff_results):
    """Visualize the test results."""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Large Matrix Coarse-to-Fine Decoding Results', fontsize=16)
    
    # Plot 1: Error progression for different matrix sizes
    ax1 = axes[0, 0]
    colors = ['red', 'blue', 'green']
    markers = ['o', 's', '^']
    
    for i, ((m, n), M_data) in enumerate(results.items()):
        for j, (M, data) in enumerate(M_data.items()):
            ax1.plot(range(M), data['level_errors'], 
                    marker=markers[j], color=colors[i], linewidth=2, markersize=8,
                    label=f'{m}x{n}, M={M}')
    
    ax1.set_title('Error Progression by Level')
    ax1.set_xlabel('Level')
    ax1.set_ylabel('Relative Error')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # Plot 2: Monotonicity test
    ax2 = axes[0, 1]
    for M, data in monotonicity_results.items():
        color = 'green' if data['is_monotonic'] else 'red'
        ax2.plot(range(M), data['errors'], marker='o', color=color, 
                linewidth=2, markersize=8, label=f'M={M}')
    
    ax2.set_title('Error Monotonicity Test')
    ax2.set_xlabel('Level')
    ax2.set_ylabel('Relative Error')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')
    
    # Plot 3: Compression vs Quality trade-off
    ax3 = axes[0, 2]
    for M, data in tradeoff_results.items():
        compressions = [level['effective_compression'] for level in data['level_data']]
        errors = [level['error'] for level in data['level_data']]
        ax3.plot(compressions, errors, marker='o', linewidth=2, markersize=8, label=f'M={M}')
    
    ax3.set_title('Compression vs Quality Trade-off')
    ax3.set_xlabel('Effective Compression Ratio')
    ax3.set_ylabel('Relative Error')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_yscale('log')
    
    # Plot 4: Computation time
    ax4 = axes[1, 0]
    for i, ((m, n), M_data) in enumerate(results.items()):
        for j, (M, data) in enumerate(M_data.items()):
            ax4.plot(range(M), data['level_times'], 
                    marker=markers[j], color=colors[i], linewidth=2, markersize=8,
                    label=f'{m}x{n}, M={M}')
    
    ax4.set_title('Computation Time by Level')
    ax4.set_xlabel('Level')
    ax4.set_ylabel('Time (seconds)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Compression ratios
    ax5 = axes[1, 1]
    M_values = list(tradeoff_results.keys())
    compression_ratios = [tradeoff_results[M]['compression_ratio'] for M in M_values]
    ax5.bar(M_values, compression_ratios, color='skyblue', alpha=0.7)
    ax5.set_title('Compression Ratio by M')
    ax5.set_xlabel('M (Number of Levels)')
    ax5.set_ylabel('Compression Ratio')
    ax5.grid(True, alpha=0.3)
    
    # Plot 6: Summary statistics
    ax6 = axes[1, 2]
    ax6.axis('off')
    
    # Calculate summary statistics
    total_tests = 0
    monotonic_tests = 0
    
    for M, data in monotonicity_results.items():
        total_tests += 1
        if data['is_monotonic']:
            monotonic_tests += 1
    
    summary_text = f"""
    Summary Statistics:
    
    Total Tests: {total_tests}
    Monotonic Tests: {monotonic_tests}
    Success Rate: {monotonic_tests/total_tests*100:.1f}%
    
    Matrix Sizes Tested:
    {', '.join([f'{m}x{n}' for (m,n) in results.keys()])}
    
    M Values Tested:
    {', '.join(map(str, M_values))}
    
    Average Error Reduction:
    {np.mean([data['errors'][-1]/data['errors'][0] if data['errors'][0] > 0 else 1 
              for data in monotonicity_results.values()]):.3f}x
    """
    
    ax6.text(0.1, 0.9, summary_text, transform=ax6.transAxes, fontsize=12,
             verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    plt.savefig('large_matrix_coarse_to_fine_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Visualization saved as 'large_matrix_coarse_to_fine_results.png'")


def main():
    """Run all large matrix tests."""
    
    print("Starting large matrix coarse-to-fine decoding tests...\n")
    
    # Run tests
    results = test_large_matrix_error_progression()
    monotonicity_results = test_error_monotonicity()
    tradeoff_results = test_compression_vs_quality_tradeoff()
    
    # Visualize results
    visualize_results(results, monotonicity_results, tradeoff_results)
    
    # Print summary
    print("\n=== Test Summary ===")
    
    total_tests = len(monotonicity_results)
    monotonic_tests = sum(1 for data in monotonicity_results.values() if data['is_monotonic'])
    
    print(f"Total monotonicity tests: {total_tests}")
    print(f"Successful tests: {monotonic_tests}")
    print(f"Success rate: {monotonic_tests/total_tests*100:.1f}%")
    
    if monotonic_tests == total_tests:
        print("🎉 All tests passed! Error decreases monotonically with more levels.")
    else:
        print("⚠️  Some tests failed. Check the results for details.")
    
    print("\n=== Large Matrix Tests Completed ===")


if __name__ == "__main__":
    main()
