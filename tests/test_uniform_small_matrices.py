#!/usr/bin/env python3
"""
Test script for coarse-to-fine decoding with uniform random variables and small matrices.

This script uses uniform random variables instead of normal random variables
and tests with small matrices to better analyze the hierarchical quantization behavior.
"""

import numpy as np
import matplotlib.pyplot as plt
from src.gemv.lattice_quantized_gemv import LatticeQuantizedGEMV


def create_uniform_scaled_matrix(m, n, M, q, lattice_type='D4'):
    """
    Create a uniform random matrix with scale proportional to the depth q^M.
    
    Parameters:
    -----------
    m, n : int
        Matrix dimensions
    M : int
        Number of hierarchical levels
    q : int
        Quantization parameter
    lattice_type : str
        Type of lattice to use
        
    Returns:
    --------
    np.ndarray
        Scaled uniform random matrix
    """
    # Create base uniform random matrix in [0, 1]
    base_matrix = np.random.uniform(0, 1, (m, n))
    
    # Scale by q^M to make hierarchical quantization more effective
    scale_factor = q ** M
    scaled_matrix = base_matrix * scale_factor
    
    print(f"Matrix scale factor: q^M = {q}^{M} = {scale_factor}")
    print(f"Matrix value range: [{scaled_matrix.min():.2f}, {scaled_matrix.max():.2f}]")
    
    return scaled_matrix


def test_uniform_small_matrices():
    """Test coarse-to-fine decoding with uniform random small matrices."""
    
    print("=== Uniform Random Small Matrices Coarse-to-Fine Test ===\n")
    
    # Test parameters - small matrices
    matrix_sizes = [
        (8, 4),      # Very small
        (16, 8),     # Small
        (32, 16),    # Medium-small
    ]
    
    M_values = [2, 3, 4]  # Different number of levels
    q = 4  # Quantization parameter
    
    results = {}
    
    for m, n in matrix_sizes:
        print(f"Testing matrix size: {m} x {n}")
        
        # Create scaled uniform matrix
        matrix = create_uniform_scaled_matrix(m, n, max(M_values), q)
        
        # Create uniform random vector (also scaled)
        vector = np.random.uniform(0, 1, n) * (q ** max(M_values))
        
        # Make vector sparse (30% non-zero elements)
        num_nonzero = max(1, int(0.3 * n))
        sparsity_pattern = np.random.choice(n, num_nonzero, replace=False)
        sparse_vector = np.zeros(n)
        sparse_vector[sparsity_pattern] = vector[sparsity_pattern]
        
        print(f"Vector value range: [{sparse_vector.min():.2f}, {sparse_vector.max():.2f}]")
        print(f"Non-zero elements: {num_nonzero}/{n}")
        
        # Compute exact result
        exact_result = matrix @ sparse_vector
        exact_norm = np.linalg.norm(exact_result)
        
        results[(m, n)] = {}
        
        for M in M_values:
            print(f"  Testing M = {M}")
            
            # Initialize processor with M levels
            processor = LatticeQuantizedGEMV(matrix, 'auto', 'D4', M)
            
            # Test each level
            level_errors = []
            level_reconstructions = []
            
            for level in range(M):
                result = processor.multiply_coarse_to_fine(sparse_vector, max_level=level)
                error = np.linalg.norm(result - exact_result) / exact_norm
                level_errors.append(error)
                level_reconstructions.append(result)
                
                print(f"    Level {level}: Error = {error:.6f}")
                print(f"      Result range: [{result.min():.2f}, {result.max():.2f}]")
            
            # Test progressive decoding
            progressive_results = processor.multiply_progressive(sparse_vector)
            progressive_errors = []
            for i, result in enumerate(progressive_results):
                error = np.linalg.norm(result - exact_result) / exact_norm
                progressive_errors.append(error)
            
            results[(m, n)][M] = {
                'level_errors': level_errors,
                'progressive_errors': progressive_errors,
                'level_reconstructions': level_reconstructions,
                'compression_ratio': processor.get_compression_ratio()
            }
            
            # Verify error decreases as levels increase
            is_monotonic = all(level_errors[i] >= level_errors[i+1] for i in range(len(level_errors)-1))
            if is_monotonic:
                print(f"    ✅ Error decreases monotonically with M = {M}")
            else:
                print(f"    ⚠️  Error does not decrease monotonically with M = {M}")
            
            print(f"    Compression ratio: {processor.get_compression_ratio():.2f}x")
            print()
    
    return results


def test_hierarchical_quantizer_uniform():
    """Test the hierarchical quantizer directly with uniform random vectors."""
    
    print("=== Hierarchical Quantizer with Uniform Random Vectors ===\n")
    
    from src.quantizers.hierarchical_nested_lattice_quantizer import HierarchicalNestedLatticeQuantizer
    from src.utils import get_d4
    from src.quantizers.closest_point import closest_point_Dn
    
    # Test parameters
    M_values = [2, 3, 4]
    q = 4
    
    results = {}
    
    for M in M_values:
        print(f"Testing M = {M}")
        
        # Setup quantizer
        G = get_d4()
        quantizer = HierarchicalNestedLatticeQuantizer(
            G=G, Q_nn=closest_point_Dn, q=q, beta=0.2,
            alpha=1/3, eps=1e-8, dither=np.zeros(4), M=M
        )
        
        # Create uniform random test vector
        scale_factor = q ** M
        x = np.random.uniform(0, 1, 4) * scale_factor
        
        print(f"  Vector scale factor: q^M = {q}^{M} = {scale_factor}")
        print(f"  Vector: {x}")
        print(f"  Vector norm: {np.linalg.norm(x):.6f}")
        
        # Encode with maximum depth
        b_list, T = quantizer.encode(x, with_dither=False)
        
        print(f"  Encoding vectors:")
        for i, b in enumerate(b_list):
            print(f"    Level {i}: {b}")
        
        # Test decoding at each level
        errors = []
        reconstructions = []
        for level in range(M):
            reconstruction = quantizer.decode_coarse_to_fine(b_list, T, with_dither=False, max_level=level)
            error = np.linalg.norm(reconstruction - x) / np.linalg.norm(x)
            errors.append(error)
            reconstructions.append(reconstruction)
            print(f"    Level {level}: Error = {error:.6f}")
            print(f"      Reconstruction: {reconstruction}")
        
        # Check monotonicity
        is_monotonic = all(errors[i] >= errors[i+1] for i in range(len(errors)-1))
        results[M] = {
            'errors': errors,
            'reconstructions': reconstructions,
            'is_monotonic': is_monotonic,
            'original_vector': x,
            'encoding_vectors': b_list
        }
        
        if is_monotonic:
            print(f"    ✅ Error decreases monotonically")
        else:
            print(f"    ⚠️  Error does not decrease monotonically")
        
        print()
    
    return results


def test_multiple_uniform_vectors():
    """Test multiple uniform random vectors to verify consistency."""
    
    print("=== Multiple Uniform Random Vectors Test ===\n")
    
    from src.quantizers.hierarchical_nested_lattice_quantizer import HierarchicalNestedLatticeQuantizer
    from src.utils import get_d4
    from src.quantizers.closest_point import closest_point_Dn
    
    # Test parameters
    M = 3
    q = 4
    num_trials = 15
    
    # Setup quantizer
    G = get_d4()
    quantizer = HierarchicalNestedLatticeQuantizer(
        G=G, Q_nn=closest_point_Dn, q=q, beta=0.2,
        alpha=1/3, eps=1e-8, dither=np.zeros(4), M=M
    )
    
    monotonic_count = 0
    error_reductions = []
    all_errors = []
    
    for trial in range(num_trials):
        # Create uniform random test vector
        scale_factor = q ** M
        x = np.random.uniform(0, 1, 4) * scale_factor
        
        # Encode with maximum depth
        b_list, T = quantizer.encode(x, with_dither=False)
        
        # Test decoding at each level
        errors = []
        for level in range(M):
            reconstruction = quantizer.decode_coarse_to_fine(b_list, T, with_dither=False, max_level=level)
            error = np.linalg.norm(reconstruction - x) / np.linalg.norm(x)
            errors.append(error)
        
        all_errors.append(errors)
        
        # Check monotonicity
        is_monotonic = all(errors[i] >= errors[i+1] for i in range(len(errors)-1))
        if is_monotonic:
            monotonic_count += 1
        
        # Calculate error reduction
        if errors[0] > 0:
            error_reduction = errors[0] / errors[-1]
            error_reductions.append(error_reduction)
        
        if trial < 5:  # Show first 5 trials
            print(f"Trial {trial+1}: Errors = {[f'{e:.6f}' for e in errors]}, Monotonic = {is_monotonic}")
    
    print(f"\nMonotonic trials: {monotonic_count}/{num_trials} = {monotonic_count/num_trials*100:.1f}%")
    
    if error_reductions:
        print(f"Average error reduction: {np.mean(error_reductions):.3f}x")
        print(f"Min error reduction: {np.min(error_reductions):.3f}x")
        print(f"Max error reduction: {np.max(error_reductions):.3f}x")
    
    # Calculate average errors per level
    avg_errors = np.mean(all_errors, axis=0)
    print(f"Average errors per level: {[f'{e:.6f}' for e in avg_errors]}")
    
    return {
        'monotonic_rate': monotonic_count/num_trials,
        'error_reductions': error_reductions,
        'all_errors': all_errors,
        'avg_errors': avg_errors
    }


def visualize_uniform_results(matrix_results, quantizer_results, vector_results):
    """Visualize the results from uniform random matrix tests."""
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Coarse-to-Fine Decoding with Uniform Random Variables', fontsize=16)
    
    # Plot 1: Matrix results
    ax1 = axes[0, 0]
    colors = ['red', 'blue', 'green']
    markers = ['o', 's', '^']
    
    for i, ((m, n), M_data) in enumerate(matrix_results.items()):
        for j, (M, data) in enumerate(M_data.items()):
            ax1.plot(range(M), data['level_errors'], 
                    marker=markers[j], color=colors[i], linewidth=2, markersize=8,
                    label=f'{m}x{n}, M={M}')
    
    ax1.set_title('Error Progression by Level (Uniform Matrices)')
    ax1.set_xlabel('Level')
    ax1.set_ylabel('Relative Error')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # Plot 2: Quantizer results
    ax2 = axes[0, 1]
    for M, data in quantizer_results.items():
        color = 'green' if data['is_monotonic'] else 'red'
        ax2.plot(range(M), data['errors'], marker='o', color=color, 
                linewidth=2, markersize=8, label=f'M={M}')
    
    ax2.set_title('Hierarchical Quantizer with Uniform Vectors')
    ax2.set_xlabel('Level')
    ax2.set_ylabel('Relative Error')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')
    
    # Plot 3: Error reduction distribution
    ax3 = axes[0, 2]
    if vector_results['error_reductions']:
        ax3.hist(vector_results['error_reductions'], bins=8, alpha=0.7, color='skyblue')
        ax3.axvline(np.mean(vector_results['error_reductions']), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(vector_results["error_reductions"]):.2f}x')
        ax3.set_title('Error Reduction Distribution')
        ax3.set_xlabel('Error Reduction Factor')
        ax3.set_ylabel('Frequency')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
    # Plot 4: Average errors per level
    ax4 = axes[1, 0]
    if len(vector_results['avg_errors']) > 0:
        ax4.plot(range(len(vector_results['avg_errors'])), vector_results['avg_errors'], 
                marker='o', color='purple', linewidth=2, markersize=8)
        ax4.set_title('Average Errors per Level (15 Trials)')
        ax4.set_xlabel('Level')
        ax4.set_ylabel('Average Relative Error')
        ax4.grid(True, alpha=0.3)
        ax4.set_yscale('log')
    
    # Plot 5: Individual trial errors
    ax5 = axes[1, 1]
    if vector_results['all_errors']:
        for i, errors in enumerate(vector_results['all_errors']):
            color = 'green' if all(errors[j] >= errors[j+1] for j in range(len(errors)-1)) else 'red'
            alpha = 0.3 if i >= 5 else 0.7  # Fade out after first 5
            ax5.plot(range(len(errors)), errors, color=color, alpha=alpha, linewidth=1)
        
        # Plot average line
        if len(vector_results['avg_errors']) > 0:
            ax5.plot(range(len(vector_results['avg_errors'])), vector_results['avg_errors'], 
                    color='black', linewidth=3, marker='o', markersize=8, label='Average')
            ax5.legend()
        
        ax5.set_title('Individual Trial Errors')
        ax5.set_xlabel('Level')
        ax5.set_ylabel('Relative Error')
        ax5.grid(True, alpha=0.3)
        ax5.set_yscale('log')
    
    # Plot 6: Summary statistics
    ax6 = axes[1, 2]
    ax6.axis('off')
    
    # Calculate summary statistics
    total_matrix_tests = 0
    monotonic_matrix_tests = 0
    
    for (m, n), M_data in matrix_results.items():
        for M, data in M_data.items():
            total_matrix_tests += 1
            if all(data['level_errors'][i] >= data['level_errors'][i+1] for i in range(len(data['level_errors'])-1)):
                monotonic_matrix_tests += 1
    
    total_quantizer_tests = len(quantizer_results)
    monotonic_quantizer_tests = sum(1 for data in quantizer_results.values() if data['is_monotonic'])
    
    summary_text = f"""
    Summary Statistics (Uniform Random):
    
    Matrix Tests:
    Total: {total_matrix_tests}
    Monotonic: {monotonic_matrix_tests}
    Success Rate: {monotonic_matrix_tests/total_matrix_tests*100:.1f}%
    
    Quantizer Tests:
    Total: {total_quantizer_tests}
    Monotonic: {monotonic_quantizer_tests}
    Success Rate: {monotonic_quantizer_tests/total_quantizer_tests*100:.1f}%
    
    Vector Tests:
    Total: 15
    Monotonic: {int(vector_results['monotonic_rate']*15)}
    Success Rate: {vector_results['monotonic_rate']*100:.1f}%
    
    Average Error Reduction: {np.mean(vector_results['error_reductions']):.3f}x
    
    Key Finding:
    Uniform random variables provide more
    controlled and predictable behavior for
    hierarchical quantization analysis.
    """
    
    ax6.text(0.1, 0.9, summary_text, transform=ax6.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    plt.savefig('uniform_small_matrices_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Visualization saved as 'uniform_small_matrices_results.png'")


def analyze_quantizer_behavior(quantizer_results):
    """Analyze the detailed behavior of the hierarchical quantizer."""
    
    print("\n=== Detailed Quantizer Behavior Analysis ===\n")
    
    for M, data in quantizer_results.items():
        print(f"M = {M}:")
        print(f"  Original vector: {data['original_vector']}")
        print(f"  Original norm: {np.linalg.norm(data['original_vector']):.6f}")
        
        for level in range(M):
            reconstruction = data['reconstructions'][level]
            error = data['errors'][level]
            print(f"  Level {level}:")
            print(f"    Reconstruction: {reconstruction}")
            print(f"    Error: {error:.6f}")
            print(f"    Reconstruction norm: {np.linalg.norm(reconstruction):.6f}")
        
        print(f"  Monotonic: {data['is_monotonic']}")
        print()


def main():
    """Run all uniform random matrix tests."""
    
    print("Starting uniform random small matrices coarse-to-fine decoding tests...\n")
    
    # Run tests
    matrix_results = test_uniform_small_matrices()
    quantizer_results = test_hierarchical_quantizer_uniform()
    vector_results = test_multiple_uniform_vectors()
    
    # Analyze detailed behavior
    analyze_quantizer_behavior(quantizer_results)
    
    # Visualize results
    visualize_uniform_results(matrix_results, quantizer_results, vector_results)
    
    # Print summary
    print("\n=== Test Summary (Uniform Random) ===")
    
    total_matrix_tests = sum(len(M_data) for M_data in matrix_results.values())
    monotonic_matrix_tests = 0
    
    for (m, n), M_data in matrix_results.items():
        for M, data in M_data.items():
            if all(data['level_errors'][i] >= data['level_errors'][i+1] for i in range(len(data['level_errors'])-1)):
                monotonic_matrix_tests += 1
    
    total_quantizer_tests = len(quantizer_results)
    monotonic_quantizer_tests = sum(1 for data in quantizer_results.values() if data['is_monotonic'])
    
    print(f"Matrix tests: {monotonic_matrix_tests}/{total_matrix_tests} monotonic ({monotonic_matrix_tests/total_matrix_tests*100:.1f}%)")
    print(f"Quantizer tests: {monotonic_quantizer_tests}/{total_quantizer_tests} monotonic ({monotonic_quantizer_tests/total_quantizer_tests*100:.1f}%)")
    print(f"Vector tests: {int(vector_results['monotonic_rate']*15)}/15 monotonic ({vector_results['monotonic_rate']*100:.1f}%)")
    
    if monotonic_matrix_tests == total_matrix_tests and monotonic_quantizer_tests == total_quantizer_tests:
        print("🎉 All tests passed! Uniform random variables provide excellent behavior.")
    else:
        print("⚠️  Some tests failed. Check the results for details.")
    
    print("\n=== Uniform Random Matrix Tests Completed ===")


if __name__ == "__main__":
    main()
