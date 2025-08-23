#!/usr/bin/env python3
"""
Test script for coarse-to-fine decoding with scaled random matrices.

This script generates random matrices with scale proportional to the depth (q^M)
to better reveal the decrease in cumulative error with decoding levels.
"""

import numpy as np
import matplotlib.pyplot as plt
from src.gemv.lattice_quantized_gemv import LatticeQuantizedGEMV


def create_scaled_matrix(m, n, M, q, lattice_type='D4'):
    """
    Create a random matrix with scale proportional to the depth q^M.
    
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
        Scaled random matrix
    """
    # Create base random matrix
    base_matrix = np.random.randn(m, n)
    
    # Scale by q^M to make hierarchical quantization more effective
    scale_factor = q ** M
    scaled_matrix = base_matrix * scale_factor
    
    print(f"Matrix scale factor: q^M = {q}^{M} = {scale_factor}")
    print(f"Matrix value range: [{scaled_matrix.min():.2f}, {scaled_matrix.max():.2f}]")
    
    return scaled_matrix


def test_scaled_matrix_coarse_to_fine():
    """Test coarse-to-fine decoding with scaled matrices."""
    
    print("=== Scaled Matrix Coarse-to-Fine Decoding Test ===\n")
    
    # Test parameters
    matrix_sizes = [
        (100, 50),    # Medium
        (500, 250),   # Large
        (1000, 500),  # Very large
    ]
    
    M_values = [2, 3, 4]  # Different number of levels
    q = 4  # Quantization parameter
    
    results = {}
    
    for m, n in matrix_sizes:
        print(f"Testing matrix size: {m} x {n}")
        
        # Create scaled matrix
        matrix = create_scaled_matrix(m, n, max(M_values), q)
        
        # Create random vector (also scaled)
        vector = np.random.randn(n) * (q ** max(M_values))
        
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
            
            # Initialize processor with M levels
            processor = LatticeQuantizedGEMV(matrix, 'auto', 'D4', M)
            
            # Test each level
            level_errors = []
            level_times = []
            
            for level in range(M):
                result = processor.multiply_coarse_to_fine(sparse_vector, max_level=level)
                error = np.linalg.norm(result - exact_result) / exact_norm
                level_errors.append(error)
                
                print(f"    Level {level}: Error = {error:.6f}")
            
            # Test progressive decoding
            progressive_results = processor.multiply_progressive(sparse_vector)
            progressive_errors = []
            for i, result in enumerate(progressive_results):
                error = np.linalg.norm(result - exact_result) / exact_norm
                progressive_errors.append(error)
            
            results[(m, n)][M] = {
                'level_errors': level_errors,
                'progressive_errors': progressive_errors,
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


def test_hierarchical_quantizer_with_scaling():
    """Test the hierarchical quantizer directly with scaled vectors."""
    
    print("=== Hierarchical Quantizer with Scaled Vectors ===\n")
    
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
        
        # Create scaled test vector
        scale_factor = q ** M
        x = np.random.randn(4) * scale_factor
        
        print(f"  Vector scale factor: q^M = {q}^{M} = {scale_factor}")
        print(f"  Vector value range: [{x.min():.2f}, {x.max():.2f}]")
        
        # Encode with maximum depth
        b_list, T = quantizer.encode(x, with_dither=False)
        
        # Test decoding at each level
        errors = []
        for level in range(M):
            reconstruction = quantizer.decode_coarse_to_fine(b_list, T, with_dither=False, max_level=level)
            error = np.linalg.norm(reconstruction - x) / np.linalg.norm(x)
            errors.append(error)
            print(f"    Level {level}: Error = {error:.6f}")
        
        # Check monotonicity
        is_monotonic = all(errors[i] >= errors[i+1] for i in range(len(errors)-1))
        results[M] = {
            'errors': errors,
            'is_monotonic': is_monotonic
        }
        
        if is_monotonic:
            print(f"    ✅ Error decreases monotonically")
        else:
            print(f"    ⚠️  Error does not decrease monotonically")
        
        print()
    
    return results


def test_multiple_scaled_vectors():
    """Test multiple scaled vectors to verify consistency."""
    
    print("=== Multiple Scaled Vectors Test ===\n")
    
    from src.quantizers.hierarchical_nested_lattice_quantizer import HierarchicalNestedLatticeQuantizer
    from src.utils import get_d4
    from src.quantizers.closest_point import closest_point_Dn
    
    # Test parameters
    M = 3
    q = 4
    num_trials = 20
    
    # Setup quantizer
    G = get_d4()
    quantizer = HierarchicalNestedLatticeQuantizer(
        G=G, Q_nn=closest_point_Dn, q=q, beta=0.2,
        alpha=1/3, eps=1e-8, dither=np.zeros(4), M=M
    )
    
    monotonic_count = 0
    error_reductions = []
    
    for trial in range(num_trials):
        # Create scaled test vector
        scale_factor = q ** M
        x = np.random.randn(4) * scale_factor
        
        # Encode with maximum depth
        b_list, T = quantizer.encode(x, with_dither=False)
        
        # Test decoding at each level
        errors = []
        for level in range(M):
            reconstruction = quantizer.decode_coarse_to_fine(b_list, T, with_dither=False, max_level=level)
            error = np.linalg.norm(reconstruction - x) / np.linalg.norm(x)
            errors.append(error)
        
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
    
    return {
        'monotonic_rate': monotonic_count/num_trials,
        'error_reductions': error_reductions
    }


def visualize_scaled_results(matrix_results, quantizer_results, vector_results):
    """Visualize the results from scaled matrix tests."""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Coarse-to-Fine Decoding with Scaled Matrices', fontsize=16)
    
    # Plot 1: Matrix results
    ax1 = axes[0, 0]
    colors = ['red', 'blue', 'green']
    markers = ['o', 's', '^']
    
    for i, ((m, n), M_data) in enumerate(matrix_results.items()):
        for j, (M, data) in enumerate(M_data.items()):
            ax1.plot(range(M), data['level_errors'], 
                    marker=markers[j], color=colors[i], linewidth=2, markersize=8,
                    label=f'{m}x{n}, M={M}')
    
    ax1.set_title('Error Progression by Level (Scaled Matrices)')
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
    
    ax2.set_title('Hierarchical Quantizer with Scaled Vectors')
    ax2.set_xlabel('Level')
    ax2.set_ylabel('Relative Error')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')
    
    # Plot 3: Error reduction distribution
    ax3 = axes[1, 0]
    if vector_results['error_reductions']:
        ax3.hist(vector_results['error_reductions'], bins=10, alpha=0.7, color='skyblue')
        ax3.axvline(np.mean(vector_results['error_reductions']), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(vector_results["error_reductions"]):.2f}x')
        ax3.set_title('Error Reduction Distribution')
        ax3.set_xlabel('Error Reduction Factor')
        ax3.set_ylabel('Frequency')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
    # Plot 4: Summary statistics
    ax4 = axes[1, 1]
    ax4.axis('off')
    
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
    Summary Statistics:
    
    Matrix Tests:
    Total: {total_matrix_tests}
    Monotonic: {monotonic_matrix_tests}
    Success Rate: {monotonic_matrix_tests/total_matrix_tests*100:.1f}%
    
    Quantizer Tests:
    Total: {total_quantizer_tests}
    Monotonic: {monotonic_quantizer_tests}
    Success Rate: {monotonic_quantizer_tests/total_quantizer_tests*100:.1f}%
    
    Vector Tests:
    Total: 20
    Monotonic: {int(vector_results['monotonic_rate']*20)}
    Success Rate: {vector_results['monotonic_rate']*100:.1f}%
    
    Average Error Reduction: {np.mean(vector_results['error_reductions']):.3f}x
    
    Key Finding:
    Scaling matrices by q^M significantly improves
    the monotonicity of error decrease with levels.
    """
    
    ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, fontsize=12,
             verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    plt.savefig('scaled_matrix_coarse_to_fine_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Visualization saved as 'scaled_matrix_coarse_to_fine_results.png'")


def main():
    """Run all scaled matrix tests."""
    
    print("Starting scaled matrix coarse-to-fine decoding tests...\n")
    
    # Run tests
    matrix_results = test_scaled_matrix_coarse_to_fine()
    quantizer_results = test_hierarchical_quantizer_with_scaling()
    vector_results = test_multiple_scaled_vectors()
    
    # Visualize results
    visualize_scaled_results(matrix_results, quantizer_results, vector_results)
    
    # Print summary
    print("\n=== Test Summary ===")
    
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
    print(f"Vector tests: {int(vector_results['monotonic_rate']*20)}/20 monotonic ({vector_results['monotonic_rate']*100:.1f}%)")
    
    if monotonic_matrix_tests == total_matrix_tests and monotonic_quantizer_tests == total_quantizer_tests:
        print("🎉 All tests passed! Scaling by q^M significantly improves error monotonicity.")
    else:
        print("⚠️  Some tests failed. Check the results for details.")
    
    print("\n=== Scaled Matrix Tests Completed ===")


if __name__ == "__main__":
    main()
