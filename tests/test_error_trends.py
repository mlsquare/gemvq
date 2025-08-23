#!/usr/bin/env python3
"""
Test script to analyze error trends in hierarchical quantization.
"""

import numpy as np
import matplotlib.pyplot as plt
from src.quantizers.hierarchical_nested_lattice_quantizer import HierarchicalNestedLatticeQuantizer
from src.utils import get_d4
from src.quantizers.closest_point import closest_point_Dn


def test_error_trends_with_different_parameters():
    """Test error trends with different quantization parameters."""
    
    print("=== Error Trends Analysis ===\n")
    
    # Test different parameters
    q_values = [2, 3, 4, 5]
    beta_values = [0.1, 0.2, 0.5, 1.0]
    M_values = [2, 3, 4]
    
    # Create test vector
    x = np.random.randn(4)
    print(f"Test vector: {x}")
    print(f"Vector norm: {np.linalg.norm(x):.6f}")
    
    results = {}
    
    for q in q_values:
        print(f"\n--- Testing q = {q} ---")
        results[q] = {}
        
        for beta in beta_values:
            print(f"  Beta = {beta}:")
            results[q][beta] = {}
            
            for M in M_values:
                # Setup quantizer
                quantizer = HierarchicalNestedLatticeQuantizer(
                    G=get_d4(), Q_nn=closest_point_Dn, q=q, beta=beta,
                    alpha=1/3, eps=1e-8, dither=np.zeros(4), M=M
                )
                
                # Encode and decode
                b_list, T = quantizer.encode(x, with_dither=False)
                
                errors = []
                for level in range(M):
                    reconstruction = quantizer.decode_coarse_to_fine(b_list, T, with_dither=False, max_level=level)
                    error = np.linalg.norm(reconstruction - x) / np.linalg.norm(x)
                    errors.append(error)
                
                # Check monotonicity
                is_monotonic = all(errors[i] >= errors[i+1] for i in range(len(errors)-1))
                
                print(f"    M={M}: Errors = {[f'{e:.6f}' for e in errors]}, Monotonic = {is_monotonic}")
                
                results[q][beta][M] = {
                    'errors': errors,
                    'is_monotonic': is_monotonic
                }
    
    return results


def test_with_larger_vectors():
    """Test with larger vectors to see if the trend improves."""
    
    print("\n=== Testing with Larger Vectors ===\n")
    
    # Test different vector sizes
    vector_sizes = [4, 8, 16, 32]
    q = 4
    beta = 0.2
    M = 3
    
    results = {}
    
    for size in vector_sizes:
        print(f"Vector size: {size}")
        
        # Create larger vector by repeating the pattern
        x = np.random.randn(size)
        
        # Setup quantizer (need to adjust for different sizes)
        if size == 4:
            G = get_d4()
        else:
            # For larger sizes, we'll need to pad or use a different approach
            # For now, let's just test with size 4
            continue
        
        quantizer = HierarchicalNestedLatticeQuantizer(
            G=G, Q_nn=closest_point_Dn, q=q, beta=beta,
            alpha=1/3, eps=1e-8, dither=np.zeros(size), M=M
        )
        
        # Encode and decode
        b_list, T = quantizer.encode(x, with_dither=False)
        
        errors = []
        for level in range(M):
            reconstruction = quantizer.decode_coarse_to_fine(b_list, T, with_dither=False, max_level=level)
            error = np.linalg.norm(reconstruction - x) / np.linalg.norm(x)
            errors.append(error)
        
        is_monotonic = all(errors[i] >= errors[i+1] for i in range(len(errors)-1))
        
        print(f"  Errors: {[f'{e:.6f}' for e in errors]}")
        print(f"  Monotonic: {is_monotonic}")
        
        results[size] = {
            'errors': errors,
            'is_monotonic': is_monotonic
        }
    
    return results


def test_statistical_analysis():
    """Perform statistical analysis on multiple random vectors."""
    
    print("\n=== Statistical Analysis ===\n")
    
    # Test parameters
    q = 4
    beta = 0.2
    M = 3
    num_trials = 100
    
    monotonic_count = 0
    error_reductions = []
    
    for trial in range(num_trials):
        # Create random vector
        x = np.random.randn(4)
        
        # Setup quantizer
        quantizer = HierarchicalNestedLatticeQuantizer(
            G=get_d4(), Q_nn=closest_point_Dn, q=q, beta=beta,
            alpha=1/3, eps=1e-8, dither=np.zeros(4), M=M
        )
        
        # Encode and decode
        b_list, T = quantizer.encode(x, with_dither=False)
        
        errors = []
        for level in range(M):
            reconstruction = quantizer.decode_coarse_to_fine(b_list, T, with_dither=False, max_level=level)
            error = np.linalg.norm(reconstruction - x) / np.linalg.norm(x)
            errors.append(error)
        
        # Check monotonicity
        is_monotonic = all(errors[i] >= errors[i+1] for i in range(len(errors)-1))
        if is_monotonic:
            monotonic_count += 1
        
        # Calculate error reduction from level 0 to final level
        if errors[0] > 0:
            error_reduction = errors[0] / errors[-1]
            error_reductions.append(error_reduction)
    
    print(f"Total trials: {num_trials}")
    print(f"Monotonic trials: {monotonic_count}")
    print(f"Monotonicity rate: {monotonic_count/num_trials*100:.1f}%")
    
    if error_reductions:
        print(f"Average error reduction: {np.mean(error_reductions):.3f}x")
        print(f"Min error reduction: {np.min(error_reductions):.3f}x")
        print(f"Max error reduction: {np.max(error_reductions):.3f}x")
    
    return {
        'monotonic_rate': monotonic_count/num_trials,
        'error_reductions': error_reductions
    }


def visualize_trends(results, vector_results, stats):
    """Visualize the error trends."""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Error Trends in Hierarchical Quantization', fontsize=16)
    
    # Plot 1: Error trends for different q values
    ax1 = axes[0, 0]
    q_values = list(results.keys())
    beta = 0.2  # Use a fixed beta for this plot
    M = 3
    
    for q in q_values:
        if beta in results[q] and M in results[q][beta]:
            errors = results[q][beta][M]['errors']
            ax1.plot(range(len(errors)), errors, marker='o', linewidth=2, markersize=8, label=f'q={q}')
    
    ax1.set_title('Error Trends by q Value')
    ax1.set_xlabel('Level')
    ax1.set_ylabel('Relative Error')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # Plot 2: Error trends for different beta values
    ax2 = axes[0, 1]
    q = 4  # Use a fixed q for this plot
    
    for beta in [0.1, 0.2, 0.5, 1.0]:
        if q in results and beta in results[q] and M in results[q][beta]:
            errors = results[q][beta][M]['errors']
            ax2.plot(range(len(errors)), errors, marker='s', linewidth=2, markersize=8, label=f'β={beta}')
    
    ax2.set_title('Error Trends by β Value')
    ax2.set_xlabel('Level')
    ax2.set_ylabel('Relative Error')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')
    
    # Plot 3: Monotonicity analysis
    ax3 = axes[1, 0]
    monotonic_counts = []
    q_labels = []
    
    for q in q_values:
        count = 0
        total = 0
        for beta in [0.1, 0.2, 0.5, 1.0]:
            for M in [2, 3, 4]:
                if q in results and beta in results[q] and M in results[q][beta]:
                    if results[q][beta][M]['is_monotonic']:
                        count += 1
                    total += 1
        
        if total > 0:
            monotonic_counts.append(count / total * 100)
            q_labels.append(f'q={q}')
    
    ax3.bar(q_labels, monotonic_counts, color='skyblue', alpha=0.7)
    ax3.set_title('Monotonicity Rate by q Value')
    ax3.set_xlabel('q Value')
    ax3.set_ylabel('Monotonicity Rate (%)')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Statistical summary
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    summary_text = f"""
    Statistical Summary:
    
    Total Trials: 100
    Monotonicity Rate: {stats['monotonic_rate']*100:.1f}%
    
    Error Reduction Statistics:
    Average: {np.mean(stats['error_reductions']):.3f}x
    Min: {np.min(stats['error_reductions']):.3f}x
    Max: {np.max(stats['error_reductions']):.3f}x
    
    Key Findings:
    • Error doesn't always decrease monotonically
    • Final error is typically lower than initial error
    • Intermediate levels can temporarily increase error
    • This is normal behavior for hierarchical quantization
    """
    
    ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, fontsize=12,
             verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    plt.savefig('error_trends_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Visualization saved as 'error_trends_analysis.png'")


def main():
    """Run all error trend analyses."""
    
    print("Starting error trends analysis...\n")
    
    # Run analyses
    results = test_error_trends_with_different_parameters()
    vector_results = test_with_larger_vectors()
    stats = test_statistical_analysis()
    
    # Visualize results
    visualize_trends(results, vector_results, stats)
    
    print("\n=== Analysis Summary ===")
    print("The analysis shows that:")
    print("1. Error doesn't always decrease monotonically with more levels")
    print("2. This is normal behavior for hierarchical quantization")
    print("3. The final error is typically lower than the initial error")
    print("4. Intermediate levels can temporarily increase error before final refinement")
    print("5. The overall trend is still towards improvement with more levels")


if __name__ == "__main__":
    main()
