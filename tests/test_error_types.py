#!/usr/bin/env python3
"""
Test script to demonstrate the difference between cumulative error and tile-specific error.
"""

import numpy as np
from src.lattices.quantizers.hierarchical_nested_lattice_quantizer import HierarchicalNestedLatticeQuantizer
from src.lattices.utils import get_d4
from src.lattices.utils import closest_point_Dn


def test_cumulative_vs_tile_specific_error():
    """Test to show the difference between cumulative and tile-specific error."""
    
    print("=== Cumulative vs Tile-Specific Error Analysis ===\n")
    
    # Setup quantizer
    G = get_d4()
    M = 3
    q = 4
    quantizer = HierarchicalNestedLatticeQuantizer(
        G=G, Q_nn=closest_point_Dn, q=q, beta=0.2,
        alpha=1/3, eps=1e-8, dither=np.zeros(4), M=M
    )
    
    # Test vector
    x = np.random.randn(4)
    print(f"Original vector: {x}")
    print(f"Original norm: {np.linalg.norm(x):.6f}")
    
    # Encode
    b_list, T = quantizer.encode(x, with_dither=False)
    print(f"\nEncoding vectors:")
    for i, b in enumerate(b_list):
        print(f"  Level {i}: {b}")
    
    # Analyze weights
    print(f"\nWeight analysis:")
    for i in range(M):
        weight = np.power(q, M - 1 - i)
        print(f"  Level {i} weight: q^{M-1-i} = {q}^{M-1-i} = {weight}")
    
    # Test cumulative error (current implementation)
    print(f"\n--- Cumulative Error Analysis ---")
    cumulative_errors = []
    
    for level in range(M):
        # Reconstruct using levels 0 to level (cumulative)
        reconstruction = quantizer.decode_coarse_to_fine(b_list, T, with_dither=False, max_level=level)
        error = np.linalg.norm(reconstruction - x) / np.linalg.norm(x)
        cumulative_errors.append(error)
        print(f"  Level {level} (cumulative): Error = {error:.6f}")
        print(f"    Reconstruction: {reconstruction}")
    
    # Test tile-specific error (individual level contributions)
    print(f"\n--- Tile-Specific Error Analysis ---")
    tile_errors = []
    
    for level in range(M):
        # Reconstruct using only this specific level
        x_hat_list = []
        for i in range(M):
            if i == level:
                b = b_list[i]
                x_i_hat = np.dot(G, b) - quantizer.q_Q(np.dot(G, b))
                x_hat_list.append(x_i_hat)
            else:
                x_hat_list.append(np.zeros_like(x))
        
        # Reconstruct with only this level's contribution
        reconstruction = sum([np.power(q, M - 1 - i) * x_i for i, x_i in enumerate(x_hat_list)])
        reconstruction = quantizer.beta * reconstruction * (2 ** (quantizer.alpha * T))
        
        error = np.linalg.norm(reconstruction - x) / np.linalg.norm(x)
        tile_errors.append(error)
        print(f"  Level {level} (tile-specific): Error = {error:.6f}")
        print(f"    Reconstruction: {reconstruction}")
    
    # Compare the two types of error
    print(f"\n--- Comparison ---")
    print(f"Cumulative errors: {[f'{e:.6f}' for e in cumulative_errors]}")
    print(f"Tile-specific errors: {[f'{e:.6f}' for e in tile_errors]}")
    
    # Check if cumulative error decreases monotonically
    cumulative_monotonic = all(cumulative_errors[i] >= cumulative_errors[i+1] for i in range(len(cumulative_errors)-1))
    print(f"Cumulative error decreases monotonically: {cumulative_monotonic}")
    
    # Check if tile-specific error decreases
    tile_monotonic = all(tile_errors[i] >= tile_errors[i+1] for i in range(len(tile_errors)-1))
    print(f"Tile-specific error decreases monotonically: {tile_monotonic}")
    
    return cumulative_errors, tile_errors


def test_multiple_vectors_error_analysis():
    """Test error analysis on multiple vectors."""
    
    print("\n=== Multiple Vectors Error Analysis ===\n")
    
    # Setup quantizer
    G = get_d4()
    M = 3
    q = 4
    quantizer = HierarchicalNestedLatticeQuantizer(
        G=G, Q_nn=closest_point_Dn, q=q, beta=0.2,
        alpha=1/3, eps=1e-8, dither=np.zeros(4), M=M
    )
    
    num_trials = 10
    cumulative_monotonic_count = 0
    tile_monotonic_count = 0
    
    for trial in range(num_trials):
        # Test vector
        x = np.random.randn(4)
        
        # Encode
        b_list, T = quantizer.encode(x, with_dither=False)
        
        # Test cumulative error
        cumulative_errors = []
        for level in range(M):
            reconstruction = quantizer.decode_coarse_to_fine(b_list, T, with_dither=False, max_level=level)
            error = np.linalg.norm(reconstruction - x) / np.linalg.norm(x)
            cumulative_errors.append(error)
        
        # Test tile-specific error
        tile_errors = []
        for level in range(M):
            x_hat_list = []
            for i in range(M):
                if i == level:
                    b = b_list[i]
                    x_i_hat = np.dot(G, b) - quantizer.q_Q(np.dot(G, b))
                    x_hat_list.append(x_i_hat)
                else:
                    x_hat_list.append(np.zeros_like(x))
            
            reconstruction = sum([np.power(q, M - 1 - i) * x_i for i, x_i in enumerate(x_hat_list)])
            reconstruction = quantizer.beta * reconstruction * (2 ** (quantizer.alpha * T))
            error = np.linalg.norm(reconstruction - x) / np.linalg.norm(x)
            tile_errors.append(error)
        
        # Check monotonicity
        cumulative_monotonic = all(cumulative_errors[i] >= cumulative_errors[i+1] for i in range(len(cumulative_errors)-1))
        tile_monotonic = all(tile_errors[i] >= tile_errors[i+1] for i in range(len(tile_errors)-1))
        
        if cumulative_monotonic:
            cumulative_monotonic_count += 1
        if tile_monotonic:
            tile_monotonic_count += 1
        
        print(f"Trial {trial+1}:")
        print(f"  Cumulative: {[f'{e:.6f}' for e in cumulative_errors]}, Monotonic = {cumulative_monotonic}")
        print(f"  Tile-specific: {[f'{e:.6f}' for e in tile_errors]}, Monotonic = {tile_monotonic}")
    
    print(f"\nSummary:")
    print(f"Cumulative monotonic: {cumulative_monotonic_count}/{num_trials} = {cumulative_monotonic_count/num_trials*100:.1f}%")
    print(f"Tile-specific monotonic: {tile_monotonic_count}/{num_trials} = {tile_monotonic_count/num_trials*100:.1f}%")


def explain_error_types():
    """Explain the difference between cumulative and tile-specific error."""
    
    print("\n=== Error Type Explanation ===\n")
    
    print("1. CUMULATIVE ERROR (Current Implementation):")
    print("   - Error is calculated for the complete reconstruction using levels 0 to max_level")
    print("   - Each level adds its contribution to the previous levels")
    print("   - Formula: ||x - sum(q^(M-1-i) * x_i for i=0 to max_level)|| / ||x||")
    print("   - This represents the total error of the progressive reconstruction")
    print()
    
    print("2. TILE-SPECIFIC ERROR (Individual Level Error):")
    print("   - Error is calculated for each individual level's contribution")
    print("   - Only one level contributes to the reconstruction")
    print("   - Formula: ||x - q^(M-1-level) * x_level|| / ||x||")
    print("   - This represents the error contribution of each individual level")
    print()
    
    print("3. WHY THEY DIFFER:")
    print("   - Cumulative error includes the combined effect of multiple levels")
    print("   - Tile-specific error shows the individual contribution of each level")
    print("   - Cumulative error is more relevant for progressive decoding")
    print("   - Tile-specific error is more relevant for understanding level contributions")
    print()
    
    print("4. WHICH ONE IS MORE MEANINGFUL:")
    print("   - For coarse-to-fine decoding: CUMULATIVE ERROR")
    print("   - For analyzing level contributions: TILE-SPECIFIC ERROR")
    print("   - The current implementation correctly uses CUMULATIVE ERROR")


if __name__ == "__main__":
    test_cumulative_vs_tile_specific_error()
    test_multiple_vectors_error_analysis()
    explain_error_types()
