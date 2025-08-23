#!/usr/bin/env python3
"""
Test script to verify the fixed hierarchical quantizer with correct MSB/LSB ordering.
"""

import numpy as np
from src.quantizers.hierarchical_nested_lattice_quantizer import HierarchicalNestedLatticeQuantizer
from src.utils import get_d4
from src.quantizers.closest_point import closest_point_Dn


def test_msb_lsb_ordering():
    """Test that the hierarchical quantizer now has correct MSB/LSB ordering."""
    
    print("=== Testing MSB/LSB Ordering ===\n")
    
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
    
    # Analyze the weights
    print(f"\nWeight analysis (should be MSB to LSB):")
    for i in range(M):
        weight = np.power(q, M - 1 - i)
        print(f"  Level {i} weight: q^{M-1-i} = {q}^{M-1-i} = {weight}")
    
    # Test decoding at each level
    print(f"\nDecoding at each level:")
    for level in range(M):
        reconstruction = quantizer.decode_coarse_to_fine(b_list, T, with_dither=False, max_level=level)
        error = np.linalg.norm(reconstruction - x) / np.linalg.norm(x)
        print(f"  Level {level}: Error = {error:.6f}, Reconstruction = {reconstruction}")
    
    # Test progressive decoding
    print(f"\nProgressive decoding:")
    progressive_reconstructions = quantizer.decode_progressive(b_list, T, with_dither=False)
    for i, reconstruction in enumerate(progressive_reconstructions):
        error = np.linalg.norm(reconstruction - x) / np.linalg.norm(x)
        print(f"  Progressive {i}: Error = {error:.6f}, Reconstruction = {reconstruction}")
    
    # Verify that error decreases monotonically
    errors = []
    for level in range(M):
        reconstruction = quantizer.decode_coarse_to_fine(b_list, T, with_dither=False, max_level=level)
        error = np.linalg.norm(reconstruction - x) / np.linalg.norm(x)
        errors.append(error)
    
    print(f"\nError progression: {[f'{e:.6f}' for e in errors]}")
    
    # Check monotonicity
    is_monotonic = all(errors[i] >= errors[i+1] for i in range(len(errors)-1))
    print(f"Error decreases monotonically: {is_monotonic}")
    
    if is_monotonic:
        print("✅ SUCCESS: Error decreases monotonically as expected!")
    else:
        print("❌ FAILURE: Error does not decrease monotonically")


def test_multiple_vectors():
    """Test with multiple random vectors to verify consistency."""
    
    print("\n=== Testing Multiple Vectors ===\n")
    
    # Setup quantizer
    G = get_d4()
    M = 3
    q = 4
    quantizer = HierarchicalNestedLatticeQuantizer(
        G=G, Q_nn=closest_point_Dn, q=q, beta=0.2,
        alpha=1/3, eps=1e-8, dither=np.zeros(4), M=M
    )
    
    num_trials = 10
    monotonic_count = 0
    
    for trial in range(num_trials):
        # Test vector
        x = np.random.randn(4)
        
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
        
        print(f"Trial {trial+1}: Errors = {[f'{e:.6f}' for e in errors]}, Monotonic = {is_monotonic}")
    
    print(f"\nMonotonic trials: {monotonic_count}/{num_trials}")
    print(f"Success rate: {monotonic_count/num_trials*100:.1f}%")


def test_different_q_values():
    """Test with different q values to verify the fix works."""
    
    print("\n=== Testing Different q Values ===\n")
    
    # Setup quantizer
    G = get_d4()
    M = 3
    
    for q in [2, 3, 4, 5]:
        print(f"Testing q = {q}")
        
        quantizer = HierarchicalNestedLatticeQuantizer(
            G=G, Q_nn=closest_point_Dn, q=q, beta=0.2,
            alpha=1/3, eps=1e-8, dither=np.zeros(4), M=M
        )
        
        # Test vector
        x = np.random.randn(4)
        
        # Encode and decode
        b_list, T = quantizer.encode(x, with_dither=False)
        
        errors = []
        for level in range(M):
            reconstruction = quantizer.decode_coarse_to_fine(b_list, T, with_dither=False, max_level=level)
            error = np.linalg.norm(reconstruction - x) / np.linalg.norm(x)
            errors.append(error)
        
        # Check monotonicity
        is_monotonic = all(errors[i] >= errors[i+1] for i in range(len(errors)-1))
        
        print(f"  Errors: {[f'{e:.6f}' for e in errors]}")
        print(f"  Monotonic: {is_monotonic}")
        print(f"  Weights: {[q**(M-1-i) for i in range(M)]}")
        print()


if __name__ == "__main__":
    test_msb_lsb_ordering()
    test_multiple_vectors()
    test_different_q_values()
