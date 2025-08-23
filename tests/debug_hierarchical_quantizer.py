#!/usr/bin/env python3
"""
Debug script to understand hierarchical quantizer behavior.
"""

import numpy as np
from src.quantizers.hierarchical_nested_lattice_quantizer import HierarchicalNestedLatticeQuantizer
from src.utils import get_d4
from src.quantizers.closest_point import closest_point_Dn


def debug_hierarchical_quantizer():
    """Debug the hierarchical quantizer behavior."""
    
    print("=== Debugging Hierarchical Quantizer ===\n")
    
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
    
    # Analyze the weights
    print(f"\nWeight analysis:")
    for i in range(M):
        weight = np.power(q, i)
        print(f"  Level {i} weight: q^{i} = {q}^{i} = {weight}")
    
    # Test with different q values
    print(f"\nTesting with different q values:")
    for q_test in [2, 3, 4, 5]:
        quantizer_test = HierarchicalNestedLatticeQuantizer(
            G=G, Q_nn=closest_point_Dn, q=q_test, beta=0.2,
            alpha=1/3, eps=1e-8, dither=np.zeros(4), M=M
        )
        
        b_list_test, T_test = quantizer_test.encode(x, with_dither=False)
        
        errors = []
        for level in range(M):
            reconstruction = quantizer_test.decode_coarse_to_fine(b_list_test, T_test, with_dither=False, max_level=level)
            error = np.linalg.norm(reconstruction - x) / np.linalg.norm(x)
            errors.append(error)
        
        print(f"  q={q_test}: Errors = {[f'{e:.6f}' for e in errors]}")
        
        # Check if monotonic
        is_monotonic = all(errors[i] >= errors[i+1] for i in range(len(errors)-1))
        print(f"    Monotonic: {is_monotonic}")


def test_simple_case():
    """Test a very simple case to understand the behavior."""
    
    print("\n=== Simple Case Test ===\n")
    
    # Use a very simple vector
    x = np.array([1.0, 0.5, 0.25, 0.125])
    print(f"Simple vector: {x}")
    
    # Setup quantizer with small q
    G = get_d4()
    M = 3
    q = 2  # Use smaller q to see the effect more clearly
    
    quantizer = HierarchicalNestedLatticeQuantizer(
        G=G, Q_nn=closest_point_Dn, q=q, beta=1.0,  # Use beta=1 for simplicity
        alpha=1/3, eps=1e-8, dither=np.zeros(4), M=M
    )
    
    # Encode
    b_list, T = quantizer.encode(x, with_dither=False)
    print(f"Encoding vectors:")
    for i, b in enumerate(b_list):
        print(f"  Level {i}: {b}")
    
    # Test each level
    print(f"\nDecoding at each level:")
    for level in range(M):
        reconstruction = quantizer.decode_coarse_to_fine(b_list, T, with_dither=False, max_level=level)
        error = np.linalg.norm(reconstruction - x) / np.linalg.norm(x)
        print(f"  Level {level}: Error = {error:.6f}")
        print(f"    Reconstruction: {reconstruction}")
        print(f"    Weights used: {[q**i for i in range(level+1)]}")


if __name__ == "__main__":
    debug_hierarchical_quantizer()
    test_simple_case()
