#!/usr/bin/env python3
"""
Demonstration script for LatticeQuant library.
This script shows basic usage of the hierarchical nested lattice quantizer.
"""

import numpy as np
from src import (
    HierarchicalNestedLatticeQuantizer, 
    NestedLatticeQuantizer,
    get_d4, 
    closest_point_Dn,
    precompute_hq_lut,
    calculate_weighted_sum,
    calculate_mse
)

def main():
    print("🚀 LatticeQuant Demo")
    print("=" * 50)
    
    # Setup parameters
    G = get_d4()  # D4 lattice
    q = 4         # Quantization parameter
    M = 2         # Hierarchical levels
    beta = 0.2    # Scaling parameter
    alpha = 1/3   # Overload scaling
    eps = 1e-8    # Perturbation
    dither = np.zeros(4)  # No dither
    
    print(f"📊 Parameters:")
    print(f"   Lattice: D4")
    print(f"   Quantization parameter (q): {q}")
    print(f"   Hierarchical levels (M): {M}")
    print(f"   Beta: {beta}")
    print(f"   Alpha: {alpha}")
    
    # Create quantizer
    quantizer = HierarchicalNestedLatticeQuantizer(
        G=G, Q_nn=closest_point_Dn, q=q, beta=beta, 
        alpha=alpha, eps=eps, dither=dither, M=M
    )
    
    print(f"\n🔧 Quantizer created successfully!")
    
    # Test quantization
    x = np.random.normal(0, 1, size=4)
    print(f"\n📥 Input vector: {x}")
    
    # Encode and decode
    encoded, T = quantizer.encode(x, with_dither=False)
    decoded = quantizer.decode(encoded, T, with_dither=False)
    
    print(f"📤 Encoded: {encoded}")
    print(f"📤 Decoded: {decoded}")
    print(f"📊 MSE: {calculate_mse(x, decoded):.6f}")
    
    # Test inner product estimation
    print(f"\n🔗 Inner Product Estimation Demo:")
    
    # Precompute lookup table
    print("   Computing lookup table...")
    lut = precompute_hq_lut(G, closest_point_Dn, q, M, eps)
    print(f"   Lookup table size: {len(lut)} entries")
    
    # Create two test vectors
    x1 = np.random.normal(0, 1, size=4)
    x2 = np.random.normal(0, 1, size=4)
    
    print(f"   Vector 1: {x1}")
    print(f"   Vector 2: {x2}")
    
    # True inner product
    true_ip = np.dot(x1, x2)
    print(f"   True inner product: {true_ip:.6f}")
    
    # Encode both vectors
    enc1, T1 = quantizer.encode(x1, with_dither=False)
    enc2, T2 = quantizer.encode(x2, with_dither=False)
    
    # Calculate scaling factor
    c = (2**(quantizer.alpha * (T1 + T2))) * (quantizer.beta**2)
    
    # Estimate inner product using lookup table
    estimated_ip = c * calculate_weighted_sum(enc1, enc2, lut, q)
    
    print(f"   Estimated inner product: {estimated_ip:.6f}")
    print(f"   Relative error: {abs(estimated_ip - true_ip)/abs(true_ip)*100:.2f}%")
    
    # Compare with classic quantizer
    print(f"\n🔄 Comparison with Classic Quantizer:")
    
    classic_quantizer = NestedLatticeQuantizer(
        G=G, Q_nn=closest_point_Dn, q=q**M, beta=beta, 
        alpha=alpha, eps=eps, dither=dither
    )
    
    # Quantize with classic method
    classic_encoded, classic_T = classic_quantizer.encode(x, with_dither=False)
    classic_decoded = classic_quantizer.decode(classic_encoded, classic_T, with_dither=False)
    
    print(f"   Classic MSE: {calculate_mse(x, classic_decoded):.6f}")
    print(f"   Hierarchical MSE: {calculate_mse(x, decoded):.6f}")
    
    print(f"\n✅ Demo completed successfully!")
    print(f"\n📚 For more examples, see the README.md file.")

if __name__ == "__main__":
    main() 