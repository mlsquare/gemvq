#!/usr/bin/env python3
"""
Demonstration of HNLQ (Hierarchical Nested Lattice Quantizer) without dither.

This script shows how to run HNLQ quantization without dithering by setting
with_dither=False in the encode and decode methods.
"""

import numpy as np
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from quantizers.lattice.hnlq import HNLQ, HNLQConfig
from quantizers.lattice.utils import get_z2, custom_round


def simple_closest_point(x):
    """Simple closest point function for testing."""
    return custom_round(x)


def demonstrate_hnlq_without_dither():
    """Demonstrate HNLQ usage without dither."""
    print("=== HNLQ Without Dither Demo ===")
    
    # Setup
    G = get_z2()  # Use simple Z2 lattice
    dither = np.random.uniform(0, 1, (1, 2))  # Dither is still required for initialization
    
    # Create configuration
    config = HNLQConfig(
        q=8,
        beta=1.0,
        alpha=0.5,
        eps=1e-8,
        M=3,
        overload=True,
        decoding="full",
        max_scaling_iterations=10
    )
    
    # Initialize HNLQ
    hnlq = HNLQ(G, simple_closest_point, config, dither)
    
    # Test input vector
    x = np.random.randn(2)
    print(f"Input vector: {x}")
    
    # Encode WITHOUT dither
    print("\n--- Encoding WITHOUT dither ---")
    b_list, T = hnlq.encode(x, with_dither=False)
    print(f"Encoding vectors: {len(b_list)} levels")
    print(f"Scaling iterations: {T}")
    
    # Decode WITHOUT dither
    print("\n--- Decoding WITHOUT dither ---")
    x_reconstructed = hnlq.decode(b_list, T, with_dither=False)
    reconstruction_error = np.linalg.norm(x - x_reconstructed)
    print(f"Reconstruction error: {reconstruction_error:.6f}")
    
    # Compare with dither version
    print("\n--- Comparison: With vs Without Dither ---")
    
    # Encode WITH dither
    b_list_with_dither, T_with_dither = hnlq.encode(x, with_dither=True)
    x_reconstructed_with_dither = hnlq.decode(b_list_with_dither, T_with_dither, with_dither=True)
    error_with_dither = np.linalg.norm(x - x_reconstructed_with_dither)
    
    print(f"Error WITH dither: {error_with_dither:.6f}")
    print(f"Error WITHOUT dither: {reconstruction_error:.6f}")
    print(f"Difference: {abs(error_with_dither - reconstruction_error):.6f}")
    
    # Test different decoding methods without dither
    print("\n--- Different Decoding Methods (Without Dither) ---")
    
    # Coarse-to-fine decoding without dither
    for level in range(hnlq.M):
        x_coarse = hnlq.decode_coarse_to_fine(b_list, T, with_dither=False, max_level=level)
        error = np.linalg.norm(x - x_coarse)
        print(f"Level {level} error (no dither): {error:.6f}")
    
    # Progressive decoding without dither
    progressive_results = hnlq.decode_progressive(b_list, T, with_dither=False)
    print(f"Progressive results (no dither): {len(progressive_results)} levels")
    
    # Test quantize convenience method without dither
    x_quantized = hnlq.quantize(x, with_dither=False)
    print(f"Quantized error (no dither): {np.linalg.norm(x - x_quantized):.6f}")
    
    print()


def demonstrate_multiple_samples():
    """Demonstrate HNLQ without dither on multiple samples."""
    print("=== Multiple Samples Demo (Without Dither) ===")
    
    # Setup
    G = get_z2()
    dither = np.random.uniform(0, 1, (1, 2))
    
    config = HNLQConfig(
        q=16,
        beta=1.0,
        alpha=0.5,
        eps=1e-8,
        M=4,
        overload=True,
        decoding="full"
    )
    
    hnlq = HNLQ(G, simple_closest_point, config, dither)
    
    # Test multiple random vectors
    n_samples = 10
    errors_without_dither = []
    errors_with_dither = []
    
    for i in range(n_samples):
        x = np.random.randn(2)
        
        # Without dither
        b_list, T = hnlq.encode(x, with_dither=False)
        x_reconstructed = hnlq.decode(b_list, T, with_dither=False)
        error_no_dither = np.linalg.norm(x - x_reconstructed)
        errors_without_dither.append(error_no_dither)
        
        # With dither
        b_list_d, T_d = hnlq.encode(x, with_dither=True)
        x_reconstructed_d = hnlq.decode(b_list_d, T_d, with_dither=True)
        error_with_dither = np.linalg.norm(x - x_reconstructed_d)
        errors_with_dither.append(error_with_dither)
        
        print(f"Sample {i+1}: No dither error: {error_no_dither:.6f}, With dither error: {error_with_dither:.6f}")
    
    print(f"\nAverage error WITHOUT dither: {np.mean(errors_without_dither):.6f}")
    print(f"Average error WITH dither: {np.mean(errors_with_dither):.6f}")
    print(f"Standard deviation WITHOUT dither: {np.std(errors_without_dither):.6f}")
    print(f"Standard deviation WITH dither: {np.std(errors_with_dither):.6f}")
    
    print()


def main():
    """Run all demonstrations."""
    print("HNLQ Without Dither Demonstration")
    print("=" * 50)
    
    demonstrate_hnlq_without_dither()
    demonstrate_multiple_samples()
    
    print("All demonstrations completed successfully!")
    print("\nKey takeaway: To run HNLQ without dither, simply set with_dither=False")
    print("in both encode() and decode() method calls.")


if __name__ == "__main__":
    main()
