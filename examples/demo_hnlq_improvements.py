#!/usr/bin/env python3
"""
Demonstration of improved HNLQ (Hierarchical Nested Lattice Quantizer) functionality.

This script showcases the improvements made to the HNLQ class, including:
1. Configuration management with HNLQConfig
2. Type hints and modern Python features
3. Better error handling and validation
4. Enhanced documentation and usability
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


def demonstrate_config_management():
    """Demonstrate the new configuration management system."""
    print("=== Configuration Management Demo ===")
    
    # Method 1: Using HNLQConfig dataclass
    config = HNLQConfig(
        q=8,
        beta=1.0,
        alpha=0.5,
        eps=1e-8,
        M=3,
        overload=True,
        decoding="full",
        max_scaling_iterations=15
    )
    print(f"Config created: {config}")
    print(f"Config as dict: {config.to_dict()}")
    
    # Method 2: Using dictionary
    config_dict = {
        'q': 16,
        'beta': 2.0,
        'alpha': 0.3,
        'eps': 1e-10,
        'M': 4,
        'overload': False,
        'decoding': 'progressive'
    }
    config_from_dict = HNLQConfig.from_dict(config_dict)
    print(f"Config from dict: {config_from_dict}")
    
    # Validation demonstration
    try:
        invalid_config = HNLQConfig(q=-1, beta=1.0, alpha=1.0, eps=1e-8, M=3)
    except ValueError as e:
        print(f"Validation caught error: {e}")
    
    print()


def demonstrate_hnlq_usage():
    """Demonstrate the improved HNLQ usage."""
    print("=== HNLQ Usage Demo ===")
    
    # Setup
    G = get_z2()  # Use simple Z2 lattice
    dither = np.random.uniform(0, 1, (1, 2))
    
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
    
    # Initialize HNLQ with new interface
    hnlq = HNLQ(G, simple_closest_point, config, dither)
    
    # Test input vector
    x = np.random.randn(2)
    print(f"Input vector: {x}")
    
    # Encode and decode
    b_list, T = hnlq.encode(x, with_dither=True)
    x_reconstructed = hnlq.decode(b_list, T, with_dither=True)
    
    print(f"Encoding vectors: {len(b_list)} levels")
    print(f"Scaling iterations: {T}")
    print(f"Reconstruction error: {np.linalg.norm(x - x_reconstructed):.6f}")
    
    # Test different decoding methods
    print("\n--- Different Decoding Methods ---")
    
    # Coarse-to-fine decoding
    for level in range(hnlq.M):
        x_coarse = hnlq.decode_coarse_to_fine(b_list, T, with_dither=True, max_level=level)
        error = np.linalg.norm(x - x_coarse)
        print(f"Level {level} error: {error:.6f}")
    
    # Progressive decoding
    progressive_results = hnlq.decode_progressive(b_list, T, with_dither=True)
    print(f"Progressive results: {len(progressive_results)} levels")
    
    # Test quantize convenience method
    x_quantized = hnlq.quantize(x, with_dither=True)
    print(f"Quantized error: {np.linalg.norm(x - x_quantized):.6f}")
    
    print()


def demonstrate_error_handling():
    """Demonstrate improved error handling."""
    print("=== Error Handling Demo ===")
    
    G = get_z2()
    dither = np.random.uniform(0, 1, (1, 2))
    
    # Test invalid configuration
    try:
        config = HNLQConfig(q=0, beta=1.0, alpha=1.0, eps=1e-8, M=3)
    except ValueError as e:
        print(f"Config validation error: {e}")
    
    # Test invalid decoding method
    try:
        config = HNLQConfig(q=8, beta=1.0, alpha=1.0, eps=1e-8, M=3, decoding="invalid")
    except ValueError as e:
        print(f"Invalid decoding method error: {e}")
    
    # Test dither dimension mismatch
    try:
        config = HNLQConfig(q=8, beta=1.0, alpha=1.0, eps=1e-8, M=3)
        wrong_dither = np.random.uniform(0, 1, (1, 3))  # Wrong dimension
        hnlq = HNLQ(G, simple_closest_point, config, wrong_dither)
    except ValueError as e:
        print(f"Dither dimension error: {e}")
    
    # Test depth validation
    config = HNLQConfig(q=8, beta=1.0, alpha=1.0, eps=1e-8, M=3)
    hnlq = HNLQ(G, simple_closest_point, config, dither)
    
    try:
        hnlq.decode_with_depth([], 0, True, depth=-1)
    except ValueError as e:
        print(f"Invalid depth error: {e}")
    
    try:
        hnlq.decode_with_depth([], 0, True, depth=10)
    except ValueError as e:
        print(f"Depth out of range error: {e}")
    
    print()


def demonstrate_properties():
    """Demonstrate the new property-based interface."""
    print("=== Properties Demo ===")
    
    G = get_z2()
    dither = np.random.uniform(0, 1, (1, 2))
    config = HNLQConfig(q=16, beta=2.0, alpha=0.3, eps=1e-10, M=4)
    
    hnlq = HNLQ(G, simple_closest_point, config, dither)
    
    print(f"Quantization parameter (q): {hnlq.q}")
    print(f"Scaling parameter (beta): {hnlq.beta}")
    print(f"Overload parameter (alpha): {hnlq.alpha}")
    print(f"Perturbation (eps): {hnlq.eps}")
    print(f"Hierarchical levels (M): {hnlq.M}")
    print(f"Overload handling: {hnlq.overload}")
    print(f"Default decoding: {hnlq.decoding}")
    
    # Show rate-distortion info
    print(f"\nRate-distortion info: {hnlq.get_rate_distortion_info()}")
    
    print()


def demonstrate_codebook_creation():
    """Demonstrate codebook creation functionality."""
    print("=== Codebook Creation Demo ===")
    
    G = get_z2()
    dither = np.random.uniform(0, 1, (1, 2))
    config = HNLQConfig(q=4, beta=1.0, alpha=0.5, eps=1e-8, M=2)  # Smaller for demo
    
    hnlq = HNLQ(G, simple_closest_point, config, dither)
    
    # Create codebook with dither
    # Note: Codebook creation has dither format compatibility issues
    # that need to be resolved between HNLQ and NLQ classes
    print("Codebook creation functionality available but requires dither format alignment")
    print("between HNLQ and NLQ classes. This is a known compatibility issue.")
    
    # Show a few entries
    print("Sample codebook entries would be generated here once dither format is aligned.")
    
    print()


def main():
    """Run all demonstrations."""
    print("HNLQ Improvements Demonstration")
    print("=" * 50)
    
    demonstrate_config_management()
    demonstrate_hnlq_usage()
    demonstrate_error_handling()
    demonstrate_properties()
    demonstrate_codebook_creation()
    
    print("All demonstrations completed successfully!")


if __name__ == "__main__":
    main()
