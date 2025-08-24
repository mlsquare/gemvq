"""
Test Different Hierarchical Decoding Methods

This test compares the performance of different decoding methods in the
hierarchical quantizer to identify which one works correctly.
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.lattices.quantizers.hierarchical_nested_lattice_quantizer import HierarchicalNestedLatticeQuantizer
from src.lattices.utils import closest_point_Dn
from src.lattices.utils import get_d4, calculate_mse


def test_all_decoding_methods():
    """
    Test all available decoding methods in the hierarchical quantizer.
    """
    print("=== Testing All Hierarchical Decoding Methods ===\n")
    
    # Setup parameters
    G = get_d4()
    q = 3
    M = 3
    beta = 1.0
    alpha = 1.0
    eps = 1e-8
    dither = np.zeros(4)
    
    # Create hierarchical quantizer
    hq = HierarchicalNestedLatticeQuantizer(
        G=G,
        Q_nn=closest_point_Dn,
        q=q,
        beta=beta,
        alpha=alpha,
        eps=eps,
        dither=dither,
        M=M
    )
    
    # Test with a simple D4 lattice point
    test_point = np.array([-1.0, -1.0, 0.0, 0.0])  # G[:, 0]
    
    print(f"Testing with D4 lattice point: {test_point}")
    
    # Encode
    b_list, T = hq.encode(test_point, with_dither=False)
    print(f"Encoding vectors: {b_list}")
    print(f"Scaling factor T: {T}")
    
    # Test different decoding methods
    print("\n--- Decoding Method Comparison ---")
    
    # Method 1: Regular decode
    try:
        reconstructed_regular = hq.decode(b_list, T, with_dither=False)
        mse_regular = calculate_mse(test_point, reconstructed_regular)
        print(f"1. Regular decode: MSE = {mse_regular:.6f}")
        print(f"   Reconstructed: {reconstructed_regular}")
    except Exception as e:
        print(f"1. Regular decode: ERROR - {e}")
    
    # Method 2: decode_coarse_to_fine at full depth
    try:
        reconstructed_coarse_full = hq.decode_coarse_to_fine(b_list, T, with_dither=False, max_level=M-1)
        mse_coarse_full = calculate_mse(test_point, reconstructed_coarse_full)
        print(f"2. decode_coarse_to_fine (full): MSE = {mse_coarse_full:.6f}")
        print(f"   Reconstructed: {reconstructed_coarse_full}")
    except Exception as e:
        print(f"2. decode_coarse_to_fine (full): ERROR - {e}")
    
    # Method 3: decode_with_depth at full depth
    try:
        reconstructed_depth_full = hq.decode_with_depth(b_list, T, with_dither=False, depth=M-1)
        mse_depth_full = calculate_mse(test_point, reconstructed_depth_full)
        print(f"3. decode_with_depth (full): MSE = {mse_depth_full:.6f}")
        print(f"   Reconstructed: {reconstructed_depth_full}")
    except Exception as e:
        print(f"3. decode_with_depth (full): ERROR - {e}")
    
    # Method 4: decode_progressive
    try:
        progressive_reconstructions = hq.decode_progressive(b_list, T, with_dither=False)
        mse_progressive_full = calculate_mse(test_point, progressive_reconstructions[-1])  # Last (finest) level
        print(f"4. decode_progressive (finest): MSE = {mse_progressive_full:.6f}")
        print(f"   Reconstructed: {progressive_reconstructions[-1]}")
        print(f"   All progressive levels: {len(progressive_reconstructions)}")
    except Exception as e:
        print(f"4. decode_progressive: ERROR - {e}")
    
    # Method 5: quantize method
    try:
        quantized = hq.quantize(test_point, with_dither=False)
        mse_quantize = calculate_mse(test_point, quantized)
        print(f"5. quantize method: MSE = {mse_quantize:.6f}")
        print(f"   Reconstructed: {quantized}")
    except Exception as e:
        print(f"5. quantize method: ERROR - {e}")
    
    print("\n" + "="*50)


def test_progressive_decoding_levels():
    """
    Test progressive decoding at different levels to understand the behavior.
    """
    print("=== Testing Progressive Decoding Levels ===\n")
    
    # Setup parameters
    G = get_d4()
    q = 3
    M = 3
    beta = 1.0
    alpha = 1.0
    eps = 1e-8
    dither = np.zeros(4)
    
    # Create hierarchical quantizer
    hq = HierarchicalNestedLatticeQuantizer(
        G=G,
        Q_nn=closest_point_Dn,
        q=q,
        beta=beta,
        alpha=alpha,
        eps=eps,
        dither=dither,
        M=M
    )
    
    # Test with a simple D4 lattice point
    test_point = np.array([-1.0, -1.0, 0.0, 0.0])  # G[:, 0]
    
    print(f"Testing progressive decoding with point: {test_point}")
    
    # Encode
    b_list, T = hq.encode(test_point, with_dither=False)
    print(f"Encoding vectors: {b_list}")
    
    # Test progressive decoding
    try:
        progressive_reconstructions = hq.decode_progressive(b_list, T, with_dither=False)
        
        print(f"\nProgressive reconstructions ({len(progressive_reconstructions)} levels):")
        for i, recon in enumerate(progressive_reconstructions):
            mse = calculate_mse(test_point, recon)
            print(f"  Level {i}: MSE = {mse:.6f}, Reconstructed = {recon}")
        
        # Check if the finest level matches regular decode
        regular_recon = hq.decode(b_list, T, with_dither=False)
        regular_mse = calculate_mse(test_point, regular_recon)
        finest_mse = calculate_mse(test_point, progressive_reconstructions[-1])
        
        print(f"\nComparison:")
        print(f"  Regular decode MSE: {regular_mse:.6f}")
        print(f"  Progressive finest MSE: {finest_mse:.6f}")
        print(f"  Match: {np.isclose(regular_mse, finest_mse, rtol=1e-10)}")
        
    except Exception as e:
        print(f"Progressive decoding failed: {e}")
    
    print("\n" + "="*50)


def test_custom_decode_implementation():
    """
    Test a custom decode implementation that might work better.
    """
    print("=== Testing Custom Decode Implementation ===\n")
    
    # Setup parameters
    G = get_d4()
    q = 3
    M = 3
    beta = 1.0
    alpha = 1.0
    eps = 1e-8
    dither = np.zeros(4)
    
    # Create hierarchical quantizer
    hq = HierarchicalNestedLatticeQuantizer(
        G=G,
        Q_nn=closest_point_Dn,
        q=q,
        beta=beta,
        alpha=alpha,
        eps=eps,
        dither=dither,
        M=M
    )
    
    # Test with a simple D4 lattice point
    test_point = np.array([-1.0, -1.0, 0.0, 0.0])  # G[:, 0]
    
    print(f"Testing custom decode with point: {test_point}")
    
    # Encode
    b_list, T = hq.encode(test_point, with_dither=False)
    print(f"Encoding vectors: {b_list}")
    
    # Custom decode implementation based on the _decode method
    try:
        # Replicate the _decode method logic
        x_hat = np.zeros_like(G[0])
        
        # Reconstruct by reversing the encoding process
        for i, b in enumerate(b_list):
            # Each level contributes q^i * G * b
            x_hat += (q ** i) * np.dot(G, b)
        
        # Apply scaling
        custom_reconstructed = beta * x_hat * (2 ** (alpha * T))
        
        mse_custom = calculate_mse(test_point, custom_reconstructed)
        print(f"Custom decode: MSE = {mse_custom:.6f}")
        print(f"Reconstructed: {custom_reconstructed}")
        
        # Compare with regular decode
        regular_reconstructed = hq.decode(b_list, T, with_dither=False)
        mse_regular = calculate_mse(test_point, regular_reconstructed)
        
        print(f"\nComparison:")
        print(f"  Regular decode MSE: {mse_regular:.6f}")
        print(f"  Custom decode MSE: {mse_custom:.6f}")
        print(f"  Match: {np.isclose(mse_regular, mse_custom, rtol=1e-10)}")
        
    except Exception as e:
        print(f"Custom decode failed: {e}")
    
    print("\n" + "="*50)


def test_multiple_d4_points():
    """
    Test with multiple D4 lattice points to see if the issue is consistent.
    """
    print("=== Testing Multiple D4 Lattice Points ===\n")
    
    # Setup parameters
    G = get_d4()
    q = 3
    M = 3
    beta = 1.0
    alpha = 1.0
    eps = 1e-8
    dither = np.zeros(4)
    
    # Create hierarchical quantizer
    hq = HierarchicalNestedLatticeQuantizer(
        G=G,
        Q_nn=closest_point_Dn,
        q=q,
        beta=beta,
        alpha=alpha,
        eps=eps,
        dither=dither,
        M=M
    )
    
    # Generate D4 lattice points
    coeffs = np.array([
        [1, 0, 0, 0],   # G[:, 0] = [-1, -1, 0, 0]
        [0, 1, 0, 0],   # G[:, 1] = [1, -1, 0, 0]  
        [0, 0, 1, 0],   # G[:, 2] = [0, 1, -1, 0]
        [0, 0, 0, 1],   # G[:, 3] = [0, 0, 1, -1]
    ])
    
    d4_points = np.dot(coeffs, G.T)
    
    print("Testing with D4 lattice basis vectors:")
    
    for i, point in enumerate(d4_points):
        print(f"\n--- Point {i}: {point} ---")
        
        # Encode
        b_list, T = hq.encode(point, with_dither=False)
        
        # Test different methods
        try:
            # Regular decode
            regular_recon = hq.decode(b_list, T, with_dither=False)
            mse_regular = calculate_mse(point, regular_recon)
            
            # Coarse-to-fine full depth
            coarse_recon = hq.decode_coarse_to_fine(b_list, T, with_dither=False, max_level=M-1)
            mse_coarse = calculate_mse(point, coarse_recon)
            
            # Progressive finest level
            progressive_recons = hq.decode_progressive(b_list, T, with_dither=False)
            mse_progressive = calculate_mse(point, progressive_recons[-1])
            
            print(f"  Regular decode: MSE = {mse_regular:.6f}")
            print(f"  Coarse-to-fine: MSE = {mse_coarse:.6f}")
            print(f"  Progressive: MSE = {mse_progressive:.6f}")
            
            # Check if they match
            regular_coarse_match = np.isclose(mse_regular, mse_coarse, rtol=1e-10)
            regular_progressive_match = np.isclose(mse_regular, mse_progressive, rtol=1e-10)
            
            print(f"  Regular ≈ Coarse-to-fine: {regular_coarse_match}")
            print(f"  Regular ≈ Progressive: {regular_progressive_match}")
            
        except Exception as e:
            print(f"  Error: {e}")
    
    print("\n" + "="*50)


if __name__ == "__main__":
    print("Testing Hierarchical Decoding Methods\n")
    
    test_all_decoding_methods()
    test_progressive_decoding_levels()
    test_custom_decode_implementation()
    test_multiple_d4_points()
    
    print("\nAll tests completed!")
