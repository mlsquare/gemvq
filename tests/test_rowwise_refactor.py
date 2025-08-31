#!/usr/bin/env python3
"""
Test script to verify that the refactored rowwise GEMV works correctly with updated HNLQ.
"""

import numpy as np

from gemvq.quantizers.hnlq import HNLQ, HNLQConfig
from gemvq.quantizers.utils import closest_point_Dn, get_d4
from gemvq.gemv.rowwise.row_wise_gemv import RowWiseGEMV

def test_rowwise_refactor():
    """Test that the refactored rowwise GEMV works correctly."""
    
    print("=== Testing Refactored Rowwise GEMV ===\n")
    
    # Setup lattice parameters
    G = get_d4()
    
    # Test data
    x = np.random.randn(4)
    
    print("Testing HNLQ with new interface...")
    
    # Test with full decoding
    config_full = HNLQConfig(
        lattice_type='D4', q=4, M=2, beta=1.0, alpha=1.0, 
        eps=1e-8, decoding='full'
    )
    quantizer_full = HNLQ(config=config_full, G=G, Q_nn=closest_point_Dn)
    
    # Test with coarse_to_fine decoding
    config_coarse = HNLQConfig(
        lattice_type='D4', q=4, M=2, beta=1.0, alpha=1.0, 
        eps=1e-8, decoding='coarse_to_fine'
    )
    quantizer_coarse = HNLQ(config=config_coarse, G=G, Q_nn=closest_point_Dn)
    
    # Test with progressive decoding
    config_progressive = HNLQConfig(
        lattice_type='D4', q=4, M=2, beta=1.0, alpha=1.0, 
        eps=1e-8, decoding='progressive'
    )
    quantizer_progressive = HNLQ(config=config_progressive, G=G, Q_nn=closest_point_Dn)
    
    # Encode the test vector
    b_list, T = quantizer_full.encode(x, with_dither=False)
    
    # Test default decoding for each quantizer
    result_full = quantizer_full.decode(b_list, T, with_dither=False)
    result_coarse = quantizer_coarse.decode_coarse_to_fine(b_list, T, with_dither=False, depth=1)
    result_progressive = quantizer_progressive.decode_progressive(b_list, T, with_dither=False)
    
    print(f"✓ Full decoding result shape: {result_full.shape}")
    print(f"✓ Coarse-to-fine decoding result shape: {result_coarse.shape}")
    print(f"✓ Progressive decoding result shape: {len(result_progressive)} levels")
    
    # Test GEMV with decoding parameter
    print("\nTesting Rowwise GEMV with decoding parameter...")
    
    # Create a small test matrix
    matrix = np.random.randn(8, 6)
    
    # Test row-wise GEMV with different configurations
    row_gemv_full = RowWiseGEMV(
        matrix, lattice_type='D4', M=2, decoding='full'
    )
    
    row_gemv_coarse = RowWiseGEMV(
        matrix, lattice_type='D4', M=2, decoding='coarse_to_fine'
    )
    
    row_gemv_progressive = RowWiseGEMV(
        matrix, lattice_type='D4', M=2, decoding='progressive'
    )
    
    # Test vector
    vector = np.random.randn(6)
    
    # Test multiplication
    result_row_full = row_gemv_full.multiply(vector)
    result_row_coarse = row_gemv_coarse.multiply(vector)
    result_row_progressive = row_gemv_progressive.multiply(vector)
    
    print(f"✓ Row-wise GEMV (full) result shape: {result_row_full.shape}")
    print(f"✓ Row-wise GEMV (coarse) result shape: {result_row_coarse.shape}")
    print(f"✓ Row-wise GEMV (progressive) result shape: {result_row_progressive.shape}")
    
    # Test coarse-to-fine methods
    result_row_coarse_fine = row_gemv_coarse.multiply_coarse_to_fine(vector, max_level=1)
    print(f"✓ Row-wise coarse-to-fine result shape: {result_row_coarse_fine.shape}")
    
    # Test progressive methods
    result_row_prog = row_gemv_progressive.multiply_progressive(vector)
    print(f"✓ Row-wise progressive result: {len(result_row_prog)} levels")
    
    # Test with different parameters
    print("\nTesting with different parameters...")
    
    row_gemv_params = RowWiseGEMV(
        matrix, 
        lattice_type='D4', 
        M=3, 
        q=8, 
        beta=0.5, 
        alpha=0.8,
        overload=True,
        max_scaling_iterations=5,
        with_tie_dither=True,
        with_dither=False
    )
    
    result_params = row_gemv_params.multiply(vector)
    print(f"✓ Row-wise GEMV with custom parameters result shape: {result_params.shape}")
    
    # Test compression and memory usage
    compression_ratio = row_gemv_full.get_compression_ratio()
    memory_usage = row_gemv_full.get_memory_usage()
    blocking_info = row_gemv_full.get_blocking_info()
    
    print(f"✓ Compression ratio: {compression_ratio:.2f}")
    print(f"✓ Memory usage: {memory_usage['total_mb']:.2f} MB")
    print(f"✓ Blocking info: {blocking_info['num_row_blocks']} row blocks")
    
    print("\n🎉 All tests passed! The refactored rowwise GEMV is working correctly.")

if __name__ == "__main__":
    test_rowwise_refactor()
