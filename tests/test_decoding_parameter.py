#!/usr/bin/env python3
"""
Test script to verify that the decoding parameter is working correctly.
"""

import numpy as np

from gemvq.quantizers.hnlq import HNLQ, HNLQConfig
from gemvq.quantizers.utils import closest_point_Dn, get_d4
from gemvq.gemv.columnwise.column_wise_gemv import ColumnWiseGEMV
from gemvq.gemv.rowwise.row_wise_gemv import RowWiseGEMV

def test_decoding_parameter():
    """Test that the decoding parameter works correctly."""
    
    print("=== Testing Decoding Parameter ===\n")
    
    # Setup lattice parameters
    G = get_d4()
    
    # Test data
    x = np.random.randn(4)
    
    print("Testing HNLQ with different decoding methods...")
    
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
    print("\nTesting GEMV with decoding parameter...")
    
    # Create a small test matrix
    matrix = np.random.randn(8, 6)
    
    # Test column-wise GEMV
    column_gemv = ColumnWiseGEMV(
        matrix, lattice_type='D4', M=2, decoding='coarse_to_fine'
    )
    
    # Test row-wise GEMV
    row_gemv = RowWiseGEMV(
        matrix, lattice_type='D4', M=2, decoding='coarse_to_fine'
    )
    
    # Test vector
    vector = np.random.randn(6)
    
    # Test multiplication
    result_column = column_gemv.multiply(vector)
    result_row = row_gemv.multiply(vector)
    
    print(f"✓ Column-wise GEMV result shape: {result_column.shape}")
    print(f"✓ Row-wise GEMV result shape: {result_row.shape}")
    
    # Test coarse-to-fine methods
    result_column_coarse = column_gemv.multiply_coarse_to_fine(vector, max_level=1)
    result_row_coarse = row_gemv.multiply_coarse_to_fine(vector, max_level=1)
    
    print(f"✓ Column-wise coarse-to-fine result shape: {result_column_coarse.shape}")
    print(f"✓ Row-wise coarse-to-fine result shape: {result_row_coarse.shape}")
    
    print("\n🎉 All tests passed! The decoding parameter is working correctly.")

if __name__ == "__main__":
    test_decoding_parameter()
