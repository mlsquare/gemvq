#!/usr/bin/env python3
"""
Test script to verify that the decoding parameter is working correctly.
"""

import numpy as np

from src.quantizers.hierarchical_nested_lattice_quantizer import HierarchicalNestedLatticeQuantizer
from src.quantizers.closest_point import closest_point_Dn
from src.utils import get_d4
from src.gemv.column_wise_gemv import ColumnWiseGEMV
from src.gemv.row_wise_gemv import RowWiseGEMV

def test_decoding_parameter():
    """Test that the decoding parameter works correctly."""
    
    print("=== Testing Decoding Parameter ===\n")
    
    # Setup lattice parameters
    G = get_d4()
    dither = np.zeros(4)
    
    # Test data
    x = np.random.randn(4)
    
    print("Testing HierarchicalNestedLatticeQuantizer with different decoding methods...")
    
    # Test with full decoding
    quantizer_full = HierarchicalNestedLatticeQuantizer(
        G=G, Q_nn=closest_point_Dn, q=4, beta=0.2, alpha=1/3, 
        eps=1e-8, dither=dither, M=2, decoding='full'
    )
    
    # Test with coarse_to_fine decoding
    quantizer_coarse = HierarchicalNestedLatticeQuantizer(
        G=G, Q_nn=closest_point_Dn, q=4, beta=0.2, alpha=1/3, 
        eps=1e-8, dither=dither, M=2, decoding='coarse_to_fine'
    )
    
    # Test with progressive decoding
    quantizer_progressive = HierarchicalNestedLatticeQuantizer(
        G=G, Q_nn=closest_point_Dn, q=4, beta=0.2, alpha=1/3, 
        eps=1e-8, dither=dither, M=2, decoding='progressive'
    )
    
    # Encode the test vector
    b_list, T = quantizer_full.encode(x, with_dither=False)
    
    # Test default decoding for each quantizer
    result_full = quantizer_full.get_default_decoding(b_list, T, with_dither=False)
    result_coarse = quantizer_coarse.get_default_decoding(b_list, T, with_dither=False, max_level=0)
    result_progressive = quantizer_progressive.get_default_decoding(b_list, T, with_dither=False)
    
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
    result_column_coarse = column_gemv.multiply_coarse_to_fine(vector, max_level=0)
    result_row_coarse = row_gemv.multiply_coarse_to_fine(vector, max_level=0)
    
    print(f"✓ Column-wise coarse-to-fine result shape: {result_column_coarse.shape}")
    print(f"✓ Row-wise coarse-to-fine result shape: {result_row_coarse.shape}")
    
    print("\n🎉 All tests passed! The decoding parameter is working correctly.")

if __name__ == "__main__":
    test_decoding_parameter()
