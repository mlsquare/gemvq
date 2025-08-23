#!/usr/bin/env python3
"""
Test script to verify the reorganization of LatticeQuant modules.

This script tests all major functionality to ensure the reorganization
was successful and all imports work correctly.
"""

import numpy as np
import sys

def test_imports():
    """Test all major imports from the reorganized modules."""
    print("Testing imports...")
    
    try:
        # Test quantizers module
        from src.quantizers.hierarchical_nested_lattice_quantizer import HierarchicalNestedLatticeQuantizer
        from src.quantizers.nested_lattice_quantizer import NestedLatticeQuantizer
        from src.quantizers.closest_point import closest_point_Dn, closest_point_A2, closest_point_E8
        print("✅ Quantizers module imports successful")
        
        # Test exps module
        from src.exps.estimate_inner_product import calculate_inner_product_distortion, plot_distortion_rate, find_best_beta
        from src.exps.compare_quantizer_distortion import run_comparison_experiment
        from src.exps.plot_reconstructed_codebook import generate_codebook
        print("✅ Exps module imports successful")
        
        # Test adaptive module
        from src.adaptive.adaptive_matvec import adaptive_matvec_multiply, create_adaptive_matvec_processor, AdaptiveColumnQuantizer
        from src.adaptive.demo_adaptive_matvec import run_comprehensive_demo
        print("✅ Adaptive module imports successful")
        
        # Test GEMV module
        from src.gemv import ColumnWiseGEMV, RowWiseGEMV, LatticeQuantizedGEMV
        from src.gemv.padder import BlockingStrategy
        from src.gemv.column_wise_gemv import column_wise_gemv
        from src.gemv.row_wise_gemv import row_wise_gemv
        from src.gemv.lattice_quantized_gemv import lattice_quantized_gemv
        print("✅ GEMV module imports successful")
        
        # Test utils module
        from src.utils import (
            get_d4,
            get_a2,
            get_e8,
            precompute_hq_lut,
            calculate_weighted_sum
        )
        print("✅ Utils module imports successful")
        
        # Test main module imports (backward compatibility)
        from src.quantizers.hierarchical_nested_lattice_quantizer import HierarchicalNestedLatticeQuantizer
        from src.quantizers.nested_lattice_quantizer import NestedLatticeQuantizer
        from src.quantizers.closest_point import closest_point_Dn
        from src.exps.estimate_inner_product import plot_distortion_rate
        from src.adaptive.adaptive_matvec import adaptive_matvec_multiply
        from src.gemv import LatticeQuantizedGEMV
        from src.utils import get_d4
        print("✅ Main module imports successful")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False

def test_basic_functionality():
    """Test basic functionality of the quantizers."""
    print("\nTesting basic functionality...")
    
    try:
        from src.quantizers.hierarchical_nested_lattice_quantizer import HierarchicalNestedLatticeQuantizer
        from src.quantizers.closest_point import closest_point_Dn
        from src.utils import get_d4
        
        # Setup parameters
        G = get_d4()
        q = 4
        M = 2
        beta = 0.2
        alpha = 1/3
        eps = 1e-8
        dither = np.zeros(4)
        
        # Create quantizer
        quantizer = HierarchicalNestedLatticeQuantizer(
            G=G, Q_nn=closest_point_Dn, q=q, beta=beta,
            alpha=alpha, eps=eps, dither=dither, M=M
        )
        
        # Test quantization
        x = np.random.normal(0, 1, size=4)
        encoded, T = quantizer.encode(x, with_dither=False)
        decoded = quantizer.decode(encoded, T, with_dither=False)
        
        mse = np.mean((x - decoded) ** 2)
        print(f"✅ Basic quantization test passed (MSE: {mse:.6f})")
        
        return True
        
    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        return False

def test_adaptive_functionality():
    """Test adaptive matrix-vector multiplication."""
    print("\nTesting adaptive functionality...")
    
    try:
        from src.adaptive.adaptive_matvec import adaptive_matvec_multiply
        
        # Create test data
        matrix = np.random.randn(4, 8)
        sparse_vector = np.zeros(8)
        sparse_vector[[0, 4, 7]] = [1.5, 2.0, -1.0]
        target_rates = [2.5, 3.0, 3.5, 2.0, 4.0, 3.2, 2.8, 3.6]
        sparsity_pattern = [0, 4, 7]
        
        # Test adaptive multiplication
        result = adaptive_matvec_multiply(
            matrix, sparse_vector, target_rates, sparsity_pattern, 'D4', 2
        )
        
        print(f"✅ Adaptive matrix-vector multiplication test passed")
        print(f"   Result shape: {result.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Adaptive functionality test failed: {e}")
        return False

def test_exps_functionality():
    """Test exps functionality."""
    print("\nTesting exps functionality...")
    
    try:
        from src.exps.estimate_inner_product import find_best_beta
        from src.utils import get_d4
        from src.quantizers.closest_point import closest_point_Dn
        
        # Test parameter optimization
        G = get_d4()
        q, m = 4, 2
        alpha = 1/3
        sig_l = np.sqrt(2) * 0.076602
        eps = 1e-8 * np.random.normal(0, 1, size=4)
        
        optimal_R, optimal_beta = find_best_beta(
            G, closest_point_Dn, q, m, alpha, sig_l, eps
        )
        
        print(f"✅ Parameter optimization test passed")
        print(f"   Optimal beta: {optimal_beta:.4f}")
        print(f"   Optimal rate: {optimal_R:.4f} bits/dimension")
        
        return True
        
    except Exception as e:
        print(f"❌ Exps functionality test failed: {e}")
        return False

def test_gemv_functionality():
    """Test GEMV functionality."""
    print("\nTesting GEMV functionality...")
    
    try:
        from src.gemv import ColumnWiseGEMV, RowWiseGEMV, LatticeQuantizedGEMV
        
        # Create test data
        matrix = np.random.randn(8, 6)
        vector = np.random.randn(6)
        
        # Test column-wise approach
        col_processor = ColumnWiseGEMV(matrix, 'D4', 2)
        col_result = col_processor.multiply(vector)
        
        # Test row-wise approach
        row_processor = RowWiseGEMV(matrix, 'D4', 2)
        row_result = row_processor.multiply(vector)
        
        # Test unified interface
        unified_processor = LatticeQuantizedGEMV(matrix, 'auto', 'D4', 2)
        unified_result = unified_processor.multiply(vector)
        
        # Compare with exact computation
        exact_result = matrix @ vector
        
        col_error = np.linalg.norm(col_result - exact_result) / np.linalg.norm(exact_result)
        row_error = np.linalg.norm(row_result - exact_result) / np.linalg.norm(exact_result)
        unified_error = np.linalg.norm(unified_result - exact_result) / np.linalg.norm(exact_result)
        
        print(f"✅ GEMV functionality test passed")
        print(f"   Column-wise error: {col_error:.6f}")
        print(f"   Row-wise error: {row_error:.6f}")
        print(f"   Unified error: {unified_error:.6f}")
        print(f"   Selected approach: {unified_processor.approach}")
        
        return True
        
    except Exception as e:
        print(f"❌ GEMV functionality test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("LatticeQuant Reorganization Test")
    print("=" * 40)
    
    tests = [
        test_imports,
        test_basic_functionality,
        test_adaptive_functionality,
        test_exps_functionality,
        test_gemv_functionality
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 40)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Reorganization successful.")
        return 0
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 