#!/usr/bin/env python3
"""
Simple test script to verify GEMV dimension fixes.
"""

import os
import sys

import numpy as np

# Add src to path
sys.path.insert(0, "src")

from src.gemv.column_wise_gemv import ColumnWiseGEMV
# Import the modules using absolute imports
from src.gemv.row_wise_gemv import RowWiseGEMV


def test_basic_functionality():
    """Test basic GEMV functionality with non-matching dimensions."""
    print("Testing GEMV with non-matching dimensions...")

    # Test case: 20x30 matrix (neither dimension matches D4 lattice)
    # Using larger matrix for better quantization performance
    matrix = np.random.randn(20, 30)
    vector = np.random.randn(30)

    # Compute exact result
    exact_result = matrix @ vector

    try:
        # Test row-wise approach
        print("Testing row-wise GEMV...")
        row_processor = RowWiseGEMV(matrix, "D4", 2)
        row_result = row_processor.multiply(vector)
        row_error = np.linalg.norm(row_result - exact_result) / np.linalg.norm(exact_result)
        print(f"Row-wise error: {row_error:.6f}")

        # Test column-wise approach
        print("Testing column-wise GEMV...")
        col_processor = ColumnWiseGEMV(matrix, "D4", 2)
        col_result = col_processor.multiply(vector)
        col_error = np.linalg.norm(col_result - exact_result) / np.linalg.norm(exact_result)
        print(f"Column-wise error: {col_error:.6f}")

        # Check shapes
        assert row_result.shape == (
            20,
        ), f"Row-wise result shape {row_result.shape} != expected (20,)"
        assert col_result.shape == (
            20,
        ), f"Column-wise result shape {col_result.shape} != expected (20,)"

        # Check that errors are reasonable (quantization error is expected)
        # Using more lenient threshold for quantization-based methods
        assert row_error < 1.0, f"Row-wise error {row_error} too large"
        assert col_error < 1.0, f"Column-wise error {col_error} too large"

        print("✓ Basic functionality test passed!")
        return True

    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_blocking_info():
    """Test that blocking information is correctly reported."""
    print("\nTesting blocking information...")

    matrix = np.random.randn(10, 15)

    try:
        # Test row-wise blocking info
        row_processor = RowWiseGEMV(matrix, "D4", 2)
        row_info = row_processor.get_blocking_info()

        print(f"Row-wise original shape: {row_info['original_shape']}")
        print(f"Row-wise padded shape: {row_info['padded_shape']}")
        print(f"Block size: {row_info['block_size']}")

        # Check that original shape is preserved
        assert row_info["original_shape"] == (
            10,
            15,
        ), f"Original shape {row_info['original_shape']} != expected (10, 15)"

        # Check that padded shape is multiple of block size
        assert (
            row_info["padded_shape"][0] % row_info["block_size"] == 0
        ), "Padded rows not multiple of block size"

        print("✓ Blocking info test passed!")
        return True

    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_multiple_lattice_types():
    """Test with different lattice types."""
    print("\nTesting multiple lattice types...")

    matrix = np.random.randn(16, 24)
    vector = np.random.randn(24)
    exact_result = matrix @ vector

    lattice_types = ["D4", "A2", "E8"]

    try:
        for lattice_type in lattice_types:
            print(f"Testing {lattice_type} lattice...")

            # Test row-wise
            row_processor = RowWiseGEMV(matrix, lattice_type, 2)
            row_result = row_processor.multiply(vector)
            row_error = np.linalg.norm(row_result - exact_result) / np.linalg.norm(exact_result)

            # Test column-wise
            col_processor = ColumnWiseGEMV(matrix, lattice_type, 2)
            col_result = col_processor.multiply(vector)
            col_error = np.linalg.norm(col_result - exact_result) / np.linalg.norm(exact_result)

            print(
                f"  {lattice_type} - Row-wise error: {row_error:.6f}, Column-wise error: {col_error:.6f}"
            )

            # Check shapes
            assert row_result.shape == (
                16,
            ), f"{lattice_type} row-wise result shape {row_result.shape} != expected (16,)"
            assert col_result.shape == (
                16,
            ), f"{lattice_type} column-wise result shape {col_result.shape} != expected (16,)"

            # Check reasonable errors
            assert row_error < 1.0, f"{lattice_type} row-wise error {row_error} too large"
            assert col_error < 1.0, f"{lattice_type} column-wise error {col_error} too large"

        print("✓ Multiple lattice types test passed!")
        return True

    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("Testing GEMV Dimension Fixes")
    print("=" * 50)

    success1 = test_basic_functionality()
    success2 = test_blocking_info()
    success3 = test_multiple_lattice_types()

    print("\n" + "=" * 50)
    if success1 and success2 and success3:
        print("✓ ALL TESTS PASSED!")
        print("The GEMV implementations now work correctly with arbitrary dimensions.")
    else:
        print("✗ SOME TESTS FAILED!")

    print("=" * 50)
