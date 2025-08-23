#!/usr/bin/env python3
"""
Tests for the GEMV (General Matrix-Vector Multiplication) module.

This module tests both column-wise and row-wise approaches for matrix-vector
multiplication using lattice quantization with blocking strategies.
"""

import numpy as np
import pytest
from typing import List, Tuple

from src.gemv.column_wise_gemv import ColumnWiseGEMV, column_wise_gemv
from src.gemv.row_wise_gemv import RowWiseGEMV, row_wise_gemv
from src.gemv.lattice_quantized_gemv import LatticeQuantizedGEMV, lattice_quantized_gemv
from src.gemv.padder import BlockingStrategy


class TestPadder:
    """Test the padder functionality."""
    
    def test_padder_initialization(self):
        """Test blocking strategy initialization with different lattice types."""
        lattice_types = ['D4', 'A2', 'E8', 'Z2', 'Z3']
        
        for lattice_type in lattice_types:
            blocking = BlockingStrategy(lattice_type)
            assert blocking.lattice_type == lattice_type
            assert blocking.block_size > 0
            assert blocking.G.shape[0] == blocking.block_size
    
    def test_get_block_indices(self):
        """Test block index generation."""
        blocking = BlockingStrategy('D4')
        vector_length = 100
        
        blocks = blocking.get_block_indices(vector_length)
        
        # Check that blocks cover the entire vector
        assert len(blocks) > 0
        assert blocks[0][0] == 0
        assert blocks[-1][1] == vector_length
        
        # Check that blocks are consecutive
        for i in range(len(blocks) - 1):
            assert blocks[i][1] == blocks[i + 1][0]
    
    def test_get_matrix_blocks_column_wise(self):
        """Test column-wise matrix blocking."""
        blocking = BlockingStrategy('D4')
        matrix = np.random.randn(50, 100)
        
        blocks = blocking.get_matrix_blocks_column_wise(matrix)
        
        # Check that blocks have correct shapes
        for block in blocks:
            assert block.shape[0] == matrix.shape[0]  # Same number of rows
            assert block.shape[1] <= blocking.block_size  # Block size constraint
    
    def test_get_matrix_blocks_row_wise(self):
        """Test row-wise matrix blocking."""
        blocking = BlockingStrategy('D4')
        matrix = np.random.randn(50, 100)
        
        blocks = blocking.get_matrix_blocks_row_wise(matrix)
        
        # Check that blocks have correct shapes
        for block in blocks:
            assert block.shape[0] <= blocking.block_size  # Block size constraint
            assert block.shape[1] == matrix.shape[1]  # Same number of columns
    
    def test_get_vector_blocks(self):
        """Test vector blocking."""
        blocking = BlockingStrategy('D4')
        vector = np.random.randn(100)
        
        blocks = blocking.get_vector_blocks(vector)
        
        # Check that blocks have correct sizes
        for block in blocks:
            assert len(block) <= blocking.block_size
    
    def test_pad_and_unpad_vector(self):
        """Test vector padding and unpadding."""
        blocking = BlockingStrategy('D4')
        original_vector = np.array([1, 2, 3])
        
        # Pad vector
        padded_vector = blocking.pad_vector(original_vector)
        assert len(padded_vector) == blocking.block_size
        assert np.array_equal(padded_vector[:3], original_vector)
        assert np.all(padded_vector[3:] == 0)
        
        # Unpad vector
        unpadded_vector = blocking.unpad_vector(padded_vector, 3)
        assert np.array_equal(unpadded_vector, original_vector)
    
    def test_get_block_indices(self):
        """Test block indices retrieval."""
        blocking = BlockingStrategy('D4')
        vector_length = 100
        
        col_blocks = blocking.get_block_indices(vector_length)
        
        assert len(col_blocks) > 0
        assert col_blocks[0][0] == 0
        assert col_blocks[-1][1] == vector_length
        
        # Check that blocks are consecutive
        for i in range(len(col_blocks) - 1):
            assert col_blocks[i][1] == col_blocks[i + 1][0]


class TestColumnWiseGEMV:
    """Test the column-wise GEMV implementation."""
    
    def test_initialization(self):
        """Test column-wise GEMV initialization."""
        matrix = np.random.randn(10, 8)
        processor = ColumnWiseGEMV(matrix, 'D4', 2)
        
        assert processor.matrix.shape == matrix.shape
        assert processor.lattice_type == 'D4'
        assert processor.M == 2
        assert len(processor.quantizers) > 0
        assert len(processor.encoded_columns) > 0
    
    def test_basic_multiplication(self):
        """Test basic matrix-vector multiplication."""
        matrix = np.random.randn(8, 6)
        vector = np.random.randn(6)
        
        processor = ColumnWiseGEMV(matrix, 'D4', 2)
        result = processor.multiply(vector)
        
        assert result.shape == (8,)
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))
    
    def test_sparsity_support(self):
        """Test sparsity support in column-wise GEMV."""
        matrix = np.random.randn(10, 8)
        vector = np.zeros(8)
        vector[[0, 3, 6]] = [1.0, 2.0, 3.0]
        sparsity_pattern = [0, 3, 6]
        
        processor = ColumnWiseGEMV(matrix, 'D4', 2)
        result = processor.multiply_with_sparsity(vector, sparsity_pattern)
        
        assert result.shape == (10,)
        assert not np.any(np.isnan(result))
    
    def test_compression_ratio(self):
        """Test compression ratio calculation."""
        matrix = np.random.randn(16, 12)
        processor = ColumnWiseGEMV(matrix, 'D4', 2)
        
        ratio = processor.get_compression_ratio()
        assert ratio > 0
        assert not np.isinf(ratio)
    
    def test_memory_usage(self):
        """Test memory usage calculation."""
        matrix = np.random.randn(16, 12)
        processor = ColumnWiseGEMV(matrix, 'D4', 2)
        
        usage = processor.get_memory_usage()
        assert 'encoded_columns_mb' in usage
        assert 'quantizers_mb' in usage
        assert 'total_mb' in usage
        assert all(v >= 0 for v in usage.values())


class TestRowWiseGEMV:
    """Test the row-wise GEMV implementation."""
    
    def test_initialization(self):
        """Test row-wise GEMV initialization."""
        matrix = np.random.randn(10, 8)
        processor = RowWiseGEMV(matrix, 'D4', 2)
        
        assert processor.matrix.shape == matrix.shape
        assert processor.lattice_type == 'D4'
        assert processor.M == 2
        assert len(processor.quantizers) > 0
        assert len(processor.encoded_rows) > 0
    
    def test_basic_multiplication(self):
        """Test basic matrix-vector multiplication."""
        matrix = np.random.randn(8, 6)
        vector = np.random.randn(6)
        
        processor = RowWiseGEMV(matrix, 'D4', 2)
        result = processor.multiply(vector)
        
        assert result.shape == (8,)
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))
    
    def test_sparsity_support(self):
        """Test sparsity support in row-wise GEMV."""
        matrix = np.random.randn(10, 8)
        vector = np.zeros(8)
        vector[[0, 3, 6]] = [1.0, 2.0, 3.0]
        sparsity_pattern = [0, 3, 6]
        
        processor = RowWiseGEMV(matrix, 'D4', 2)
        result = processor.multiply_with_sparsity(vector, sparsity_pattern)
        
        assert result.shape == (10,)
        assert not np.any(np.isnan(result))
    
    def test_lookup_table_support(self):
        """Test lookup table support in row-wise GEMV."""
        matrix = np.random.randn(8, 6)
        vector = np.random.randn(6)
        
        processor = RowWiseGEMV(matrix, 'D4', 2)
        result = processor.multiply_with_lookup(vector)
        
        assert result.shape == (8,)
        assert not np.any(np.isnan(result))
    
    def test_compression_ratio(self):
        """Test compression ratio calculation."""
        matrix = np.random.randn(16, 12)
        processor = RowWiseGEMV(matrix, 'D4', 2)
        
        ratio = processor.get_compression_ratio()
        assert ratio > 0
        assert not np.isinf(ratio)
    
    def test_memory_usage(self):
        """Test memory usage calculation."""
        matrix = np.random.randn(16, 12)
        processor = RowWiseGEMV(matrix, 'D4', 2)
        
        usage = processor.get_memory_usage()
        assert 'encoded_rows_mb' in usage
        assert 'quantizers_mb' in usage
        assert 'total_mb' in usage
        assert all(v >= 0 for v in usage.values())


class TestLatticeQuantizedGEMV:
    """Test the unified GEMV interface."""
    
    def test_initialization(self):
        """Test unified GEMV initialization."""
        matrix = np.random.randn(10, 8)
        
        # Test auto approach
        processor = LatticeQuantizedGEMV(matrix, 'auto', 'D4', 2)
        assert processor.approach in ['column', 'row']
        
        # Test explicit approaches
        col_processor = LatticeQuantizedGEMV(matrix, 'column', 'D4', 2)
        assert col_processor.approach == 'column'
        
        row_processor = LatticeQuantizedGEMV(matrix, 'row', 'D4', 2)
        assert row_processor.approach == 'row'
    
    def test_approach_selection(self):
        """Test automatic approach selection."""
        # Tall matrix should prefer column-wise
        tall_matrix = np.random.randn(20, 8)
        processor = LatticeQuantizedGEMV(tall_matrix, 'auto', 'D4', 2)
        assert processor.approach == 'column'
        
        # Wide matrix should prefer row-wise
        wide_matrix = np.random.randn(8, 20)
        processor = LatticeQuantizedGEMV(wide_matrix, 'auto', 'D4', 2)
        assert processor.approach == 'row'
    
    def test_basic_multiplication(self):
        """Test basic matrix-vector multiplication."""
        matrix = np.random.randn(8, 6)
        vector = np.random.randn(6)
        
        processor = LatticeQuantizedGEMV(matrix, 'auto', 'D4', 2)
        result = processor.multiply(vector)
        
        assert result.shape == (8,)
        assert not np.any(np.isnan(result))
    
    def test_sparsity_support(self):
        """Test sparsity support."""
        matrix = np.random.randn(10, 8)
        vector = np.zeros(8)
        vector[[0, 3, 6]] = [1.0, 2.0, 3.0]
        sparsity_pattern = [0, 3, 6]
        
        processor = LatticeQuantizedGEMV(matrix, 'auto', 'D4', 2)
        result = processor.multiply(vector, sparsity_pattern)
        
        assert result.shape == (10,)
        assert not np.any(np.isnan(result))
    
    def test_approach_comparison(self):
        """Test approach comparison functionality."""
        matrix = np.random.randn(12, 8)
        vector = np.random.randn(8)
        
        processor = LatticeQuantizedGEMV(matrix, 'auto', 'D4', 2)
        comparison = processor.compare_approaches(vector)
        
        assert 'column_wise' in comparison
        assert 'row_wise' in comparison
        assert 'recommended_approach' in comparison
        assert comparison['recommended_approach'] in ['column', 'row']
    
    def test_get_info_methods(self):
        """Test information retrieval methods."""
        matrix = np.random.randn(10, 8)
        processor = LatticeQuantizedGEMV(matrix, 'auto', 'D4', 2)
        
        # Test approach info
        approach_info = processor.get_approach_info()
        assert 'selected_approach' in approach_info
        assert 'matrix_shape' in approach_info
        assert 'aspect_ratio' in approach_info
        
        # Test blocking info
        blocking_info = processor.get_blocking_info()
        assert 'lattice_type' in blocking_info
        assert 'block_size' in blocking_info
        
        # Test compression ratio
        ratio = processor.get_compression_ratio()
        assert ratio > 0
        
        # Test memory usage
        usage = processor.get_memory_usage()
        assert 'total_mb' in usage


class TestFunctionInterfaces:
    """Test the function-level interfaces."""
    
    def test_column_wise_gemv_function(self):
        """Test the column_wise_gemv function."""
        matrix = np.random.randn(8, 6)
        vector = np.random.randn(6)
        
        result = column_wise_gemv(matrix, vector, 'D4', 2)
        assert result.shape == (8,)
        assert not np.any(np.isnan(result))
    
    def test_row_wise_gemv_function(self):
        """Test the row_wise_gemv function."""
        matrix = np.random.randn(8, 6)
        vector = np.random.randn(6)
        
        result = row_wise_gemv(matrix, vector, 'D4', 2)
        assert result.shape == (8,)
        assert not np.any(np.isnan(result))
    
    def test_lattice_quantized_gemv_function(self):
        """Test the lattice_quantized_gemv function."""
        matrix = np.random.randn(8, 6)
        vector = np.random.randn(6)
        
        result = lattice_quantized_gemv(matrix, vector, 'auto', 'D4', 2)
        assert result.shape == (8,)
        assert not np.any(np.isnan(result))


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    def test_invalid_lattice_type(self):
        """Test handling of invalid lattice type."""
        matrix = np.random.randn(8, 6)
        
        with pytest.raises(ValueError):
            ColumnWiseGEMV(matrix, 'INVALID', 2)
        
        with pytest.raises(ValueError):
            RowWiseGEMV(matrix, 'INVALID', 2)
    
    def test_dimension_mismatch(self):
        """Test handling of dimension mismatch."""
        matrix = np.random.randn(8, 6)
        vector = np.random.randn(8)  # Wrong dimension
        
        processor = ColumnWiseGEMV(matrix, 'D4', 2)
        with pytest.raises(ValueError):
            processor.multiply(vector)
        
        processor = RowWiseGEMV(matrix, 'D4', 2)
        with pytest.raises(ValueError):
            processor.multiply(vector)
    
    def test_empty_matrix(self):
        """Test handling of empty matrix."""
        matrix = np.array([]).reshape(0, 0)
        
        with pytest.raises(ValueError):
            ColumnWiseGEMV(matrix, 'D4', 2)
        
        with pytest.raises(ValueError):
            RowWiseGEMV(matrix, 'D4', 2)


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__]) 