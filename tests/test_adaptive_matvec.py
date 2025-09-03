"""
Tests for adaptive matrix-vector multiplication module.

This module contains comprehensive tests for the adaptive matrix-vector
multiplication functionality using hierarchical nested quantizers.
"""

import numpy as np
import pytest
from src.adaptive.adaptive_matvec import (
    AdaptiveColumnQuantizer,
    AdaptiveLookupTable,
    SparseMatVecProcessor,
    adaptive_matvec_multiply,
    create_adaptive_matvec_processor,
)


class TestAdaptiveColumnQuantizer:
    """Test cases for AdaptiveColumnQuantizer class."""

    def test_initialization(self):
        """Test quantizer initialization with different parameters."""
        target_rates = [2.0, 4.0, 6.0]
        quantizer = AdaptiveColumnQuantizer(target_rates, "D4", M=2)

        assert len(quantizer.quantizers) == 3
        assert quantizer.lattice_type == "D4"
        assert quantizer.M == 2
        assert quantizer.dimension == 4  # D4 lattice dimension

    def test_lattice_setup(self):
        """Test lattice setup for different lattice types."""
        target_rates = [3.0]

        # Test D4 lattice
        quantizer_d4 = AdaptiveColumnQuantizer(target_rates, "D4")
        assert quantizer_d4.dimension == 4

        # Test A2 lattice
        quantizer_a2 = AdaptiveColumnQuantizer(target_rates, "A2")
        assert quantizer_a2.dimension == 2

        # Test E8 lattice
        quantizer_e8 = AdaptiveColumnQuantizer(target_rates, "E8")
        assert quantizer_e8.dimension == 8

        # Test invalid lattice type
        with pytest.raises(ValueError):
            AdaptiveColumnQuantizer(target_rates, "INVALID")

    def test_rate_to_parameters(self):
        """Test conversion from bit rate to quantization parameters."""
        target_rates = [2.0, 4.0, 6.0]
        quantizer = AdaptiveColumnQuantizer(target_rates, "D4", M=2)

        for rate in target_rates:
            q, beta = quantizer._rate_to_parameters(rate)
            assert q >= 2  # Minimum quantization parameter
            assert beta > 0  # Positive scaling
            assert beta <= 1.0  # Reasonable scaling range

    def test_encode_decode_column(self):
        """Test encoding and decoding of a single column."""
        target_rates = [4.0]
        quantizer = AdaptiveColumnQuantizer(target_rates, "D4", M=2)

        # Create test column
        column = np.random.randn(4)

        # Encode column
        encoding, scaling = quantizer.encode_column(column, 0)

        # Check encoding structure
        assert isinstance(encoding, tuple)
        assert len(encoding) == 2  # M=2 levels
        assert scaling >= 0

        # Decode column
        decoded = quantizer.decode_column(encoding, 0, scaling)

        # Check decoded structure
        assert decoded.shape == column.shape
        assert decoded.dtype == np.float64

    def test_invalid_column_index(self):
        """Test error handling for invalid column indices."""
        target_rates = [3.0]
        quantizer = AdaptiveColumnQuantizer(target_rates, "D4")

        column = np.random.randn(4)

        # Test encoding with invalid index
        with pytest.raises(ValueError):
            quantizer.encode_column(column, 1)  # Index 1 not initialized

        # Test decoding with invalid index
        encoding = (np.array([0, 0, 0, 0]), np.array([0, 0, 0, 0]))
        with pytest.raises(ValueError):
            quantizer.decode_column(encoding, 1, 0)


class TestAdaptiveLookupTable:
    """Test cases for AdaptiveLookupTable class."""

    def test_initialization(self):
        """Test lookup table initialization."""
        lookup_manager = AdaptiveLookupTable("D4", max_rate=6.0, M=2)

        assert lookup_manager.lattice_type == "D4"
        assert lookup_manager.max_rate == 6.0
        assert lookup_manager.M == 2
        assert len(lookup_manager.lookup_tables) > 0

    def test_table_creation(self):
        """Test lookup table creation for different rates."""
        lookup_manager = AdaptiveLookupTable("D4", max_rate=4.0, M=2)

        # Test table creation for different rates
        for rate in [2.0, 3.0, 4.0]:
            table = lookup_manager.get_table(rate)
            assert isinstance(table, dict)
            assert len(table) > 0

    def test_rate_to_parameters(self):
        """Test parameter conversion in lookup table."""
        lookup_manager = AdaptiveLookupTable("D4", max_rate=6.0, M=2)

        for rate in [2.0, 4.0, 6.0]:
            q, beta = lookup_manager._rate_to_parameters(rate)
            assert q >= 2
            assert beta > 0


class TestSparseMatVecProcessor:
    """Test cases for SparseMatVecProcessor class."""

    def test_initialization(self):
        """Test processor initialization."""
        matrix = np.random.randn(10, 5)
        target_rates = [2.0, 3.0, 4.0, 3.5, 2.5]
        sparsity_pattern = [0, 2, 4]

        processor = SparseMatVecProcessor(
            matrix, target_rates, sparsity_pattern, "D4", 2
        )

        assert processor.m == 10
        assert processor.n == 5
        assert processor.sparsity_pattern == [0, 2, 4]
        assert len(processor.encoded_columns) == 5

    def test_matrix_encoding(self):
        """Test matrix column encoding."""
        matrix = np.random.randn(8, 3)
        target_rates = [3.0, 4.0, 3.5]

        processor = SparseMatVecProcessor(matrix, target_rates, "D4", 2)

        # Check that all columns are encoded
        assert len(processor.encoded_columns) == 3
        assert len(processor.overload_scalings) == 3

        # Check encoding structure
        for i in range(3):
            encoding = processor.encoded_columns[i]
            assert isinstance(encoding, tuple)
            assert len(encoding) == 2  # M=2 levels

    def test_matvec_computation(self):
        """Test matrix-vector multiplication."""
        matrix = np.random.randn(10, 5)
        target_rates = [2.0, 3.0, 4.0, 3.5, 2.5]
        sparsity_pattern = [0, 2, 4]

        processor = SparseMatVecProcessor(
            matrix, target_rates, sparsity_pattern, "D4", 2
        )

        # Create sparse vector
        sparse_vector = np.zeros(5)
        sparse_vector[sparsity_pattern] = [1.0, 2.0, 3.0]

        # Compute matrix-vector product
        result = processor.compute_matvec(sparse_vector)

        assert result.shape == (10,)
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))

    def test_matvec_with_lookup(self):
        """Test matrix-vector multiplication using lookup tables."""
        matrix = np.random.randn(8, 4)
        target_rates = [2.5, 3.0, 3.5, 4.0]
        sparsity_pattern = [1, 3]

        processor = SparseMatVecProcessor(
            matrix, target_rates, sparsity_pattern, "D4", 2
        )

        # Create sparse vector
        sparse_vector = np.zeros(4)
        sparse_vector[sparsity_pattern] = [1.5, 2.5]

        # Compute using lookup tables
        result = processor.compute_matvec_with_lookup(sparse_vector)

        assert result.shape == (8,)
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))

    def test_dimension_mismatch(self):
        """Test error handling for dimension mismatches."""
        matrix = np.random.randn(10, 5)
        target_rates = [2.0, 3.0, 4.0, 3.5, 2.5]

        processor = SparseMatVecProcessor(matrix, target_rates, "D4", 2)

        # Test with wrong vector dimension
        wrong_vector = np.random.randn(3)  # Wrong dimension

        with pytest.raises(ValueError):
            processor.compute_matvec(wrong_vector)

    def test_compression_ratio(self):
        """Test compression ratio calculation."""
        matrix = np.random.randn(20, 10)
        target_rates = [2.0] * 10

        processor = SparseMatVecProcessor(matrix, target_rates, "D4", 2)

        ratio = processor.get_compression_ratio()
        assert ratio > 0
        assert not np.isinf(ratio)

    def test_memory_usage(self):
        """Test memory usage calculation."""
        matrix = np.random.randn(15, 8)
        target_rates = [3.0] * 8

        processor = SparseMatVecProcessor(matrix, target_rates, "D4", 2)

        memory_info = processor.get_memory_usage()

        assert "encoded_columns_mb" in memory_info
        assert "lookup_tables_mb" in memory_info
        assert "total_mb" in memory_info

        for key, value in memory_info.items():
            assert value >= 0


class TestFactoryFunctions:
    """Test cases for factory functions."""

    def test_create_adaptive_matvec_processor(self):
        """Test processor creation factory function."""
        matrix = np.random.randn(12, 6)
        target_rates = [2.5, 3.0, 3.5, 2.0, 4.0, 3.2]
        sparsity_pattern = [0, 3, 5]

        processor = create_adaptive_matvec_processor(
            matrix, target_rates, sparsity_pattern, "D4", 2
        )

        assert isinstance(processor, SparseMatVecProcessor)
        assert processor.m == 12
        assert processor.n == 6
        assert processor.sparsity_pattern == [0, 3, 5]

    def test_adaptive_matvec_multiply(self):
        """Test adaptive matrix-vector multiplication function."""
        matrix = np.random.randn(10, 5)
        vector = np.random.randn(5)
        target_rates = [2.0, 3.0, 4.0, 3.5, 2.5]
        sparsity_pattern = [0, 2, 4]

        # Make vector sparse
        sparse_vector = np.zeros(5)
        sparse_vector[sparsity_pattern] = vector[sparsity_pattern]

        # Test without lookup tables
        result1 = adaptive_matvec_multiply(
            matrix, sparse_vector, target_rates, sparsity_pattern, "D4", 2, False
        )

        # Test with lookup tables
        result2 = adaptive_matvec_multiply(
            matrix, sparse_vector, target_rates, sparsity_pattern, "D4", 2, True
        )

        assert result1.shape == (10,)
        assert result2.shape == (10,)
        assert not np.any(np.isnan(result1))
        assert not np.any(np.isnan(result2))


class TestIntegration:
    """Integration tests for the complete workflow."""

    def test_complete_workflow(self):
        """Test complete adaptive matrix-vector multiplication workflow."""
        # Setup test data
        m, n = 20, 10
        matrix = np.random.randn(m, n)
        target_rates = np.random.uniform(2.0, 5.0, n)
        sparsity_pattern = [0, 3, 7, 9]

        # Create sparse vector
        sparse_vector = np.zeros(n)
        sparse_vector[sparsity_pattern] = [1.0, 2.0, 3.0, 4.0]

        # Perform adaptive multiplication
        result = adaptive_matvec_multiply(
            matrix, sparse_vector, target_rates, sparsity_pattern, "D4", 2
        )

        # Compare with exact computation
        exact_result = matrix @ sparse_vector
        error = np.linalg.norm(result - exact_result) / np.linalg.norm(exact_result)

        # Check reasonable error bounds
        assert error < 0.1  # 10% relative error threshold
        assert result.shape == exact_result.shape

    def test_different_lattice_types(self):
        """Test with different lattice types."""
        matrix = np.random.randn(8, 4)
        target_rates = [2.5, 3.0, 3.5, 4.0]
        sparsity_pattern = [0, 2]
        sparse_vector = np.zeros(4)
        sparse_vector[sparsity_pattern] = [1.0, 2.0]

        lattice_types = ["D4", "A2", "E8"]

        for lattice_type in lattice_types:
            result = adaptive_matvec_multiply(
                matrix, sparse_vector, target_rates, sparsity_pattern, lattice_type, 2
            )

            assert result.shape == (8,)
            assert not np.any(np.isnan(result))

    def test_varying_hierarchical_levels(self):
        """Test with different numbers of hierarchical levels."""
        matrix = np.random.randn(10, 5)
        target_rates = [3.0] * 5
        sparsity_pattern = [0, 2, 4]
        sparse_vector = np.zeros(5)
        sparse_vector[sparsity_pattern] = [1.0, 2.0, 3.0]

        for M in [1, 2, 3]:
            result = adaptive_matvec_multiply(
                matrix, sparse_vector, target_rates, sparsity_pattern, "D4", M
            )

            assert result.shape == (10,)
            assert not np.any(np.isnan(result))


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__])
