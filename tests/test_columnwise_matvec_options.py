"""
Tests for Columnwise Matrix-Vector Multiplication Options

This module tests the different columnwise matvec processors to ensure they
work correctly with various input scenarios.
"""

import numpy as np
import pytest

from src.gemv.columnwise.columnwise_matvec_factory import (
    create_processor,
    create_standard_dot_processor,
    create_lookup_table_processor,
    create_adaptive_processor,
    get_available_processors,
    get_processor_info
)


class TestColumnwiseMatVecOptions:
    """Test class for columnwise matvec options."""
    
    def __init__(self):
        """Initialize test data."""
        # Create test matrix and vector
        self.matrix = np.random.randn(16, 12)
        self.x = np.random.randn(12)
        
        # Create sparse vector
        self.sparse_x = self.x.copy()
        self.sparse_x[::2] = 0.0  # Make every other element zero
    
    def setup_method(self):
        """Setup test data."""
        # Create test matrix and vector
        self.matrix = np.random.randn(16, 12)
        self.x = np.random.randn(12)
        
        # Create sparse vector
        self.sparse_x = self.x.copy()
        self.sparse_x[::2] = 0.0  # Make every other element zero
    
    def test_processor_creation(self):
        """Test that all processors can be created successfully."""
        processors = [
            create_standard_dot_processor(self.matrix),
            create_lookup_table_processor(self.matrix),
            create_adaptive_processor(self.matrix)
        ]
        
        for processor in processors:
            assert processor is not None
            assert hasattr(processor, 'compute_matvec')
            assert hasattr(processor, 'get_compression_stats')
    
    def test_basic_matvec_computation(self):
        """Test basic matrix-vector multiplication."""
        processors = {
            'standard_dot': create_processor(self.matrix, 'standard_dot'),
            'lookup_table': create_processor(self.matrix, 'lookup_table'),
            'adaptive': create_processor(self.matrix, 'adaptive')
        }
        
        # Compute reference result
        reference_result = self.matrix @ self.x
        
        for name, processor in processors.items():
            result = processor.compute_matvec(self.x)
            
            # Check shape
            assert result.shape == reference_result.shape
            
            # Check accuracy (should be close to reference)
            error = np.linalg.norm(result - reference_result) / np.linalg.norm(reference_result)
            assert error < 1e-10, f"Large error in {name}: {error}"
    
    def test_sparsity_handling(self):
        """Test handling of sparse vectors."""
        processor = create_adaptive_processor(self.matrix)
        
        # Test with sparse vector
        result_sparse = processor.compute_matvec(self.sparse_x)
        result_dense = processor.compute_matvec(self.x)
        
        # Results should be different for sparse vs dense
        assert not np.allclose(result_sparse, result_dense)
        
        # But sparse result should be correct
        reference_sparse = self.matrix @ self.sparse_x
        error = np.linalg.norm(result_sparse - reference_sparse) / np.linalg.norm(reference_sparse)
        assert error < 1e-10
    
    def test_padding_handling(self):
        """Test handling of non-multiple dimensions."""
        # Create matrix with dimensions not multiples of 4
        matrix_odd = np.random.randn(15, 13)
        x_odd = np.random.randn(13)
        
        processor = create_standard_dot_processor(matrix_odd)
        
        # Should handle padding automatically
        result = processor.compute_matvec(x_odd)
        reference = matrix_odd @ x_odd
        
        assert result.shape == reference.shape
        error = np.linalg.norm(result - reference) / np.linalg.norm(reference)
        assert error < 1e-10
    
    def test_decoding_depths(self):
        """Test different decoding depths."""
        processor = create_standard_dot_processor(self.matrix, fixed_depth=False)
        
        # Test with different decoding depths
        depths = [0, 1, 2]
        
        for depth in depths:
            decoding_depths = [depth] * (self.matrix.shape[1] // 4)  # Assuming D4 lattice
            result = processor.compute_matvec(self.x, decoding_depths=decoding_depths)
            
            # Should produce valid result
            assert result.shape == (self.matrix.shape[0],)
            assert not np.any(np.isnan(result))
            assert not np.any(np.isinf(result))
    
    def test_adaptive_strategy_selection(self):
        """Test adaptive strategy selection."""
        processor = create_adaptive_processor(self.matrix)
        
        # Test with different input characteristics
        test_vectors = [
            self.x,  # Dense
            self.sparse_x,  # Sparse
            np.zeros_like(self.x),  # All zeros
            np.ones_like(self.x),  # All ones
        ]
        
        for x_test in test_vectors:
            result = processor.compute_matvec(x_test)
            
            # Should produce valid result
            assert result.shape == (self.matrix.shape[0],)
            assert not np.any(np.isnan(result))
            assert not np.any(np.isinf(result))
    
    def test_compression_stats(self):
        """Test compression statistics."""
        processor = create_standard_dot_processor(self.matrix)
        
        stats = processor.get_compression_stats()
        
        # Check that stats contain expected keys
        expected_keys = ['compression_ratio', 'memory_usage_mb', 'computation_time']
        for key in expected_keys:
            assert key in stats
        
        # Check that values are reasonable
        assert stats['compression_ratio'] > 0
        assert stats['memory_usage_mb'] >= 0
    
    def test_matrix_info(self):
        """Test matrix information retrieval."""
        processor = create_standard_dot_processor(self.matrix)
        
        info = processor.get_matrix_info()
        
        # Check that info contains expected keys
        expected_keys = ['original_shape', 'padded_shape', 'lattice_dimension', 'num_blocks']
        for key in expected_keys:
            assert key in info
        
        # Check that shapes are correct
        assert info['original_shape'] == self.matrix.shape
        assert info['lattice_dimension'] == 4  # D4 lattice
    
    def test_factory_functions(self):
        """Test factory functions."""
        # Test get_available_processors
        available = get_available_processors()
        assert 'standard_dot' in available
        assert 'lookup_table' in available
        assert 'adaptive' in available
        
        # Test get_processor_info
        for processor_type in available:
            info = get_processor_info(processor_type)
            assert 'name' in info
            assert 'description' in info
    
    def test_error_handling(self):
        """Test error handling."""
        # Test with wrong vector size
        processor = create_standard_dot_processor(self.matrix)
        wrong_x = np.random.randn(10)  # Wrong size
        
        with pytest.raises(ValueError):
            processor.compute_matvec(wrong_x)
        
        # Test with invalid processor type
        with pytest.raises(ValueError):
            create_processor(self.matrix, 'invalid_type')
    
    def test_performance_tracking(self):
        """Test performance tracking."""
        processor = create_adaptive_processor(self.matrix)
        
        # Run computation
        result = processor.compute_matvec(self.x)
        
        # Get performance stats
        stats = processor.get_performance_stats()
        
        # Check that performance data is recorded
        assert 'computation_time' in stats
        assert stats['computation_time'] > 0
        assert 'selected_strategy' in stats
    
    def test_lookup_table_strategies(self):
        """Test different lookup table strategies."""
        strategies = ['layer_wise_histogram', 'inner_product', 'hybrid']
        
        for strategy in strategies:
            processor = create_lookup_table_processor(
                self.matrix, 
                table_strategy=strategy
            )
            
            result = processor.compute_matvec(self.x)
            reference = self.matrix @ self.x
            
            # Check accuracy
            error = np.linalg.norm(result - reference) / np.linalg.norm(reference)
            assert error < 1e-10, f"Large error in {strategy}: {error}"
    
    def test_adaptive_benchmarking(self):
        """Test adaptive processor benchmarking."""
        processor = create_adaptive_processor(self.matrix)
        
        # Run benchmark
        benchmark_results = processor.benchmark_strategies(self.x, num_runs=2)
        
        # Check that benchmark results are valid
        assert len(benchmark_results) > 0
        
        for strategy_name, stats in benchmark_results.items():
            assert 'mean_time' in stats
            assert 'std_time' in stats
            assert stats['mean_time'] > 0


def test_edge_cases():
    """Test edge cases."""
    # Test with very small matrix
    small_matrix = np.random.randn(4, 4)
    small_x = np.random.randn(4)
    
    processor = create_standard_dot_processor(small_matrix)
    result = processor.compute_matvec(small_x)
    
    assert result.shape == (4,)
    
    # Test with very large matrix (small test)
    large_matrix = np.random.randn(32, 32)
    large_x = np.random.randn(32)
    
    processor = create_lookup_table_processor(large_matrix)
    result = processor.compute_matvec(large_x)
    
    assert result.shape == (32,)
    
    # Test with all-zero matrix
    zero_matrix = np.zeros((8, 8))
    x = np.random.randn(8)
    
    processor = create_adaptive_processor(zero_matrix)
    result = processor.compute_matvec(x)
    
    assert np.allclose(result, 0)


if __name__ == "__main__":
    # Run tests
    test_class = TestColumnwiseMatVecOptions()
    
    print("Running columnwise matvec options tests...")
    
    # Run all test methods
    test_methods = [method for method in dir(test_class) if method.startswith('test_')]
    
    for method_name in test_methods:
        print(f"Testing {method_name}...")
        method = getattr(test_class, method_name)
        method()
        print(f"✓ {method_name} passed")
    
    # Run edge case tests
    print("Testing edge cases...")
    test_edge_cases()
    print("✓ Edge cases passed")
    
    print("All tests passed successfully!")
