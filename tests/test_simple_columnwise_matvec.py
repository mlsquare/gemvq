"""
Tests for Simplified Columnwise Matrix-Vector Multiplication

This module tests the simplified columnwise matvec processors to ensure they
work correctly with various input scenarios.
"""

import numpy as np
import pytest

from src.gemv.simple_columnwise_matvec import (
    create_simple_processor,
    SimpleColumnwiseMatVecProcessor
)


class TestSimpleColumnwiseMatVec:
    """Test class for simplified columnwise matvec."""
    
    def setup_method(self):
        """Setup test data."""
        # Create test matrix and vector
        self.matrix = np.random.randn(16, 12)
        self.x = np.random.randn(12)
        
        # Create sparse vector
        self.sparse_x = self.x.copy()
        self.sparse_x[::2] = 0.0  # Make every other element zero
    
    def test_processor_creation(self):
        """Test that processors can be created successfully."""
        strategies = ["standard_dot", "adaptive_depth", "lookup_table"]
        
        for strategy in strategies:
            processor = create_simple_processor(self.matrix, strategy=strategy)
            assert processor is not None
            assert hasattr(processor, 'compute_matvec')
            assert hasattr(processor, 'get_compression_stats')
            assert processor.strategy == strategy
    
    def test_basic_matvec_computation(self):
        """Test basic matrix-vector multiplication."""
        strategies = ["standard_dot", "adaptive_depth", "lookup_table"]
        
        # Compute reference result
        reference_result = self.matrix @ self.x
        
        for strategy in strategies:
            processor = create_simple_processor(self.matrix, strategy=strategy)
            result = processor.compute_matvec(self.x)
            
            # Check shape
            assert result.shape == reference_result.shape
            
            # Check accuracy (should be close to reference)
            error = np.linalg.norm(result - reference_result) / np.linalg.norm(reference_result)
            assert error < 1.0, f"Large error in {strategy}: {error}"
    
    def test_sparsity_handling(self):
        """Test handling of sparse vectors."""
        processor = create_simple_processor(self.matrix, strategy="adaptive_depth")
        
        # Test with sparse vector
        result_sparse = processor.compute_matvec(self.sparse_x)
        result_dense = processor.compute_matvec(self.x)
        
        # Results should be different for sparse vs dense
        assert not np.allclose(result_sparse, result_dense)
        
        # But sparse result should be correct
        reference_sparse = self.matrix @ self.sparse_x
        error = np.linalg.norm(result_sparse - reference_sparse) / np.linalg.norm(reference_sparse)
        assert error < 1.0
    
    def test_matrix_info(self):
        """Test matrix information retrieval."""
        processor = create_simple_processor(self.matrix, strategy="standard_dot")
        
        info = processor.get_matrix_info()
        
        # Check that info contains expected keys
        expected_keys = ['original_shape', 'lattice_type', 'strategy', 'M', 'q']
        for key in expected_keys:
            assert key in info
        
        # Check that shapes are correct
        assert info['original_shape'] == self.matrix.shape
        assert info['lattice_type'] == 'D4'
        assert info['strategy'] == 'standard_dot'
    
    def test_compression_stats(self):
        """Test compression statistics."""
        processor = create_simple_processor(self.matrix, strategy="standard_dot")
        
        stats = processor.get_compression_stats()
        
        # Check that stats contain expected keys
        expected_keys = ['compression_ratio', 'memory_usage_mb', 'computation_time']
        for key in expected_keys:
            assert key in stats
        
        # Check that values are reasonable
        assert stats['compression_ratio'] > 0
        assert stats['memory_usage_mb'] >= 0
    
    def test_error_handling(self):
        """Test error handling."""
        processor = create_simple_processor(self.matrix, strategy="standard_dot")
        
        # Test with wrong vector size
        wrong_x = np.random.randn(10)  # Wrong size
        
        try:
            processor.compute_matvec(wrong_x)
            assert False, "Should have raised ValueError for wrong vector size"
        except ValueError:
            pass  # Expected
        
        # Test with invalid strategy
        try:
            create_simple_processor(self.matrix, strategy="invalid_strategy")
            assert False, "Should have raised ValueError for invalid strategy"
        except ValueError:
            pass  # Expected
    
    def test_performance_tracking(self):
        """Test performance tracking."""
        processor = create_simple_processor(self.matrix, strategy="adaptive_depth")
        
        # Run computation
        result = processor.compute_matvec(self.x)
        
        # Get performance stats
        stats = processor.get_compression_stats()
        
        # Check that performance data is recorded
        assert 'computation_time' in stats
        assert stats['computation_time'] > 0
    
    def test_different_strategies(self):
        """Test that different strategies produce valid results."""
        strategies = ["standard_dot", "adaptive_depth", "lookup_table"]
        
        results = {}
        for strategy in strategies:
            processor = create_simple_processor(self.matrix, strategy=strategy)
            results[strategy] = processor.compute_matvec(self.x)
        
        # All results should have the same shape
        shapes = [result.shape for result in results.values()]
        assert len(set(shapes)) == 1
        
        # All results should be finite
        for strategy, result in results.items():
            assert np.all(np.isfinite(result)), f"Non-finite values in {strategy}"


def test_edge_cases():
    """Test edge cases."""
    # Test with very small matrix
    small_matrix = np.random.randn(4, 4)
    small_x = np.random.randn(4)
    
    processor = create_simple_processor(small_matrix, strategy="standard_dot")
    result = processor.compute_matvec(small_x)
    
    assert result.shape == (4,)
    
    # Test with very large matrix (small test)
    large_matrix = np.random.randn(32, 32)
    large_x = np.random.randn(32)
    
    processor = create_simple_processor(large_matrix, strategy="lookup_table")
    result = processor.compute_matvec(large_x)
    
    assert result.shape == (32,)
    
    # Test with all-zero matrix
    zero_matrix = np.zeros((8, 8))
    x = np.random.randn(8)
    
    processor = create_simple_processor(zero_matrix, strategy="adaptive_depth")
    result = processor.compute_matvec(x)
    
    assert np.allclose(result, 0)


if __name__ == "__main__":
    # Run tests
    test_class = TestSimpleColumnwiseMatVec()
    
    print("Running simplified columnwise matvec tests...")
    
    # Setup test data
    test_class.setup_method()
    
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
