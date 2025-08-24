"""
Demo script for the new GEMV structure.

This script demonstrates the usage of the reorganized GEMV module with
columnwise, rowwise, and SVD processors.
"""

import numpy as np
import time
from typing import Dict, Any

from .base import create_gemv_processor, GEMVFactory
from .columnwise import ColumnwiseGEMVProcessor
from .rowwise import RowwiseGEMVProcessor
from .svd import SVDGEMVProcessor


def create_test_matrix(m: int = 100, n: int = 50) -> np.ndarray:
    """Create a test matrix with controlled properties."""
    np.random.seed(42)
    matrix = np.random.randn(m, n)
    # Make it somewhat structured for better SVD performance
    matrix = matrix / np.linalg.norm(matrix, 'fro')
    return matrix


def create_test_vector(n: int = 50) -> np.ndarray:
    """Create a test vector."""
    np.random.seed(123)
    return np.random.randn(n)


def benchmark_processor(processor: Any, matrix: np.ndarray, vector: np.ndarray, 
                       num_runs: int = 5) -> Dict[str, Any]:
    """Benchmark a GEMV processor."""
    # Warm up
    _ = processor(matrix, vector)
    
    # Benchmark
    times = []
    for _ in range(num_runs):
        start_time = time.time()
        result = processor(matrix, vector)
        end_time = time.time()
        times.append(end_time - start_time)
    
    # Compute reference result
    reference = matrix @ vector
    
    # Compute error
    error = np.linalg.norm(result - reference) / np.linalg.norm(reference)
    
    return {
        'mean_time': np.mean(times),
        'std_time': np.std(times),
        'error': error,
        'result': result,
        'info': processor.get_processor_info()
    }


def demo_columnwise_processor():
    """Demo the columnwise processor."""
    print("=" * 60)
    print("COLUMNWISE GEMV PROCESSOR DEMO")
    print("=" * 60)
    
    # Create test data
    matrix = create_test_matrix(100, 50)
    vector = create_test_vector(50)
    
    # Create processor using factory
    processor = create_gemv_processor(
        'columnwise',
        lattice_type='D4',
        M=2,
        q=4,
        beta=0.2,
        alpha=1/3,
        decoding='full'
    )
    
    # Benchmark
    results = benchmark_processor(processor, matrix, vector)
    
    print(f"Mean computation time: {results['mean_time']:.6f} ± {results['std_time']:.6f} seconds")
    print(f"Relative error: {results['error']:.2e}")
    print(f"Processor info: {results['info']}")
    print()


def demo_rowwise_processor():
    """Demo the rowwise processor."""
    print("=" * 60)
    print("ROWWISE GEMV PROCESSOR DEMO")
    print("=" * 60)
    
    # Create test data
    matrix = create_test_matrix(100, 50)
    vector = create_test_vector(50)
    
    # Create processor using factory
    processor = create_gemv_processor(
        'rowwise',
        lattice_type='D4',
        M=2,
        q=4,
        beta=0.2,
        alpha=1/3,
        decoding='full'
    )
    
    # Benchmark
    results = benchmark_processor(processor, matrix, vector)
    
    print(f"Mean computation time: {results['mean_time']:.6f} ± {results['std_time']:.6f} seconds")
    print(f"Relative error: {results['error']:.2e}")
    print(f"Processor info: {results['info']}")
    print()


def demo_svd_processor():
    """Demo the SVD processor."""
    print("=" * 60)
    print("SVD GEMV PROCESSOR DEMO")
    print("=" * 60)
    
    # Create test data
    matrix = create_test_matrix(100, 50)
    vector = create_test_vector(50)
    
    # Create processor using factory
    processor = create_gemv_processor(
        'svd',
        lattice_type='D4',
        M=2,
        q=4,
        beta=0.2,
        alpha=1/3,
        decoding='full',
        svd_rank=20,  # Truncated SVD
        quantize_svd=True
    )
    
    # Benchmark
    results = benchmark_processor(processor, matrix, vector)
    
    print(f"Mean computation time: {results['mean_time']:.6f} ± {results['std_time']:.6f} seconds")
    print(f"Relative error: {results['error']:.2e}")
    print(f"Processor info: {results['info']}")
    print()


def demo_factory_capabilities():
    """Demo the factory capabilities."""
    print("=" * 60)
    print("GEMV FACTORY DEMO")
    print("=" * 60)
    
    # Show available processors
    available = GEMVFactory.get_available_processors()
    print(f"Available processor types: {available}")
    
    # Show processor info
    for processor_type in available:
        info = GEMVFactory.get_processor_info(processor_type)
        print(f"\n{processor_type.upper()} processor:")
        print(f"  Class: {info['class'].__name__}")
        print(f"  Module: {info['module']}")
        print(f"  Docstring: {info['docstring'][:100]}...")
    
    print()


def main():
    """Run all demos."""
    print("GEMV NEW STRUCTURE DEMO")
    print("=" * 80)
    
    # Demo factory capabilities
    demo_factory_capabilities()
    
    # Demo each processor type
    demo_columnwise_processor()
    demo_rowwise_processor()
    demo_svd_processor()
    
    print("=" * 80)
    print("DEMO COMPLETED")


if __name__ == "__main__":
    main()
