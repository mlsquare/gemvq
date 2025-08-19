"""
Adaptive Matrix-Vector Multiplication Module

This module contains specialized implementations for adaptive matrix-vector
multiplication using hierarchical nested lattice quantizers with column-wise
encoding. It provides efficient sparse matrix-vector multiplication with
adaptive bit rates for each column.

Core Classes:
    - AdaptiveColumnQuantizer: Column-wise adaptive quantization
    - AdaptiveLookupTable: Efficient lookup table management
    - SparseMatVecProcessor: Sparse matrix-vector processing

Core Functions:
    - adaptive_matvec_multiply: Main adaptive multiplication function
    - create_adaptive_matvec_processor: Processor factory function

Demo and Analysis:
    - run_comprehensive_demo: Complete demonstration suite
    - demo_basic_usage: Basic usage examples
    - demo_lattice_comparison: Lattice type comparison
    - demo_hierarchical_levels: Hierarchical level analysis
    - demo_sparsity_impact: Sparsity impact analysis
    - demo_rate_allocation: Rate allocation strategies

This module implements the adaptive approach described in the paper, providing
efficient matrix-vector multiplication for sparse vectors with adaptive
quantization rates per column.
"""

from .adaptive_matvec import (
    adaptive_matvec_multiply,
    create_adaptive_matvec_processor,
    AdaptiveColumnQuantizer,
    AdaptiveLookupTable,
    SparseMatVecProcessor
)

from .demo_adaptive_matvec import (
    run_comprehensive_demo,
    demo_basic_usage,
    demo_lattice_comparison,
    demo_hierarchical_levels,
    demo_sparsity_impact,
    demo_rate_allocation
)

__all__ = [
    # Core adaptive matvec functions
    'adaptive_matvec_multiply',
    'create_adaptive_matvec_processor',
    'AdaptiveColumnQuantizer',
    'AdaptiveLookupTable',
    'SparseMatVecProcessor',
    
    # Demo and analysis functions
    'run_comprehensive_demo',
    'demo_basic_usage',
    'demo_lattice_comparison',
    'demo_hierarchical_levels',
    'demo_sparsity_impact',
    'demo_rate_allocation'
] 