"""
Adaptive Matrix-Vector Multiplication Module

This module implements adaptive matrix-vector multiplication using hierarchical
nested lattice quantizers. The approach encodes matrix W once with maximum bit rate,
then adaptively decodes columns based on bit budget for each input vector x,
exploiting the hierarchical levels M for variable precision decoding.

Core Classes:
    - FixedMatrixQuantizer: Fixed matrix encoding with adaptive decoding
    - AdaptiveMatVecProcessor: Adaptive matrix-vector processing

Core Functions:
    - adaptive_matvec_multiply: Main adaptive multiplication function
    - adaptive_matvec_multiply_sparse: Sparse vector multiplication
    - create_adaptive_matvec_processor: Processor factory function

Demo and Analysis:
    - demo_basic_functionality: Basic functionality demonstration
    - demo_sparsity_handling: Sparsity handling demonstration
    - demo_hierarchical_levels: Hierarchical level analysis
    - demo_rate_distortion_tradeoff: Rate-distortion analysis
    - demo_lookup_table_efficiency: Lookup table efficiency
    - demo_memory_usage: Memory usage analysis

This module implements the new approach where W is encoded once and columns
are decoded adaptively based on bit budget requirements.
"""

from .adaptive_matvec import (
    adaptive_matvec_multiply,
    adaptive_matvec_multiply_sparse,
    create_adaptive_matvec_processor,
    FixedMatrixQuantizer,
    AdaptiveMatVecProcessor
)

from .demo_adaptive_matvec import (
    demo_basic_functionality,
    demo_sparsity_handling,
    demo_hierarchical_levels,
    demo_rate_distortion_tradeoff,
    demo_lookup_table_efficiency,
    demo_memory_usage,
    plot_results,
    main as run_demo
)

__all__ = [
    # Core adaptive matvec functions
    'adaptive_matvec_multiply',
    'adaptive_matvec_multiply_sparse',
    'create_adaptive_matvec_processor',
    'FixedMatrixQuantizer',
    'AdaptiveMatVecProcessor',
    
    # Demo and analysis functions
    'demo_basic_functionality',
    'demo_sparsity_handling',
    'demo_hierarchical_levels',
    'demo_rate_distortion_tradeoff',
    'demo_lookup_table_efficiency',
    'demo_memory_usage',
    'plot_results',
    'run_demo'
] 