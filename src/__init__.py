"""
LatticeQuant: High-Rate Nested-Lattice Quantized Matrix Multiplication

A Python library implementing hierarchical nested lattice quantization for
efficient matrix multiplication with small lookup tables.

Package Structure:
    - src.quantizers: Core quantizer implementations and lattice algorithms
    - src.applications: Matrix multiplication applications and analysis tools
    - src.adaptive: Adaptive matrix-vector multiplication with fixed encoding
    - src.utils: Utility functions and lattice generators

Main Classes:
    - HierarchicalNestedLatticeQuantizer: Multi-level hierarchical quantization
    - NestedLatticeQuantizer: Classic nested lattice quantization
    - FixedMatrixQuantizer: Fixed matrix encoding with adaptive decoding
    - AdaptiveMatVecProcessor: Adaptive matrix-vector processing

Main Functions:
    - closest_point_Dn, closest_point_A2, closest_point_E8: Lattice algorithms
    - get_d4, get_a2, get_e8: Lattice generator matrices
    - precompute_hq_lut: Lookup table generation
    - calculate_weighted_sum: Inner product estimation
    - adaptive_matvec_multiply: Adaptive matrix-vector multiplication

For detailed usage examples, see the README.md file.
For reorganization details, see REORGANIZATION_SUMMARY.md.
"""

# Core quantizer classes
from .quantizers import (
    HierarchicalNestedLatticeQuantizer,
    NestedLatticeQuantizer,
    closest_point_Dn,
    closest_point_A2, 
    closest_point_E8,
    custom_round
)

# Utility functions
from .utils import (
    get_d4,
    get_a2,
    get_e8,
    get_z2,
    get_z3,
    get_d2,
    get_d3,
    precompute_hq_lut,
    calculate_weighted_sum,
    calculate_mse,
    calculate_t_entropy
)

# Analysis modules
from .applications import (
    calculate_inner_product_distortion,
    plot_distortion_rate,
    find_best_beta,
    calculate_mse_and_overload_for_samples,
    plot_distortion_rho,
    generate_rho_correlated_samples,
    calculate_distortion,
    run_comparison_experiment,
    calculate_rate_and_distortion,
    generate_codebook,
    compare_codebooks,
    plot_with_voronoi
)

# Adaptive matrix-vector multiplication
from .adaptive import (
    FixedMatrixQuantizer,
    AdaptiveMatVecProcessor,
    create_adaptive_matvec_processor,
    adaptive_matvec_multiply,
    adaptive_matvec_multiply_sparse,
    demo_basic_functionality,
    demo_sparsity_handling,
    demo_hierarchical_levels,
    demo_rate_distortion_tradeoff,
    demo_lookup_table_efficiency,
    demo_memory_usage,
    plot_results,
    run_demo
)

# Version information
__version__ = "1.0.0"
__author__ = "LatticeQuant Contributors"
__email__ = "contact@latticequant.org"

# Main exports
__all__ = [
    # Core classes
    "HierarchicalNestedLatticeQuantizer",
    "NestedLatticeQuantizer",
    
    # Lattice algorithms
    "closest_point_Dn",
    "closest_point_A2", 
    "closest_point_E8",
    "custom_round",
    
    # Lattice generators
    "get_d4",
    "get_a2",
    "get_e8",
    "get_z2",
    "get_z3",
    "get_d2",
    "get_d3",
    
    # Utility functions
    "precompute_hq_lut",
    "calculate_weighted_sum",
    "calculate_mse",
    "calculate_t_entropy",
    
    # Analysis functions
    "calculate_inner_product_distortion",
    "plot_distortion_rate",
    "find_best_beta",
    "calculate_mse_and_overload_for_samples",
    "plot_distortion_rho",
    "generate_rho_correlated_samples",
    "calculate_distortion",
    "run_comparison_experiment",
    "calculate_rate_and_distortion",
    "generate_codebook",
    "compare_codebooks",
    "plot_with_voronoi",
    
    # Adaptive matrix-vector multiplication
    "FixedMatrixQuantizer",
    "AdaptiveMatVecProcessor",
    "create_adaptive_matvec_processor",
    "adaptive_matvec_multiply",
    "adaptive_matvec_multiply_sparse",
    "demo_basic_functionality",
    "demo_sparsity_handling",
    "demo_hierarchical_levels",
    "demo_rate_distortion_tradeoff",
    "demo_lookup_table_efficiency",
    "demo_memory_usage",
    "plot_results",
    "run_demo"
]
