"""
LatticeQuant: High-Rate Nested-Lattice Quantized Matrix Multiplication

A Python library implementing hierarchical nested lattice quantization for
efficient matrix multiplication with small lookup tables.

Main Classes:
    - HierarchicalNestedLatticeQuantizer: Multi-level hierarchical quantization
    - NestedLatticeQuantizer: Classic nested lattice quantization

Main Functions:
    - closest_point_Dn, closest_point_A2, closest_point_E8: Lattice algorithms
    - get_d4, get_a2, get_e8: Lattice generator matrices
    - precompute_hq_lut: Lookup table generation
    - calculate_weighted_sum: Inner product estimation

For detailed usage examples, see the README.md file.
"""

# Core quantizer classes
from .hierarchical_nested_lattice_quantizer import HierarchicalNestedLatticeQuantizer
from .nested_lattice_quantizer import NestedLatticeQuantizer

# Lattice algorithms
from .closest_point import (
    closest_point_Dn,
    closest_point_A2, 
    closest_point_E8,
    custom_round,
    upscale,
    downscale
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
from .estimate_inner_product import (
    plot_distortion_rate,
    find_best_beta,
    calculate_inner_product_distortion,
    distortion_rate_theoretical
)

from .estimate_correlated_inner_product import (
    plot_distortion_rho,
    generate_rho_correlated_samples,
    calculate_distortion
)

from .compare_quantizer_distortion import (
    run_comparison_experiment,
    calculate_rate_and_distortion
)

from .plot_reconstructed_codebook import (
    generate_codebook,
    compare_codebooks,
    plot_with_voronoi
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
    "upscale",
    "downscale",
    
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
    "plot_distortion_rate",
    "find_best_beta",
    "calculate_inner_product_distortion",
    "distortion_rate_theoretical",
    "plot_distortion_rho",
    "generate_rho_correlated_samples",
    "calculate_distortion",
    "run_comparison_experiment",
    "calculate_rate_and_distortion",
    "generate_codebook",
    "compare_codebooks",
    "plot_with_voronoi"
]
