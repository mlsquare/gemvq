"""
Applications Module

This module contains applications of lattice quantizers for matrix operations,
analysis, and visualization. It provides tools for evaluating quantizer performance
and analyzing quantization effects on matrix operations.

Analysis Functions:
    - calculate_inner_product_distortion: Rate-distortion analysis
    - plot_distortion_rate: Visualization of performance curves
    - find_best_beta: Parameter optimization
    - calculate_mse_and_overload_for_samples: Performance evaluation

Correlation Analysis:
    - plot_distortion_rho: Correlation impact analysis
    - generate_rho_correlated_samples: Correlated data generation
    - calculate_distortion: Distortion calculation

Comparison Tools:
    - run_comparison_experiment: Quantizer comparison
    - calculate_rate_and_distortion: Performance metrics

Visualization:
    - generate_codebook: Codebook generation and visualization
    - compare_codebooks: Quantization method comparison
    - plot_with_voronoi: Voronoi diagram visualization

This module builds on the core quantizers to provide comprehensive analysis
and evaluation tools for lattice quantization applications.
"""

from .estimate_inner_product import (
    calculate_inner_product_distortion,
    plot_distortion_rate,
    find_best_beta,
    calculate_mse_and_overload_for_samples
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

__all__ = [
    # Inner product estimation
    'calculate_inner_product_distortion',
    'plot_distortion_rate',
    'find_best_beta',
    'calculate_mse_and_overload_for_samples',
    
    # Correlated inner product estimation
    'plot_distortion_rho',
    'generate_rho_correlated_samples',
    'calculate_distortion',
    
    # Distortion comparison
    'run_comparison_experiment',
    'calculate_rate_and_distortion',
    
    # Visualization
    'generate_codebook',
    'compare_codebooks',
    'plot_with_voronoi'
] 