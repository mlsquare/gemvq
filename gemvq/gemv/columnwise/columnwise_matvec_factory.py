"""
Factory Functions for Columnwise Matrix-Vector Multiplication

This module provides factory functions to easily create different types of
columnwise matrix-vector multiplication processors.
"""

from typing import Dict, List, Optional, Union

import numpy as np

from ..adaptive_processor import AdaptiveProcessor
from ..utils.lookup_table_processor import LookupTableProcessor
from .columnwise_matvec_processor import ColumnwiseMatVecProcessor
from .standard_dot_processor import StandardDotProcessor


def create_standard_dot_processor(
    matrix: np.ndarray,
    lattice_type: str = "D4",
    M: int = 2,
    q: int = 4,
    beta: float = 0.2,
    alpha: float = 1 / 3,
    eps: float = 1e-8,
    fixed_depth: bool = True,
    adaptive_depth: bool = False,
    sparsity_threshold: float = 1e-10,
    decoding: str = "full",
) -> StandardDotProcessor:
    """
    Create a standard dot product processor.

    This processor uses np.dot for computation without lookup tables.

    Parameters:
    -----------
    matrix : np.ndarray
        Input matrix W (m x n).
    lattice_type : str
        Type of lattice to use ('D4', 'A2', 'E8', 'Z2', 'Z3').
    M : int
        Number of hierarchical levels.
    q : int
        Quantization parameter.
    beta : float
        Scaling parameter for quantization.
    alpha : float
        Scaling parameter for overload handling.
    eps : float
        Small perturbation parameter.
    fixed_depth : bool
        If True, use same depth for all columns.
    adaptive_depth : bool
        If True, adapt depth based on sparsity pattern.
    sparsity_threshold : float
        Threshold for considering elements as zero.
    decoding : str
        Default decoding method.

    Returns:
    --------
    StandardDotProcessor
        Configured standard dot processor.
    """
    return StandardDotProcessor(
        matrix=matrix,
        lattice_type=lattice_type,
        M=M,
        q=q,
        beta=beta,
        alpha=alpha,
        eps=eps,
        fixed_depth=fixed_depth,
        adaptive_depth=adaptive_depth,
        sparsity_threshold=sparsity_threshold,
        decoding=decoding,
    )


def create_lookup_table_processor(
    matrix: np.ndarray,
    lattice_type: str = "D4",
    M: int = 2,
    q: int = 4,
    beta: float = 0.2,
    alpha: float = 1 / 3,
    eps: float = 1e-8,
    table_strategy: str = "layer_wise_histogram",
    precompute_tables: bool = True,
    sparsity_threshold: float = 1e-10,
    decoding: str = "full",
) -> LookupTableProcessor:
    """
    Create a lookup table processor.

    This processor uses precomputed lookup tables for fast computation.

    Parameters:
    -----------
    matrix : np.ndarray
        Input matrix W (m x n).
    lattice_type : str
        Type of lattice to use.
    M : int
        Number of hierarchical levels.
    q : int
        Quantization parameter.
    beta : float
        Scaling parameter for quantization.
    alpha : float
        Scaling parameter for overload handling.
    eps : float
        Small perturbation parameter.
    table_strategy : str
        Strategy for lookup tables: "layer_wise_histogram", "inner_product", or "hybrid".
    precompute_tables : bool
        Whether to precompute lookup tables.
    sparsity_threshold : float
        Threshold for considering elements as zero.
    decoding : str
        Default decoding method.

    Returns:
    --------
    LookupTableProcessor
        Configured lookup table processor.
    """
    return LookupTableProcessor(
        matrix=matrix,
        lattice_type=lattice_type,
        M=M,
        q=q,
        beta=beta,
        alpha=alpha,
        eps=eps,
        table_strategy=table_strategy,
        precompute_tables=precompute_tables,
        sparsity_threshold=sparsity_threshold,
        decoding=decoding,
    )


def create_adaptive_processor(
    matrix: np.ndarray,
    lattice_type: str = "D4",
    M: int = 2,
    q: int = 4,
    beta: float = 0.2,
    alpha: float = 1 / 3,
    eps: float = 1e-8,
    adaptation_threshold: float = 0.1,
    sparsity_threshold: float = 1e-10,
    decoding: str = "full",
    enable_standard_dot: bool = True,
    enable_lookup_tables: bool = True,
    enable_adaptive_depth: bool = True,
) -> AdaptiveProcessor:
    """
    Create an adaptive processor.

    This processor dynamically chooses between different computation strategies.

    Parameters:
    -----------
    matrix : np.ndarray
        Input matrix W (m x n).
    lattice_type : str
        Type of lattice to use.
    M : int
        Number of hierarchical levels.
    q : int
        Quantization parameter.
    beta : float
        Scaling parameter for quantization.
    alpha : float
        Scaling parameter for overload handling.
    eps : float
        Small perturbation parameter.
    adaptation_threshold : float
        Sparsity threshold for switching strategies.
    sparsity_threshold : float
        Threshold for considering elements as zero.
    decoding : str
        Default decoding method.
    enable_standard_dot : bool
        Whether to enable standard dot product strategy.
    enable_lookup_tables : bool
        Whether to enable lookup table strategies.
    enable_adaptive_depth : bool
        Whether to enable adaptive depth decoding.

    Returns:
    --------
    AdaptiveProcessor
        Configured adaptive processor.
    """
    return AdaptiveProcessor(
        matrix=matrix,
        lattice_type=lattice_type,
        M=M,
        q=q,
        beta=beta,
        alpha=alpha,
        eps=eps,
        adaptation_threshold=adaptation_threshold,
        sparsity_threshold=sparsity_threshold,
        decoding=decoding,
        enable_standard_dot=enable_standard_dot,
        enable_lookup_tables=enable_lookup_tables,
        enable_adaptive_depth=enable_adaptive_depth,
    )


def create_processor(
    matrix: np.ndarray, processor_type: str = "adaptive", **kwargs
) -> ColumnwiseMatVecProcessor:
    """
    Create a columnwise matvec processor of the specified type.

    This is a convenience function that creates processors based on type string.

    Parameters:
    -----------
    matrix : np.ndarray
        Input matrix W (m x n).
    processor_type : str
        Type of processor: "standard_dot", "lookup_table", or "adaptive".
    **kwargs
        Additional arguments passed to the specific processor constructor.

    Returns:
    --------
    ColumnwiseMatVecProcessor
        Configured processor of the specified type.

    Raises:
    -------
    ValueError
        If processor_type is not supported.
    """
    if processor_type == "standard_dot":
        return create_standard_dot_processor(matrix, **kwargs)
    elif processor_type == "lookup_table":
        return create_lookup_table_processor(matrix, **kwargs)
    elif processor_type == "adaptive":
        return create_adaptive_processor(matrix, **kwargs)
    else:
        raise ValueError(
            f"Unsupported processor type: {processor_type}. "
            f"Supported types: standard_dot, lookup_table, adaptive"
        )


def get_available_processors() -> List[str]:
    """Get list of available processor types."""
    return ["standard_dot", "lookup_table", "adaptive"]


def get_processor_info(processor_type: str) -> Dict[str, str]:
    """
    Get information about a processor type.

    Parameters:
    -----------
    processor_type : str
        Type of processor.

    Returns:
    --------
    Dict[str, str]
        Information about the processor type.
    """
    info = {
        "standard_dot": {
            "name": "Standard Dot Processor",
            "description": "Uses np.dot for computation without lookup tables",
            "best_for": "Simple cases, high sparsity, small matrices",
            "memory_usage": "Low",
            "computation_speed": "Medium",
        },
        "lookup_table": {
            "name": "Lookup Table Processor",
            "description": "Uses precomputed lookup tables for fast computation",
            "best_for": "Large matrices, low sparsity, repeated computations",
            "memory_usage": "High",
            "computation_speed": "High",
        },
        "adaptive": {
            "name": "Adaptive Processor",
            "description": "Dynamically chooses between different strategies",
            "best_for": "Variable input characteristics, optimal performance",
            "memory_usage": "Medium",
            "computation_speed": "High",
        },
    }

    return info.get(
        processor_type, {"name": "Unknown", "description": "Unknown processor type"}
    )
