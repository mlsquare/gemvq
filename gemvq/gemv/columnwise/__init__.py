"""
Column-wise matrix-vector multiplication implementations.

This module provides column-wise processing strategies for matrix-vector
multiplication with lattice quantization support.
"""

from .columnwise_matvec_processor import ColumnwiseMatVecProcessor

__all__ = ["ColumnwiseMatVecProcessor"]
