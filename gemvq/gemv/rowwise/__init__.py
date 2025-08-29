"""
Row-wise matrix-vector multiplication implementations.

This module provides row-wise processing strategies for matrix-vector
multiplication with lattice quantization support.
"""

from .rowwise_processor import RowwiseGEMVProcessor

__all__ = [
    "RowwiseGEMVProcessor"
]
