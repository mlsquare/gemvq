"""
SVD-based matrix-vector multiplication implementations.

This module provides SVD-based processing strategies for matrix-vector
multiplication with lattice quantization support.
"""

from .svd_gemv_processor import SVDGEMVProcessor

__all__ = [
    "SVDGEMVProcessor"
]
