"""
GEMV (General Matrix-Vector) module.

This module provides various strategies for matrix-vector multiplication including
columnwise, rowwise, and SVD-based approaches with support for quantization.
"""

from .base import GEMVProcessor, GEMVFactory, create_gemv_processor
from .columnwise import ColumnwiseGEMVProcessor
from .rowwise import RowwiseGEMVProcessor
from .svd import SVDGEMVProcessor
from .utils import BlockingStrategy

# Register processors with the factory
GEMVFactory.register_processor('columnwise', ColumnwiseGEMVProcessor)
GEMVFactory.register_processor('rowwise', RowwiseGEMVProcessor)
GEMVFactory.register_processor('svd', SVDGEMVProcessor)

__all__ = [
    'GEMVProcessor',
    'GEMVFactory', 
    'create_gemv_processor',
    'ColumnwiseGEMVProcessor',
    'RowwiseGEMVProcessor',
    'SVDGEMVProcessor',
    'BlockingStrategy'
]
