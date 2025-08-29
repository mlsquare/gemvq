"""
GEMV-Q: Matrix-vector multiplication with lattice quantizers.

A comprehensive library for efficient matrix-vector multiplication using
different lattice quantizers, including nested lattice quantizers and
hierarchical nested lattice quantizers.
"""

__version__ = "0.1.0"
__author__ = "MLSquare"
__email__ = "contact@mlsquare.com"

# Import main classes for easier access
from .quantizers.nlq import NLQ
from .quantizers.hnlq import HNLQ
from .quantizers.utils import get_d4, get_a2, get_e8, closest_point_Dn

__all__ = [
    "NLQ",
    "HNLQ", 
    "get_d4",
    "get_a2",
    "get_e8",
    "closest_point_Dn"
]
