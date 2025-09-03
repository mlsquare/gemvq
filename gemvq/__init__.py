"""
GEMV-Q: Matrix-vector multiplication with lattice quantizers.

A comprehensive library for efficient matrix-vector multiplication using
different lattice quantizers, including nested lattice quantizers and
hierarchical nested lattice quantizers.
"""

__version__ = "0.1.0"
__author__ = "MLSquare"
__email__ = "contact@mlsquare.com"

from .quantizers.hnlq import HNLQ

# Import main classes for easier access
from .quantizers.nlq import NLQ
from .quantizers.utils import closest_point_Dn, get_a2, get_d4, get_e8

__all__ = ["NLQ", "HNLQ", "get_d4", "get_e8", "closest_point_Dn"]
