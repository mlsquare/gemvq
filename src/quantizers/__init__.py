"""
Core Quantizers Module

This module contains the fundamental quantizer implementations and lattice algorithms
that form the foundation of the LatticeQuant library.

Classes:
    - NestedLatticeQuantizer: Classic single-level lattice quantization
    - HierarchicalNestedLatticeQuantizer: Multi-level hierarchical quantization

Functions:
    - closest_point_Dn: Dₙ lattice closest point algorithm
    - closest_point_A2: A₂ lattice closest point algorithm  
    - closest_point_E8: E₈ lattice closest point algorithm
    - custom_round: Custom rounding function for quantization

This module provides the core quantization functionality used by all other modules
in the LatticeQuant library.
"""

from .nested_lattice_quantizer import NestedLatticeQuantizer
from .hierarchical_nested_lattice_quantizer import HierarchicalNestedLatticeQuantizer
from .closest_point import (
    closest_point_Dn,
    closest_point_A2,
    closest_point_E8,
    custom_round
)

__all__ = [
    'NestedLatticeQuantizer',
    'HierarchicalNestedLatticeQuantizer',
    'closest_point_Dn',
    'closest_point_A2',
    'closest_point_E8',
    'custom_round'
] 