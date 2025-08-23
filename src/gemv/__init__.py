"""
General Matrix-Vector Multiplication (GEMV) Module

This module implements efficient matrix-vector multiplication using lattice quantization
with both column-wise and row-wise approaches. It includes blocking strategies based on
lattice dimensions for optimal performance.

Main Components:
    - ColumnWiseGEMV: Matrix-vector multiplication as linear combination of columns
    - RowWiseGEMV: Matrix-vector multiplication as series of dot products
    - BlockingStrategy: Efficient blocking based on lattice dimensions
    - LatticeQuantizedGEMV: Unified interface for both approaches

Usage:
    from src.gemv import ColumnWiseGEMV, RowWiseGEMV, LatticeQuantizedGEMV
"""
