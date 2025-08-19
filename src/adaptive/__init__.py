"""
Adaptive Matrix-Vector Multiplication Module

This module contains specialized implementations for adaptive matrix-vector
multiplication using hierarchical nested lattice quantizers with column-wise
encoding. It provides efficient sparse matrix-vector multiplication with
adaptive bit rates for each column.

The module includes:
- Layer-wise histogram matrix-vector multiplication (from ada_matmul.md)
- Adaptive column-based quantization
- Sparse matrix-vector processing

For usage examples, see the individual module files:
- layer_wise_histogram_matvec.py
- standalone_layer_wise_histogram.py
- adaptive_matvec.py
- demo_adaptive_matvec.py
""" 