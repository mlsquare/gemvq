"""
Blocking Strategy for Lattice-Quantized Matrix-Vector Multiplication

This module implements efficient blocking strategies for matrix-vector multiplication
using lattice quantization. The blocking is based on lattice dimensions to optimize
memory access patterns and computational efficiency.
"""

from typing import List, Tuple

import numpy as np

from ..lattices.utils import (closest_point_A2, closest_point_Dn,
                              closest_point_E8)
from ..lattices.utils import get_a2, get_d4, get_e8, get_z2, get_z3


class BlockingStrategy:
    """
    Blocking strategy for efficient matrix-vector multiplication.

    This class manages the division of vectors into blocks based on lattice dimensions,
    optimizing memory access patterns and computational efficiency for lattice-quantized
    matrix-vector multiplication.
    """

    def __init__(self, lattice_type: str = "D4"):
        """
        Initialize the blocking strategy.

        Parameters:
        -----------
        lattice_type : str
            Type of lattice to use ('D4', 'A2', 'E8', 'Z2', 'Z3').
        """
        self.lattice_type = lattice_type
        self.G, self.Q_nn = self._setup_lattice(lattice_type)
        self.block_size = self.G.shape[0]  # Dimension of the lattice

    def _setup_lattice(self, lattice_type: str) -> Tuple[np.ndarray, callable]:
        """Setup lattice generator matrix and closest point function."""
        if lattice_type == "D4":
            return get_d4(), closest_point_Dn
        elif lattice_type == "A2":
            return get_a2(), closest_point_A2
        elif lattice_type == "E8":
            return get_e8(), closest_point_E8
        elif lattice_type == "Z2":
            return get_z2(), lambda x: np.round(x)
        elif lattice_type == "Z3":
            return get_z3(), lambda x: np.round(x)
        else:
            raise ValueError(f"Unsupported lattice type: {lattice_type}")

    def get_block_indices(self, vector_length: int) -> List[Tuple[int, int]]:
        """
        Get block indices for a vector of given length.

        Parameters:
        -----------
        vector_length : int
            Length of the vector to be blocked.

        Returns:
        --------
        List[Tuple[int, int]]
            List of (start_idx, end_idx) tuples for each block.
        """
        blocks = []
        for start_idx in range(0, vector_length, self.block_size):
            end_idx = min(start_idx + self.block_size, vector_length)
            blocks.append((start_idx, end_idx))
        return blocks

    def get_matrix_blocks_column_wise(self, matrix: np.ndarray) -> List[np.ndarray]:
        """
        Get matrix blocks for column-wise approach.
        Each column is divided into blocks of size equal to lattice dimension.

        Parameters:
        -----------
        matrix : np.ndarray
            Input matrix to be blocked.

        Returns:
        --------
        List[np.ndarray]
            List of matrix blocks.
        """
        m, n = matrix.shape
        blocks = []

        # Block by columns
        for start_col in range(0, n, self.block_size):
            end_col = min(start_col + self.block_size, n)
            block = matrix[:, start_col:end_col]
            blocks.append(block)

        return blocks

    def get_matrix_blocks_row_wise(self, matrix: np.ndarray) -> List[np.ndarray]:
        """
        Get matrix blocks for row-wise approach.
        Each row is divided into blocks of size equal to lattice dimension.

        Parameters:
        -----------
        matrix : np.ndarray
            Input matrix to be blocked.

        Returns:
        --------
        List[np.ndarray]
            List of matrix blocks.
        """
        m, n = matrix.shape
        blocks = []

        # Block by rows
        for start_row in range(0, m, self.block_size):
            end_row = min(start_row + self.block_size, m)
            block = matrix[start_row:end_row, :]
            blocks.append(block)

        return blocks

    def get_vector_blocks(self, vector: np.ndarray) -> List[np.ndarray]:
        """
        Get vector blocks based on the blocking strategy.

        Parameters:
        -----------
        vector : np.ndarray
            Input vector to be blocked.

        Returns:
        --------
        List[np.ndarray]
            List of vector blocks.
        """
        n = len(vector)
        blocks = []

        for start_idx in range(0, n, self.block_size):
            end_idx = min(start_idx + self.block_size, n)
            block = vector[start_idx:end_idx]
            blocks.append(block)

        return blocks

    def pad_vector(self, vector: np.ndarray) -> np.ndarray:
        """
        Pad vector to make its length divisible by block_size.
        Works for both column-wise and row-wise approaches since vector is always column-shaped.

        Parameters:
        -----------
        vector : np.ndarray
            Input vector to pad.

        Returns:
        --------
        np.ndarray
            Padded vector.
        """
        n = len(vector)
        padding_needed = (self.block_size - (n % self.block_size)) % self.block_size

        if padding_needed == 0:
            return vector

        padded = np.zeros(n + padding_needed)
        padded[:n] = vector
        return padded

    def pad_matrix_for_column_wise(self, matrix: np.ndarray) -> np.ndarray:
        """
        Pad matrix for column-wise approach.
        Pads the matrix columns to make the number of columns divisible by block_size.

        Parameters:
        -----------
        matrix : np.ndarray
            Input matrix to pad.

        Returns:
        --------
        np.ndarray
            Padded matrix.
        """
        m, n = matrix.shape
        padding_needed = (self.block_size - (n % self.block_size)) % self.block_size

        if padding_needed == 0:
            return matrix

        padded = np.zeros((m, n + padding_needed))
        padded[:, :n] = matrix
        return padded

    def pad_matrix_for_row_wise(self, matrix: np.ndarray) -> np.ndarray:
        """
        Pad matrix for row-wise approach.
        Pads the matrix rows to make the number of rows divisible by block_size.

        Parameters:
        -----------
        matrix : np.ndarray
            Input matrix to pad.

        Returns:
        --------
        np.ndarray
            Padded matrix.
        """
        m, n = matrix.shape
        padding_needed = (self.block_size - (m % self.block_size)) % self.block_size

        if padding_needed == 0:
            return matrix

        padded = np.zeros((m + padding_needed, n))
        padded[:m, :] = matrix
        return padded

    def unpad_vector(self, padded_vector: np.ndarray, original_length: int) -> np.ndarray:
        """
        Remove padding from a vector.

        Parameters:
        -----------
        padded_vector : np.ndarray
            Padded vector to unpad.
        original_length : int
            Original length of the vector.

        Returns:
        --------
        np.ndarray
            Unpadded vector.
        """
        return padded_vector[:original_length]

    def unpad_matrix(
        self, padded_matrix: np.ndarray, original_shape: Tuple[int, int]
    ) -> np.ndarray:
        """
        Remove padding from a matrix.

        Parameters:
        -----------
        padded_matrix : np.ndarray
            Padded matrix to unpad.
        original_shape : Tuple[int, int]
            Original shape of the matrix (rows, columns).

        Returns:
        --------
        np.ndarray
            Unpadded matrix.
        """
        m, n = original_shape
        return padded_matrix[:m, :n]
