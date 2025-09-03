"""
Columnwise Matrix-Vector Multiplication Processor

This module implements various strategies for columnwise matrix-vector multiplication
y = Wx where the operation is treated as a linear combination of matrix columns.
Supports sparse vectors, padding, quantization options, and different computation strategies.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from ...quantizers.hnlq import HNLQ
from ...quantizers.utils import (
    closest_point_A2,
    closest_point_Dn,
    closest_point_E8,
    get_a2,
    get_d4,
    get_e8,
    get_z2,
    get_z3,
)
from ..utils.padder import BlockingStrategy


class ColumnwiseMatVecProcessor(ABC):
    """
    Base class for columnwise matrix-vector multiplication.

    This abstract base class provides the core interface and common functionality
    for different matvec computation strategies.
    """

    def __init__(
        self,
        matrix: np.ndarray,
        lattice_type: str = "D4",
        M: int = 2,
        q: int = 4,
        beta: float = 0.2,
        alpha: float = 1 / 3,
        eps: float = 1e-8,
        use_lookup: bool = False,
        quantize_x: bool = False,
        sparsity_threshold: float = 1e-10,
        decoding: str = "full",
    ):
        """
        Initialize the columnwise matvec processor.

        Parameters:
        -----------
        matrix : np.ndarray
            Input matrix W (m x n).
        lattice_type : str
            Type of lattice to use ('D4', 'A2', 'E8', 'Z2', 'Z3').
        M : int
            Number of hierarchical levels.
        q : int
            Quantization parameter.
        beta : float
            Scaling parameter for quantization.
        alpha : float
            Scaling parameter for overload handling.
        eps : float
            Small perturbation parameter.
        use_lookup : bool
            Whether to use lookup tables for computation.
        quantize_x : bool
            Whether to quantize the input vector x.
        sparsity_threshold : float
            Threshold for considering elements as zero.
        decoding : str
            Default decoding method ('full', 'coarse_to_fine', 'progressive').
        """
        self.original_matrix = matrix.copy()
        self.original_m, self.original_n = matrix.shape
        self.lattice_type = lattice_type
        self.M = M
        self.q = q
        self.beta = beta
        self.alpha = alpha
        self.eps = eps
        self.use_lookup = use_lookup
        self.quantize_x = quantize_x
        self.sparsity_threshold = sparsity_threshold
        self.decoding = decoding

        # Setup lattice parameters
        self.G, self.Q_nn = self._setup_lattice(lattice_type)
        self.dimension = self.G.shape[0]

        # Initialize blocking strategy
        self.blocking = BlockingStrategy(lattice_type)

        # Setup padding
        self.padded_matrix, self.padded_shape = self._setup_padding()
        self.m, self.n = self.padded_shape

        # Initialize quantizers for matrix columns
        self.column_quantizers = {}
        self.encoded_columns = {}
        self._initialize_column_quantizers()

        # Performance tracking
        self.stats = {
            "compression_ratio": 0.0,
            "memory_usage_mb": 0.0,
            "computation_time": 0.0,
        }

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

    def _setup_padding(self) -> Tuple[np.ndarray, Tuple[int, int]]:
        """Setup padding for matrix dimensions."""
        # Calculate padding needed
        m_pad = (self.dimension - (self.original_m % self.dimension)) % self.dimension
        n_pad = (self.dimension - (self.original_n % self.dimension)) % self.dimension

        # Create padded matrix
        padded_m = self.original_m + m_pad
        padded_n = self.original_n + n_pad
        padded_matrix = np.zeros((padded_m, padded_n))
        padded_matrix[: self.original_m, : self.original_n] = self.original_matrix

        return padded_matrix, (padded_m, padded_n)

    def _initialize_column_quantizers(self):
        """Initialize quantizers for each column block."""
        n_blocks = self.n // self.dimension

        for block_idx in range(n_blocks):
            start_col = block_idx * self.dimension
            end_col = start_col + self.dimension

            # Create quantizer for this block
            # Create dither vector for this block
            dither = np.zeros(self.dimension)

            quantizer = HNLQ(
                G=self.G,
                Q_nn=self.Q_nn,
                M=self.M,
                q=self.q,
                beta=self.beta,
                alpha=self.alpha,
                eps=self.eps,
                dither=dither,
            )

            self.column_quantizers[block_idx] = quantizer

            # Encode each column in the block
            self.encoded_columns[block_idx] = []

            for col_idx in range(start_col, end_col):
                column = self.padded_matrix[:, col_idx]

                # Pad column if necessary to make it divisible by lattice dimension
                if len(column) % self.dimension != 0:
                    padding_needed = self.dimension - (len(column) % self.dimension)
                    column = np.pad(column, (0, padding_needed), mode="constant")

                # Block the column into chunks of lattice dimension
                column_chunks = []
                for i in range(0, len(column), self.dimension):
                    chunk = column[i : i + self.dimension]
                    column_chunks.append(chunk)

                # Encode each chunk
                column_encodings = []
                for chunk in column_chunks:
                    encoding, T = quantizer.encode(chunk, with_dither=False)
                    column_encodings.append(encoding)

                self.encoded_columns[block_idx].append(column_encodings)

    def _pad_vector(self, x: np.ndarray) -> np.ndarray:
        """Pad vector x to match matrix column blocking."""
        if len(x) != self.original_n:
            raise ValueError(
                f"Vector length {len(x)} != matrix columns {self.original_n}"
            )

        # Calculate padding needed
        n_pad = (self.dimension - (self.original_n % self.dimension)) % self.dimension

        if n_pad == 0:
            return x.copy()

        # Pad with zeros
        padded_x = np.zeros(self.original_n + n_pad)
        padded_x[: self.original_n] = x
        return padded_x

    def _detect_sparsity(self, x: np.ndarray) -> Tuple[List[int], float]:
        """Detect sparsity pattern and sparsity ratio."""
        non_zero_indices = [
            i for i, val in enumerate(x) if abs(val) > self.sparsity_threshold
        ]
        sparsity_ratio = 1.0 - len(non_zero_indices) / len(x)
        return non_zero_indices, sparsity_ratio

    def _get_block_indices(self, vector_indices: List[int]) -> List[int]:
        """Get block indices corresponding to vector indices."""
        block_indices = set()
        for idx in vector_indices:
            block_idx = idx // self.dimension
            block_indices.add(block_idx)
        return sorted(list(block_indices))

    @abstractmethod
    def compute_matvec(
        self,
        x: np.ndarray,
        decoding_depths: Optional[List[int]] = None,
        sparsity_pattern: Optional[List[int]] = None,
    ) -> np.ndarray:
        """
        Compute y = Wx using columnwise approach.

        Parameters:
        -----------
        x : np.ndarray
            Input vector x.
        decoding_depths : List[int], optional
            Decoding depth for each column block (0 to M-1).
        sparsity_pattern : List[int], optional
            Indices of non-zero elements in x.

        Returns:
        --------
        np.ndarray
            Result vector y.
        """
        pass

    def get_compression_stats(self) -> Dict[str, float]:
        """Get compression and performance statistics."""
        # Calculate original storage
        original_bits = self.original_m * self.original_n * 32  # Assuming 32-bit floats

        # Calculate encoded storage
        encoded_bits = 0
        for block_idx, quantizer in self.column_quantizers.items():
            for column_encodings in self.encoded_columns[block_idx]:
                for chunk_encoding in column_encodings:
                    for level_encoding in chunk_encoding:
                        encoded_bits += len(level_encoding) * np.log2(quantizer.q)

        self.stats["compression_ratio"] = (
            original_bits / encoded_bits if encoded_bits > 0 else float("inf")
        )

        # Calculate memory usage
        encoded_size = 0
        for block_encodings in self.encoded_columns.values():
            for column_encodings in block_encodings:
                for chunk_encoding in column_encodings:
                    for level_encoding in chunk_encoding:
                        encoded_size += len(level_encoding)
        self.stats["memory_usage_mb"] = (
            encoded_size * 4 / (1024 * 1024)
        )  # Assuming 4 bytes per element

        return self.stats.copy()

    def get_matrix_info(self) -> Dict[str, Union[int, float]]:
        """Get information about the matrix and padding."""
        return {
            "original_shape": (self.original_m, self.original_n),
            "padded_shape": (self.m, self.n),
            "lattice_dimension": self.dimension,
            "num_blocks": self.n // self.dimension,
            "padding_rows": self.m - self.original_m,
            "padding_cols": self.n - self.original_n,
        }
