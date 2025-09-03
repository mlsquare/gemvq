"""
Rowwise GEMV processor.

This module implements rowwise matrix-vector multiplication where the operation
is treated as a series of dot products between matrix rows and the input vector.
"""

from typing import Any, Dict, List, Optional, Tuple, Union

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
from ..base.gemv_processor import GEMVProcessor


class RowwiseGEMVProcessor(GEMVProcessor):
    """
    Rowwise GEMV processor.

    Implements matrix-vector multiplication as a series of dot products between
    matrix rows and the input vector, with support for quantization and different
    computation strategies.
    """

    def __init__(
        self,
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
        **kwargs,
    ):
        """
        Initialize the rowwise GEMV processor.

        Parameters:
        -----------
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
        **kwargs : Additional configuration parameters
        """
        super().__init__(
            lattice_type=lattice_type,
            M=M,
            q=q,
            beta=beta,
            alpha=alpha,
            eps=eps,
            use_lookup=use_lookup,
            quantize_x=quantize_x,
            sparsity_threshold=sparsity_threshold,
            decoding=decoding,
            **kwargs,
        )

        # Initialize state variables
        self.original_matrix = None
        self.original_m = None
        self.original_n = None
        self.G = None
        self.Q_nn = None
        self.dimension = None
        self.padded_matrix = None
        self.padded_shape = None
        self.m = None
        self.n = None
        self.row_quantizers = {}
        self.encoded_rows = {}
        self.stats = {
            "compression_ratio": 0.0,
            "memory_usage_mb": 0.0,
            "computation_time": 0.0,
        }

    def _validate_config(self) -> None:
        """Validate the configuration parameters."""
        lattice_type = self.config.get("lattice_type", "D4")
        if lattice_type not in ["D4", "A2", "E8", "Z2", "Z3"]:
            raise ValueError(f"Unsupported lattice type: {lattice_type}")

        M = self.config.get("M", 2)
        if M < 1:
            raise ValueError(f"M must be >= 1, got {M}")

        q = self.config.get("q", 4)
        if q < 1:
            raise ValueError(f"q must be >= 1, got {q}")

        beta = self.config.get("beta", 0.2)
        if beta <= 0:
            raise ValueError(f"beta must be > 0, got {beta}")

        alpha = self.config.get("alpha", 1 / 3)
        if alpha <= 0:
            raise ValueError(f"alpha must be > 0, got {alpha}")

        eps = self.config.get("eps", 1e-8)
        if eps <= 0:
            raise ValueError(f"eps must be > 0, got {eps}")

        sparsity_threshold = self.config.get("sparsity_threshold", 1e-10)
        if sparsity_threshold < 0:
            raise ValueError(
                f"sparsity_threshold must be >= 0, got {sparsity_threshold}"
            )

        decoding = self.config.get("decoding", "full")
        if decoding not in ["full", "coarse_to_fine", "progressive"]:
            raise ValueError(f"Unsupported decoding method: {decoding}")

    def preprocess_matrix(self, matrix: np.ndarray) -> np.ndarray:
        """
        Preprocess the matrix for rowwise computation.

        Args:
            matrix: Input matrix

        Returns:
            Preprocessed matrix
        """
        # Store original matrix and setup lattice parameters
        self.original_matrix = matrix.copy()
        self.original_m, self.original_n = matrix.shape

        # Setup lattice parameters
        lattice_type = self.config.get("lattice_type", "D4")
        self.G, self.Q_nn = self._setup_lattice(lattice_type)
        self.dimension = self.G.shape[0]

        # Setup padding
        self.padded_matrix, self.padded_shape = self._setup_padding()
        self.m, self.n = self.padded_shape

        # Initialize quantizers for matrix rows
        self._initialize_row_quantizers()

        return self.padded_matrix

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

    def _initialize_row_quantizers(self):
        """Initialize quantizers for each row block."""
        m_blocks = self.m // self.dimension

        for block_idx in range(m_blocks):
            start_row = block_idx * self.dimension
            end_row = start_row + self.dimension

            # Create quantizer for this block
            dither = np.zeros(self.dimension)

            quantizer = HNLQ(
                G=self.G,
                Q_nn=self.Q_nn,
                M=self.config.get("M", 2),
                q=self.config.get("q", 4),
                beta=self.config.get("beta", 0.2),
                alpha=self.config.get("alpha", 1 / 3),
                eps=self.config.get("eps", 1e-8),
                dither=dither,
            )

            self.row_quantizers[block_idx] = quantizer

            # Encode each row in the block
            self.encoded_rows[block_idx] = []

            for row_idx in range(start_row, end_row):
                row = self.padded_matrix[row_idx, :]

                # Pad row if necessary to make it divisible by lattice dimension
                if len(row) % self.dimension != 0:
                    padding_needed = self.dimension - (len(row) % self.dimension)
                    row = np.pad(row, (0, padding_needed), mode="constant")

                # Block the row into chunks of lattice dimension
                row_chunks = []
                for i in range(0, len(row), self.dimension):
                    chunk = row[i : i + self.dimension]
                    row_chunks.append(chunk)

                # Encode each chunk
                row_encodings = []
                for chunk in row_chunks:
                    encoding, T = quantizer.encode(chunk, with_dither=False)
                    row_encodings.append((encoding, T))

                self.encoded_rows[block_idx].append(row_encodings)

    def _pad_vector(self, x: np.ndarray) -> np.ndarray:
        """Pad vector x to match matrix row blocking."""
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

    def process(self, matrix: np.ndarray, vector: np.ndarray) -> np.ndarray:
        """
        Process the rowwise matrix-vector multiplication.

        Args:
            matrix: Input matrix (already preprocessed)
            vector: Input vector

        Returns:
            Result vector
        """
        # Pad vector if necessary
        padded_x = self._pad_vector(vector)

        # Initialize result vector
        result = np.zeros(self.original_m)

        # Process each row block
        m_blocks = self.m // self.dimension

        for block_idx in range(m_blocks):
            start_row = block_idx * self.dimension
            end_row = start_row + self.dimension

            # Get quantizer for this block
            quantizer = self.row_quantizers[block_idx]

            # Process each row in the block
            for row_idx in range(start_row, end_row):
                if row_idx >= self.original_m:
                    break

                # Get encoded row
                row_encodings = self.encoded_rows[block_idx][row_idx - start_row]

                # Decode row
                decoding_method = self.config.get("decoding", "full")
                decoded_row = self._decode_row(
                    row_encodings, quantizer, decoding_method
                )

                # Compute dot product with vector
                dot_product = np.dot(
                    decoded_row[: self.original_n], padded_x[: self.original_n]
                )

                # Store result
                result[row_idx] = dot_product

        return result

    def _decode_row(
        self, row_encodings: List, quantizer: HNLQ, decoding_method: str
    ) -> np.ndarray:
        """Decode a row using the specified decoding method."""
        decoded_chunks = []

        for encoding, T in row_encodings:
            if decoding_method == "full":
                chunk = quantizer.decode(encoding, T, with_dither=False)
            elif decoding_method == "coarse_to_fine":
                chunk = quantizer.decode_coarse_to_fine(encoding, T, with_dither=False)
            elif decoding_method == "progressive":
                chunk = quantizer.decode_progressive(encoding, T, with_dither=False)
            else:
                chunk = quantizer.decode(encoding, T, with_dither=False)

            decoded_chunks.append(chunk)

        return np.concatenate(decoded_chunks)

    def get_processor_info(self) -> Dict[str, Any]:
        """
        Get information about the processor configuration.

        Returns:
            Dictionary containing processor information
        """
        return {
            "type": "rowwise",
            "lattice_type": self.config.get("lattice_type", "D4"),
            "M": self.config.get("M", 2),
            "q": self.config.get("q", 4),
            "beta": self.config.get("beta", 0.2),
            "alpha": self.config.get("alpha", 1 / 3),
            "eps": self.config.get("eps", 1e-8),
            "use_lookup": self.config.get("use_lookup", False),
            "quantize_x": self.config.get("quantize_x", False),
            "sparsity_threshold": self.config.get("sparsity_threshold", 1e-10),
            "decoding": self.config.get("decoding", "full"),
            "stats": self.stats,
        }
