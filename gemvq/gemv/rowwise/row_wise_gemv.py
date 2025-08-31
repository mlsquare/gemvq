"""
Row-Wise General Matrix-Vector Multiplication (GEMV)

This module implements matrix-vector multiplication as a series of dot products
using lattice quantization with blocking strategies. The approach treats Wx as
a collection of inner products between matrix rows and the input vector.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np

from ...quantizers.utils import (closest_point_A2, closest_point_Dn,
                              closest_point_E8)
from ...quantizers.hnlq import HNLQ, HNLQConfig
from ...quantizers.utils import get_a2, get_d4, get_e8, get_z2, get_z3
from ..utils.padder import BlockingStrategy


class RowWiseGEMV:
    """
    Row-wise matrix-vector multiplication using lattice quantization.

    This class implements matrix-vector multiplication as a series of dot products
    between quantized matrix rows and the input vector, with efficient blocking
    strategies based on lattice dimensions.
    """

    def __init__(
        self,
        matrix: np.ndarray,
        lattice_type: str = "D4",
        M: int = 2,
        alpha: float = 1.0,
        eps: float = 1e-8,
        q: int = 4,
        beta: float = 1.0,
        decoding: str = "full",
        decoding_depths: Optional[List[int]] = None,
        overload: bool = True,
        max_scaling_iterations: int = 10,
        with_tie_dither: bool = True,
        with_dither: bool = False,
    ):
        """
        Initialize the row-wise GEMV processor.

        Parameters:
        -----------
        matrix : np.ndarray
            Input matrix W (m x n).
        lattice_type : str
            Type of lattice to use ('D4', 'A2', 'E8', 'Z2', 'Z3').
        M : int
            Number of hierarchical levels.
        alpha : float
            Scaling parameter for overload handling.
        eps : float
            Small perturbation parameter.
        q : int
            Quantization parameter.
        beta : float
            Scaling parameter for quantization.
        decoding : str, optional
            Default decoding method to use ('full', 'coarse_to_fine', 'progressive').
            Default is 'full'.
        decoding_depths : List[int], optional
            Decoding depth for each row (1 to M). If None, uses M for all rows.
            Only used when decoding='adaptive_depth'.
        overload : bool, optional
            Whether to handle overload by scaling. Default is True.
        max_scaling_iterations : int, optional
            Maximum number of scaling iterations to prevent infinite loops. Default is 10.
        with_tie_dither : bool, optional
            Whether to add dither to the input for tie breaking. Default is True.
        with_dither : bool, optional
            Whether to add dither to the input for randomized quantization. Default is False.
        """
        self.original_matrix = matrix
        self.original_m, self.original_n = matrix.shape
        self.lattice_type = lattice_type
        self.M = M
        self.alpha = alpha
        self.eps = eps
        self.q = q
        self.beta = beta
        self.decoding = decoding
        self.overload = overload
        self.max_scaling_iterations = max_scaling_iterations
        self.with_tie_dither = with_tie_dither
        self.with_dither = with_dither

        # Set up decoding depths for each row
        if decoding_depths is None:
            # Default to full depth (M) for all rows
            self.decoding_depths = [M] * self.original_m
        else:
            if len(decoding_depths) != self.original_m:
                raise ValueError(
                    f"decoding_depths length ({len(decoding_depths)}) must match number of rows ({self.original_m})"
                )
            # Validate decoding depths
            for i, depth in enumerate(decoding_depths):
                if depth < 1 or depth > M:
                    raise ValueError(f"decoding_depths[{i}] = {depth} must be between 1 and {M}")
            self.decoding_depths = decoding_depths.copy()

        # Initialize blocking strategy
        self.blocking = BlockingStrategy(lattice_type)

        # Setup lattice parameters
        self.G, self.Q_nn = self._setup_lattice(lattice_type)
        self.dimension = self.G.shape[0]

        # Pad matrix for row-wise processing
        self.matrix = self.blocking.pad_matrix_for_row_wise(matrix)
        self.m, self.n = self.matrix.shape

        # Initialize quantizers for each block
        self.quantizers = {}
        self.encoded_rows = {}
        self.overload_scalings = {}

        # Pre-encode matrix rows
        self._encode_matrix()

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

    def _encode_matrix(self):
        """Encode all matrix rows using blocking strategy."""
        # Get row blocks
        row_blocks = self.blocking.get_block_indices(self.m)

        for block_idx, (start_row, end_row) in enumerate(row_blocks):
            # Get matrix block
            matrix_block = self.matrix[start_row:end_row, :]

            # Create HNLQ configuration for this block
            config = HNLQConfig(
                lattice_type=self.lattice_type,
                q=self.q,
                M=self.M,
                beta=self.beta,
                alpha=self.alpha,
                eps=self.eps,
                overload=self.overload,
                decoding=self.decoding,
                max_scaling_iterations=self.max_scaling_iterations,
                with_tie_dither=self.with_tie_dither,
                with_dither=self.with_dither,
            )

            # Create quantizer for this block
            self.quantizers[block_idx] = HNLQ(
                config=config,
                G=self.G,
                Q_nn=self.Q_nn,
            )

            # Encode each row in the block
            self.encoded_rows[block_idx] = []
            self.overload_scalings[block_idx] = []

            for row_idx in range(matrix_block.shape[0]):
                row = matrix_block[row_idx, :]

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
                row_scalings = []
                for chunk in row_chunks:
                    encoding, scaling = self.quantizers[block_idx].encode(chunk, with_dither=self.with_dither)
                    row_encodings.append(encoding)
                    row_scalings.append(scaling)

                self.encoded_rows[block_idx].append(row_encodings)
                self.overload_scalings[block_idx].append(row_scalings)

    def multiply(self, vector: np.ndarray) -> np.ndarray:
        """
        Perform row-wise matrix-vector multiplication.

        Parameters:
        -----------
        vector : np.ndarray
            Input vector x.

        Returns:
        --------
        np.ndarray
            Result vector y = Wx.
        """
        if len(vector) != self.original_n:
            raise ValueError(f"Vector dimension {len(vector)} != matrix columns {self.original_n}")

        # Pad vector to match padded matrix
        padded_vector = self.blocking.pad_vector(vector)

        # Initialize result vector (will be unpadded later)
        result = np.zeros(self.m)

        # Get row blocks
        row_blocks = self.blocking.get_block_indices(self.m)

        # Process each row block
        for block_idx, (start_row, end_row) in enumerate(row_blocks):
            # Process each row in the block
            for row_idx in range(len(self.encoded_rows[block_idx])):
                # Decode all chunks of the row
                decoded_chunks = []
                for chunk_idx in range(len(self.encoded_rows[block_idx][row_idx])):
                    decoded_chunk = self.quantizers[block_idx].decode(
                        self.encoded_rows[block_idx][row_idx][chunk_idx],
                        self.overload_scalings[block_idx][row_idx][chunk_idx],
                        with_dither=self.with_dither,
                    )
                    decoded_chunks.append(decoded_chunk)

                # Concatenate all chunks to reconstruct the full row
                decoded_row = np.concatenate(decoded_chunks)

                # Unpad decoded row to match padded vector length
                if len(decoded_row) > len(padded_vector):
                    decoded_row = decoded_row[: len(padded_vector)]

                # Compute dot product with padded vector
                dot_product = np.dot(decoded_row, padded_vector)
                result[start_row + row_idx] = dot_product

        # Unpad result to original matrix rows
        result = self.blocking.unpad_vector(result, self.original_m)

        return result

    def multiply_with_sparsity(
        self, vector: np.ndarray, sparsity_pattern: Optional[List[int]] = None
    ) -> np.ndarray:
        """
        Perform row-wise matrix-vector multiplication with sparsity support.

        Parameters:
        -----------
        vector : np.ndarray
            Input vector x.
        sparsity_pattern : List[int], optional
            Indices of non-zero elements in the vector.

        Returns:
        --------
        np.ndarray
            Result vector y = Wx.
        """
        if len(vector) != self.original_n:
            raise ValueError(f"Vector dimension {len(vector)} != matrix columns {self.original_n}")

        # Pad vector to match padded matrix
        padded_vector = self.blocking.pad_vector(vector)

        # Initialize result vector
        result = np.zeros(self.m)

        # Use provided sparsity pattern or find non-zero elements
        if sparsity_pattern is None:
            sparsity_pattern = np.where(np.abs(vector) > 1e-10)[0].tolist()

        # Create sparse vector for efficient dot product
        sparse_vector = np.zeros(len(padded_vector))
        sparse_vector[sparsity_pattern] = vector[sparsity_pattern]

        # Get row blocks
        row_blocks = self.blocking.get_block_indices(self.m)

        # Process each row block
        for block_idx, (start_row, end_row) in enumerate(row_blocks):
            # Process each row in the block
            for row_idx in range(len(self.encoded_rows[block_idx])):
                # Decode all chunks of the row
                decoded_chunks = []
                for chunk_idx in range(len(self.encoded_rows[block_idx][row_idx])):
                    decoded_chunk = self.quantizers[block_idx].decode(
                        self.encoded_rows[block_idx][row_idx][chunk_idx],
                        self.overload_scalings[block_idx][row_idx][chunk_idx],
                        with_dither=self.with_dither,
                    )
                    decoded_chunks.append(decoded_chunk)

                # Concatenate all chunks to reconstruct the full row
                decoded_row = np.concatenate(decoded_chunks)

                # Unpad decoded row to match padded vector length
                if len(decoded_row) > len(sparse_vector):
                    decoded_row = decoded_row[: len(sparse_vector)]

                # Compute dot product with sparse vector (only non-zero elements)
                dot_product = 0.0
                for idx in sparsity_pattern:
                    if idx < len(decoded_row):
                        dot_product += decoded_row[idx] * sparse_vector[idx]

                result[start_row + row_idx] = dot_product

        # Unpad result to original matrix rows
        result = self.blocking.unpad_vector(result, self.original_m)

        return result

    def multiply_coarse_to_fine(
        self,
        vector: np.ndarray,
        max_level: int = None,
        sparsity_pattern: Optional[List[int]] = None,
    ) -> np.ndarray:
        """
        Perform row-wise matrix-vector multiplication with coarse-to-fine decoding.

        This method allows decoding at different levels of detail, where higher
        max_level means coarser reconstruction (less detail).

        Parameters:
        -----------
        vector : np.ndarray
            Input vector x.
        max_level : int, optional
            Maximum level to decode up to (1 <= max_level <= M).
            If None, decodes all levels (equivalent to multiply method).
            Higher max_level means coarser reconstruction.
        sparsity_pattern : List[int], optional
            Indices of non-zero elements in the vector.

        Returns:
        --------
        np.ndarray
            Result vector y = Wx at the specified level of detail.
        """
        if len(vector) != self.original_n:
            raise ValueError(f"Vector dimension {len(vector)} != matrix columns {self.original_n}")

        # Pad vector to match padded matrix
        padded_vector = self.blocking.pad_vector(vector)

        # Initialize result vector (will be unpadded later)
        result = np.zeros(self.m)

        # Get row blocks
        row_blocks = self.blocking.get_block_indices(self.m)

        # Process each row block
        for block_idx, (start_row, end_row) in enumerate(row_blocks):
            # Process each row in the block
            for row_idx in range(len(self.encoded_rows[block_idx])):
                # Decode all chunks of the row at specified level
                decoded_chunks = []
                for chunk_idx in range(len(self.encoded_rows[block_idx][row_idx])):
                    decoded_chunk = self.quantizers[block_idx].decode_coarse_to_fine(
                        self.encoded_rows[block_idx][row_idx][chunk_idx],
                        self.overload_scalings[block_idx][row_idx][chunk_idx],
                        with_dither=self.with_dither,
                        depth=max_level,
                    )
                    decoded_chunks.append(decoded_chunk)

                # Concatenate all chunks to reconstruct the full row
                decoded_row = np.concatenate(decoded_chunks)

                # Unpad decoded row to match padded vector length
                if len(decoded_row) > len(padded_vector):
                    decoded_row = decoded_row[: len(padded_vector)]

                # Compute dot product with padded vector
                dot_product = np.dot(decoded_row, padded_vector)
                result[start_row + row_idx] = dot_product

        # Unpad result to original matrix rows
        result = self.blocking.unpad_vector(result, self.original_m)

        return result

    def multiply_progressive(
        self, vector: np.ndarray, sparsity_pattern: Optional[List[int]] = None
    ) -> List[np.ndarray]:
        """
        Perform row-wise matrix-vector multiplication with progressive refinement.

        This method returns a list of results at each level of detail,
        from coarsest to finest.

        Parameters:
        -----------
        vector : np.ndarray
            Input vector x.
        sparsity_pattern : List[int], optional
            Indices of non-zero elements in the vector.

        Returns:
        --------
        List[np.ndarray]
            List of result vectors at each level of detail, from coarsest to finest.
        """
        if len(vector) != self.original_n:
            raise ValueError(f"Vector dimension {len(vector)} != matrix columns {self.original_n}")

        # Pad vector to match padded matrix
        padded_vector = self.blocking.pad_vector(vector)

        # Initialize results for each level
        results = [np.zeros(self.m) for _ in range(self.M)]

        # Get row blocks
        row_blocks = self.blocking.get_block_indices(self.m)

        # Process each row block
        for block_idx, (start_row, end_row) in enumerate(row_blocks):
            # Process each row in the block
            for row_idx in range(len(self.encoded_rows[block_idx])):
                # Get progressive reconstructions for all chunks
                progressive_chunks = []
                for chunk_idx in range(len(self.encoded_rows[block_idx][row_idx])):
                    chunk_reconstructions = self.quantizers[block_idx].decode_progressive(
                        self.encoded_rows[block_idx][row_idx][chunk_idx],
                        self.overload_scalings[block_idx][row_idx][chunk_idx],
                        with_dither=self.with_dither,
                    )
                    progressive_chunks.append(chunk_reconstructions)

                # For each level, concatenate chunks and compute dot product
                for level_idx in range(self.M):
                    decoded_chunks = [
                        chunk_reconstructions[level_idx]
                        for chunk_reconstructions in progressive_chunks
                    ]
                    decoded_row = np.concatenate(decoded_chunks)

                    # Unpad decoded row to match padded vector length
                    if len(decoded_row) > len(padded_vector):
                        decoded_row = decoded_row[: len(padded_vector)]

                    # Compute dot product with padded vector
                    dot_product = np.dot(decoded_row, padded_vector)
                    results[level_idx][start_row + row_idx] = dot_product

        # Unpad all results to original matrix rows
        results = [self.blocking.unpad_vector(result, self.original_m) for result in results]

        return results

    def multiply_with_lookup(
        self, vector: np.ndarray, lookup_tables: Optional[Dict] = None
    ) -> np.ndarray:
        """
        Perform row-wise matrix-vector multiplication using lookup tables.

        Parameters:
        -----------
        vector : np.ndarray
            Input vector x.
        lookup_tables : Dict, optional
            Precomputed lookup tables for inner product estimation.

        Returns:
        --------
        np.ndarray
            Result vector y = Wx.
        """
        if len(vector) != self.original_n:
            raise ValueError(f"Vector dimension {len(vector)} != matrix columns {self.original_n}")

        # Pad vector to match padded matrix
        padded_vector = self.blocking.pad_vector(vector)

        # Initialize result vector
        result = np.zeros(self.m)

        # Get row blocks
        row_blocks = self.blocking.get_block_indices(self.m)

        # Process each row block
        for block_idx, (start_row, end_row) in enumerate(row_blocks):
            # Process each row in the block
            for row_idx in range(len(self.encoded_rows[block_idx])):
                # Estimate inner product using lookup tables if available
                if lookup_tables is not None and block_idx in lookup_tables:
                    dot_product = self._estimate_inner_product_with_lookup(
                        self.encoded_rows[block_idx][row_idx],
                        padded_vector,
                        lookup_tables[block_idx],
                    )
                else:
                    # Fall back to regular decoding
                    decoded_chunks = []
                    for chunk_idx in range(len(self.encoded_rows[block_idx][row_idx])):
                        decoded_chunk = self.quantizers[block_idx].decode(
                            self.encoded_rows[block_idx][row_idx][chunk_idx],
                            self.overload_scalings[block_idx][row_idx][chunk_idx],
                            with_dither=self.with_dither,
                        )
                        decoded_chunks.append(decoded_chunk)

                    # Concatenate all chunks to reconstruct the full row
                    decoded_row = np.concatenate(decoded_chunks)

                    # Unpad decoded row to match padded vector length
                    if len(decoded_row) > len(padded_vector):
                        decoded_row = decoded_row[: len(padded_vector)]

                    # Compute dot product
                    dot_product = np.dot(decoded_row, padded_vector)

                result[start_row + row_idx] = dot_product

        # Unpad result to original matrix rows
        result = self.blocking.unpad_vector(result, self.original_m)

        return result

    def _estimate_inner_product_with_lookup(
        self, encoding: Tuple, vector: np.ndarray, lookup_table: Dict
    ) -> float:
        """
        Estimate inner product using lookup table.

        Parameters:
        -----------
        encoding : tuple
            Encoded row representation.
        vector : np.ndarray
            Input vector.
        lookup_table : dict
            Lookup table for inner product estimation.

        Returns:
        --------
        float
            Estimated inner product value.
        """
        # This is a simplified implementation
        # In practice, this would use the full hierarchical lookup table computation
        # from the paper for more accurate estimation

        # For now, decode and compute exact dot product
        decoded_row = self.quantizers[0].decode(encoding, 0, with_dither=self.with_dither)

        if len(decoded_row) > len(vector):
            decoded_row = decoded_row[: len(vector)]

        return np.dot(decoded_row, vector)

    def get_compression_ratio(self) -> float:
        """
        Calculate compression ratio achieved by the encoding.

        Returns:
        --------
        float
            Compression ratio (original bits / encoded bits).
        """
        # Calculate original storage
        original_bits = self.original_m * self.original_n * 32  # Assuming 32-bit floats

        # Calculate encoded storage
        encoded_bits = 0
        for block_idx in self.encoded_rows:
            for row_encodings in self.encoded_rows[block_idx]:
                for chunk_encoding in row_encodings:
                    # Count bits in encoding vectors
                    for level_encoding in chunk_encoding:
                        encoded_bits += len(level_encoding) * np.log2(self.q)

        return original_bits / encoded_bits if encoded_bits > 0 else float("inf")

    def get_memory_usage(self) -> Dict[str, float]:
        """
        Get memory usage statistics.

        Returns:
        --------
        Dict
            Dictionary with memory usage information.
        """
        # Calculate encoded size
        encoded_size = sum(
            sum(len(level_encoding) for level_encoding in chunk_encoding)
            for block_encodings in self.encoded_rows.values()
            for row_encodings in block_encodings
            for chunk_encoding in row_encodings
        )

        # Calculate quantizer storage
        quantizer_size = len(self.quantizers) * self.dimension * 8  # Assuming 8 bytes per element

        return {
            "encoded_rows_mb": encoded_size * 4 / (1024 * 1024),
            "quantizers_mb": quantizer_size / (1024 * 1024),
            "total_mb": (encoded_size * 4 + quantizer_size) / (1024 * 1024),
        }

    def get_blocking_info(self) -> Dict:
        """
        Get information about the blocking strategy used.

        Returns:
        --------
        Dict
            Dictionary with blocking information.
        """
        col_blocks = self.blocking.get_block_indices(self.n)
        row_blocks = self.blocking.get_block_indices(self.m)

        return {
            "lattice_type": self.lattice_type,
            "block_size": self.blocking.block_size,
            "num_col_blocks": len(col_blocks),
            "num_row_blocks": len(row_blocks),
            "col_block_indices": col_blocks,
            "row_block_indices": row_blocks,
            "total_blocks": len(col_blocks) * len(row_blocks),
            "original_shape": (self.original_m, self.original_n),
            "padded_shape": (self.m, self.n),
        }


def row_wise_gemv(
    matrix: np.ndarray,
    vector: np.ndarray,
    lattice_type: str = "D4",
    M: int = 2,
    sparsity_pattern: Optional[List[int]] = None,
    use_lookup: bool = False,
    **kwargs
) -> np.ndarray:
    """
    Perform row-wise matrix-vector multiplication.

    Parameters:
    -----------
    matrix : np.ndarray
        Input matrix W.
    vector : np.ndarray
        Input vector x.
    lattice_type : str
        Type of lattice to use.
    M : int
        Number of hierarchical levels.
    sparsity_pattern : List[int], optional
        Indices of non-zero elements in the vector.
    use_lookup : bool
        Whether to use lookup tables for computation.
    **kwargs : Additional parameters for HNLQ configuration

    Returns:
    --------
    np.ndarray
        Result vector y = Wx.
    """
    processor = RowWiseGEMV(matrix, lattice_type, M, **kwargs)

    if use_lookup:
        return processor.multiply_with_lookup(vector)
    elif sparsity_pattern is not None:
        return processor.multiply_with_sparsity(vector, sparsity_pattern)
    else:
        return processor.multiply(vector)
