"""
Column-Wise General Matrix-Vector Multiplication (GEMV)

This module implements matrix-vector multiplication as a linear combination of columns
using lattice quantization with blocking strategies. The approach treats Wx as a
weighted sum of matrix columns, where each column is quantized using hierarchical
nested lattice quantizers.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np

from ...lattices.utils import (closest_point_A2, closest_point_Dn,
                              closest_point_E8)
from ...lattices.quantizers.hierarchical_nested_lattice_quantizer import \
    HierarchicalNestedLatticeQuantizer
from ...lattices.utils import get_a2, get_d4, get_e8, get_z2, get_z3
from ..utils.padder import BlockingStrategy


class ColumnWiseGEMV:
    """
    Column-wise matrix-vector multiplication using lattice quantization.

    This class implements matrix-vector multiplication as a linear combination
    of quantized matrix columns, with efficient blocking strategies based on
    lattice dimensions.
    """

    def __init__(
        self,
        matrix: np.ndarray,
        lattice_type: str = "D4",
        M: int = 2,
        alpha: float = 1 / 3,
        eps: float = 1e-8,
        q: int = 4,
        beta: float = 0.2,
        decoding: str = "full",
        decoding_depths: Optional[List[int]] = None,
    ):
        """
        Initialize the column-wise GEMV processor.

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
            Decoding depth for each column (0 to M-1). If None, uses M-1 for all columns.
            Only used when decoding='adaptive_depth'.
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

        # Set up decoding depths for each column
        if decoding_depths is None:
            # Default to full depth (M-1) for all columns
            self.decoding_depths = [M - 1] * self.original_n
        else:
            if len(decoding_depths) != self.original_n:
                raise ValueError(
                    f"decoding_depths length ({len(decoding_depths)}) must match number of columns ({self.original_n})"
                )
            # Validate decoding depths
            for i, depth in enumerate(decoding_depths):
                if depth < 0 or depth >= M:
                    raise ValueError(f"decoding_depths[{i}] = {depth} must be between 0 and {M-1}")
            self.decoding_depths = decoding_depths.copy()

        # Initialize blocking strategy
        self.blocking = BlockingStrategy(lattice_type)

        # Setup lattice parameters
        self.G, self.Q_nn = self._setup_lattice(lattice_type)
        self.dimension = self.G.shape[0]

        # Pad matrix for column-wise processing
        self.matrix = self.blocking.pad_matrix_for_column_wise(matrix)
        self.m, self.n = self.matrix.shape

        # Initialize quantizers for each block
        self.quantizers = {}
        self.encoded_columns = {}
        self.overload_scalings = {}

        # Pre-encode matrix columns
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
        """Encode all matrix columns using blocking strategy."""
        # Get column blocks
        col_blocks = self.blocking.get_block_indices(self.n)

        for block_idx, (start_col, end_col) in enumerate(col_blocks):
            # Get matrix block
            matrix_block = self.matrix[:, start_col:end_col]

            # Create quantizer for this block
            dither = np.zeros(self.dimension)
            self.quantizers[block_idx] = HierarchicalNestedLatticeQuantizer(
                G=self.G,
                Q_nn=self.Q_nn,
                q=self.q,
                beta=self.beta,
                alpha=self.alpha,
                eps=self.eps,
                dither=dither,
                M=self.M,
                decoding=self.decoding,
            )

            # Encode each column in the block
            self.encoded_columns[block_idx] = []
            self.overload_scalings[block_idx] = []

            for col_idx in range(matrix_block.shape[1]):
                column = matrix_block[:, col_idx]

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
                column_scalings = []
                for chunk in column_chunks:
                    encoding, scaling = self.quantizers[block_idx].encode(chunk, with_dither=False)
                    column_encodings.append(encoding)
                    column_scalings.append(scaling)

                self.encoded_columns[block_idx].append(column_encodings)
                self.overload_scalings[block_idx].append(column_scalings)

    def multiply(self, vector: np.ndarray) -> np.ndarray:
        """
        Perform column-wise matrix-vector multiplication.

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

        # Get vector blocks
        vector_blocks = self.blocking.get_vector_blocks(padded_vector)
        col_blocks = self.blocking.get_block_indices(self.n)

        # Process each block
        for block_idx, (start_col, end_col) in enumerate(col_blocks):
            vector_block = vector_blocks[block_idx]

            # Process each column in the block
            for col_idx, weight in enumerate(vector_block):
                if abs(weight) > 1e-10:  # Check for non-zero
                    # Decode all chunks of the column
                    decoded_chunks = []
                    for chunk_idx in range(len(self.encoded_columns[block_idx][col_idx])):
                        decoded_chunk = self.quantizers[block_idx].get_default_decoding(
                            self.encoded_columns[block_idx][col_idx][chunk_idx],
                            self.overload_scalings[block_idx][col_idx][chunk_idx],
                            with_dither=False,
                        )
                        decoded_chunks.append(decoded_chunk)

                    # Concatenate all chunks to reconstruct the full column
                    decoded_column = np.concatenate(decoded_chunks)

                    # Unpad decoded column to match padded result length
                    if len(decoded_column) > len(result):
                        decoded_column = decoded_column[: len(result)]

                    # Add weighted column to result
                    result += weight * decoded_column

        # Unpad result to original matrix rows
        result = self.blocking.unpad_vector(result, self.original_m)

        return result

    def multiply_with_sparsity(
        self, vector: np.ndarray, sparsity_pattern: Optional[List[int]] = None
    ) -> np.ndarray:
        """
        Perform column-wise matrix-vector multiplication with sparsity support.

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

        # Group sparse indices by blocks
        col_blocks = self.blocking.get_block_indices(self.n)
        block_sparsity = {}

        for idx in sparsity_pattern:
            block_idx = idx // self.blocking.block_size
            if block_idx not in block_sparsity:
                block_sparsity[block_idx] = []
            block_sparsity[block_idx].append(idx % self.blocking.block_size)

        # Process only non-zero blocks
        for block_idx, local_indices in block_sparsity.items():
            for local_idx in local_indices:
                global_idx = block_idx * self.blocking.block_size + local_idx
                weight = padded_vector[global_idx]

                # Decode all chunks of the column
                decoded_chunks = []
                for chunk_idx in range(len(self.encoded_columns[block_idx][local_idx])):
                    decoded_chunk = self.quantizers[block_idx].decode(
                        self.encoded_columns[block_idx][local_idx][chunk_idx],
                        self.overload_scalings[block_idx][local_idx][chunk_idx],
                        with_dither=False,
                    )
                    decoded_chunks.append(decoded_chunk)

                # Concatenate all chunks to reconstruct the full column
                decoded_column = np.concatenate(decoded_chunks)

                # Unpad decoded column to match padded result length
                if len(decoded_column) > len(result):
                    decoded_column = decoded_column[: len(result)]

                # Add weighted column to result
                result += weight * decoded_column

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
        Perform column-wise matrix-vector multiplication with coarse-to-fine decoding.

        This method allows decoding at different levels of detail, where higher
        max_level means coarser reconstruction (less detail).

        Parameters:
        -----------
        vector : np.ndarray
            Input vector x.
        max_level : int, optional
            Maximum level to decode up to (0 <= max_level < M).
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

        # Get vector blocks
        vector_blocks = self.blocking.get_vector_blocks(padded_vector)
        col_blocks = self.blocking.get_block_indices(self.n)

        # Process each block
        for block_idx, (start_col, end_col) in enumerate(col_blocks):
            vector_block = vector_blocks[block_idx]

            # Process each column in the block
            for col_idx, weight in enumerate(vector_block):
                if abs(weight) > 1e-10:  # Check for non-zero
                    # Decode all chunks of the column at specified level
                    decoded_chunks = []
                    for chunk_idx in range(len(self.encoded_columns[block_idx][col_idx])):
                        decoded_chunk = self.quantizers[block_idx].decode_coarse_to_fine(
                            self.encoded_columns[block_idx][col_idx][chunk_idx],
                            self.overload_scalings[block_idx][col_idx][chunk_idx],
                            with_dither=False,
                            max_level=max_level,
                        )
                        decoded_chunks.append(decoded_chunk)

                    # Concatenate all chunks to reconstruct the full column
                    decoded_column = np.concatenate(decoded_chunks)

                    # Unpad decoded column to match padded result length
                    if len(decoded_column) > len(result):
                        decoded_column = decoded_column[: len(result)]

                    # Add weighted column to result
                    result += weight * decoded_column

        # Unpad result to original matrix rows
        result = self.blocking.unpad_vector(result, self.original_m)

        return result

    def multiply_progressive(
        self, vector: np.ndarray, sparsity_pattern: Optional[List[int]] = None
    ) -> List[np.ndarray]:
        """
        Perform column-wise matrix-vector multiplication with progressive refinement.

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

        # Get vector blocks
        vector_blocks = self.blocking.get_vector_blocks(padded_vector)
        col_blocks = self.blocking.get_block_indices(self.n)

        # Process each block
        for block_idx, (start_col, end_col) in enumerate(col_blocks):
            vector_block = vector_blocks[block_idx]

            # Process each column in the block
            for col_idx, weight in enumerate(vector_block):
                if abs(weight) > 1e-10:  # Check for non-zero
                    # Get progressive reconstructions for all chunks
                    progressive_chunks = []
                    for chunk_idx in range(len(self.encoded_columns[block_idx][col_idx])):
                        chunk_reconstructions = self.quantizers[block_idx].decode_progressive(
                            self.encoded_columns[block_idx][col_idx][chunk_idx],
                            self.overload_scalings[block_idx][col_idx][chunk_idx],
                            with_dither=False,
                        )
                        progressive_chunks.append(chunk_reconstructions)

                    # For each level, concatenate chunks and add to result
                    for level_idx in range(self.M):
                        decoded_chunks = [
                            chunk_reconstructions[level_idx]
                            for chunk_reconstructions in progressive_chunks
                        ]
                        decoded_column = np.concatenate(decoded_chunks)

                        # Unpad decoded column to match padded result length
                        if len(decoded_column) > len(results[level_idx]):
                            decoded_column = decoded_column[: len(results[level_idx])]

                        # Add weighted column to result for this level
                        results[level_idx] += weight * decoded_column

        # Unpad all results to original matrix rows
        results = [self.blocking.unpad_vector(result, self.original_m) for result in results]

        return results

    def multiply_with_adaptive_depths(
        self, vector: np.ndarray, decoding_depths: Optional[List[int]] = None
    ) -> np.ndarray:
        """
        Perform column-wise matrix-vector multiplication with adaptive decoding depths.

        This method allows specifying different decoding depths for each column,
        providing fine-grained control over the reconstruction quality vs speed tradeoff.

        Parameters:
        -----------
        vector : np.ndarray
            Input vector x.
        decoding_depths : List[int], optional
            Decoding depth for each column (0 to M-1). If None, uses self.decoding_depths.
            Higher depth means finer reconstruction.

        Returns:
        --------
        np.ndarray
            Result vector y = Wx with adaptive decoding depths.
        """
        if len(vector) != self.original_n:
            raise ValueError(f"Vector dimension {len(vector)} != matrix columns {self.original_n}")

        # Use provided depths or default depths
        if decoding_depths is None:
            depths = self.decoding_depths
        else:
            if len(decoding_depths) != self.original_n:
                raise ValueError(
                    f"decoding_depths length ({len(decoding_depths)}) must match vector length ({self.original_n})"
                )
            depths = decoding_depths

        # Pad vector to match padded matrix
        padded_vector = self.blocking.pad_vector(vector)

        # Initialize result vector (will be unpadded later)
        result = np.zeros(self.m)

        # Get vector blocks
        vector_blocks = self.blocking.get_vector_blocks(padded_vector)
        col_blocks = self.blocking.get_block_indices(self.n)

        # Process each block
        for block_idx, (start_col, end_col) in enumerate(col_blocks):
            vector_block = vector_blocks[block_idx]

            # Process each column in the block
            for col_idx, weight in enumerate(vector_block):
                if abs(weight) > 1e-10:  # Check for non-zero
                    # Get the global column index
                    global_col_idx = start_col + col_idx
                    if global_col_idx < len(depths):
                        depth = depths[global_col_idx]
                    else:
                        depth = self.M - 1  # Default to full depth

                    # Decode all chunks of the column with specified depth
                    decoded_chunks = []
                    for chunk_idx in range(len(self.encoded_columns[block_idx][col_idx])):
                        decoded_chunk = self.quantizers[block_idx].decode_with_depth(
                            self.encoded_columns[block_idx][col_idx][chunk_idx],
                            self.overload_scalings[block_idx][col_idx][chunk_idx],
                            with_dither=False,
                            depth=depth,
                        )
                        decoded_chunks.append(decoded_chunk)

                    # Concatenate all chunks to reconstruct the full column
                    decoded_column = np.concatenate(decoded_chunks)

                    # Unpad decoded column to match padded result length
                    if len(decoded_column) > len(result):
                        decoded_column = decoded_column[: len(result)]

                    # Add weighted column to result
                    result += weight * decoded_column

        # Unpad result to original matrix rows
        result = self.blocking.unpad_vector(result, self.original_m)

        return result

    def multiply_with_lookup(
        self, vector: np.ndarray, lookup_tables: Optional[Dict] = None
    ) -> np.ndarray:
        """
        Perform column-wise matrix-vector multiplication using lookup tables.

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

        # Get vector blocks
        vector_blocks = self.blocking.get_vector_blocks(padded_vector)
        col_blocks = self.blocking.get_block_indices(self.n)

        # Process each block
        for block_idx, (start_col, end_col) in enumerate(col_blocks):
            vector_block = vector_blocks[block_idx]

            # Process each column in the block
            for col_idx, weight in enumerate(vector_block):
                if abs(weight) > 1e-10:  # Check for non-zero
                    # Estimate column contribution using lookup tables if available
                    if lookup_tables is not None and block_idx in lookup_tables:
                        column_contribution = self._estimate_column_contribution_with_lookup(
                            self.encoded_columns[block_idx][col_idx],
                            weight,
                            lookup_tables[block_idx],
                        )
                    else:
                        # Fall back to regular decoding
                        decoded_chunks = []
                        for chunk_idx in range(len(self.encoded_columns[block_idx][col_idx])):
                            decoded_chunk = self.quantizers[block_idx].decode(
                                self.encoded_columns[block_idx][col_idx][chunk_idx],
                                self.overload_scalings[block_idx][col_idx][chunk_idx],
                                with_dither=False,
                            )
                            decoded_chunks.append(decoded_chunk)

                        # Concatenate all chunks to reconstruct the full column
                        decoded_column = np.concatenate(decoded_chunks)

                        # Unpad decoded column to match padded result length
                        if len(decoded_column) > len(result):
                            decoded_column = decoded_column[: len(result)]

                        # Compute weighted column contribution
                        column_contribution = weight * decoded_column

                    # Add weighted column to result
                    result += column_contribution

        # Unpad result to original matrix rows
        result = self.blocking.unpad_vector(result, self.original_m)

        return result

    def _estimate_column_contribution_with_lookup(
        self, encoding: Tuple, weight: float, lookup_table: Dict
    ) -> np.ndarray:
        """
        Estimate column contribution using lookup table.

        Parameters:
        -----------
        encoding : tuple
            Encoded column representation.
        weight : float
            Weight from input vector.
        lookup_table : dict
            Lookup table for column contribution estimation.

        Returns:
        --------
        np.ndarray
            Estimated column contribution.
        """
        # This is a simplified implementation
        # In practice, this would use the full hierarchical lookup table computation
        # from the paper for more accurate estimation

        # For now, decode and compute exact column contribution
        decoded_chunks = []
        for chunk_encoding in encoding:
            decoded_chunk = self.quantizers[0].decode(chunk_encoding, 1, with_dither=False)
            decoded_chunks.append(decoded_chunk)

        # Concatenate all chunks to reconstruct the full column
        decoded_column = np.concatenate(decoded_chunks)

        return weight * decoded_column

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
        for block_idx in self.encoded_columns:
            for column_encodings in self.encoded_columns[block_idx]:
                for chunk_encoding in column_encodings:
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
            for block_encodings in self.encoded_columns.values()
            for column_encodings in block_encodings
            for chunk_encoding in column_encodings
        )

        # Calculate quantizer storage
        quantizer_size = len(self.quantizers) * self.dimension * 8  # Assuming 8 bytes per element

        return {
            "encoded_columns_mb": encoded_size * 4 / (1024 * 1024),
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


def column_wise_gemv(
    matrix: np.ndarray,
    vector: np.ndarray,
    lattice_type: str = "D4",
    M: int = 2,
    sparsity_pattern: Optional[List[int]] = None,
    use_lookup: bool = False,
) -> np.ndarray:
    """
    Perform column-wise matrix-vector multiplication.

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

    Returns:
    --------
    np.ndarray
        Result vector y = Wx.
    """
    processor = ColumnWiseGEMV(matrix, lattice_type, M)

    if use_lookup:
        return processor.multiply_with_lookup(vector)
    elif sparsity_pattern is not None:
        return processor.multiply_with_sparsity(vector, sparsity_pattern)
    else:
        return processor.multiply(vector)
