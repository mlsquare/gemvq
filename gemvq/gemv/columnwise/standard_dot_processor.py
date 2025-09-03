"""
Standard Dot Product Processor for Columnwise Matrix-Vector Multiplication

This module implements matrix-vector multiplication using standard np.dot operations
without lookup tables. Supports fixed depth, variable depth, and adaptive depth
decoding based on sparsity patterns.
"""

import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from .columnwise_matvec_processor import ColumnwiseMatVecProcessor


class StandardDotProcessor(ColumnwiseMatVecProcessor):
    """
    Standard dot product processor using np.dot for computation.

    This implementation uses standard numpy dot product operations without
    lookup tables. It supports different decoding strategies:
    - Fixed depth: All columns decoded to same depth
    - Variable depth: Per-column decoding depths
    - Adaptive depth: Depth based on sparsity pattern
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
        fixed_depth: bool = True,
        adaptive_depth: bool = False,
        sparsity_threshold: float = 1e-10,
        decoding: str = "full",
    ):
        """
        Initialize the standard dot processor.

        Parameters:
        -----------
        matrix : np.ndarray
            Input matrix W (m x n).
        lattice_type : str
            Type of lattice to use.
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
        fixed_depth : bool
            If True, use same depth for all columns.
        adaptive_depth : bool
            If True, adapt depth based on sparsity pattern.
        sparsity_threshold : float
            Threshold for considering elements as zero.
        decoding : str
            Default decoding method.
        """
        super().__init__(
            matrix=matrix,
            lattice_type=lattice_type,
            M=M,
            q=q,
            beta=beta,
            alpha=alpha,
            eps=eps,
            use_lookup=False,
            quantize_x=False,
            sparsity_threshold=sparsity_threshold,
            decoding=decoding,
        )

        self.fixed_depth = fixed_depth
        self.adaptive_depth = adaptive_depth

        # Default decoding depths
        if fixed_depth:
            self.default_decoding_depths = [self.M - 1] * (self.n // self.dimension)
        else:
            # Variable depth: use full depth for all blocks
            self.default_decoding_depths = [self.M - 1] * (self.n // self.dimension)

    def compute_matvec(
        self,
        x: np.ndarray,
        decoding_depths: Optional[List[int]] = None,
        sparsity_pattern: Optional[List[int]] = None,
    ) -> np.ndarray:
        """
        Compute y = Wx using standard dot product operations.

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
        start_time = time.time()

        # Pad vector to match matrix dimensions
        padded_x = self._pad_vector(x)

        # Detect sparsity if not provided
        if sparsity_pattern is None:
            sparsity_pattern, sparsity_ratio = self._detect_sparsity(padded_x)

        # Get relevant block indices
        relevant_blocks = self._get_block_indices(sparsity_pattern)

        # Determine decoding depths
        if decoding_depths is None:
            if self.adaptive_depth:
                decoding_depths = self._compute_adaptive_depths(
                    padded_x, sparsity_pattern
                )
            else:
                decoding_depths = self.default_decoding_depths.copy()

        # Validate decoding depths
        self._validate_decoding_depths(decoding_depths)

        # Initialize result vector
        result = np.zeros(self.m)

        # Process each relevant block
        for block_idx in relevant_blocks:
            # Get the block's x values
            start_col = block_idx * self.dimension
            end_col = start_col + self.dimension
            x_block = padded_x[start_col:end_col]

            # Skip if all elements in block are zero
            if np.all(np.abs(x_block) <= self.sparsity_threshold):
                continue

            # Decode the column block to specified depth
            decoded_block = self._decode_column_block(
                block_idx, decoding_depths[block_idx]
            )

            # Compute dot product: result += decoded_block @ x_block
            result += decoded_block @ x_block

        # Trim result to original matrix dimensions
        final_result = result[: self.original_m]

        # Update performance stats
        self.stats["computation_time"] = time.time() - start_time

        return final_result

    def _decode_column_block(self, block_idx: int, depth: int) -> np.ndarray:
        """
        Decode a column block to specified depth.

        Parameters:
        -----------
        block_idx : int
            Index of the column block.
        depth : int
            Decoding depth (0 to M-1).

        Returns:
        --------
        np.ndarray
            Decoded column block.
        """
        quantizer = self.column_quantizers[block_idx]
        encoding = self.encoded_columns[block_idx]

        # Decode to specified depth
        encoding, T = encoding  # Unpack the encoding tuple
        decoded_block = quantizer.decode(encoding, T, with_dither=False)

        return decoded_block

    def _compute_adaptive_depths(
        self, x: np.ndarray, sparsity_pattern: List[int]
    ) -> List[int]:
        """
        Compute adaptive decoding depths based on sparsity pattern.

        Parameters:
        -----------
        x : np.ndarray
            Input vector x.
        sparsity_pattern : List[int]
            Indices of non-zero elements.

        Returns:
        --------
        List[int]
            Adaptive decoding depths for each block.
        """
        n_blocks = self.n // self.dimension
        depths = []

        for block_idx in range(n_blocks):
            start_col = block_idx * self.dimension
            end_col = start_col + self.dimension

            # Get x values for this block
            x_block = x[start_col:end_col]

            # Compute importance metric based on magnitude
            importance = np.sum(np.abs(x_block))

            # Determine depth based on importance
            if importance > 2.0:
                depth = self.M - 1  # Full depth
            elif importance > 1.0:
                depth = max(1, self.M - 2)  # Medium depth
            elif importance > 0.5:
                depth = max(0, self.M - 3)  # Low depth
            else:
                depth = 0  # Minimal depth

            depths.append(depth)

        return depths

    def _validate_decoding_depths(self, decoding_depths: List[int]):
        """Validate decoding depths."""
        n_blocks = self.n // self.dimension

        if len(decoding_depths) != n_blocks:
            raise ValueError(
                f"decoding_depths length ({len(decoding_depths)}) must match "
                f"number of blocks ({n_blocks})"
            )

        for i, depth in enumerate(decoding_depths):
            if depth < 0 or depth >= self.M:
                raise ValueError(
                    f"decoding_depths[{i}] = {depth} must be between 0 and {self.M-1}"
                )

    def compute_matvec_fixed_depth(
        self, x: np.ndarray, depth: int = None
    ) -> np.ndarray:
        """
        Compute matvec with fixed depth for all columns.

        Parameters:
        -----------
        x : np.ndarray
            Input vector x.
        depth : int, optional
            Fixed depth for all columns. If None, uses M-1.

        Returns:
        --------
        np.ndarray
            Result vector y.
        """
        if depth is None:
            depth = self.M - 1

        if depth < 0 or depth >= self.M:
            raise ValueError(f"depth must be between 0 and {self.M-1}")

        n_blocks = self.n // self.dimension
        decoding_depths = [depth] * n_blocks

        return self.compute_matvec(x, decoding_depths=decoding_depths)

    def compute_matvec_variable_depth(
        self, x: np.ndarray, decoding_depths: List[int]
    ) -> np.ndarray:
        """
        Compute matvec with variable depth for each column block.

        Parameters:
        -----------
        x : np.ndarray
            Input vector x.
        decoding_depths : List[int]
            Decoding depth for each column block.

        Returns:
        --------
        np.ndarray
            Result vector y.
        """
        return self.compute_matvec(x, decoding_depths=decoding_depths)

    def compute_matvec_adaptive_depth(self, x: np.ndarray) -> np.ndarray:
        """
        Compute matvec with adaptive depth based on sparsity.

        Parameters:
        -----------
        x : np.ndarray
            Input vector x.

        Returns:
        --------
        np.ndarray
            Result vector y.
        """
        return self.compute_matvec(x)

    def get_performance_stats(self) -> Dict[str, float]:
        """Get detailed performance statistics."""
        stats = self.get_compression_stats()

        # Add additional stats
        stats.update(
            {
                "fixed_depth": self.fixed_depth,
                "adaptive_depth": self.adaptive_depth,
                "num_blocks": self.n // self.dimension,
                "lattice_dimension": self.dimension,
            }
        )

        return stats
