"""
Unified Lattice-Quantized General Matrix-Vector Multiplication (GEMV)

This module provides a unified interface for both column-wise and row-wise
matrix-vector multiplication using lattice quantization with blocking strategies.
It allows users to choose the most appropriate approach based on their specific
requirements and matrix characteristics.
"""

from typing import Dict, List, Literal, Optional, Tuple

import numpy as np

from .column_wise_gemv import ColumnWiseGEMV
from .padder import BlockingStrategy
from .row_wise_gemv import RowWiseGEMV


class LatticeQuantizedGEMV:
    """
    Unified interface for lattice-quantized matrix-vector multiplication.

    This class provides both column-wise and row-wise approaches for matrix-vector
    multiplication using lattice quantization, with automatic selection based on
    matrix characteristics and user preferences.
    """

    def __init__(
        self,
        matrix: np.ndarray,
        approach: Literal["column", "row", "auto"] = "auto",
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
        Initialize the unified GEMV processor.

        Parameters:
        -----------
        matrix : np.ndarray
            Input matrix W (m x n).
        approach : str
            Approach to use ('column', 'row', or 'auto' for automatic selection).
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
            Decoding depths for each column (column-wise) or row (row-wise).
            If None, uses M-1 for all vectors.
        """
        self.matrix = matrix
        self.m, self.n = matrix.shape
        self.approach = approach
        self.lattice_type = lattice_type
        self.M = M
        self.alpha = alpha
        self.eps = eps
        self.q = q
        self.beta = beta
        self.decoding = decoding
        self.decoding_depths = decoding_depths

        # Initialize blocking strategy
        self.blocking = BlockingStrategy(lattice_type)

        # Determine approach
        if approach == "auto":
            self.approach = self._select_optimal_approach()

        # Initialize appropriate processor
        if self.approach == "column":
            self.processor = ColumnWiseGEMV(
                matrix, lattice_type, M, alpha, eps, q, beta, decoding, decoding_depths
            )
        else:  # row
            self.processor = RowWiseGEMV(
                matrix, lattice_type, M, alpha, eps, q, beta, decoding, decoding_depths
            )

    def _select_optimal_approach(self) -> Literal["column", "row"]:
        """
        Automatically select the optimal approach based on matrix characteristics.

        Returns:
        --------
        str
            Selected approach ('column' or 'row').
        """
        # Simple heuristic: use column-wise for tall matrices, row-wise for wide matrices
        aspect_ratio = self.m / self.n

        if aspect_ratio > 1.5:  # Tall matrix
            return "column"
        elif aspect_ratio < 0.67:  # Wide matrix
            return "row"
        else:  # Square-ish matrix
            # Default to column-wise for better sparsity handling
            return "column"

    def multiply(
        self,
        vector: np.ndarray,
        sparsity_pattern: Optional[List[int]] = None,
        use_lookup: bool = False,
    ) -> np.ndarray:
        """
        Perform matrix-vector multiplication using the selected approach.

        Parameters:
        -----------
        vector : np.ndarray
            Input vector x.
        sparsity_pattern : List[int], optional
            Indices of non-zero elements in the vector.
        use_lookup : bool
            Whether to use lookup tables for computation (row-wise only).

        Returns:
        --------
        np.ndarray
            Result vector y = Wx.
        """
        if self.approach == "column":
            if sparsity_pattern is not None:
                return self.processor.multiply_with_sparsity(vector, sparsity_pattern)
            else:
                return self.processor.multiply(vector)
        else:  # row
            if use_lookup:
                return self.processor.multiply_with_lookup(vector)
            elif sparsity_pattern is not None:
                return self.processor.multiply_with_sparsity(vector, sparsity_pattern)
            else:
                return self.processor.multiply(vector)

    def multiply_coarse_to_fine(
        self,
        vector: np.ndarray,
        max_level: int = None,
        sparsity_pattern: Optional[List[int]] = None,
        use_lookup: bool = False,
    ) -> np.ndarray:
        """
        Perform matrix-vector multiplication with coarse-to-fine decoding.

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
        use_lookup : bool
            Whether to use lookup tables for computation (row-wise only).

        Returns:
        --------
        np.ndarray
            Result vector y = Wx at the specified level of detail.
        """
        if self.approach == "column":
            return self.processor.multiply_coarse_to_fine(vector, max_level, sparsity_pattern)
        else:  # row
            if use_lookup:
                # For row-wise with lookup, we need to fall back to regular multiply
                # since lookup tables don't support coarse-to-fine decoding
                return self.processor.multiply_with_lookup(vector)
            else:
                return self.processor.multiply_coarse_to_fine(vector, max_level, sparsity_pattern)

    def multiply_progressive(
        self,
        vector: np.ndarray,
        sparsity_pattern: Optional[List[int]] = None,
        use_lookup: bool = False,
    ) -> List[np.ndarray]:
        """
        Perform matrix-vector multiplication with progressive refinement.

        This method returns a list of results at each level of detail,
        from coarsest to finest.

        Parameters:
        -----------
        vector : np.ndarray
            Input vector x.
        sparsity_pattern : List[int], optional
            Indices of non-zero elements in the vector.
        use_lookup : bool
            Whether to use lookup tables for computation (row-wise only).

        Returns:
        --------
        List[np.ndarray]
            List of result vectors at each level of detail, from coarsest to finest.
        """
        if self.approach == "column":
            return self.processor.multiply_progressive(vector, sparsity_pattern)
        else:  # row
            if use_lookup:
                # For row-wise with lookup, we need to fall back to regular multiply
                # since lookup tables don't support progressive decoding
                result = self.processor.multiply_with_lookup(vector)
                return [result]  # Return single result as list
            else:
                return self.processor.multiply_progressive(vector, sparsity_pattern)

    def multiply_with_adaptive_depths(
        self, vector: np.ndarray, decoding_depths: Optional[List[int]] = None
    ) -> np.ndarray:
        """
        Perform matrix-vector multiplication with adaptive decoding depths.

        Parameters:
        -----------
        vector : np.ndarray
            Input vector x.
        decoding_depths : List[int], optional
            Decoding depths for each column (column-wise) or row (row-wise).
            If None, uses the processor's default depths.

        Returns:
        --------
        np.ndarray
            Result vector y = Wx with adaptive depths.
        """
        if hasattr(self.processor, "multiply_with_adaptive_depths"):
            return self.processor.multiply_with_adaptive_depths(vector, decoding_depths)
        else:
            # Fall back to regular multiplication
            return self.processor.multiply(vector)

    def get_compression_ratio(self) -> float:
        """
        Calculate compression ratio achieved by the encoding.

        Returns:
        --------
        float
            Compression ratio (original bits / encoded bits).
        """
        return self.processor.get_compression_ratio()

    def get_memory_usage(self) -> Dict[str, float]:
        """
        Get memory usage statistics.

        Returns:
        --------
        Dict
            Dictionary with memory usage information.
        """
        return self.processor.get_memory_usage()

    def get_blocking_info(self) -> Dict:
        """
        Get information about the blocking strategy used.

        Returns:
        --------
        Dict
            Dictionary with blocking information.
        """
        return self.processor.get_blocking_info()

    def get_approach_info(self) -> Dict:
        """
        Get information about the selected approach.

        Returns:
        --------
        Dict
            Dictionary with approach information.
        """
        return {
            "selected_approach": self.approach,
            "matrix_shape": (self.m, self.n),
            "aspect_ratio": self.m / self.n,
            "lattice_type": self.lattice_type,
            "block_size": self.blocking.block_size,
        }

    def compare_approaches(
        self, vector: np.ndarray, sparsity_pattern: Optional[List[int]] = None
    ) -> Dict:
        """
        Compare column-wise and row-wise approaches for the given vector.

        Parameters:
        -----------
        vector : np.ndarray
            Input vector for comparison.
        sparsity_pattern : List[int], optional
            Indices of non-zero elements in the vector.

        Returns:
        --------
        Dict
            Dictionary with comparison results.
        """
        import time

        # Test column-wise approach
        col_processor = ColumnWiseGEMV(
            self.matrix, self.lattice_type, self.M, self.alpha, self.eps, self.q, self.beta
        )

        start_time = time.time()
        if sparsity_pattern is not None:
            col_result = col_processor.multiply_with_sparsity(vector, sparsity_pattern)
        else:
            col_result = col_processor.multiply(vector)
        col_time = time.time() - start_time

        # Test row-wise approach
        row_processor = RowWiseGEMV(
            self.matrix, self.lattice_type, self.M, self.alpha, self.eps, self.q, self.beta
        )

        start_time = time.time()
        if sparsity_pattern is not None:
            row_result = row_processor.multiply_with_sparsity(vector, sparsity_pattern)
        else:
            row_result = row_processor.multiply(vector)
        row_time = time.time() - start_time

        # Compare results
        col_error = np.linalg.norm(col_result - row_result) / np.linalg.norm(row_result)

        return {
            "column_wise": {
                "time": col_time,
                "compression_ratio": col_processor.get_compression_ratio(),
                "memory_usage": col_processor.get_memory_usage(),
            },
            "row_wise": {
                "time": row_time,
                "compression_ratio": row_processor.get_compression_ratio(),
                "memory_usage": row_processor.get_memory_usage(),
            },
            "result_difference": col_error,
            "recommended_approach": "column" if col_time < row_time else "row",
        }


def lattice_quantized_gemv(
    matrix: np.ndarray,
    vector: np.ndarray,
    approach: Literal["column", "row", "auto"] = "auto",
    lattice_type: str = "D4",
    M: int = 2,
    sparsity_pattern: Optional[List[int]] = None,
    use_lookup: bool = False,
) -> np.ndarray:
    """
    Perform lattice-quantized matrix-vector multiplication.

    Parameters:
    -----------
    matrix : np.ndarray
        Input matrix W.
    vector : np.ndarray
        Input vector x.
    approach : str
        Approach to use ('column', 'row', or 'auto').
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
    processor = LatticeQuantizedGEMV(matrix, approach, lattice_type, M)

    return processor.multiply(vector, sparsity_pattern, use_lookup)


def compare_gemv_approaches(
    matrix: np.ndarray,
    vector: np.ndarray,
    lattice_type: str = "D4",
    M: int = 2,
    sparsity_pattern: Optional[List[int]] = None,
) -> Dict:
    """
    Compare column-wise and row-wise approaches for matrix-vector multiplication.

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

    Returns:
    --------
    Dict
        Dictionary with comparison results.
    """
    processor = LatticeQuantizedGEMV(matrix, "auto", lattice_type, M)
    return processor.compare_approaches(vector, sparsity_pattern)


# Example usage and testing functions
def example_usage():
    """Example usage of the unified GEMV interface."""
    # Create test matrix and vector
    m, n = 100, 50
    matrix = np.random.randn(m, n)
    vector = np.random.randn(n)

    # Make vector sparse (only 10 non-zero elements)
    sparsity_pattern = np.random.choice(n, 10, replace=False)
    sparse_vector = np.zeros(n)
    sparse_vector[sparsity_pattern] = vector[sparsity_pattern]

    # Test unified interface
    processor = LatticeQuantizedGEMV(matrix, "auto", "D4", 2)
    result = processor.multiply(sparse_vector, sparsity_pattern)

    # Compare with exact computation
    exact_result = matrix @ sparse_vector
    error = np.linalg.norm(result - exact_result) / np.linalg.norm(exact_result)

    print(f"Approach used: {processor.approach}")
    print(f"Relative error: {error:.6f}")
    print(f"Compression ratio: {processor.get_compression_ratio():.2f}")
    print(f"Memory usage: {processor.get_memory_usage()}")

    return result, exact_result, error


if __name__ == "__main__":
    # Run example
    result, exact_result, error = example_usage()
    print("Example completed successfully!")
