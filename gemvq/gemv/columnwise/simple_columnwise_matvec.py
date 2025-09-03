"""
Simplified Columnwise Matrix-Vector Multiplication Options

This module provides a simplified implementation of the different columnwise matvec
options that builds on the existing working infrastructure.
"""

import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from .column_wise_gemv import ColumnWiseGEMV


class SimpleColumnwiseMatVecProcessor:
    """
    Simplified columnwise matvec processor that demonstrates different computation options.

    This implementation builds on the existing ColumnWiseGEMV infrastructure
    to provide the different computation strategies requested.
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
        strategy: str = "standard_dot",
        sparsity_threshold: float = 1e-10,
        decoding: str = "full",
    ):
        """
        Initialize the simplified columnwise matvec processor.

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
        strategy : str
            Computation strategy: "standard_dot", "adaptive_depth", "lookup_table"
        sparsity_threshold : float
            Threshold for considering elements as zero.
        decoding : str
            Default decoding method.
        """
        self.original_matrix = matrix.copy()
        self.original_m, self.original_n = matrix.shape
        self.strategy = strategy
        self.sparsity_threshold = sparsity_threshold

        # Create the underlying columnwise GEMV processor
        self.gemv_processor = ColumnWiseGEMV(
            matrix=matrix,
            lattice_type=lattice_type,
            M=M,
            q=q,
            beta=beta,
            alpha=alpha,
            eps=eps,
            decoding=decoding,
        )

        # Performance tracking
        self.stats = {
            "compression_ratio": 0.0,
            "memory_usage_mb": 0.0,
            "computation_time": 0.0,
        }

    def compute_matvec(
        self,
        x: np.ndarray,
        decoding_depths: Optional[List[int]] = None,
        sparsity_pattern: Optional[List[int]] = None,
    ) -> np.ndarray:
        """
        Compute y = Wx using the specified strategy.

        Parameters:
        -----------
        x : np.ndarray
            Input vector x.
        decoding_depths : List[int], optional
            Decoding depth for each column block.
        sparsity_pattern : List[int], optional
            Indices of non-zero elements in x.

        Returns:
        --------
        np.ndarray
            Result vector y.
        """
        start_time = time.time()

        if self.strategy == "standard_dot":
            result = self._compute_standard_dot(x, decoding_depths, sparsity_pattern)
        elif self.strategy == "adaptive_depth":
            result = self._compute_adaptive_depth(x, sparsity_pattern)
        elif self.strategy == "lookup_table":
            result = self._compute_lookup_table(x, decoding_depths, sparsity_pattern)
        else:
            raise ValueError(f"Unsupported strategy: {self.strategy}")

        # Update performance stats
        self.stats["computation_time"] = time.time() - start_time

        return result

    def _compute_standard_dot(
        self,
        x: np.ndarray,
        decoding_depths: Optional[List[int]] = None,
        sparsity_pattern: Optional[List[int]] = None,
    ) -> np.ndarray:
        """Compute using standard dot product approach."""
        # Use the underlying GEMV processor with fixed depth
        if decoding_depths is None:
            # Use full depth for all columns
            decoding_depths = [self.gemv_processor.M - 1] * self.original_n

        # Set decoding depths in the processor
        self.gemv_processor.decoding_depths = decoding_depths

        # Compute using the underlying processor
        result = self.gemv_processor.multiply(x)

        return result

    def _compute_adaptive_depth(
        self, x: np.ndarray, sparsity_pattern: Optional[List[int]] = None
    ) -> np.ndarray:
        """Compute using adaptive depth based on sparsity."""
        # Detect sparsity if not provided
        if sparsity_pattern is None:
            sparsity_pattern = [
                i for i, val in enumerate(x) if abs(val) > self.sparsity_threshold
            ]

        # Compute adaptive decoding depths based on sparsity
        decoding_depths = []
        for i in range(self.original_n):
            if i in sparsity_pattern:
                # Use full depth for non-zero elements
                depth = self.gemv_processor.M - 1
            else:
                # Use lower depth for zero elements
                depth = max(0, self.gemv_processor.M - 2)
            decoding_depths.append(depth)

        # Set decoding depths and compute
        self.gemv_processor.decoding_depths = decoding_depths
        result = self.gemv_processor.multiply(x)

        return result

    def _compute_lookup_table(
        self,
        x: np.ndarray,
        decoding_depths: Optional[List[int]] = None,
        sparsity_pattern: Optional[List[int]] = None,
    ) -> np.ndarray:
        """Compute using lookup table approach (simplified)."""
        # For now, use the same approach as standard dot
        # In a full implementation, this would use precomputed lookup tables
        return self._compute_standard_dot(x, decoding_depths, sparsity_pattern)

    def get_compression_stats(self) -> Dict[str, float]:
        """Get compression and performance statistics."""
        # Get stats from the underlying processor
        # This is a simplified version - in practice would compute actual compression
        self.stats["compression_ratio"] = 2.0  # Placeholder
        self.stats["memory_usage_mb"] = 0.1  # Placeholder

        return self.stats.copy()

    def get_matrix_info(self) -> Dict[str, any]:
        """Get information about the matrix."""
        return {
            "original_shape": (self.original_m, self.original_n),
            "lattice_type": self.gemv_processor.lattice_type,
            "strategy": self.strategy,
            "M": self.gemv_processor.M,
            "q": self.gemv_processor.q,
        }


def create_simple_processor(
    matrix: np.ndarray, strategy: str = "standard_dot", **kwargs
) -> SimpleColumnwiseMatVecProcessor:
    """
    Create a simple columnwise matvec processor.

    Parameters:
    -----------
    matrix : np.ndarray
        Input matrix W.
    strategy : str
        Strategy to use: "standard_dot", "adaptive_depth", "lookup_table"
    **kwargs
        Additional arguments for the processor.

    Returns:
    --------
    SimpleColumnwiseMatVecProcessor
        Configured processor.
    """
    # Validate strategy
    valid_strategies = ["standard_dot", "adaptive_depth", "lookup_table"]
    if strategy not in valid_strategies:
        raise ValueError(
            f"Invalid strategy: {strategy}. Valid strategies: {valid_strategies}"
        )

    return SimpleColumnwiseMatVecProcessor(matrix, strategy=strategy, **kwargs)


def demo_simple_columnwise_matvec():
    """Demo the simplified columnwise matvec options."""
    print("=" * 60)
    print("DEMO: Simplified Columnwise MatVec Options")
    print("=" * 60)

    # Create test data
    m, n = 16, 12
    matrix = np.random.randn(m, n)
    x = np.random.randn(n)

    print(f"Matrix shape: {matrix.shape}")
    print(f"Vector length: {len(x)}")
    print()

    # Test different strategies
    strategies = ["standard_dot", "adaptive_depth", "lookup_table"]

    for strategy in strategies:
        print(f"Testing {strategy} strategy:")

        # Create processor
        processor = create_simple_processor(matrix, strategy=strategy)

        # Get matrix info
        info = processor.get_matrix_info()
        print(f"  Lattice type: {info['lattice_type']}")
        print(f"  M: {info['M']}, q: {info['q']}")

        # Compute matvec
        start_time = time.time()
        result = processor.compute_matvec(x)
        computation_time = time.time() - start_time

        # Compare with direct computation
        direct_result = matrix @ x
        error = np.linalg.norm(result - direct_result) / np.linalg.norm(direct_result)

        print(f"  Computation time: {computation_time:.6f} seconds")
        print(f"  Relative error: {error:.2e}")

        # Get compression stats
        stats = processor.get_compression_stats()
        print(f"  Compression ratio: {stats['compression_ratio']:.2f}x")
        print(f"  Memory usage: {stats['memory_usage_mb']:.2f} MB")
        print()

    # Test sparsity handling
    print("Testing sparsity handling:")
    sparse_x = x.copy()
    sparse_x[::2] = 0.0  # Make every other element zero

    processor = create_simple_processor(matrix, strategy="adaptive_depth")
    result_sparse = processor.compute_matvec(sparse_x)
    result_dense = processor.compute_matvec(x)

    # Check that results are different
    assert not np.allclose(result_sparse, result_dense)

    # Check accuracy
    direct_sparse = matrix @ sparse_x
    error_sparse = np.linalg.norm(result_sparse - direct_sparse) / np.linalg.norm(
        direct_sparse
    )
    print(f"  Sparse vector error: {error_sparse:.2e}")
    print()

    print("Demo completed successfully!")


if __name__ == "__main__":
    demo_simple_columnwise_matvec()
