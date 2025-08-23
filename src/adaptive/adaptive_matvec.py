"""
Adaptive Matrix-Vector Multiplication with Hierarchical Nested Quantizers

This module implements adaptive matrix-vector multiplication using hierarchical
nested lattice quantizers with column-wise encoding. The approach interprets
matrix-vector multiplication as a linear combination of column vectors, where
each column can have different target bit rates and some vector elements are
known to be zero a priori.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np

from ..quantizers.closest_point import (closest_point_A2, closest_point_Dn,
                                        closest_point_E8)
from ..quantizers.hierarchical_nested_lattice_quantizer import \
    HierarchicalNestedLatticeQuantizer
from ..utils import get_a2, get_d4, get_e8, get_z2, get_z3


class AdaptiveColumnQuantizer:
    """
    Adaptive column quantizer that handles different bit rates for each column.

    This class manages the quantization of individual matrix columns with
    adaptive parameters based on target bit rates. It provides efficient
    encoding and decoding operations for sparse matrix-vector multiplication.
    """

    def __init__(
        self,
        target_rates: List[float],
        lattice_type: str = "D4",
        M: int = 2,
        alpha: float = 1 / 3,
        eps: float = 1e-8,
        decoding: str = "full",
    ):
        """
        Initialize the adaptive column quantizer.

        Parameters:
        -----------
        target_rates : List[float]
            Target bit rates for each column (bits per dimension).
        lattice_type : str
            Type of lattice to use ('D4', 'A2', 'E8', 'Z2', 'Z3').
        M : int
            Number of hierarchical levels.
        alpha : float
            Scaling parameter for overload handling.
        eps : float
            Small perturbation parameter.
        decoding : str, optional
            Default decoding method to use ('full', 'coarse_to_fine', 'progressive').
            Default is 'full'.
        """
        self.target_rates = target_rates
        self.lattice_type = lattice_type
        self.M = M
        self.alpha = alpha
        self.eps = eps
        self.decoding = decoding

        # Setup lattice parameters
        self.G, self.Q_nn = self._setup_lattice(lattice_type)
        self.dimension = self.G.shape[0]

        # Initialize quantizers for each column
        self.quantizers = {}
        self._initialize_quantizers()

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

    def _initialize_quantizers(self):
        """Initialize quantizers for each column with adaptive parameters."""
        for i, rate in enumerate(self.target_rates):
            # Convert bit rate to quantization parameters
            q, beta = self._rate_to_parameters(rate)

            # Create dither vector
            dither = np.zeros(self.dimension)

            # Initialize quantizer
            self.quantizers[i] = HierarchicalNestedLatticeQuantizer(
                G=self.G,
                Q_nn=self.Q_nn,
                q=q,
                beta=beta,
                alpha=self.alpha,
                eps=self.eps,
                dither=dither,
                M=self.M,
                decoding=self.decoding,
            )

    def _rate_to_parameters(self, rate: float) -> Tuple[int, float]:
        """
        Convert target bit rate to quantization parameters.

        Parameters:
        -----------
        rate : float
            Target bit rate in bits per dimension.

        Returns:
        --------
        tuple
            (q, beta) where q is quantization parameter and beta is scaling.
        """
        # Simple parameter mapping - can be optimized based on rate-distortion analysis
        q = max(2, int(2 ** (rate / self.M)))
        beta = 1.0 / (2 ** (rate / 4))  # Scaling based on rate

        return q, beta

    def encode_column(self, column: np.ndarray, col_idx: int) -> Tuple[Tuple, int]:
        """
        Encode a single column with adaptive parameters.

        Parameters:
        -----------
        column : np.ndarray
            Column vector to encode.
        col_idx : int
            Column index for parameter selection.

        Returns:
        --------
        tuple
            (encoding_vectors, overload_scaling) where encoding_vectors is a
            tuple of M encoding vectors and overload_scaling is the scaling factor.
        """
        if col_idx not in self.quantizers:
            raise ValueError(f"No quantizer initialized for column {col_idx}")

        quantizer = self.quantizers[col_idx]
        return quantizer.encode(column, with_dither=False)

    def decode_column(self, encoding: Tuple, col_idx: int, overload_scaling: int) -> np.ndarray:
        """
        Decode a single column with adaptive parameters.

        Parameters:
        -----------
        encoding : tuple
            Tuple of M encoding vectors.
        col_idx : int
            Column index for parameter selection.
        overload_scaling : int
            Scaling factor applied during encoding.

        Returns:
        --------
        np.ndarray
            Decoded column vector.
        """
        if col_idx not in self.quantizers:
            raise ValueError(f"No quantizer initialized for column {col_idx}")

        quantizer = self.quantizers[col_idx]
        return quantizer.decode(encoding, overload_scaling, with_dither=False)


class AdaptiveLookupTable:
    """
    Adaptive lookup table manager for different bit rates.

    This class manages precomputed lookup tables for efficient inner product
    estimation with different quantization parameters.
    """

    def __init__(self, lattice_type: str = "D4", max_rate: float = 8.0, M: int = 2):
        """
        Initialize the adaptive lookup table manager.

        Parameters:
        -----------
        lattice_type : str
            Type of lattice to use.
        max_rate : float
            Maximum bit rate to support.
        M : int
            Number of hierarchical levels.
        """
        self.lattice_type = lattice_type
        self.max_rate = max_rate
        self.M = M

        # Setup lattice
        self.G, self.Q_nn = self._setup_lattice(lattice_type)
        self.dimension = self.G.shape[0]

        # Precompute lookup tables for different rates
        self.lookup_tables = {}
        self._precompute_tables()

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

    def _precompute_tables(self):
        """Precompute lookup tables for different bit rates."""
        # Generate rate points for table precomputation
        rates = np.linspace(1.0, self.max_rate, 20)

        for rate in rates:
            q, _ = self._rate_to_parameters(rate)
            # Precompute lookup table for this rate
            # This is a simplified version - actual implementation would use
            # the precompute_hq_lut function from utils
            self.lookup_tables[rate] = self._create_lookup_table(q)

    def _rate_to_parameters(self, rate: float) -> Tuple[int, float]:
        """Convert bit rate to quantization parameters."""
        q = max(2, int(2 ** (rate / self.M)))
        beta = 1.0 / (2 ** (rate / 4))
        return q, beta

    def _create_lookup_table(self, q: int) -> Dict:
        """Create lookup table for given quantization parameter."""
        # Simplified lookup table creation
        # In practice, this would use the full hierarchical lookup table
        table = {}
        for i in range(q):
            for j in range(q):
                table[(i, j)] = i * j  # Simplified inner product
        return table

    def get_table(self, rate: float) -> Dict:
        """
        Get lookup table for given bit rate.

        Parameters:
        -----------
        rate : float
            Target bit rate.

        Returns:
        --------
        dict
            Lookup table for the specified rate.
        """
        # Find closest precomputed rate
        rates = list(self.lookup_tables.keys())
        closest_rate = min(rates, key=lambda x: abs(x - rate))
        return self.lookup_tables[closest_rate]


class SparseMatVecProcessor:
    """
    Sparse matrix-vector multiplication processor with adaptive quantization.

    This class implements efficient matrix-vector multiplication for sparse
    vectors using adaptive hierarchical nested quantization with column-wise
    encoding.
    """

    def __init__(
        self,
        matrix: np.ndarray,
        target_rates: List[float],
        sparsity_pattern: Optional[List[int]] = None,
        lattice_type: str = "D4",
        M: int = 2,
        decoding: str = "full",
    ):
        """
        Initialize the sparse matrix-vector processor.

        Parameters:
        -----------
        matrix : np.ndarray
            Input matrix A (m x n).
        target_rates : List[float]
            Target bit rates for each column.
        sparsity_pattern : List[int], optional
            Indices of non-zero elements in the vector.
        lattice_type : str
            Type of lattice to use.
        M : int
            Number of hierarchical levels.
        """
        self.matrix = matrix
        self.m, self.n = matrix.shape
        self.target_rates = target_rates
        self.sparsity_pattern = sparsity_pattern or list(range(self.n))
        self.lattice_type = lattice_type
        self.M = M
        self.decoding = decoding

        # Initialize components
        self.column_quantizer = AdaptiveColumnQuantizer(
            target_rates, lattice_type, M, decoding=decoding
        )
        self.lookup_manager = AdaptiveLookupTable(lattice_type, max(target_rates), M)

        # Pre-encode matrix columns
        self.encoded_columns = {}
        self.overload_scalings = {}
        self._encode_matrix()

    def _encode_matrix(self):
        """Encode all matrix columns with adaptive parameters."""
        for i in range(self.n):
            column = self.matrix[:, i]
            encoding, scaling = self.column_quantizer.encode_column(column, i)
            self.encoded_columns[i] = encoding
            self.overload_scalings[i] = scaling

    def compute_matvec(self, sparse_vector: np.ndarray) -> np.ndarray:
        """
        Compute matrix-vector product using encoded columns.

        Parameters:
        -----------
        sparse_vector : np.ndarray
            Sparse input vector x.

        Returns:
        --------
        np.ndarray
            Result vector y = Ax.
        """
        if len(sparse_vector) != self.n:
            raise ValueError(f"Vector dimension {len(sparse_vector)} != matrix columns {self.n}")

        # Initialize result vector
        result = np.zeros(self.m)

        # Process only non-zero elements
        for i in self.sparsity_pattern:
            if abs(sparse_vector[i]) > 1e-10:  # Check for non-zero
                # Decode column
                decoded_column = self.column_quantizer.decode_column(
                    self.encoded_columns[i], i, self.overload_scalings[i]
                )

                # Add weighted column to result
                result += sparse_vector[i] * decoded_column

        return result

    def compute_matvec_with_lookup(self, sparse_vector: np.ndarray) -> np.ndarray:
        """
        Compute matrix-vector product using lookup tables for efficiency.

        Parameters:
        -----------
        sparse_vector : np.ndarray
            Sparse input vector x.

        Returns:
        --------
        np.ndarray
            Result vector y = Ax.
        """
        if len(sparse_vector) != self.n:
            raise ValueError(f"Vector dimension {len(sparse_vector)} != matrix columns {self.n}")

        # Initialize result vector
        result = np.zeros(self.m)

        # Process only non-zero elements using lookup tables
        for i in self.sparsity_pattern:
            if abs(sparse_vector[i]) > 1e-10:
                # Get lookup table for this column's rate
                lookup_table = self.lookup_manager.get_table(self.target_rates[i])

                # Estimate inner product using lookup table
                # This is a simplified version - actual implementation would use
                # the full hierarchical lookup table computation
                estimated_contribution = self._estimate_column_contribution(
                    i, sparse_vector[i], lookup_table
                )
                result += estimated_contribution

        return result

    def _estimate_column_contribution(
        self, col_idx: int, weight: float, lookup_table: Dict
    ) -> np.ndarray:
        """
        Estimate column contribution using lookup table.

        Parameters:
        -----------
        col_idx : int
            Column index.
        weight : float
            Weight from sparse vector.
        lookup_table : dict
            Lookup table for inner product estimation.

        Returns:
        --------
        np.ndarray
            Estimated column contribution.
        """
        # Simplified estimation - in practice, this would use the full
        # hierarchical lookup table computation from the paper
        encoding = self.encoded_columns[col_idx]

        # For now, decode the column and multiply by weight
        # This can be optimized using the actual lookup table computation
        decoded_column = self.column_quantizer.decode_column(
            encoding, col_idx, self.overload_scalings[col_idx]
        )

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
        original_bits = self.m * self.n * 32  # Assuming 32-bit floats

        # Calculate encoded storage
        encoded_bits = 0
        for i in range(self.n):
            encoding = self.encoded_columns[i]
            # Count bits in encoding vectors
            for level_encoding in encoding:
                encoded_bits += len(level_encoding) * np.log2(self.column_quantizer.quantizers[i].q)

        return original_bits / encoded_bits if encoded_bits > 0 else float("inf")

    def get_memory_usage(self) -> Dict[str, float]:
        """
        Get memory usage statistics.

        Returns:
        --------
        dict
            Dictionary with memory usage information.
        """
        # Calculate various memory components
        encoded_size = sum(
            sum(len(level_encoding) for level_encoding in encoding)
            for encoding in self.encoded_columns.values()
        )

        lookup_size = sum(len(table) for table in self.lookup_manager.lookup_tables.values())

        return {
            "encoded_columns_mb": encoded_size * 4 / (1024 * 1024),  # Assuming 4 bytes per element
            "lookup_tables_mb": lookup_size * 8 / (1024 * 1024),  # Assuming 8 bytes per entry
            "total_mb": (encoded_size * 4 + lookup_size * 8) / (1024 * 1024),
        }


def create_adaptive_matvec_processor(
    matrix: np.ndarray,
    target_rates: List[float],
    sparsity_pattern: Optional[List[int]] = None,
    lattice_type: str = "D4",
    M: int = 2,
) -> SparseMatVecProcessor:
    """
    Factory function to create an adaptive matrix-vector processor.

    Parameters:
    -----------
    matrix : np.ndarray
        Input matrix A.
    target_rates : List[float]
        Target bit rates for each column.
    sparsity_pattern : List[int], optional
        Indices of non-zero elements in the vector.
    lattice_type : str
        Type of lattice to use.
    M : int
        Number of hierarchical levels.

    Returns:
    --------
    SparseMatVecProcessor
        Configured processor for adaptive matrix-vector multiplication.
    """
    return SparseMatVecProcessor(
        matrix=matrix,
        target_rates=target_rates,
        sparsity_pattern=sparsity_pattern,
        lattice_type=lattice_type,
        M=M,
    )


def adaptive_matvec_multiply(
    matrix: np.ndarray,
    vector: np.ndarray,
    target_rates: List[float],
    sparsity_pattern: Optional[List[int]] = None,
    lattice_type: str = "D4",
    M: int = 2,
    use_lookup: bool = False,
) -> np.ndarray:
    """
    Perform adaptive matrix-vector multiplication.

    Parameters:
    -----------
    matrix : np.ndarray
        Input matrix A.
    vector : np.ndarray
        Input vector x.
    target_rates : List[float]
        Target bit rates for each column.
    sparsity_pattern : List[int], optional
        Indices of non-zero elements in the vector.
    lattice_type : str
        Type of lattice to use.
    M : int
        Number of hierarchical levels.
    use_lookup : bool
        Whether to use lookup tables for computation.

    Returns:
    --------
    np.ndarray
        Result vector y = Ax.
    """
    # Create processor
    processor = create_adaptive_matvec_processor(
        matrix, target_rates, sparsity_pattern, lattice_type, M
    )

    # Perform multiplication
    if use_lookup:
        return processor.compute_matvec_with_lookup(vector)
    else:
        return processor.compute_matvec(vector)


# Example usage and testing functions
def example_usage():
    """Example usage of the adaptive matrix-vector multiplication."""
    # Create test matrix and vector
    m, n = 100, 50
    matrix = np.random.randn(m, n)
    vector = np.random.randn(n)

    # Make vector sparse (only 10 non-zero elements)
    sparsity_pattern = np.random.choice(n, 10, replace=False)
    sparse_vector = np.zeros(n)
    sparse_vector[sparsity_pattern] = vector[sparsity_pattern]

    # Define target rates (different for each column)
    target_rates = np.random.uniform(2.0, 6.0, n)

    # Perform adaptive matrix-vector multiplication
    result = adaptive_matvec_multiply(
        matrix, sparse_vector, target_rates, sparsity_pattern.tolist(), "D4", 2
    )

    # Compare with exact computation
    exact_result = matrix @ sparse_vector
    error = np.linalg.norm(result - exact_result) / np.linalg.norm(exact_result)

    print(f"Relative error: {error:.6f}")
    print(f"Result shape: {result.shape}")

    return result, exact_result, error


if __name__ == "__main__":
    # Run example
    result, exact_result, error = example_usage()
    print("Example completed successfully!")
