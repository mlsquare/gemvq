"""
Adaptive Matrix-Vector Multiplication with Hierarchical Nested Quantizers

This module implements adaptive matrix-vector multiplication using hierarchical
nested lattice quantizers. The approach encodes matrix W once with maximum bit rate,
then adaptively decodes columns based on bit budget for each input vector x,
exploiting the hierarchical levels M for variable precision decoding.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional, Union
from ..quantizers.hierarchical_nested_lattice_quantizer import HierarchicalNestedLatticeQuantizer
from ..utils import get_d4, get_a2, get_e8, get_z2, get_z3, precompute_hq_lut, calculate_weighted_sum
from ..quantizers.closest_point import closest_point_Dn, closest_point_A2, closest_point_E8


class FixedMatrixQuantizer:
    """
    Fixed matrix quantizer that encodes W once with maximum bit rate.
    
    This class encodes the entire matrix W using hierarchical nested quantization
    with maximum precision, then provides adaptive decoding capabilities based
    on bit budget requirements for different input vectors x.
    """
    
    def __init__(self, matrix: np.ndarray, lattice_type: str = 'D4', 
                 max_rate: float = 8.0, M: int = 4, alpha: float = 1/3, eps: float = 1e-8):
        """
        Initialize the fixed matrix quantizer.
        
        Parameters:
        -----------
        matrix : np.ndarray
            Input matrix W to be encoded (m x n).
        lattice_type : str
            Type of lattice to use ('D4', 'A2', 'E8', 'Z2', 'Z3').
        max_rate : float
            Maximum bit rate for encoding (bits per dimension).
        M : int
            Number of hierarchical levels.
        alpha : float
            Scaling parameter for overload handling.
        eps : float
            Small perturbation parameter.
        """
        self.matrix = matrix
        self.m, self.n = matrix.shape
        self.lattice_type = lattice_type
        self.max_rate = max_rate
        self.M = M
        self.alpha = alpha
        self.eps = eps
        
        # Setup lattice parameters
        self.G, self.Q_nn = self._setup_lattice(lattice_type)
        self.dimension = self.G.shape[0]
        
        # Convert max rate to quantization parameters
        self.q, self.beta = self._rate_to_parameters(max_rate)
        
        # Create dither vector
        self.dither = np.zeros(self.dimension)
        
        # Initialize quantizer with max rate
        self.quantizer = HierarchicalNestedLatticeQuantizer(
            G=self.G, Q_nn=self.Q_nn, q=self.q, beta=self.beta,
            alpha=self.alpha, eps=self.eps, dither=self.dither, M=self.M
        )
        
        # Encode all columns
        self.encoded_columns = {}
        self.overload_scalings = {}
        self._encode_matrix()
        
        # Precompute lookup tables for different hierarchical levels
        self.lookup_tables = {}
        self._precompute_lookup_tables()
    
    def _setup_lattice(self, lattice_type: str) -> Tuple[np.ndarray, callable]:
        """Setup lattice generator matrix and closest point function."""
        if lattice_type == 'D4':
            return get_d4(), closest_point_Dn
        elif lattice_type == 'A2':
            return get_a2(), closest_point_A2
        elif lattice_type == 'E8':
            return get_e8(), closest_point_E8
        elif lattice_type == 'Z2':
            return get_z2(), lambda x: np.round(x)
        elif lattice_type == 'Z3':
            return get_z3(), lambda x: np.round(x)
        else:
            raise ValueError(f"Unsupported lattice type: {lattice_type}")
    
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
        # Convert rate to quantization parameter
        q = max(2, int(2 ** (rate / self.M)))
        beta = 1.0 / (2 ** (rate / 4))  # Scaling based on rate
        
        return q, beta
    
    def _encode_matrix(self):
        """Encode all matrix columns with maximum precision."""
        for i in range(self.n):
            column = self.matrix[:, i]
            
            # Process column in blocks that match lattice dimension
            encoded_blocks = []
            scaling_blocks = []
            
            for j in range(0, self.m, self.dimension):
                # Extract block of size dimension
                end_idx = min(j + self.dimension, self.m)
                block = column[j:end_idx]
                
                # Pad block if necessary to match lattice dimension
                if len(block) < self.dimension:
                    block = np.pad(block, (0, self.dimension - len(block)), mode='constant')
                
                # Encode block
                encoding, scaling = self.quantizer.encode(block, with_dither=False)
                encoded_blocks.append(encoding)
                scaling_blocks.append(scaling)
            
            self.encoded_columns[i] = encoded_blocks
            self.overload_scalings[i] = scaling_blocks
    
    def _precompute_lookup_tables(self):
        """Precompute lookup tables for different hierarchical levels."""
        # Precompute lookup table for the maximum rate
        self.lookup_tables[self.max_rate] = precompute_hq_lut(
            self.G, self.Q_nn, self.q, self.M, self.eps
        )
        
        # Precompute lookup tables for lower rates (fewer hierarchical levels)
        for m in range(1, self.M):
            # Calculate effective rate for m levels
            effective_rate = self.max_rate * m / self.M
            self.lookup_tables[effective_rate] = precompute_hq_lut(
                self.G, self.Q_nn, self.q, m, self.eps
            )
    
    def decode_column_adaptive(self, col_idx: int, target_rate: float) -> np.ndarray:
        """
        Decode a column adaptively based on target bit rate.
        
        Parameters:
        -----------
        col_idx : int
            Column index to decode.
        target_rate : float
            Target bit rate for decoding (can be less than max_rate).
            
        Returns:
        --------
        np.ndarray
            Decoded column vector with precision based on target_rate.
        """
        if col_idx >= self.n:
            raise ValueError(f"Column index {col_idx} out of range [0, {self.n-1}]")
        
        encoded_blocks = self.encoded_columns[col_idx]
        scaling_blocks = self.overload_scalings[col_idx]
        
        # Determine how many hierarchical levels to use based on target rate
        levels_to_use = self.M
        if target_rate < self.max_rate:
            levels_to_use = max(1, int(self.M * target_rate / self.max_rate))
        
        # Decode each block
        decoded_blocks = []
        for block_idx, (encoding, scaling) in enumerate(zip(encoded_blocks, scaling_blocks)):
            if levels_to_use >= self.M:
                # Use full precision (all M levels)
                decoded_block = self.quantizer.decode(encoding, scaling, with_dither=False)
            else:
                # Use partial precision based on target rate
                decoded_block = self._decode_partial(encoding, scaling, levels_to_use)
            
            decoded_blocks.append(decoded_block)
        
        # Combine blocks back into full column
        full_column = np.concatenate(decoded_blocks)
        return full_column[:self.m]  # Trim to original column size
    
    def _decode_partial(self, encoding: Tuple, scaling: int, levels: int) -> np.ndarray:
        """
        Decode using only a subset of hierarchical levels.
        
        Parameters:
        -----------
        encoding : tuple
            Full encoding with M levels.
        scaling : int
            Overload scaling factor.
        levels : int
            Number of hierarchical levels to use.
            
        Returns:
        --------
        np.ndarray
            Partially decoded vector.
        """
        # Take only the first 'levels' encoding vectors
        partial_encoding = encoding[:levels]
        
        # Create a temporary quantizer with fewer levels
        temp_quantizer = HierarchicalNestedLatticeQuantizer(
            G=self.G, Q_nn=self.Q_nn, q=self.q, beta=self.beta,
            alpha=self.alpha, eps=self.eps, dither=self.dither, M=levels
        )
        
        return temp_quantizer.decode(partial_encoding, scaling, with_dither=False)
    
    def estimate_inner_product_adaptive(self, col_idx: int, weight: float, 
                                      target_rate: float) -> np.ndarray:
        """
        Estimate inner product adaptively using lookup tables.
        
        Parameters:
        -----------
        col_idx : int
            Column index.
        weight : float
            Weight from input vector x.
        target_rate : float
            Target bit rate for estimation.
            
        Returns:
        --------
        np.ndarray
            Estimated column contribution.
        """
        encoded_blocks = self.encoded_columns[col_idx]
        scaling_blocks = self.overload_scalings[col_idx]
        
        # Find appropriate lookup table
        if target_rate >= self.max_rate:
            lookup_table = self.lookup_tables[self.max_rate]
            levels_to_use = self.M
        else:
            # Find closest precomputed rate
            rates = list(self.lookup_tables.keys())
            closest_rate = min(rates, key=lambda x: abs(x - target_rate))
            lookup_table = self.lookup_tables[closest_rate]
            levels_to_use = max(1, int(self.M * closest_rate / self.max_rate))
        
        # Process each block
        estimated_blocks = []
        for encoding, scaling in zip(encoded_blocks, scaling_blocks):
            # Use only the first 'levels_to_use' encoding vectors
            partial_encoding = encoding[:levels_to_use]
            
            # Estimate inner product using lookup table
            estimated_block = calculate_weighted_sum(
                partial_encoding, weight, lookup_table, self.G, self.Q_nn, 
                self.q, self.beta, self.alpha, scaling, levels_to_use, self.eps
            )
            estimated_blocks.append(estimated_block)
        
        # Combine blocks
        full_contribution = np.concatenate(estimated_blocks)
        return full_contribution[:self.m]  # Trim to original column size


class AdaptiveMatVecProcessor:
    """
    Adaptive matrix-vector multiplication processor.
    
    This class implements efficient matrix-vector multiplication where W is
    encoded once with maximum precision, and columns are decoded adaptively
    based on bit budget for each input vector x.
    """
    
    def __init__(self, matrix: np.ndarray, lattice_type: str = 'D4', 
                 max_rate: float = 8.0, M: int = 4):
        """
        Initialize the adaptive matrix-vector processor.
        
        Parameters:
        -----------
        matrix : np.ndarray
            Input matrix W (m x n).
        lattice_type : str
            Type of lattice to use.
        max_rate : float
            Maximum bit rate for encoding W.
        M : int
            Number of hierarchical levels.
        """
        self.matrix = matrix
        self.m, self.n = matrix.shape
        self.lattice_type = lattice_type
        self.max_rate = max_rate
        self.M = M
        
        # Initialize fixed matrix quantizer
        self.quantizer = FixedMatrixQuantizer(
            matrix, lattice_type, max_rate, M
        )
    
    def compute_matvec(self, vector: np.ndarray, column_rates: List[float],
                      use_lookup: bool = False) -> np.ndarray:
        """
        Compute matrix-vector product with adaptive column decoding.
        
        Parameters:
        -----------
        vector : np.ndarray
            Input vector x.
        column_rates : List[float]
            Target bit rates for each column (can be less than max_rate).
        use_lookup : bool
            Whether to use lookup tables for computation.
            
        Returns:
        --------
        np.ndarray
            Result vector y = Wx.
        """
        if len(vector) != self.n:
            raise ValueError(f"Vector dimension {len(vector)} != matrix columns {self.n}")
        
        if len(column_rates) != self.n:
            raise ValueError(f"Column rates length {len(column_rates)} != matrix columns {self.n}")
        
        # Initialize result vector
        result = np.zeros(self.m)
        
        # Process each column with its target rate
        for i in range(self.n):
            if abs(vector[i]) > 1e-10:  # Check for non-zero
                if use_lookup:
                    # Use lookup table for efficient computation
                    contribution = self.quantizer.estimate_inner_product_adaptive(
                        i, vector[i], column_rates[i]
                    )
                else:
                    # Decode column and multiply
                    decoded_column = self.quantizer.decode_column_adaptive(
                        i, column_rates[i]
                    )
                    contribution = vector[i] * decoded_column
                
                result += contribution
        
        return result
    
    def compute_matvec_sparse(self, sparse_vector: np.ndarray, 
                            non_zero_indices: List[int],
                            column_rates: List[float],
                            use_lookup: bool = False) -> np.ndarray:
        """
        Compute matrix-vector product for sparse vectors efficiently.
        
        Parameters:
        -----------
        sparse_vector : np.ndarray
            Sparse input vector x.
        non_zero_indices : List[int]
            Indices of non-zero elements in x.
        column_rates : List[float]
            Target bit rates for each column.
        use_lookup : bool
            Whether to use lookup tables for computation.
            
        Returns:
        --------
        np.ndarray
            Result vector y = Wx.
        """
        if len(sparse_vector) != self.n:
            raise ValueError(f"Vector dimension {len(sparse_vector)} != matrix columns {self.n}")
        
        # Initialize result vector
        result = np.zeros(self.m)
        
        # Process only non-zero elements
        for i in non_zero_indices:
            if i >= self.n:
                continue
                
            if use_lookup:
                # Use lookup table for efficient computation
                contribution = self.quantizer.estimate_inner_product_adaptive(
                    i, sparse_vector[i], column_rates[i]
                )
            else:
                # Decode column and multiply
                decoded_column = self.quantizer.decode_column_adaptive(
                    i, column_rates[i]
                )
                contribution = sparse_vector[i] * decoded_column
            
            result += contribution
        
        return result
    
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
        for encoded_blocks in self.encoded_columns.values():
            # Count bits in encoding vectors for all blocks
            for block_encoding in encoded_blocks:
                for level_encoding in block_encoding:
                    encoded_bits += len(level_encoding) * np.log2(self.q)
        
        return original_bits / encoded_bits if encoded_bits > 0 else float('inf')
    
    def get_memory_usage(self) -> Dict[str, float]:
        """
        Get memory usage statistics.
        
        Returns:
        --------
        dict
            Dictionary with memory usage information.
        """
        # Calculate encoded matrix size
        encoded_size = sum(
            sum(sum(len(level_encoding) for level_encoding in block_encoding)
                for block_encoding in encoded_blocks)
            for encoded_blocks in self.encoded_columns.values()
        )
        
        # Calculate lookup table sizes
        lookup_size = sum(
            len(table) for table in self.lookup_tables.values()
        )
        
        return {
            'encoded_matrix_mb': encoded_size * 4 / (1024 * 1024),  # Assuming 4 bytes per element
            'lookup_tables_mb': lookup_size * 8 / (1024 * 1024),    # Assuming 8 bytes per entry
            'total_mb': (encoded_size * 4 + lookup_size * 8) / (1024 * 1024)
        }


def create_adaptive_matvec_processor(matrix: np.ndarray, lattice_type: str = 'D4',
                                   max_rate: float = 8.0, M: int = 4) -> AdaptiveMatVecProcessor:
    """
    Factory function to create an adaptive matrix-vector processor.
    
    Parameters:
    -----------
    matrix : np.ndarray
        Input matrix W.
    lattice_type : str
        Type of lattice to use.
    max_rate : float
        Maximum bit rate for encoding W.
    M : int
        Number of hierarchical levels.
        
    Returns:
    --------
    AdaptiveMatVecProcessor
        Configured processor for adaptive matrix-vector multiplication.
    """
    return AdaptiveMatVecProcessor(matrix, lattice_type, max_rate, M)


def adaptive_matvec_multiply(matrix: np.ndarray, vector: np.ndarray,
                           column_rates: List[float],
                           lattice_type: str = 'D4', max_rate: float = 8.0, 
                           M: int = 4, use_lookup: bool = False) -> np.ndarray:
    """
    Perform adaptive matrix-vector multiplication.
    
    Parameters:
    -----------
    matrix : np.ndarray
        Input matrix W.
    vector : np.ndarray
        Input vector x.
    column_rates : List[float]
        Target bit rates for each column.
    lattice_type : str
        Type of lattice to use.
    max_rate : float
        Maximum bit rate for encoding W.
    M : int
        Number of hierarchical levels.
    use_lookup : bool
        Whether to use lookup tables for computation.
        
    Returns:
    --------
    np.ndarray
        Result vector y = Wx.
    """
    # Create processor
    processor = create_adaptive_matvec_processor(matrix, lattice_type, max_rate, M)
    
    # Perform multiplication
    return processor.compute_matvec(vector, column_rates, use_lookup)


def adaptive_matvec_multiply_sparse(matrix: np.ndarray, sparse_vector: np.ndarray,
                                  non_zero_indices: List[int],
                                  column_rates: List[float],
                                  lattice_type: str = 'D4', max_rate: float = 8.0,
                                  M: int = 4, use_lookup: bool = False) -> np.ndarray:
    """
    Perform adaptive matrix-vector multiplication for sparse vectors.
    
    Parameters:
    -----------
    matrix : np.ndarray
        Input matrix W.
    sparse_vector : np.ndarray
        Sparse input vector x.
    non_zero_indices : List[int]
        Indices of non-zero elements in x.
    column_rates : List[float]
        Target bit rates for each column.
    lattice_type : str
        Type of lattice to use.
    max_rate : float
        Maximum bit rate for encoding W.
    M : int
        Number of hierarchical levels.
    use_lookup : bool
        Whether to use lookup tables for computation.
        
    Returns:
    --------
    np.ndarray
        Result vector y = Wx.
    """
    # Create processor
    processor = create_adaptive_matvec_processor(matrix, lattice_type, max_rate, M)
    
    # Perform multiplication
    return processor.compute_matvec_sparse(sparse_vector, non_zero_indices, column_rates, use_lookup)


# Example usage and testing functions
def example_usage():
    """Example usage of the adaptive matrix-vector multiplication."""
    # Create test matrix and vector
    m, n = 100, 50
    matrix = np.random.randn(m, n)
    vector = np.random.randn(n)
    
    # Define column rates (can be less than max_rate)
    max_rate = 8.0
    column_rates = np.random.uniform(2.0, max_rate, n)
    
    # Perform adaptive matrix-vector multiplication
    result = adaptive_matvec_multiply(
        matrix, vector, column_rates.tolist(), 'D4', max_rate, 4, use_lookup=False
    )
    
    # Compare with exact computation
    exact_result = matrix @ vector
    error = np.linalg.norm(result - exact_result) / np.linalg.norm(exact_result)
    
    print(f"Relative error: {error:.6f}")
    print(f"Result shape: {result.shape}")
    
    return result, exact_result, error


def example_sparse_usage():
    """Example usage with sparse vectors."""
    # Create test matrix and sparse vector
    m, n = 100, 50
    matrix = np.random.randn(m, n)
    
    # Create sparse vector (only 10 non-zero elements)
    sparsity = 10
    non_zero_indices = np.random.choice(n, sparsity, replace=False)
    sparse_vector = np.zeros(n)
    sparse_vector[non_zero_indices] = np.random.randn(sparsity)
    
    # Define column rates
    max_rate = 8.0
    column_rates = np.random.uniform(2.0, max_rate, n)
    
    # Perform adaptive sparse matrix-vector multiplication
    result = adaptive_matvec_multiply_sparse(
        matrix, sparse_vector, non_zero_indices.tolist(), 
        column_rates.tolist(), 'D4', max_rate, 4, use_lookup=False
    )
    
    # Compare with exact computation
    exact_result = matrix @ sparse_vector
    error = np.linalg.norm(result - exact_result) / np.linalg.norm(exact_result)
    
    print(f"Sparse relative error: {error:.6f}")
    print(f"Sparsity: {sparsity}/{n} = {sparsity/n:.2f}")
    
    return result, exact_result, error


if __name__ == "__main__":
    # Run examples
    print("=== Dense Vector Example ===")
    result1, exact1, error1 = example_usage()
    
    print("\n=== Sparse Vector Example ===")
    result2, exact2, error2 = example_sparse_usage()
    
    print("\nExamples completed successfully!") 