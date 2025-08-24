"""
SVD GEMV processor.

This module implements matrix-vector multiplication in the SVD domain where
W = U * S * V^T, and the computation is done efficiently.
"""

from typing import Dict, List, Optional, Tuple, Union, Any
import numpy as np
from scipy import linalg

from ..base.gemv_processor import GEMVProcessor
from ...lattices.utils import (closest_point_A2, closest_point_Dn, closest_point_E8,
                              get_a2, get_d4, get_e8, get_z2, get_z3)
from ...lattices.quantizers.hierarchical_nested_lattice_quantizer import \
    HierarchicalNestedLatticeQuantizer


class SVDGEMVProcessor(GEMVProcessor):
    """
    SVD GEMV processor.
    
    Implements matrix-vector multiplication in the SVD domain where W = U * S * V^T.
    The computation y = Wx is performed as y = U * (S * (V^T * x)) for efficiency.
    """

    def __init__(
        self,
        lattice_type: str = "D4",
        M: int = 2,
        q: int = 4,
        beta: float = 0.2,
        alpha: float = 1/3,
        eps: float = 1e-8,
        use_lookup: bool = False,
        quantize_x: bool = False,
        sparsity_threshold: float = 1e-10,
        decoding: str = "full",
        svd_rank: Optional[int] = None,
        quantize_svd: bool = True,
        **kwargs
    ):
        """
        Initialize the SVD GEMV processor.

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
        svd_rank : Optional[int]
            Rank for truncated SVD. If None, full SVD is used.
        quantize_svd : bool
            Whether to quantize the SVD components (U, S, V).
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
            svd_rank=svd_rank,
            quantize_svd=quantize_svd,
            **kwargs
        )
        
        # Initialize state variables
        self.original_matrix = None
        self.original_m = None
        self.original_n = None
        self.G = None
        self.Q_nn = None
        self.dimension = None
        self.U = None
        self.S = None
        self.Vt = None
        self.svd_rank = None
        self.quantized_components = {}
        self.stats = {
            'compression_ratio': 0.0,
            'memory_usage_mb': 0.0,
            'computation_time': 0.0,
            'svd_error': 0.0
        }

    def _validate_config(self) -> None:
        """Validate the configuration parameters."""
        lattice_type = self.config.get('lattice_type', 'D4')
        if lattice_type not in ['D4', 'A2', 'E8', 'Z2', 'Z3']:
            raise ValueError(f"Unsupported lattice type: {lattice_type}")
        
        M = self.config.get('M', 2)
        if M < 1:
            raise ValueError(f"M must be >= 1, got {M}")
        
        q = self.config.get('q', 4)
        if q < 1:
            raise ValueError(f"q must be >= 1, got {q}")
        
        beta = self.config.get('beta', 0.2)
        if beta <= 0:
            raise ValueError(f"beta must be > 0, got {beta}")
        
        alpha = self.config.get('alpha', 1/3)
        if alpha <= 0:
            raise ValueError(f"alpha must be > 0, got {alpha}")
        
        eps = self.config.get('eps', 1e-8)
        if eps <= 0:
            raise ValueError(f"eps must be > 0, got {eps}")
        
        sparsity_threshold = self.config.get('sparsity_threshold', 1e-10)
        if sparsity_threshold < 0:
            raise ValueError(f"sparsity_threshold must be >= 0, got {sparsity_threshold}")
        
        decoding = self.config.get('decoding', 'full')
        if decoding not in ['full', 'coarse_to_fine', 'progressive']:
            raise ValueError(f"Unsupported decoding method: {decoding}")
        
        svd_rank = self.config.get('svd_rank')
        if svd_rank is not None and svd_rank < 1:
            raise ValueError(f"svd_rank must be >= 1, got {svd_rank}")

    def preprocess_matrix(self, matrix: np.ndarray) -> np.ndarray:
        """
        Preprocess the matrix by computing SVD and optionally quantizing components.
        
        Args:
            matrix: Input matrix
            
        Returns:
            Preprocessed matrix (SVD components stored internally)
        """
        # Store original matrix
        self.original_matrix = matrix.copy()
        self.original_m, self.original_n = matrix.shape
        
        # Setup lattice parameters
        lattice_type = self.config.get('lattice_type', 'D4')
        self.G, self.Q_nn = self._setup_lattice(lattice_type)
        self.dimension = self.G.shape[0]
        
        # Compute SVD
        self._compute_svd()
        
        # Quantize SVD components if requested
        if self.config.get('quantize_svd', True):
            self._quantize_svd_components()
        
        return matrix  # Return original matrix for compatibility

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

    def _compute_svd(self):
        """Compute SVD decomposition of the matrix."""
        svd_rank = self.config.get('svd_rank')
        
        if svd_rank is not None:
            # Truncated SVD
            U, S, Vt = linalg.svd(self.original_matrix, full_matrices=False)
            self.U = U[:, :svd_rank]
            self.S = S[:svd_rank]
            self.Vt = Vt[:svd_rank, :]
            self.svd_rank = svd_rank
        else:
            # Full SVD
            U, S, Vt = linalg.svd(self.original_matrix, full_matrices=False)
            self.U = U
            self.S = S
            self.Vt = Vt
            self.svd_rank = len(S)
        
        # Compute SVD reconstruction error
        reconstructed = self.U @ np.diag(self.S) @ self.Vt
        self.stats['svd_error'] = np.linalg.norm(self.original_matrix - reconstructed, 'fro')

    def _quantize_svd_components(self):
        """Quantize the SVD components using lattice quantization."""
        quantizer = HierarchicalNestedLatticeQuantizer(
            G=self.G,
            Q_nn=self.Q_nn,
            M=self.config.get('M', 2),
            q=self.config.get('q', 4),
            beta=self.config.get('beta', 0.2),
            alpha=self.config.get('alpha', 1/3),
            eps=self.config.get('eps', 1e-8),
            dither=np.zeros(self.dimension)
        )
        
        # Quantize U matrix
        self.quantized_components['U'] = self._quantize_matrix(self.U, quantizer)
        
        # Quantize S vector
        self.quantized_components['S'] = self._quantize_vector(self.S, quantizer)
        
        # Quantize V^T matrix
        self.quantized_components['Vt'] = self._quantize_matrix(self.Vt, quantizer)

    def _quantize_matrix(self, matrix: np.ndarray, quantizer: HierarchicalNestedLatticeQuantizer) -> Dict:
        """Quantize a matrix using the given quantizer."""
        m, n = matrix.shape
        encoded_matrix = []
        
        for i in range(m):
            row_encodings = []
            for j in range(0, n, self.dimension):
                chunk = matrix[i, j:j+self.dimension]
                if len(chunk) < self.dimension:
                    chunk = np.pad(chunk, (0, self.dimension - len(chunk)), mode='constant')
                encoding, T = quantizer.encode(chunk, with_dither=False)
                row_encodings.append((encoding, T))
            encoded_matrix.append(row_encodings)
        
        return {
            'encodings': encoded_matrix,
            'shape': matrix.shape,
            'quantizer': quantizer
        }

    def _quantize_vector(self, vector: np.ndarray, quantizer: HierarchicalNestedLatticeQuantizer) -> Dict:
        """Quantize a vector using the given quantizer."""
        encoded_vector = []
        
        for i in range(0, len(vector), self.dimension):
            chunk = vector[i:i+self.dimension]
            if len(chunk) < self.dimension:
                chunk = np.pad(chunk, (0, self.dimension - len(chunk)), mode='constant')
            encoding, T = quantizer.encode(chunk, with_dither=False)
            encoded_vector.append((encoding, T))
        
        return {
            'encodings': encoded_vector,
            'length': len(vector),
            'quantizer': quantizer
        }

    def _decode_matrix(self, quantized_matrix: Dict, decoding_method: str) -> np.ndarray:
        """Decode a quantized matrix."""
        encodings = quantized_matrix['encodings']
        shape = quantized_matrix['shape']
        quantizer = quantized_matrix['quantizer']
        
        decoded_matrix = np.zeros(shape)
        
        for i, row_encodings in enumerate(encodings):
            for j, (encoding, T) in enumerate(row_encodings):
                if decoding_method == 'full':
                    chunk = quantizer.decode(encoding, T, with_dither=False)
                elif decoding_method == 'coarse_to_fine':
                    chunk = quantizer.decode_coarse_to_fine(encoding, T, with_dither=False)
                elif decoding_method == 'progressive':
                    chunk = quantizer.decode_progressive(encoding, T, with_dither=False)
                else:
                    chunk = quantizer.decode(encoding, T, with_dither=False)
                
                start_col = j * self.dimension
                end_col = min(start_col + self.dimension, shape[1])
                decoded_matrix[i, start_col:end_col] = chunk[:end_col - start_col]
        
        return decoded_matrix

    def _decode_vector(self, quantized_vector: Dict, decoding_method: str) -> np.ndarray:
        """Decode a quantized vector."""
        encodings = quantized_vector['encodings']
        length = quantized_vector['length']
        quantizer = quantized_vector['quantizer']
        
        decoded_chunks = []
        for encoding, T in encodings:
            if decoding_method == 'full':
                chunk = quantizer.decode(encoding, T, with_dither=False)
            elif decoding_method == 'coarse_to_fine':
                chunk = quantizer.decode_coarse_to_fine(encoding, T, with_dither=False)
            elif decoding_method == 'progressive':
                chunk = quantizer.decode_progressive(encoding, T, with_dither=False)
            else:
                chunk = quantizer.decode(encoding, T, with_dither=False)
            decoded_chunks.append(chunk)
        
        decoded_vector = np.concatenate(decoded_chunks)
        return decoded_vector[:length]

    def process(self, matrix: np.ndarray, vector: np.ndarray) -> np.ndarray:
        """
        Process the SVD matrix-vector multiplication.
        
        Args:
            matrix: Input matrix (not used, SVD components stored internally)
            vector: Input vector
            
        Returns:
            Result vector
        """
        if len(vector) != self.original_n:
            raise ValueError(f"Vector length {len(vector)} != matrix columns {self.original_n}")
        
        decoding_method = self.config.get('decoding', 'full')
        
        # Get SVD components (quantized or original)
        if self.config.get('quantize_svd', True) and self.quantized_components:
            # Use quantized components
            U = self._decode_matrix(self.quantized_components['U'], decoding_method)
            S = self._decode_vector(self.quantized_components['S'], decoding_method)
            Vt = self._decode_matrix(self.quantized_components['Vt'], decoding_method)
        else:
            # Use original components
            U = self.U
            S = self.S
            Vt = self.Vt
        
        # Compute y = U * (S * (V^T * x))
        # Step 1: V^T * x
        step1 = Vt @ vector
        
        # Step 2: S * (V^T * x)
        step2 = S * step1
        
        # Step 3: U * (S * (V^T * x))
        result = U @ step2
        
        return result

    def get_processor_info(self) -> Dict[str, Any]:
        """
        Get information about the processor configuration.
        
        Returns:
            Dictionary containing processor information
        """
        return {
            'type': 'svd',
            'lattice_type': self.config.get('lattice_type', 'D4'),
            'M': self.config.get('M', 2),
            'q': self.config.get('q', 4),
            'beta': self.config.get('beta', 0.2),
            'alpha': self.config.get('alpha', 1/3),
            'eps': self.config.get('eps', 1e-8),
            'use_lookup': self.config.get('use_lookup', False),
            'quantize_x': self.config.get('quantize_x', False),
            'sparsity_threshold': self.config.get('sparsity_threshold', 1e-10),
            'decoding': self.config.get('decoding', 'full'),
            'svd_rank': self.svd_rank,
            'quantize_svd': self.config.get('quantize_svd', True),
            'stats': self.stats
        }
