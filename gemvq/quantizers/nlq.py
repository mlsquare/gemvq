import numpy as np
from typing import Tuple, Optional, Callable, Dict, Any, List
import warnings
from dataclasses import dataclass
from .utils import generate_tie_dither



@dataclass
class QuantizerConfig:
    """
    Configuration class for nested lattice quantizer parameters.
    
    This class provides a structured way to manage quantizer parameters
    with built-in validation and default values.
    
    Attributes:
    -----------
    q : int
        Quantization parameter (alphabet size).
    beta : float
        Scaling parameter for quantization.
    alpha : float
        Scaling parameter for overload handling.
    eps : float
        Small perturbation parameter.
    overload : bool
        Whether to handle overload by scaling.
    max_scaling_iterations : int
        Maximum number of scaling iterations.
    with_tie_dither : Boolean
        Whether to add dither to the input for tie breaking.
    with_dither : Boolean
        Whether to add dither to the input for randomized quantization.
    """
    q: int
    beta: float
    alpha: float
    eps: float
    overload: bool = True
    max_scaling_iterations: int = 10
    with_tie_dither: bool = True
    with_dither: bool = False
    
    def __post_init__(self):
        """Validate parameters after initialization."""
        if self.q <= 0:
            raise ValueError("Quantization parameter q must be positive")
        if self.beta <= 0:
            raise ValueError("Scaling parameter beta must be positive")
        if self.alpha <= 0:
            raise ValueError("Scaling parameter alpha must be positive")
        if self.max_scaling_iterations <= 0:
            raise ValueError("max_scaling_iterations must be positive")
        if not isinstance(self.with_tie_dither, bool):
            raise ValueError("with_tie_dither must be a Boolean")
        if not isinstance(self.with_dither, bool):
            raise ValueError("with_dither must be a Boolean")
        
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'QuantizerConfig':
        """Create configuration from dictionary."""
        return cls(**config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'q': self.q,
            'beta': self.beta,
            'alpha': self.alpha,
            'eps': self.eps,
            'overload': self.overload,
            'max_scaling_iterations': self.max_scaling_iterations,
            'with_tie_dither': self.with_tie_dither,
            'with_dither': self.with_dither
        }


class NLQ:
    """
    A nested lattice quantizer for vector quantization using lattice structures.
    
    This class implements a nested lattice quantization scheme that can handle
    overload conditions by scaling the input vector. The quantizer uses a
    generator matrix G to define the lattice structure and supports dithering
    for randomized quantization.
    
    Attributes:
    -----------
    G : numpy.ndarray
        Generator matrix defining the lattice structure.
    Q_nn : Callable
        Closest point function for the lattice (e.g., closest_point_Dn).
    q : int
        Quantization parameter (alphabet size).
    beta : float
        Scaling parameter for quantization.
    alpha : float
        Scaling parameter for overload handling.
    eps : float
        Small perturbation parameter added to input before quantization for tie breaking.
    with_tie_dither : Boolean
        Whether to add dither to the input for tie breaking.
    with_dither : Boolean or None
        Dither vector for randomized quantization. Generated on-demand when with_dither=True.
    overload : bool
        Whether to handle overload by scaling the input vector.
    G_inv : numpy.ndarray
        Inverse of the generator matrix, precomputed for efficiency.
    max_scaling_iterations : int
        Maximum number of scaling iterations to prevent infinite loops.
    
    Notes:
    ------
    The quantizer works by:
    1. Scaling the input by beta
    2. Adding dither if with_dither=True (otherwise adds zeros)
    3. Finding the closest lattice point using Q_nn
    4. Encoding the result modulo q
    5. If overload occurs and overload=True, scale the input and repeat
    
    Dither is generated on-demand when with_dither=True and can be reset using reset_dither().
    
    References:
    -----------
    Conway, J. H., & Sloane, N. J. A. (1982). Fast quantizing and decoding
    algorithms for lattice quantizers and codes. IEEE Transactions on
    Information Theory, 28(2), 227-232.
    """
    
    def __init__(
        self, 
        G: np.ndarray, 
        Q_nn: Callable[[np.ndarray], np.ndarray], 
        q: int, 
        beta: float, 
        alpha: float, 
        eps: float,
        with_tie_dither: bool = True,
        with_dither: bool = False,
        M: Optional[int] = None, 
        overload: bool = True,
        max_scaling_iterations: int = 10,
        config: Optional[QuantizerConfig] = None
    ) -> None:
        """
        Initialize the nested lattice quantizer.
        
        Parameters:
        -----------
        G : numpy.ndarray
            Generator matrix for the lattice.
        Q_nn : Callable
            Closest point function for the lattice.
        q : int
            Quantization parameter (alphabet size).
        beta : float
            Scaling parameter for quantization.
        alpha : float
            Scaling parameter for overload handling.
        eps : float
            Small perturbation parameter for breaking ties in the lattice.
        with_tie_dither : Boolean
            Whether to add dither to the input for tie breaking.
        with_dither : Boolean
            Whether to add dither to the input for randomized quantization.
        M : int, optional
            Number of hierarchical levels (unused in this implementation).
        overload : bool, optional
            Whether to handle overload by scaling. Default is True.
        max_scaling_iterations : int, optional
            Maximum number of scaling iterations to prevent infinite loops.
            Default is 10.
        config : QuantizerConfig, optional
            Configuration object. If provided, overrides individual parameters.
            
        Raises:
        -------
        ValueError
            If G is not a square matrix or if dither dimension doesn't match.
        """
        if G.shape[0] != G.shape[1]:
            raise ValueError("Generator matrix G must be square")
        
        # Use config if provided, otherwise use individual parameters
        if config is not None:
            self.config = config
            self.q = config.q
            self.beta = config.beta
            self.alpha = config.alpha
            self.eps = config.eps
            self.overload = config.overload
            self.max_scaling_iterations = config.max_scaling_iterations
            self.with_tie_dither = config.with_tie_dither
            self.with_dither = config.with_dither
        else:
            self.config = QuantizerConfig(q, beta, alpha, eps, overload, max_scaling_iterations, with_tie_dither, with_dither)
            self.q = q
            self.beta = beta
            self.alpha = alpha
            self.eps = eps
            self.overload = overload
            self.max_scaling_iterations = max_scaling_iterations
            self.with_tie_dither = with_tie_dither
            self.with_dither = with_dither
            
        if self.with_tie_dither:
            self._original_eps = generate_tie_dither(G.shape[0])
        else:
            self._original_eps = eps
            
        # Store original eps for backward compatibility
        self.G = G
        self.Q_nn = lambda x: Q_nn(x + self._original_eps)
        
        # Initialize dither as None - will be generated on demand when with_dither=True
        # At this time, this feature is not supported.
        self.dither = np.zeros(self.G.shape[0])
        
        self.G_inv = np.linalg.inv(G)
        
        # Precompute some values for efficiency
        self._dim = G.shape[0]
        self._q_squared = self.q ** 2

    def _encode(self, x: np.ndarray, with_dither: bool) -> Tuple[np.ndarray, bool]:
        """
        Encode a vector to lattice coordinates.
        
        Parameters:
        -----------
        x : numpy.ndarray
            Input vector to encode.
        with_dither : bool
            Whether to apply dithering during encoding.
            
        Returns:
        --------
        tuple
            (encoded_vector, overload_error) where encoded_vector is the
            quantized coordinates and overload_error indicates if overload occurred.
        """
        # Ensure input is a numpy array
        x = np.asarray(x, dtype=np.float64)
        
        x_tag = (x / self.beta)
        if with_dither:    
            x_tag = x_tag + self.dither
        
        t = self.Q_nn(x_tag)
        y = np.dot(self.G_inv, t)
        enc = np.mod(np.round(y), self.q).astype(int)
        
        # More robust overload detection
        try:
            overload_error = not np.allclose(self.Q_nn(t / self.q), 0, atol=1e-8)
        except:
            # Fallback overload detection
            overload_error = np.any(np.abs(t) >= self.q)
            
        return enc, overload_error

    def encode(self, x: np.ndarray, with_dither: bool = False) -> Tuple[np.ndarray, int]:
        """
        Encode a vector with overload handling.
        
        Parameters:
        -----------
        x : numpy.ndarray
            Input vector to encode.
        with_dither : bool, optional
            Whether to apply dithering during encoding. Default is False.
            
        Returns:
        --------
        tuple
            (encoded_vector, scaling_count) where encoded_vector is the
            quantized coordinates and scaling_count is the number of scaling
            operations applied to handle overload.
            
        Raises:
        -------
        RuntimeWarning
            If maximum scaling iterations are reached.
        """
        enc, did_overload = self._encode(x, with_dither)
        t = 0
        
        if self.overload:
            while did_overload and t < self.max_scaling_iterations:
                t += 1
                x = np.asarray(x, dtype=np.float64) / (2 ** self.alpha)
                enc, did_overload = self._encode(x, with_dither)
            
            if t >= self.max_scaling_iterations and did_overload:
                warnings.warn(
                    f"Maximum scaling iterations ({self.max_scaling_iterations}) reached. "
                    "Consider increasing max_scaling_iterations or adjusting parameters.",
                    RuntimeWarning
                )
        else:
            enc, did_overload = self._encode(x, with_dither)
            
        return enc, t

    def _decode(self, y: np.ndarray, with_dither: bool) -> np.ndarray:
        """
        Decode lattice coordinates back to a vector.
        
        Parameters:
        -----------
        y : numpy.ndarray
            Encoded vector (lattice coordinates).
        with_dither : bool
            Whether dithering was applied during encoding.
            
        Returns:
        --------
        numpy.ndarray
            Decoded vector.
        """
        # Ensure input is a numpy array
        y = np.asarray(y, dtype=np.float64)
        x_p = np.dot(self.G, y)
        
        if with_dither: 
            x_p = x_p - self.dither
        
        x_pp = self.q * self.Q_nn(x_p / self.q)
        return self.beta * (x_p - x_pp)

    def decode(self, enc: np.ndarray, T: int, with_dither: bool = False) -> np.ndarray:
        """
        Decode an encoded vector with scaling compensation.
        
        Parameters:
        -----------
        enc : numpy.ndarray
            Encoded vector.
        T : int
            Number of scaling operations applied during encoding.
        with_dither : bool, optional
            Whether dithering was applied during encoding. Default is False.
            
        Returns:
        --------
        numpy.ndarray
            Decoded vector with scaling compensation applied.
        """
        if T < 0:
            raise ValueError("Scaling count T must be non-negative")
        return self._decode(enc, with_dither) * (2 ** (self.alpha * T))

    def quantize(self, x: np.ndarray, with_dither: bool = False) -> np.ndarray:
        """
        Quantize a vector (encode and decode).
        
        Parameters:
        -----------
        x : numpy.ndarray
            Input vector to quantize.
        with_dither : bool, optional
            Whether to apply dithering. Default is False.
            
        Returns:
        --------
        numpy.ndarray
            Quantized vector.
        """
        enc, T = self.encode(x, with_dither)
        return self.decode(enc, T, with_dither)

    def create_codebook(self, with_dither: bool = False) -> Dict[Tuple[int, ...], np.ndarray]:
        """
        Create a codebook mapping encoded vectors to lattice points.
        
        Parameters:
        -----------
        with_dither : bool, optional
            Whether to apply dithering when creating the codebook. Default is False.
            
        Returns:
        --------
        dict
            Dictionary mapping encoded vector tuples to corresponding lattice points.
        """
        d = self.G.shape[0]
        codebook = {}
        
        # Use itertools.product for more efficient generation
        from itertools import product
        encoding_vectors = np.array(list(product(*[range(self.q)] * d)))
        
        for enc in encoding_vectors:
            lattice_point = self.decode(enc, 0, with_dither)
            codebook[tuple(enc)] = lattice_point
            
        return codebook
    
    def get_quantization_error(self, x: np.ndarray, with_dither: bool = False) -> float:
        """
        Calculate the quantization error for a given vector.
        
        Parameters:
        -----------
        x : numpy.ndarray
            Input vector.
        with_dither : bool, optional
            Whether to apply dithering. Default is False.
            
        Returns:
        --------
        float
            Mean squared error between input and quantized output.
        """
        x_quantized = self.quantize(x, with_dither)
        return np.mean((x - x_quantized) ** 2)
    
    def get_rate(self, T_values: List[int]) -> float:
        """
        Calculate the rate (bits per dimension) for given scaling counts.
        
        Parameters:
        -----------
        T_values : list
            List of scaling counts from encoding multiple vectors.
            
        Returns:
        --------
        float
            Rate in bits per dimension.
        """
        from .utils import calculate_t_entropy
        H_T, _ = calculate_t_entropy(T_values, self.q)
        return H_T + np.log2(self.q ** self.G.shape[0])
    
    def batch_encode(self, X: np.ndarray, with_dither: bool = False) -> Tuple[List[np.ndarray], List[int]]:
        """
        Encode multiple vectors efficiently.
        
        Parameters:
        -----------
        X : numpy.ndarray
            Input matrix where each row is a vector to encode.
        with_dither : bool, optional
            Whether to apply dithering. Default is False.
            
        Returns:
        --------
        tuple
            (encoded_vectors, scaling_counts) where encoded_vectors is a list
            of encoded vectors and scaling_counts is a list of scaling counts.
        """
        if X.ndim == 1:
            X = X.reshape(1, -1)
            
        encoded_vectors = []
        scaling_counts = []
        
        for i in range(X.shape[0]):
            enc, T = self.encode(X[i], with_dither)
            encoded_vectors.append(enc)
            scaling_counts.append(T)
            
        return encoded_vectors, scaling_counts
    
    def batch_decode(self, encoded_vectors: List[np.ndarray], scaling_counts: List[int], 
                    with_dither: bool = False) -> np.ndarray:
        """
        Decode multiple vectors efficiently.
        
        Parameters:
        -----------
        encoded_vectors : list
            List of encoded vectors.
        scaling_counts : list
            List of scaling counts corresponding to each encoded vector.
        with_dither : bool, optional
            Whether dithering was applied during encoding. Default is False.
            
        Returns:
        --------
        numpy.ndarray
            Matrix where each row is a decoded vector.
        """
        if len(encoded_vectors) != len(scaling_counts):
            raise ValueError("Number of encoded vectors must match number of scaling counts")
            
        decoded_vectors = []
        for enc, T in zip(encoded_vectors, scaling_counts):
            decoded = self.decode(enc, T, with_dither)
            decoded_vectors.append(decoded)
            
        return np.array(decoded_vectors)
    
    def get_lattice_volume(self) -> float:
        """
        Calculate the volume of the fundamental cell of the lattice.
        
        Returns:
        --------
        float
            Volume of the fundamental cell.
        """
        return abs(np.linalg.det(self.G))
    
    def get_packing_density(self) -> float:
        """
        Calculate the packing density of the lattice.
        
        Returns:
        --------
        float
            Packing density (volume of sphere / volume of fundamental cell).
        """
        # This is a simplified calculation - actual packing density depends on the lattice type
        import math
        sphere_volume = np.pi ** (self._dim / 2) / math.gamma(self._dim / 2 + 1)
        return sphere_volume / self.get_lattice_volume()
    
    def __repr__(self) -> str:
        """String representation of the quantizer."""
        return (f"NLQ(dim={self._dim}, q={self.q}, "
                f"beta={self.beta:.3f}, alpha={self.alpha:.3f}, "
                f"overload={self.overload})")
    
    def __str__(self) -> str:
        """String representation of the quantizer."""
        return self.__repr__()
    
    @classmethod
    def create_z2_quantizer(cls, q: int, beta: float = 1.0, alpha: float = 1.0, 
                           eps: float = 1e-8, overload: bool = True, 
                           with_tie_dither: bool = True, with_dither: bool = False) -> 'NLQ':
        """
        Create a quantizer for the Z² lattice (identity matrix).
        
        Parameters:
        -----------
        q : int
            Quantization parameter.
        beta : float, optional
            Scaling parameter. Default is 1.0.
        alpha : float, optional
            Scaling parameter for overload handling. Default is 1.0.
        eps : float, optional
            Small perturbation parameter. Default is 1e-8.
        overload : bool, optional
            Whether to handle overload by scaling. Default is True.
            
        Returns:
        --------
        NestedLatticeQuantizer
            Configured quantizer for Z² lattice.
        """
        from .utils import get_z2
        
        def closest_point_zn(x):
            return np.floor(x + 0.5)
        
        G = get_z2()

        config = QuantizerConfig(q, beta, alpha, eps, overload, 10, with_tie_dither, with_dither)
        
        return cls(G, closest_point_zn, q, beta, alpha, eps, with_tie_dither=with_tie_dither, with_dither=with_dither, config=config)
    
    @classmethod
    def create_d4_quantizer(cls, q: int, beta: float = 1.0, alpha: float = 1.0,
                           eps: float = 1e-8, overload: bool = True,
                           with_tie_dither: bool = True, with_dither: bool = False) -> 'NLQ':
        """
        Create a quantizer for the D₄ lattice.
        
        Parameters:
        -----------
        q : int
            Quantization parameter.
        beta : float, optional
            Scaling parameter. Default is 1.0.
        alpha : float, optional
            Scaling parameter for overload handling. Default is 1.0.
        eps : float, optional
            Small perturbation parameter. Default is 1e-8.
        overload : bool, optional
            Whether to handle overload by scaling. Default is True.
            
        Returns:
        --------
        NestedLatticeQuantizer
            Configured quantizer for D₄ lattice.
        """
        from .utils import get_d4, closest_point_Dn
        
        G = get_d4()
        config = QuantizerConfig(q, beta, alpha, eps, overload, 10, with_tie_dither, with_dither)
        
        return cls(G, closest_point_Dn, q, beta, alpha, eps, with_tie_dither=with_tie_dither, with_dither=with_dither, config=config)
    
    @classmethod
    def create_e8_quantizer(cls, q: int, beta: float = 1.0, alpha: float = 1.0,
                           eps: float = 1e-8, overload: bool = True,
                           with_tie_dither: bool = True, with_dither: bool = False) -> 'NLQ':
        """
        Create a quantizer for the E₈ lattice.
        
        Parameters:
        -----------
        q : int
            Quantization parameter.
        beta : float, optional
            Scaling parameter. Default is 1.0.
        alpha : float, optional
            Scaling parameter for overload handling. Default is 1.0.
        eps : float, optional
            Small perturbation parameter. Default is 1e-8.
        overload : bool, optional
            Whether to handle overload by scaling. Default is True.
            
        Returns:
        --------
        NestedLatticeQuantizer
            Configured quantizer for E₈ lattice.
        """
        from .utils import get_e8, closest_point_E8
        
        G = get_e8()
        config = QuantizerConfig(q, beta, alpha, eps, overload, 10, with_tie_dither, with_dither)
        
        return cls(G, closest_point_E8, q, beta, alpha, eps, with_tie_dither=with_tie_dither, with_dither=with_dither, config=config)
    
    def get_config(self) -> QuantizerConfig:
        """
        Get the current configuration of the quantizer.
        
        Returns:
        --------
        QuantizerConfig
            Current configuration.
        """
        return self.config
    
    def update_config(self, new_config: QuantizerConfig) -> None:
        """
        Update the quantizer configuration.
        
        Parameters:
        -----------
        new_config : QuantizerConfig
            New configuration to apply.
        """
        self.config = new_config
        self.q = new_config.q
        self.beta = new_config.beta
        self.alpha = new_config.alpha
        self.eps = new_config.eps
        self.overload = new_config.overload
        self.max_scaling_iterations = new_config.max_scaling_iterations
        self._q_squared = self.q ** 2