import numpy as np
from typing import Tuple, Optional, Callable, Dict, Any, List, Union
from dataclasses import dataclass
import warnings

from .utils import custom_round
from .nlq import NLQ as NQ


@dataclass
class HNLQConfig:
    """
    Configuration class for hierarchical nested lattice quantizer parameters.
    
    This class provides a structured way to manage HNLQ parameters
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
    M : int
        Number of hierarchical levels.
    overload : bool
        Whether to handle overload by scaling.
    decoding : str
        Default decoding method ('full', 'coarse_to_fine', 'progressive').
    max_scaling_iterations : int
        Maximum number of scaling iterations to prevent infinite loops.
    """
    q: int
    beta: float
    alpha: float
    eps: float
    M: int
    overload: bool = True
    decoding: str = "full"
    max_scaling_iterations: int = 10
    
    def __post_init__(self):
        """Validate parameters after initialization."""
        if self.q <= 0:
            raise ValueError("Quantization parameter q must be positive")
        if self.beta <= 0:
            raise ValueError("Scaling parameter beta must be positive")
        if self.alpha <= 0:
            raise ValueError("Scaling parameter alpha must be positive")
        if self.M <= 0:
            raise ValueError("Number of hierarchical levels M must be positive")
        if self.max_scaling_iterations <= 0:
            raise ValueError("max_scaling_iterations must be positive")
        if self.decoding not in ["full", "coarse_to_fine", "progressive"]:
            raise ValueError(f"Unknown decoding method: {self.decoding}")
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'HNLQConfig':
        """Create configuration from dictionary."""
        return cls(**config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'q': self.q,
            'beta': self.beta,
            'alpha': self.alpha,
            'eps': self.eps,
            'M': self.M,
            'overload': self.overload,
            'decoding': self.decoding,
            'max_scaling_iterations': self.max_scaling_iterations
        }


class HNLQ:
    """
    Hierarchical Nested Lattice Quantizer implementing multi-level quantization.

    This class implements a hierarchical quantization approach that uses multiple
    levels of quantization to achieve better rate-distortion performance. The
    hierarchical structure allows for successive refinement and efficient inner
    product estimation using small lookup tables.

    Attributes:
    -----------
    G : numpy.ndarray
        Generator matrix for the lattice.
    Q_nn : Callable
        Closest point function for the lattice.
    config : HNLQConfig
        Configuration object containing all quantizer parameters.
    dither : numpy.ndarray
        Dither vector for randomized quantization.
    G_inv : numpy.ndarray
        Inverse of the generator matrix.

    Notes:
    ------
    The hierarchical quantizer uses M levels of quantization, where each level
    provides a refinement of the previous level. This approach enables efficient
    inner product estimation and achieves better rate-distortion performance
    compared to single-level quantization.
    """

    def __init__(
        self, 
        G: np.ndarray, 
        Q_nn: Callable, 
        config: Union[HNLQConfig, Dict[str, Any]], 
        dither: np.ndarray
    ):
        """
        Initialize the Hierarchical Nested Lattice Quantizer.

        Parameters:
        -----------
        G : numpy.ndarray
            Generator matrix for the lattice.
        Q_nn : Callable
            Closest point function for the lattice (e.g., closest_point_Dn).
        config : HNLQConfig or Dict[str, Any]
            Configuration object or dictionary with quantizer parameters.
        dither : numpy.ndarray
            Dither vector for randomized quantization.
        """
        if isinstance(config, dict):
            config = HNLQConfig.from_dict(config)
        
        self.G = G
        # Handle eps as either scalar or vector for backward compatibility
        if np.isscalar(config.eps):
            self.Q_nn = lambda x: Q_nn(x + config.eps)
        else:
            self.Q_nn = lambda x: Q_nn(x + config.eps)
        self.config = config
        self.dither = dither
        self.G_inv = np.linalg.inv(G)
        
        # Validate dither dimensions and ensure correct format
        if dither.ndim == 2:
            if dither.shape[1] != G.shape[0]:
                raise ValueError(f"Dither dimensions {dither.shape} don't match generator matrix {G.shape}")
        else:
            if dither.shape[0] != G.shape[0]:
                raise ValueError(f"Dither dimensions {dither.shape} don't match generator matrix {G.shape}")
            # Convert to 2D format for consistency
            dither = dither.reshape(1, -1)

    # ============================================================================
    # Properties
    # ============================================================================

    @property
    def q(self) -> int:
        """Get quantization parameter."""
        return self.config.q
    
    @property
    def beta(self) -> float:
        """Get scaling parameter."""
        return self.config.beta
    
    @beta.setter
    def beta(self, value: float) -> None:
        """Set scaling parameter."""
        self.config.beta = value
    
    @property
    def alpha(self) -> float:
        """Get overload scaling parameter."""
        return self.config.alpha
    
    @alpha.setter
    def alpha(self, value: float) -> None:
        """Set overload scaling parameter."""
        self.config.alpha = value
    
    @property
    def eps(self) -> float:
        """Get perturbation parameter."""
        return self.config.eps
    
    @property
    def M(self) -> int:
        """Get number of hierarchical levels."""
        return self.config.M
    
    @property
    def overload(self) -> bool:
        """Get overload handling flag."""
        return self.config.overload
    
    @property
    def decoding(self) -> str:
        """Get default decoding method."""
        return self.config.decoding

    # ============================================================================
    # Encoding Methods
    # ============================================================================

    def _encode(self, x: np.ndarray, with_dither: bool) -> Tuple[Tuple[np.ndarray, ...], bool]:
        """
        Internal encoding function that performs hierarchical quantization.

        This method implements the core hierarchical encoding algorithm,
        producing M levels of encoding vectors.

        Parameters:
        -----------
        x : numpy.ndarray
            Input vector to be quantized.
        with_dither : bool
            Whether to apply dithering during quantization.

        Returns:
        --------
        Tuple[Tuple[numpy.ndarray, ...], bool]
            (encoding_vectors, overload_error) where encoding_vectors is a
            tuple of M encoding vectors and overload_error indicates if
            overload occurred.
        """
        # Ensure x is a column vector
        x = np.asarray(x).flatten()
        x = x / self.beta
        if with_dither:
            x = x + self.dither.flatten()
        x_l = x
        encoding_vectors = []

        for _ in range(self.M):
            x_l = self.Q_nn(x_l)
            b_i = custom_round(np.mod(np.dot(self.G_inv, x_l), self.q)).astype(int)
            encoding_vectors.append(b_i)
            x_l = x_l / self.q

        overload_error = not np.allclose(self.Q_nn(x_l), 0, atol=1e-8)
        return tuple(encoding_vectors), overload_error

    def encode(self, x: np.ndarray, with_dither: bool) -> Tuple[Tuple[np.ndarray, ...], int]:
        """
        Encode a vector using hierarchical nested lattice quantization.

        This method quantizes the input vector using M hierarchical levels
        and handles overload by scaling the vector until quantization succeeds.

        Parameters:
        -----------
        x : numpy.ndarray
            Input vector to be quantized.
        with_dither : bool
            Whether to apply dithering during quantization.

        Returns:
        --------
        Tuple[Tuple[numpy.ndarray, ...], int]
            (b_list, T) where b_list is a tuple of M encoding vectors and
            T is the number of scaling operations performed to handle overload.
        """
        b_list, did_overload = self._encode(x, with_dither)
        t = 0
        
        if self.overload:
            while did_overload and t < self.config.max_scaling_iterations:
                t += 1
                x = x / (2**self.alpha)
                b_list, did_overload = self._encode(x, with_dither)
            
            if did_overload:
                warnings.warn(
                    f"Overload not resolved after {self.config.max_scaling_iterations} iterations. "
                    "Consider increasing max_scaling_iterations or adjusting parameters."
                )
        else:
            b_list, did_overload = self._encode(x, with_dither)
            
        return b_list, t

    # ============================================================================
    # Decoding Methods
    # ============================================================================

    def _decode(self, b_list: Tuple[np.ndarray, ...], with_dither: bool) -> np.ndarray:
        """
        Internal decoding function that performs hierarchical reconstruction.

        This method reconstructs the original vector from M levels of
        encoding vectors using the hierarchical decoding algorithm.

        Parameters:
        -----------
        b_list : Tuple[numpy.ndarray, ...]
            Tuple of M encoding vectors.
        with_dither : bool
            Whether dithering was applied during encoding.

        Returns:
        --------
        numpy.ndarray
            Reconstructed vector.
        """
        x_hat_list = []
        for b in b_list:
            # Use q_Q to properly handle quantization error
            x_i_hat = np.dot(self.G, b) - self.q_Q(np.dot(self.G, b))
            x_hat_list.append(x_i_hat)
        
        x_hat = sum([np.power(self.q, i) * x_i for i, x_i in enumerate(x_hat_list)])
        
        if with_dither:
            x_hat = x_hat - self.dither
            
        return self.beta * x_hat

    def decode(
        self, 
        b_list: Tuple[np.ndarray, ...], 
        T: int, 
        with_dither: bool
    ) -> np.ndarray:
        """
        Decode hierarchical encoding vectors back to the original space.

        This method reconstructs the original vector from its hierarchical
        encoding, accounting for any scaling that was applied during encoding.

        Parameters:
        -----------
        b_list : Tuple[numpy.ndarray, ...]
            Tuple of M encoding vectors.
        T : int
            Number of scaling operations that were applied during encoding.
        with_dither : bool
            Whether dithering was applied during encoding.

        Returns:
        --------
        numpy.ndarray
            Reconstructed vector.
        """
        return self._decode(b_list, with_dither) * (2 ** (self.alpha * T))

    def decode_coarse_to_fine(
        self, 
        b_list: Tuple[np.ndarray, ...], 
        T: int, 
        with_dither: bool, 
        max_level: Optional[int] = None
    ) -> np.ndarray:
        """
        Decode hierarchical encoding vectors with coarse-to-fine reconstruction.

        This method allows decoding from coarse to fine levels, where higher M
        means coarser quantization. The reconstruction can be stopped at any
        level from M-1 down to 0, providing progressive refinement.

        Parameters:
        -----------
        b_list : Tuple[numpy.ndarray, ...]
            Tuple of M encoding vectors.
        T : int
            Number of scaling operations that were applied during encoding.
        with_dither : bool
            Whether dithering was applied during encoding.
        max_level : int, optional
            Maximum level to decode up to (0 <= max_level < M).
            If None, decodes all levels (equivalent to decode method).
            Higher max_level means coarser reconstruction.

        Returns:
        --------
        numpy.ndarray
            Reconstructed vector at the specified level of detail.
        """
        if max_level is None:
            # Default behavior: decode all levels
            return self.decode(b_list, T, with_dither)

        if not (0 <= max_level < self.M):
            raise ValueError(f"max_level must be between 0 and {self.M-1}, got {max_level}")

        # Use the same reconstruction formula as _decode method for consistency
        x_hat_list = []
        for i in range(max_level + 1):
            b = b_list[i]
            # Use q_Q to properly handle quantization error (same as _decode)
            x_i_hat = np.dot(self.G, b) - self.q_Q(np.dot(self.G, b))
            x_hat_list.append(x_i_hat)
        
        x_hat = sum([np.power(self.q, i) * x_i for i, x_i in enumerate(x_hat_list)])
        
        if with_dither:
            x_hat = x_hat - self.dither

        return self.beta * x_hat * (2 ** (self.alpha * T))

    def decode_progressive(
        self, 
        b_list: Tuple[np.ndarray, ...], 
        T: int, 
        with_dither: bool
    ) -> List[np.ndarray]:
        """
        Generate progressive reconstructions from coarse to fine.

        This method returns a list of reconstructions at each level,
        from the coarsest (level 0) to the finest (level M-1).

        Parameters:
        -----------
        b_list : Tuple[numpy.ndarray, ...]
            Tuple of M encoding vectors.
        T : int
            Number of scaling operations that were applied during encoding.
        with_dither : bool
            Whether dithering was applied during encoding.

        Returns:
        --------
        List[numpy.ndarray]
            List of reconstructed vectors, from coarsest to finest.
        """
        reconstructions = []
        for level in range(self.M):  # From 0 to M-1
            reconstruction = self.decode_coarse_to_fine(b_list, T, with_dither, level)
            reconstructions.append(reconstruction)
        return reconstructions

    def get_default_decoding(
        self, 
        b_list: Tuple[np.ndarray, ...], 
        T: int, 
        with_dither: bool, 
        max_level: Optional[int] = None
    ) -> Union[np.ndarray, List[np.ndarray]]:
        """
        Get the default decoding based on the decoding parameter.

        Parameters:
        -----------
        b_list : Tuple[numpy.ndarray, ...]
            Tuple of M encoding vectors.
        T : int
            Number of scaling operations that were applied during encoding.
        with_dither : bool
            Whether dithering was applied during encoding.
        max_level : int, optional
            Maximum level for coarse-to-fine decoding.

        Returns:
        --------
        numpy.ndarray or List[numpy.ndarray]
            Decoded result based on the decoding parameter.
        """
        if self.decoding == "full":
            return self.decode(b_list, T, with_dither)
        elif self.decoding == "coarse_to_fine":
            return self.decode_coarse_to_fine(b_list, T, with_dither, max_level)
        elif self.decoding == "progressive":
            return self.decode_progressive(b_list, T, with_dither)
        else:
            raise ValueError(f"Unknown decoding method: {self.decoding}")

    def decode_with_depth(
        self, 
        b_list: Tuple[np.ndarray, ...], 
        T: int, 
        with_dither: bool, 
        depth: int
    ) -> np.ndarray:
        """
        Decode with a specific depth level.

        Parameters:
        -----------
        b_list : Tuple[numpy.ndarray, ...]
            Tuple of M encoding vectors.
        T : int
            Number of scaling operations that were applied during encoding.
        with_dither : bool
            Whether dithering was applied during encoding.
        depth : int
            Decoding depth (0 to M-1). Higher depth means finer reconstruction.

        Returns:
        --------
        numpy.ndarray
            Decoded result at the specified depth.
        """
        if depth < 0 or depth >= self.M:
            raise ValueError(f"Depth must be between 0 and {self.M-1}, got {depth}")

        return self.decode_coarse_to_fine(b_list, T, with_dither, max_level=depth)

    # ============================================================================
    # Combined Methods
    # ============================================================================

    def quantize(self, x: np.ndarray, with_dither: bool) -> np.ndarray:
        """
        Complete hierarchical quantization process: encode and decode a vector.

        This is a convenience method that performs both encoding and decoding
        in a single call, returning the quantized version of the input vector.

        Parameters:
        -----------
        x : numpy.ndarray
            Input vector to be quantized.
        with_dither : bool
            Whether to apply dithering during quantization.

        Returns:
        --------
        numpy.ndarray
            Quantized version of the input vector.
        """
        b_list, T = self.encode(x, with_dither)
        return self.decode(b_list, T, with_dither)

    # ============================================================================
    # Utility Methods
    # ============================================================================

    def q_Q(self, x: np.ndarray) -> np.ndarray:
        """
        Quantization function that maps x to the nearest lattice point.

        Parameters:
        -----------
        x : numpy.ndarray
            Input vector.

        Returns:
        --------
        numpy.ndarray
            Quantized vector.
        """
        return self.q * self.Q_nn(x / self.q)

    def create_q_codebook(self, with_dither: bool) -> Dict[Tuple[int, ...], np.ndarray]:
        """
        Create a codebook for the hierarchical quantizer.

        This method creates a codebook by using the nested lattice quantizer
        with appropriate parameters. The codebook maps encoding vectors to
        their corresponding lattice points.

        Parameters:
        -----------
        with_dither : bool
            Whether to apply dithering when creating the codebook.

        Returns:
        --------
        Dict[Tuple[int, ...], numpy.ndarray]
            Dictionary mapping encoding vectors (as tuples) to lattice points.

        Notes:
        ------
        The codebook is created using the nested lattice quantizer with
        the same parameters as this hierarchical quantizer, providing
        a mapping for lookup-based operations.
        """
        if with_dither:
            nq = NQ(
                self.G, self.Q_nn, self.q, self.beta, self.alpha, 
                eps=self.eps, dither=self.dither
            )
        else:
            # Create zero dither with correct dimensions for NLQ (1D array)
            dither = np.zeros(self.G.shape[0])
            nq = NQ(
                self.G, self.Q_nn, self.q, self.beta, self.alpha, 
                eps=self.eps, dither=dither
            )
        return nq.create_codebook(with_dither)

    def get_rate_distortion_info(self) -> Dict[str, Any]:
        """
        Get information about the rate-distortion characteristics of the quantizer.
        
        Returns:
        --------
        Dict[str, Any]
            Dictionary containing rate-distortion information.
        """
        return {
            'q': self.q,
            'M': self.M,
            'beta': self.beta,
            'alpha': self.alpha,
            'overload': self.overload,
            'decoding': self.decoding,
            'max_scaling_iterations': self.config.max_scaling_iterations
        }
