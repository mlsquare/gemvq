import numpy as np
from .closest_point import custom_round
from .nested_lattice_quantizer import NestedLatticeQuantizer as NQ


class HierarchicalNestedLatticeQuantizer:
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
    Q_nn : function
        Closest point function for the lattice.
    q : int
        Quantization parameter (alphabet size).
    beta : float
        Scaling parameter for quantization.
    alpha : float
        Scaling parameter for overload handling.
    eps : float
        Small perturbation parameter.
    dither : numpy.ndarray
        Dither vector for randomized quantization.
    M : int
        Number of hierarchical levels.
    G_inv : numpy.ndarray
        Inverse of the generator matrix.
        
    Notes:
    ------
    The hierarchical quantizer uses M levels of quantization, where each level
    provides a refinement of the previous level. This approach enables efficient
    inner product estimation and achieves better rate-distortion performance
    compared to single-level quantization.
    """
    
    def __init__(self, G, Q_nn, q, beta, alpha, eps, dither, M):
        """
        Initialize the Hierarchical Nested Lattice Quantizer.
        
        Parameters:
        -----------
        G : numpy.ndarray
            Generator matrix for the lattice.
        Q_nn : function
            Closest point function for the lattice (e.g., closest_point_Dn).
        q : int
            Quantization parameter (alphabet size).
        beta : float
            Scaling parameter for quantization.
        alpha : float
            Scaling parameter for overload handling.
        eps : float
            Small perturbation parameter.
        dither : numpy.ndarray
            Dither vector for randomized quantization.
        M : int
            Number of hierarchical levels.
        """
        self.G = G
        self.Q_nn = lambda x: Q_nn(x + eps)
        self.q = q
        self.beta = beta
        self.alpha = alpha
        self.eps = eps
        self.dither = dither
        self.M = M
        self.G_inv = np.linalg.inv(G)

    def _encode(self, x, with_dither):
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
        tuple
            (encoding_vectors, overload_error) where encoding_vectors is a
            tuple of M encoding vectors and overload_error indicates if
            overload occurred.
        """
        x = (x / self.beta)
        if with_dither:
            x = x + self.dither
        x_l = x
        encoding_vectors = []

        for _ in range(self.M):
            x_l = self.Q_nn(x_l)
            b_i = custom_round(np.mod(np.dot(self.G_inv, x_l), self.q)).astype(int)
            encoding_vectors.append(b_i)
            x_l = x_l / self.q

        overload_error = not np.allclose(self.Q_nn(x_l), 0, atol=1e-8)
        return tuple(encoding_vectors), overload_error

    def encode(self, x, with_dither):
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
        tuple
            (b_list, T) where b_list is a tuple of M encoding vectors and
            T is the number of scaling operations performed to handle overload.
        """
        b_list, did_overload = self._encode(x, with_dither)
        t = 0
        while did_overload:
            t += 1
            x = x / (2 ** self.alpha)
            b_list, did_overload = self._encode(x, with_dither)
        return b_list, t

    def q_Q(self, x):
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

    def _decode(self, b_list, with_dither):
        """
        Internal decoding function that performs hierarchical reconstruction.
        
        This method reconstructs the original vector from M levels of
        encoding vectors using the hierarchical decoding algorithm.
        
        Parameters:
        -----------
        b_list : tuple
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
            x_i_hat = np.dot(self.G, b) - self.q_Q(np.dot(self.G, b))
            x_hat_list.append(x_i_hat)
        x_hat = sum([np.power(self.q, i) * x_i for i, x_i in enumerate(x_hat_list)])
        if with_dither:
            x_hat = x_hat - self.dither
        return self.beta * x_hat

    def decode(self, b_list, T, with_dither):
        """
        Decode hierarchical encoding vectors back to the original space.
        
        This method reconstructs the original vector from its hierarchical
        encoding, accounting for any scaling that was applied during encoding.
        
        Parameters:
        -----------
        b_list : tuple
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

    def quantize(self, x, with_dither):
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

    def create_q_codebook(self, with_dither):
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
        dict
            Dictionary mapping encoding vectors (as tuples) to lattice points.
            
        Notes:
        ------
        The codebook is created using the nested lattice quantizer with
        the same parameters as this hierarchical quantizer, providing
        a mapping for lookup-based operations.
        """
        if with_dither:
            nq = NQ(self.G, self.Q_nn, self.q, self.beta, self.alpha, eps=self.eps, dither=self.dither)
        else:
            dither = np.array([[0]*len(self.G)])
            nq = NQ(self.G, self.Q_nn, self.q, self.beta, self.alpha, eps=self.eps, dither=dither)
        return nq.create_codebook(with_dither)
