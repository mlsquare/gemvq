import numpy as np


class NestedLatticeQuantizer:
    """
    Classic Nested Lattice Quantizer implementing standard Voronoi code quantization.

    This class implements the traditional nested lattice quantization approach
    where vectors are quantized using a single-level Voronoi code. It serves as
    a reference implementation for comparison with the hierarchical approach.

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
    dither : numpy.ndarray
        Dither vector for randomized quantization.
    G_inv : numpy.ndarray
        Inverse of the generator matrix.

    Notes:
    ------
    This quantizer uses a single-level quantization approach where vectors
    are mapped to the nearest lattice point in the Voronoi region. It provides
    a baseline for comparison with hierarchical quantization methods.
    """

    def __init__(self, G, Q_nn, q, beta, alpha, eps, dither, M=None, decoding="full"):
        """
        Initialize the Nested Lattice Quantizer.

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
        M : int, optional
            Number of levels (not used in classic quantizer, kept for compatibility).
        decoding : str, optional
            Default decoding method to use ('full').
            Default is 'full'.
        """
        self.G = G
        self.Q_nn = lambda x: Q_nn(x + eps)
        self.q = q
        self.beta = beta
        self.alpha = alpha
        self.dither = dither
        self.decoding = decoding
        self.G_inv = np.linalg.inv(G)

    def get_default_decoding(self, enc, T, with_dither):
        """
        Get the default decoding based on the decoding parameter.

        Parameters:
        -----------
        enc : numpy.ndarray
            Encoding vector to be decoded.
        T : int
            Number of scaling operations that were applied during encoding.
        with_dither : bool
            Whether dithering was applied during encoding.

        Returns:
        --------
        numpy.ndarray
            Decoded result based on the decoding parameter.
        """
        if self.decoding == "full":
            return self.decode(enc, T, with_dither)
        else:
            raise ValueError(f"Unknown decoding method: {self.decoding}")

    def _encode(self, x, with_dither):
        """
        Internal encoding function that performs the core quantization.

        Parameters:
        -----------
        x : numpy.ndarray
            Input vector to be quantized.
        with_dither : bool
            Whether to apply dithering during quantization.

        Returns:
        --------
        tuple
            (enc, overload_error) where enc is the encoding vector and
            overload_error indicates if overload occurred.
        """
        x_tag = x / self.beta
        if with_dither:
            x_tag = x_tag + self.dither
        t = self.Q_nn(x_tag)
        t = self.Q_nn(x_tag)
        y = np.dot(self.G_inv, t)
        enc = np.mod(np.round(y), self.q).astype(int)

        overload_error = not np.allclose(self.Q_nn(t / self.q), 0, atol=1e-8)
        return enc, overload_error

    def encode(self, x, with_dither):
        """
        Encode a vector using nested lattice quantization with overload handling.

        This method quantizes the input vector and handles overload by scaling
        the vector until quantization succeeds.

        Parameters:
        -----------
        x : numpy.ndarray
            Input vector to be quantized.
        with_dither : bool
            Whether to apply dithering during quantization.

        Returns:
        --------
        tuple
            (enc, T) where enc is the encoding vector and T is the number
            of scaling operations performed to handle overload.
        """
        enc, did_overload = self._encode(x, with_dither)
        t = 0
        while did_overload:
            t += 1
            x = x / (2**self.alpha)
            enc, did_overload = self._encode(x, with_dither)
        return enc, t

    def _decode(self, y, with_dither):
        """
        Internal decoding function that performs the core reconstruction.

        Parameters:
        -----------
        y : numpy.ndarray
            Encoding vector to be decoded.
        with_dither : bool
            Whether dithering was applied during encoding.

        Returns:
        --------
        numpy.ndarray
            Reconstructed vector.
        """
        x_p = np.dot(self.G, y)
        if with_dither:
            x_p = x_p - self.dither
        x_pp = self.q * self.Q_nn(x_p / self.q)
        return self.beta * (x_p - x_pp)

    def decode(self, enc, T, with_dither):
        """
        Decode an encoding vector back to the original space.

        This method reconstructs the original vector from its encoding,
        accounting for any scaling that was applied during encoding.

        Parameters:
        -----------
        enc : numpy.ndarray
            Encoding vector to be decoded.
        T : int
            Number of scaling operations that were applied during encoding.
        with_dither : bool
            Whether dithering was applied during encoding.

        Returns:
        --------
        numpy.ndarray
            Reconstructed vector.
        """
        return self._decode(enc, with_dither) * (2 ** (self.alpha * T))

    def quantize(self, x):
        """
        Complete quantization process: encode and decode a vector.

        This is a convenience method that performs both encoding and decoding
        in a single call, returning the quantized version of the input vector.

        Parameters:
        -----------
        x : numpy.ndarray
            Input vector to be quantized.

        Returns:
        --------
        numpy.ndarray
            Quantized version of the input vector.
        """
        enc, T = self.encode(x)
        return self.decode(enc, T)

    def create_codebook(self, with_dither):
        """
        Create a codebook mapping encoding vectors to lattice points.

        This method generates all possible encoding vectors and maps them
        to their corresponding lattice points in the reconstruction space.

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
        The codebook contains all possible quantized representations and
        can be used for analysis or lookup-based decoding.
        """
        d = self.G.shape[0]
        codebook = {}
        encoding_vectors = np.array(np.meshgrid(*[range(self.q)] * d)).T.reshape(-1, d)
        for enc in encoding_vectors:
            lattice_point = self.decode(enc, 0, with_dither)
            codebook[tuple(enc)] = lattice_point
        return codebook
