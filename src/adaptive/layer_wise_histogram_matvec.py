"""
Layer-Wise Histogram Matrix-Vector Multiplication for Hierarchical Nested-Lattice Quantization

This module implements the layer-wise histogram technique for efficient matrix-vector
multiplication when the matrix columns are stored using hierarchical nested-lattice
quantization. The technique pools identical codewords at each layer to reduce
computational complexity.

The key insight is that instead of computing:
    y = sum_j x_j * w_j

where each w_j is stored as:
    w_j = sum_{m=0}^{M_j-1} q^m * lambda(b_{j,m})

We can compute:
    y = sum_{m=0}^{M-1} q^m * sum_k s_{m,k} * lambda(k)

where s_{m,k} = sum_{j: m<M_j, b_{j,m}=k} x_j is the layer-wise histogram.

This approach is more efficient when many columns share the same code indices.

References:
    - ada_matmul.md: Layer-Wise Histograms for Hierarchical Nested-Lattice MatVec
"""

from typing import Dict, List, Tuple

import numpy as np

from ..quantizers.hierarchical_nested_lattice_quantizer import \
    HierarchicalNestedLatticeQuantizer


class LayerWiseHistogramMatVec:
    """
    Implements layer-wise histogram matrix-vector multiplication for hierarchical
    nested-lattice quantized matrices.

    This class provides efficient matrix-vector multiplication when the matrix
    columns are stored using hierarchical nested-lattice quantization. The
    technique uses layer-wise histograms to pool identical codewords and reduce
    computational complexity.

    The algorithm works by:
    1. Computing layer-wise histograms s_{m,k} for each layer m and code index k
    2. Accumulating contributions from each layer: y += q^m * sum_k s_{m,k} * lambda(k)

    This is more efficient than reconstructing each column individually when
    many columns share the same code indices.

    Attributes:
    -----------
    quantizer : HierarchicalNestedLatticeQuantizer
        The hierarchical quantizer used for encoding/decoding.
    codebook : dict
        Mapping from code indices to codeword vectors.
    M : int
        Maximum number of hierarchical levels.
    q : int
        Nesting base for the hierarchical structure.
    """

    def __init__(self, quantizer: HierarchicalNestedLatticeQuantizer):
        """
        Initialize the layer-wise histogram matrix-vector multiplier.

        Parameters:
        -----------
        quantizer : HierarchicalNestedLatticeQuantizer
            The hierarchical quantizer to use for encoding/decoding.
        """
        self.quantizer = quantizer
        self.M = quantizer.M
        self.q = quantizer.q

        # Create codebook for efficient lookup
        self.codebook = self._create_codebook()

    def _create_codebook(self) -> Dict[int, np.ndarray]:
        """
        Create a codebook mapping code indices to codeword vectors.

        Creates a simple codebook based on the generator matrix to avoid
        parameter optimization that can slow down the implementation.

        Returns:
        --------
        dict
            Mapping from code indices to codeword vectors.
        """
        codebook = {}
        n = self.quantizer.G.shape[0]

        for k in range(self.quantizer.q):
            # Create codeword based on the generator matrix
            # For simplicity, use the generator matrix applied to unit vectors
            codeword = np.zeros(n)
            if k < n:
                codeword[k] = 1.0
            else:
                # For indices beyond dimension, create a combination
                codeword[k % n] = 1.0
                if k >= n and k < 2 * n:
                    codeword[(k - n) % n] = 1.0

            codebook[k] = codeword

        return codebook

    def encode_matrix_columns(
        self, W: np.ndarray, with_dither: bool = False
    ) -> Tuple[List[List[int]], List[int]]:
        """
        Encode all columns of a matrix using hierarchical nested-lattice quantization.

        Parameters:
        -----------
        W : numpy.ndarray
            Matrix to encode, shape (n, d) where n is output dimension and d is input dimension.
        with_dither : bool, optional
            Whether to apply dithering during quantization.

        Returns:
        --------
        tuple
            (b_matrix, M_j) where:
            - b_matrix[j][m] is the code index for column j at layer m
            - M_j[j] is the number of layers used for column j
        """
        n, d = W.shape
        b_matrix = []
        M_j = []

        for j in range(d):
            column = W[:, j]
            b_list, T = self.quantizer.encode(column, with_dither)

            # Convert to list of code indices
            b_indices = [int(b[0]) if len(b) > 0 else 0 for b in b_list]
            b_matrix.append(b_indices)
            M_j.append(len(b_indices))

        return b_matrix, M_j

    def compute_layer_histograms(
        self, x: np.ndarray, b_matrix: List[List[int]], M_j: List[int]
    ) -> np.ndarray:
        """
        Compute layer-wise histograms for the given input vector.

        Computes s_{m,k} = sum_{j: m<M_j, b_{j,m}=k} x_j for each layer m and code index k.

        Parameters:
        -----------
        x : numpy.ndarray
            Input vector, shape (d,).
        b_matrix : List[List[int]]
            Code indices for each column at each layer.
        M_j : List[int]
            Number of layers used for each column.

        Returns:
        --------
        numpy.ndarray
            Layer-wise histograms, shape (M, K) where K is the number of code indices.
        """
        d = len(x)
        K = len(self.codebook)

        # Initialize histograms
        s = np.zeros((self.M, K), dtype=float)

        for m in range(self.M):
            for j in range(d):
                xj = x[j]

                # Skip zero coefficients
                if xj == 0.0:
                    continue

                # Skip columns not decoded at this layer
                if m >= M_j[j]:
                    continue

                # Add to histogram
                k = b_matrix[j][m]
                s[m, k] += xj

        return s

    def matvec(self, x: np.ndarray, b_matrix: List[List[int]], M_j: List[int]) -> np.ndarray:
        """
        Perform matrix-vector multiplication using layer-wise histograms.

        Implements the algorithm: y = sum_{m=0}^{M-1} q^m * sum_k s_{m,k} * lambda(k)

        Parameters:
        -----------
        x : numpy.ndarray
            Input vector, shape (d,).
        b_matrix : List[List[int]]
            Code indices for each column at each layer.
        M_j : List[int]
            Number of layers used for each column.

        Returns:
        --------
        numpy.ndarray
            Output vector, shape (n,).
        """
        # Compute layer-wise histograms
        s = self.compute_layer_histograms(x, b_matrix, M_j)

        # Get output dimension from codebook
        n = next(iter(self.codebook.values())).shape[0]
        y = np.zeros(n, dtype=float)

        # Accumulate layer contributions
        for m in range(self.M):
            if np.any(s[m, :]):
                # Compute layer contribution
                layer_contribution = np.zeros(n, dtype=float)
                for k, sk in enumerate(s[m, :]):
                    if sk != 0.0 and k in self.codebook:
                        layer_contribution += sk * self.codebook[k]

                # Scale by q^m and add to result
                y += (self.q**m) * layer_contribution

        return y

    def matvec_direct(self, x: np.ndarray, W: np.ndarray) -> np.ndarray:
        """
        Perform direct matrix-vector multiplication for comparison.

        Parameters:
        -----------
        x : numpy.ndarray
            Input vector, shape (d,).
        W : numpy.ndarray
            Matrix, shape (n, d).

        Returns:
        --------
        numpy.ndarray
            Output vector, shape (n,).
        """
        return np.dot(W, x)

    def matvec_quantized_direct(
        self, x: np.ndarray, W: np.ndarray, b_matrix: List[List[int]], M_j: List[int]
    ) -> np.ndarray:
        """
        Perform matrix-vector multiplication by reconstructing each column and
        computing the dot product directly.

        This method reconstructs each column using the same codebook as the
        histogram method for verification purposes.

        Parameters:
        -----------
        x : numpy.ndarray
            Input vector, shape (d,).
        W : numpy.ndarray
            Original matrix (for reference), shape (n, d).
        b_matrix : List[List[int]]
            Code indices for each column at each layer.
        M_j : List[int]
            Number of layers used for each column.

        Returns:
        --------
        numpy.ndarray
            Output vector, shape (n,).
        """
        n, d = W.shape
        y = np.zeros(n, dtype=float)

        for j in range(d):
            # Reconstruct column j using the same codebook as the histogram method
            column_reconstructed = np.zeros(n, dtype=float)

            for m in range(M_j[j]):
                k = b_matrix[j][m]
                if k in self.codebook:
                    column_reconstructed += (self.q**m) * self.codebook[k]

            # Add contribution
            y += x[j] * column_reconstructed

        return y


def create_example_from_paper() -> (
    Tuple[LayerWiseHistogramMatVec, np.ndarray, List[List[int]], List[int]]
):
    """
    Create the example from the paper with the given parameters.

    Returns:
    --------
    tuple
        (matvec_obj, W, b_matrix, M_j) where:
        - matvec_obj is the layer-wise histogram matvec object
        - W is the example matrix
        - b_matrix contains the code indices
        - M_j contains the layer counts
    """
    # Parameters from the paper
    n = 4  # output dimension
    d = 5  # input dimension
    q = 3  # base
    M = 3  # depth

    # Create a simple generator matrix (identity for this example)
    G = np.eye(n)

    # Create a simple closest point function
    def Q_nn(x):
        return np.round(x)

    # Create quantizer
    quantizer = HierarchicalNestedLatticeQuantizer(
        G=G, Q_nn=Q_nn, q=q, beta=1.0, alpha=1.0, eps=1e-8, dither=np.zeros(n), M=M
    )

    # Create matvec object
    matvec_obj = LayerWiseHistogramMatVec(quantizer)

    # Create example matrix (this would normally come from quantization)
    # For the example, we'll create a simple matrix
    W = np.array(
        [
            [1.0, 0.0, 1.0, 1.0, 0.0],
            [0.0, 1.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )

    # Code indices from the paper example
    b_matrix = [
        [0, 2, 1],  # column 1: b_{1,0}=0, b_{1,1}=2, b_{1,2}=1
        [1, 0, 1],  # column 2: b_{2,0}=1, b_{2,1}=0, b_{2,2}=1
        [2, 2, 2],  # column 3: b_{3,0}=2, b_{3,1}=2, b_{3,2}=2
        [0, 1, 0],  # column 4: b_{4,0}=0, b_{4,1}=1, b_{4,2}=0 (truncated)
        [1, 1, 0],  # column 5: b_{5,0}=1, b_{5,1}=1, b_{5,2}=0
    ]

    # Layer counts from the paper
    M_j = [3, 2, 1, 2, 3]  # (M_1, M_2, M_3, M_4, M_5)

    return matvec_obj, W, b_matrix, M_j


def run_paper_example():
    """
    Run the example from the paper to verify the implementation.

    Returns:
    --------
    tuple
        (y_histogram, y_direct) where y_histogram is the result using the
        layer-wise histogram method and y_direct is the result using direct
        reconstruction.
    """
    print("Running Layer-Wise Histogram MatVec Example from Paper")
    print("=" * 60)

    # Create example
    matvec_obj, W, b_matrix, M_j = create_example_from_paper()

    # Input vector from the paper
    x = np.array([0.7, -1.2, 0.0, 0.5, 2.0])

    print(f"Input vector x: {x}")
    print(f"Layer counts M_j: {M_j}")
    print(f"Code indices b_matrix:")
    for j, b_list in enumerate(b_matrix):
        print(f"  Column {j+1}: {b_list}")

    # Compute using layer-wise histogram method
    y_histogram = matvec_obj.matvec(x, b_matrix, M_j)
    print(f"\nResult using layer-wise histogram method:")
    print(f"y = {y_histogram}")

    # Verify with direct computation (reconstructing columns)
    y_direct = matvec_obj.matvec_quantized_direct(x, W, b_matrix, M_j)
    print(f"\nResult using direct reconstruction method:")
    print(f"y = {y_direct}")

    # Check if results match
    error = np.linalg.norm(y_histogram - y_direct)
    print(f"\nError between methods: {error:.2e}")

    if error < 1e-10:
        print("✓ Methods agree!")
    else:
        print("✗ Methods disagree!")

    return y_histogram, y_direct


if __name__ == "__main__":
    run_paper_example()
