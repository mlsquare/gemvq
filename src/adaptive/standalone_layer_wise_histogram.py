"""
Standalone Layer-Wise Histogram Matrix-Vector Multiplication Implementation.

This is a standalone implementation that doesn't import from the main src module
to avoid triggering parameter optimization on import. It implements the same
algorithm as the main implementation but with a simplified codebook.

The layer-wise histogram technique efficiently computes matrix-vector multiplication
when matrix columns are stored using hierarchical nested-lattice quantization by
pooling identical codewords at each layer.

References:
    - ada_matmul.md: Layer-Wise Histograms for Hierarchical Nested-Lattice MatVec
"""

import numpy as np
from typing import List, Tuple, Dict


class StandaloneLayerWiseHistogramMatVec:
    """
    Standalone implementation of layer-wise histogram matrix-vector multiplication.
    
    This class implements the algorithm described in ada_matmul.md without
    depending on the main quantizer modules to avoid parameter optimization.
    
    The algorithm computes:
        y = sum_{m=0}^{M-1} q^m * sum_k s_{m,k} * lambda(k)
    
    where s_{m,k} = sum_{j: m<M_j, b_{j,m}=k} x_j is the layer-wise histogram.
    
    Attributes:
    -----------
    n : int
        Output dimension.
    q : int
        Nesting base for hierarchical structure.
    M : int
        Maximum number of hierarchical levels.
    codebook : dict
        Mapping from code indices to codeword vectors.
    """
    
    def __init__(self, n: int, q: int, M: int):
        """
        Initialize the standalone layer-wise histogram matvec.
        
        Parameters:
        -----------
        n : int
            Output dimension.
        q : int
            Nesting base for hierarchical structure.
        M : int
            Maximum number of hierarchical levels.
        """
        self.n = n
        self.q = q
        self.M = M
        
        # Create simple codebook
        self.codebook = self._create_simple_codebook()
    
    def _create_simple_codebook(self) -> Dict[int, np.ndarray]:
        """
        Create a simple codebook for demonstration.
        
        Creates codewords based on unit vectors and their combinations.
        This avoids the complexity of full lattice quantization while
        still demonstrating the layer-wise histogram technique.
        
        Returns:
        --------
        dict
            Mapping from code indices to codeword vectors.
        """
        codebook = {}
        
        for k in range(self.q):
            # Create simple codewords
            codeword = np.zeros(self.n)
            if k < self.n:
                codeword[k] = 1.0
            else:
                # For indices beyond dimension, create combinations
                codeword[k % self.n] = 1.0
                if k >= self.n and k < 2*self.n:
                    codeword[(k - self.n) % self.n] = 1.0
            
            codebook[k] = codeword
        
        return codebook
    
    def compute_layer_histograms(self, x: np.ndarray, b_matrix: List[List[int]], 
                                M_j: List[int]) -> np.ndarray:
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
    
    def matvec(self, x: np.ndarray, b_matrix: List[List[int]], 
               M_j: List[int]) -> np.ndarray:
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
        
        y = np.zeros(self.n, dtype=float)
        
        # Accumulate layer contributions
        for m in range(self.M):
            if np.any(s[m, :]):
                # Compute layer contribution
                layer_contribution = np.zeros(self.n, dtype=float)
                for k, sk in enumerate(s[m, :]):
                    if sk != 0.0 and k in self.codebook:
                        layer_contribution += sk * self.codebook[k]
                
                # Scale by q^m and add to result
                y += (self.q ** m) * layer_contribution
        
        return y
    
    def matvec_direct(self, x: np.ndarray, b_matrix: List[List[int]], 
                     M_j: List[int]) -> np.ndarray:
        """
        Perform matrix-vector multiplication by reconstructing each column.
        
        This method reconstructs each column using the same codebook as the
        histogram method for verification purposes.
        
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
        d = len(x)
        y = np.zeros(self.n, dtype=float)
        
        for j in range(d):
            # Reconstruct column j using the same codebook
            column_reconstructed = np.zeros(self.n, dtype=float)
            
            for m in range(M_j[j]):
                k = b_matrix[j][m]
                if k in self.codebook:
                    column_reconstructed += (self.q ** m) * self.codebook[k]
            
            # Add contribution
            y += x[j] * column_reconstructed
        
        return y


def run_paper_example_standalone():
    """
    Run the paper example using the standalone implementation.
    
    Returns:
    --------
    tuple
        (y_histogram, y_direct) where y_histogram is the result using the
        layer-wise histogram method and y_direct is the result using direct
        reconstruction.
    """
    print("Standalone Layer-Wise Histogram MatVec Example from Paper")
    print("=" * 60)
    
    # Parameters from the paper
    n = 4  # output dimension
    q = 3  # base
    M = 3  # depth
    
    # Create matvec object
    matvec_obj = StandaloneLayerWiseHistogramMatVec(n, q, M)
    
    # Code indices from the paper example
    b_matrix = [
        [0, 2, 1],  # column 1: b_{1,0}=0, b_{1,1}=2, b_{1,2}=1
        [1, 0, 1],  # column 2: b_{2,0}=1, b_{2,1}=0, b_{2,2}=1
        [2, 2, 2],  # column 3: b_{3,0}=2, b_{3,1}=2, b_{3,2}=2
        [0, 1, 0],  # column 4: b_{4,0}=0, b_{4,1}=1, b_{4,2}=0 (truncated)
        [1, 1, 0]   # column 5: b_{5,0}=1, b_{5,1}=1, b_{5,2}=0
    ]
    
    # Layer counts from the paper
    M_j = [3, 2, 1, 2, 3]  # (M_1, M_2, M_3, M_4, M_5)
    
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
    
    # Verify with direct computation
    y_direct = matvec_obj.matvec_direct(x, b_matrix, M_j)
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
    run_paper_example_standalone() 