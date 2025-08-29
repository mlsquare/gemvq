# GEMV (General Matrix-Vector Multiplication) Module

This module implements efficient matrix-vector multiplication using lattice quantization with both column-wise and row-wise approaches, featuring intelligent blocking strategies based on lattice dimensions.

## Overview

The GEMV module provides two complementary approaches for matrix-vector multiplication:

1. **Column-wise GEMV**: Treats matrix-vector multiplication as a linear combination of quantized matrix columns
2. **Row-wise GEMV**: Treats matrix-vector multiplication as a series of dot products between quantized matrix rows and the input vector

Both approaches use hierarchical nested lattice quantizers with efficient blocking strategies to optimize memory access patterns and computational efficiency.

## Key Features

- **Dual Approaches**: Both column-wise and row-wise matrix-vector multiplication
- **Blocking Strategy**: Efficient blocking based on lattice dimensions (D4=4, A2=2, E8=8, Z2=2, Z3=3)
- **Sparsity Support**: Optimized computation for sparse input vectors
- **Lookup Tables**: Optional lookup table support for faster inner product estimation
- **Unified Interface**: Automatic approach selection based on matrix characteristics
- **Multiple Lattices**: Support for D4, A2, E8, Z2, and Z3 lattices
- **Compression Analysis**: Built-in compression ratio and memory usage analysis

## Architecture

```
src/gemv/
├── __init__.py                    # Module exports
├── padder.py                      # Blocking strategy implementation
├── column_wise_gemv.py           # Column-wise GEMV implementation
├── row_wise_gemv.py              # Row-wise GEMV implementation
├── lattice_quantized_gemv.py     # Unified interface
├── demo_gemv.py                  # Demonstration script
└── README.md                     # This file
```

## Quick Start

### Basic Usage

```python
import numpy as np
from src.gemv import LatticeQuantizedGEMV

# Create test data
matrix = np.random.randn(100, 50)
vector = np.random.randn(50)

# Use unified interface with automatic approach selection
processor = LatticeQuantizedGEMV(matrix, 'auto', 'D4', 2)
result = processor.multiply(vector)

# Compare with exact computation
exact_result = matrix @ vector
error = np.linalg.norm(result - exact_result) / np.linalg.norm(exact_result)
print(f"Relative error: {error:.6f}")
```

### Column-wise Approach

```python
from src.gemv import ColumnWiseGEMV

# Initialize column-wise processor
processor = ColumnWiseGEMV(matrix, 'D4', 2)

# Basic multiplication
result = processor.multiply(vector)

# With sparsity support
sparsity_pattern = [0, 5, 10, 15]  # Indices of non-zero elements
result = processor.multiply_with_sparsity(vector, sparsity_pattern)
```

### Row-wise Approach

```python
from src.gemv import RowWiseGEMV

# Initialize row-wise processor
processor = RowWiseGEMV(matrix, 'D4', 2)

# Basic multiplication
result = processor.multiply(vector)

# With lookup tables for faster computation
result = processor.multiply_with_lookup(vector)
```

### Blocking Strategy

```python
from src.gemv import BlockingStrategy

# Initialize blocking strategy
blocking = BlockingStrategy('D4')  # Block size = 4

# Get block indices
vector_length = 100
col_blocks = blocking.get_block_indices(vector_length)
print(f"Block size: {blocking.block_size}")
print(f"Number of column blocks: {len(col_blocks)}")
```

## API Reference

### LatticeQuantizedGEMV

The main unified interface for matrix-vector multiplication.

```python
class LatticeQuantizedGEMV:
    def __init__(self, matrix, approach='auto', lattice_type='D4', M=2, 
                 alpha=1/3, eps=1e-8, q=4, beta=0.2):
        """
        Initialize the unified GEMV processor.
        
        Parameters:
        -----------
        matrix : np.ndarray
            Input matrix W (m x n).
        approach : str
            Approach to use ('column', 'row', or 'auto').
        lattice_type : str
            Type of lattice to use ('D4', 'A2', 'E8', 'Z2', 'Z3').
        M : int
            Number of hierarchical levels.
        alpha : float
            Scaling parameter for overload handling.
        eps : float
            Small perturbation parameter.
        q : int
            Quantization parameter.
        beta : float
            Scaling parameter for quantization.
        """
    
    def multiply(self, vector, sparsity_pattern=None, use_lookup=False):
        """Perform matrix-vector multiplication."""
    
    def compare_approaches(self, vector, sparsity_pattern=None):
        """Compare column-wise and row-wise approaches."""
    
    def get_compression_ratio(self):
        """Calculate compression ratio achieved by the encoding."""
    
    def get_memory_usage(self):
        """Get memory usage statistics."""
    
    def get_blocking_info(self):
        """Get information about the blocking strategy used."""
    
    def get_approach_info(self):
        """Get information about the selected approach."""
```

### ColumnWiseGEMV

Column-wise matrix-vector multiplication implementation.

```python
class ColumnWiseGEMV:
    def __init__(self, matrix, lattice_type='D4', M=2, alpha=1/3, 
                 eps=1e-8, q=4, beta=0.2):
        """Initialize the column-wise GEMV processor."""
    
    def multiply(self, vector):
        """Perform column-wise matrix-vector multiplication."""
    
    def multiply_with_sparsity(self, vector, sparsity_pattern=None):
        """Perform multiplication with sparsity support."""
```

### RowWiseGEMV

Row-wise matrix-vector multiplication implementation.

```python
class RowWiseGEMV:
    def __init__(self, matrix, lattice_type='D4', M=2, alpha=1/3, 
                 eps=1e-8, q=4, beta=0.2):
        """Initialize the row-wise GEMV processor."""
    
    def multiply(self, vector):
        """Perform row-wise matrix-vector multiplication."""
    
    def multiply_with_sparsity(self, vector, sparsity_pattern=None):
        """Perform multiplication with sparsity support."""
    
    def multiply_with_lookup(self, vector, lookup_tables=None):
        """Perform multiplication using lookup tables."""
```

### BlockingStrategy

Efficient blocking strategy based on lattice dimensions.

```python
class BlockingStrategy:
    def __init__(self, lattice_type='D4'):
        """Initialize the blocking strategy."""
    
    def get_block_indices(self, vector_length):
        """Get block indices for a vector of given length."""
    
    def get_matrix_blocks_column_wise(self, matrix):
        """Get matrix blocks for column-wise approach."""
    
    def get_matrix_blocks_row_wise(self, matrix):
        """Get matrix blocks for row-wise approach."""
    
    def get_vector_blocks(self, vector):
        """Get vector blocks based on the blocking strategy."""
    
    def pad_vector(self, vector):
        """Pad vector to make its length divisible by block_size."""
    
    def pad_matrix_for_column_wise(self, matrix):
        """Pad matrix columns for column-wise approach."""
    
    def pad_matrix_for_row_wise(self, matrix):
        """Pad matrix rows for row-wise approach."""
    
    def unpad_vector(self, padded_vector, original_length):
        """Remove padding from a vector."""
    
    def unpad_matrix(self, padded_matrix, original_shape):
        """Remove padding from a matrix."""
```

## Function Interfaces

For convenience, the module also provides function-level interfaces:

```python
# Column-wise GEMV
result = column_wise_gemv(matrix, vector, 'D4', 2, sparsity_pattern=None)

# Row-wise GEMV
result = row_wise_gemv(matrix, vector, 'D4', 2, sparsity_pattern=None, use_lookup=False)

# Unified GEMV
result = lattice_quantized_gemv(matrix, vector, 'auto', 'D4', 2, 
                               sparsity_pattern=None, use_lookup=False)

# Compare approaches
comparison = compare_gemv_approaches(matrix, vector, 'D4', 2, sparsity_pattern=None)
```

## Approach Selection

The unified interface automatically selects the optimal approach based on matrix characteristics:

- **Column-wise**: Preferred for tall matrices (aspect ratio > 1.5) and better sparsity handling
- **Row-wise**: Preferred for wide matrices (aspect ratio < 0.67) and when lookup tables are available
- **Auto**: Automatically selects based on matrix shape and characteristics

## Blocking Strategy

The blocking strategy divides vectors and matrices into blocks based on lattice dimensions:

- **D4 lattice**: Block size = 4
- **A2 lattice**: Block size = 2  
- **E8 lattice**: Block size = 8
- **Z2 lattice**: Block size = 2
- **Z3 lattice**: Block size = 3

This ensures optimal memory access patterns and computational efficiency for lattice-quantized operations.

## Performance Considerations

1. **Memory Usage**: Column-wise approach may use more memory for wide matrices
2. **Computation Time**: Row-wise approach with lookup tables can be faster for dense vectors
3. **Sparsity**: Column-wise approach is more efficient for sparse vectors
4. **Blocking**: Larger block sizes (E8) may be more efficient for larger matrices

## Examples

### Basic Example

```python
import numpy as np
from src.gemv import LatticeQuantizedGEMV

# Create test data
matrix = np.random.randn(64, 32)
vector = np.random.randn(32)

# Use unified interface
processor = LatticeQuantizedGEMV(matrix, 'auto', 'D4', 2)
result = processor.multiply(vector)

print(f"Approach used: {processor.approach}")
print(f"Compression ratio: {processor.get_compression_ratio():.2f}")
print(f"Memory usage: {processor.get_memory_usage()}")
```

### Sparsity Example

```python
# Create sparse vector
sparse_vector = np.zeros(32)
sparse_vector[[0, 8, 16, 24]] = [1.0, 2.0, 3.0, 4.0]
sparsity_pattern = [0, 8, 16, 24]

# Use with sparsity support
result = processor.multiply(sparse_vector, sparsity_pattern)
```

### Performance Comparison

```python
# Compare approaches
comparison = processor.compare_approaches(vector)
print(f"Column-wise time: {comparison['column_wise']['time']:.4f}s")
print(f"Row-wise time: {comparison['row_wise']['time']:.4f}s")
print(f"Recommended: {comparison['recommended_approach']}")
```

## Testing

Run the comprehensive test suite:

```bash
python -m pytest tests/test_gemv.py -v
```

Or run the demo script:

```bash
python src/gemv/demo_gemv.py
```

## Integration

The GEMV module integrates seamlessly with the existing LatticeQuant framework:

```python
from src.gemv import LatticeQuantizedGEMV
from src.adaptive import adaptive_matvec_multiply
from src.quantizers.lattice import HNLQ

# Use GEMV for general matrix-vector multiplication
gemv_processor = LatticeQuantizedGEMV(matrix, 'auto', 'D4', 2)
result = gemv_processor.multiply(vector)

# Use adaptive module for specialized sparse operations
adaptive_result = adaptive_matvec_multiply(matrix, vector, target_rates, sparsity_pattern)
```

## References

This module builds upon the hierarchical nested lattice quantization framework described in the LatticeQuant paper, extending it to efficient matrix-vector multiplication with blocking strategies. 