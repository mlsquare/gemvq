# Coarse-to-Fine Decoding in Lattice Quantization

## Overview

This document describes the implementation of coarse-to-fine decoding functionality in the lattice quantization module. This feature allows decoding from coarse to fine levels, where higher M means coarser quantization, and the reconstruction can be stopped at any level from M-1 down to 0.

## Key Concepts

### Hierarchical Quantization Levels

In the hierarchical nested lattice quantizer, the encoding process creates M levels of quantization:

- **Level 0**: Coarsest level (highest weight in reconstruction)
- **Level 1**: Intermediate level
- **Level M-1**: Finest level (lowest weight in reconstruction)

The reconstruction formula is:
```
x_hat = sum([q^i * x_i_hat for i in range(M)])
```

Where `q^i` gives the highest weight to level 0 and the lowest weight to level M-1.

### Coarse-to-Fine Decoding

The coarse-to-fine decoding allows stopping the reconstruction at any level:

- **max_level = 0**: Use only the coarsest level (level 0)
- **max_level = 1**: Use levels 0 and 1
- **max_level = M-1**: Use all levels (full reconstruction)

## Implementation

### 1. HierarchicalNestedLatticeQuantizer

#### New Methods Added:

```python
def decode_coarse_to_fine(self, b_list, T, with_dither, max_level: int = None):
    """
    Decode hierarchical encoding vectors with coarse-to-fine reconstruction.
    
    Parameters:
    -----------
    max_level : int, optional
        Maximum level to decode up to (0 <= max_level < M).
        If None, decodes all levels (equivalent to decode method).
        Higher max_level means finer reconstruction.
    """
```

```python
def decode_progressive(self, b_list, T, with_dither):
    """
    Generate progressive reconstructions from coarse to fine.
    
    Returns:
    --------
    list
        List of reconstructed vectors, from coarsest to finest.
    """
```

### 2. ColumnWiseGEMV and RowWiseGEMV

#### New Methods Added:

```python
def multiply_coarse_to_fine(self, vector, max_level=None, sparsity_pattern=None):
    """
    Perform matrix-vector multiplication with coarse-to-fine decoding.
    """

def multiply_progressive(self, vector, sparsity_pattern=None):
    """
    Perform matrix-vector multiplication with progressive refinement.
    """
```

### 3. LatticeQuantizedGEMV (Unified Interface)

#### New Methods Added:

```python
def multiply_coarse_to_fine(self, vector, max_level=None, sparsity_pattern=None, use_lookup=False):
    """
    Perform matrix-vector multiplication with coarse-to-fine decoding.
    """

def multiply_progressive(self, vector, sparsity_pattern=None, use_lookup=False):
    """
    Perform matrix-vector multiplication with progressive refinement.
    """
```

## Usage Examples

### Basic Usage

```python
from src.gemv.lattice_quantized_gemv import LatticeQuantizedGEMV
import numpy as np

# Create processor with M=3 levels
matrix = np.random.randn(64, 32)
vector = np.random.randn(32)
processor = LatticeQuantizedGEMV(matrix, 'auto', 'D4', M=3)

# Decode at different levels
result_coarse = processor.multiply_coarse_to_fine(vector, max_level=0)  # Coarsest
result_medium = processor.multiply_coarse_to_fine(vector, max_level=1)  # Medium
result_fine = processor.multiply_coarse_to_fine(vector, max_level=2)    # Finest
result_full = processor.multiply_coarse_to_fine(vector, max_level=None) # All levels
```

### Progressive Decoding

```python
# Get all reconstructions from coarse to fine
progressive_results = processor.multiply_progressive(vector)

# progressive_results[0] = coarsest reconstruction (level 0)
# progressive_results[1] = medium reconstruction (levels 0, 1)
# progressive_results[2] = finest reconstruction (levels 0, 1, 2)
```

### With Sparsity Support

```python
# Make vector sparse
sparse_vector = np.zeros(32)
sparse_vector[0] = 1.0
sparsity_pattern = [0]

# Decode with sparsity
result = processor.multiply_coarse_to_fine(sparse_vector, max_level=1, 
                                         sparsity_pattern=sparsity_pattern)
```

## Error Analysis

The coarse-to-fine decoding provides a natural trade-off between:

1. **Compression**: Higher max_level means more levels used, requiring more bits
2. **Quality**: Higher max_level means finer reconstruction, lower error

### Typical Error Progression

For a well-designed quantizer:
- Level 0 (coarsest): Highest error, lowest bit rate
- Level 1: Medium error, medium bit rate  
- Level M-1 (finest): Lowest error, highest bit rate

## Applications

### 1. Progressive Transmission

```python
# Send coarse result first, then refine
coarse_result = processor.multiply_coarse_to_fine(vector, max_level=0)
# Transmit coarse_result...

# Later, send refinement
refined_result = processor.multiply_coarse_to_fine(vector, max_level=1)
# Transmit additional bits for refinement...
```

### 2. Adaptive Quality Control

```python
# Start with coarse reconstruction
result = processor.multiply_coarse_to_fine(vector, max_level=0)
error = compute_error(result, target)

# If error is too high, use more levels
if error > threshold:
    result = processor.multiply_coarse_to_fine(vector, max_level=1)
```

### 3. Resource-Constrained Environments

```python
# In low-power scenarios, use fewer levels
if power_level == 'low':
    max_level = 0  # Coarsest reconstruction
elif power_level == 'medium':
    max_level = 1  # Medium reconstruction
else:
    max_level = None  # Full reconstruction
```

## Performance Characteristics

### Computational Complexity

- **Level 0**: O(1) reconstruction per vector element
- **Level 1**: O(2) reconstruction per vector element
- **Level M-1**: O(M) reconstruction per vector element

### Memory Usage

- **Storage**: All M levels are stored regardless of decoding level
- **Runtime**: Only the required levels are used in reconstruction

### Compression Ratio

The compression ratio depends on the number of levels used:

```
Compression Ratio = Original Bits / (Bits per level × Number of levels used)
```

## Testing

The implementation includes comprehensive tests in `test_coarse_to_fine.py`:

1. **Hierarchical Quantizer Tests**: Verify basic coarse-to-fine decoding
2. **Column-wise GEMV Tests**: Test column-wise approach
3. **Row-wise GEMV Tests**: Test row-wise approach  
4. **Unified Interface Tests**: Test the unified interface
5. **Edge Case Tests**: Test boundary conditions

Run tests with:
```bash
python test_coarse_to_fine.py
```

## Demo

A demonstration script `demo_coarse_to_fine.py` shows:

1. **Coarse-to-Fine Decoding**: Results at different levels
2. **Progressive Refinement**: Step-by-step improvement
3. **Compression vs Quality Trade-off**: Analysis across different M values
4. **Visualization**: Plots showing error reduction and reconstruction quality

Run demo with:
```bash
python demo_coarse_to_fine.py
```

## Limitations

1. **Lookup Tables**: Row-wise approach with lookup tables doesn't support coarse-to-fine decoding
2. **Memory Overhead**: All M levels must be stored even if only coarse levels are used
3. **Error Monotonicity**: Error doesn't always decrease monotonically with more levels (depends on quantizer design)

## Future Enhancements

1. **Adaptive Level Selection**: Automatically choose optimal max_level based on error tolerance
2. **Selective Level Storage**: Store only required levels to save memory
3. **Lookup Table Support**: Extend coarse-to-fine decoding to lookup-based approaches
4. **Error Bounds**: Provide theoretical error bounds for each level

## Conclusion

The coarse-to-fine decoding functionality provides a flexible framework for progressive reconstruction in lattice quantization. It enables applications that require adaptive quality control, progressive transmission, and resource-constrained computation while maintaining the benefits of hierarchical quantization.
