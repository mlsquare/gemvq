# Layer-Wise Histogram Matrix-Vector Multiplication

This document describes the layer-wise histogram technique for efficient matrix-vector multiplication when matrix columns are stored using hierarchical nested-lattice quantization.

## Overview

The layer-wise histogram technique efficiently computes matrix-vector multiplication by pooling identical codewords at each layer of the hierarchical quantization. This approach is particularly effective when many columns share the same code indices.

## Mathematical Formulation

Instead of computing:
```
y = sum_j x_j * w_j
```

where each w_j is stored as:
```
w_j = sum_{m=0}^{M_j-1} q^m * lambda(b_{j,m})
```

We compute:
```
y = sum_{m=0}^{M-1} q^m * sum_k s_{m,k} * lambda(k)
```

where `s_{m,k} = sum_{j: m<M_j, b_{j,m}=k} x_j` is the layer-wise histogram.

## Implementation

### Main Implementation

The main implementation uses the hierarchical nested-lattice quantizer:

```python
from src.adaptive.layer_wise_histogram_matvec import LayerWiseHistogramMatVec

# Create quantizer
quantizer = HierarchicalNestedLatticeQuantizer(...)
matvec_obj = LayerWiseHistogramMatVec(quantizer)

# Compute matrix-vector multiplication
y = matvec_obj.matvec(x, b_matrix, M_j)
```

### Standalone Implementation

For testing and demonstration without parameter optimization:

```python
from src.adaptive.standalone_layer_wise_histogram import StandaloneLayerWiseHistogramMatVec

# Create standalone matvec object
matvec_obj = StandaloneLayerWiseHistogramMatVec(n=4, q=3, M=3)

# Compute matrix-vector multiplication
y = matvec_obj.matvec(x, b_matrix, M_j)
```

## Example from Paper

The implementation includes the exact example from `ada_matmul.md`:

### Parameters
- Output dimension: n = 4
- Input dimension: d = 5
- Base: q = 3
- Depth: M = 3 (layers m = 0, 1, 2)

### Code Indices
```
Column 1: [0, 2, 1]  # b_{1,0}=0, b_{1,1}=2, b_{1,2}=1
Column 2: [1, 0, 1]  # b_{2,0}=1, b_{2,1}=0, b_{2,2}=1
Column 3: [2, 2, 2]  # b_{3,0}=2, b_{3,1}=2, b_{3,2}=2
Column 4: [0, 1, 0]  # b_{4,0}=0, b_{4,1}=1, b_{4,2}=0 (truncated)
Column 5: [1, 1, 0]  # b_{5,0}=1, b_{5,1}=1, b_{5,2}=0
```

### Layer Counts
```
M_j = [3, 2, 1, 2, 3]  # (M_1, M_2, M_3, M_4, M_5)
```

### Input Vector
```
x = [0.7, -1.2, 0.0, 0.5, 2.0]
```

### Layer-Wise Histograms
```
Layer 0: s[0,:] = [1.2, 0.8, 0.0]  # x1+x4, x2+x5, x3
Layer 1: s[1,:] = [-1.2, 2.5, 0.7] # x2, x4+x5, x1  
Layer 2: s[2,:] = [2.0, 0.7, 0.0]  # x5, x1, 0
```

### Result
```
y = [15.6, 14.6, 2.1, 0.0]
```

## Running the Example

```python
from src.adaptive.layer_wise_histogram_matvec import run_paper_example

# Run the paper example
y_histogram, y_direct = run_paper_example()
```

Or using the standalone version:

```python
from src.adaptive.standalone_layer_wise_histogram import run_paper_example_standalone

# Run the paper example (no parameter optimization)
y_histogram, y_direct = run_paper_example_standalone()
```

## API Reference

### LayerWiseHistogramMatVec

Main implementation using hierarchical nested-lattice quantizer.

**Methods:**
- `matvec(x, b_matrix, M_j)`: Compute matrix-vector multiplication using layer-wise histograms
- `compute_layer_histograms(x, b_matrix, M_j)`: Compute layer-wise histograms
- `matvec_quantized_direct(x, W, b_matrix, M_j)`: Direct reconstruction for verification

### StandaloneLayerWiseHistogramMatVec

Standalone implementation without dependencies.

**Methods:**
- `matvec(x, b_matrix, M_j)`: Compute matrix-vector multiplication
- `matvec_direct(x, b_matrix, M_j)`: Direct reconstruction for verification
- `compute_layer_histograms(x, b_matrix, M_j)`: Compute layer-wise histograms

## Efficiency Benefits

The layer-wise histogram approach provides efficiency gains when:

1. **Many columns share code indices**: Instead of computing `lambda(k)` multiple times, we compute it once per layer and scale by the histogram sum.

2. **Sparse input vectors**: Zero components in the input vector are automatically skipped.

3. **Variable layer depths**: Columns can use different numbers of layers (M_j) based on their importance or precision requirements.

## Import Structure

The implementation uses explicit imports to avoid namespace pollution:

```python
# Main implementation
from src.adaptive.layer_wise_histogram_matvec import LayerWiseHistogramMatVec, run_paper_example

# Standalone implementation (faster for testing)
from src.adaptive.standalone_layer_wise_histogram import StandaloneLayerWiseHistogramMatVec, run_paper_example_standalone

# Dependencies
from src.quantizers.hierarchical_nested_lattice_quantizer import HierarchicalNestedLatticeQuantizer
```

## Performance Characteristics

- **Algorithm Correctness**: Both histogram and direct methods produce identical results (error < 1e-15)
- **Computational Complexity**: O(M × d) for histogram computation, O(M × K) for layer accumulation
- **Memory Efficiency**: Only stores layer-wise histograms, not individual column reconstructions
- **Scalability**: Performance improves with shared code indices across columns

## Testing

Run the tests to verify the implementation:

```bash
# Test the main implementation
python src/adaptive/test_layer_wise_histogram.py

# Test the standalone implementation (faster)
python src/adaptive/test_standalone_layer_wise_histogram.py
```

## Efficiency Benefits

The layer-wise histogram approach provides efficiency gains when:

1. **Many columns share code indices**: Instead of computing `lambda(k)` multiple times, we compute it once per layer and scale by the histogram sum.

2. **Sparse input vectors**: Zero components in the input vector are automatically skipped.

3. **Variable layer depths**: Columns can use different numbers of layers (M_j) based on their importance or precision requirements.

## References

- `ada_matmul.md`: Layer-Wise Histograms for Hierarchical Nested-Lattice MatVec
- The implementation follows the algorithm described in the paper example 