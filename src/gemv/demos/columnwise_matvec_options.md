# Columnwise Matrix-Vector Multiplication Options

## Overview

This document outlines different approaches for implementing columnwise matrix-vector multiplication `y = Wx` where the operation is treated as a linear combination of matrix columns. The implementation must handle:

1. **Sparse vectors**: Some elements of `x` may be zero
2. **Padding requirements**: Matrix dimensions need not be multiples of lattice size
3. **Quantization options**: `x` may be quantized or used as-is
4. **Blocking strategies**: Columns and corresponding `x` blocks are processed together
5. **Computation strategies**: Various options for actual dot product computation

## Problem Formulation

Given:
- Matrix `W` of shape `(m, n)` where columns are quantized using hierarchical nested lattice quantizers
- Vector `x` of shape `(n,)` which may be sparse
- Lattice dimension `d` (e.g., 4 for D4 lattice)

The goal is to compute:
```
y = Wx = Σ_{j=1}^n x_j * w_j
```

where each column `w_j` is stored in quantized form:
```
w_j = Σ_{m=0}^{M_j-1} q^m * λ(b_{j,m})
```

## Key Constraints and Requirements

### 1. Sparsity Handling
- Vector `x` may have zero elements that should be skipped
- Non-zero pattern may be known a priori or detected at runtime
- Zero elements contribute nothing regardless of their column's quantization

### 2. Padding Strategy
- Matrix dimensions `m` and `n` need not be multiples of lattice dimension `d`
- Columns must be padded to lattice dimension boundaries
- Vector `x` must be padded to match column blocking
- Padding should be handled transparently

### 3. Quantization Options for x
- **Option A**: Use `x` as-is (no quantization)
- **Option B**: Quantize `x` using the same lattice quantizer
- **Option C**: Use different quantization for `x` vs `W`

### 4. Blocking Strategy
- Columns are processed in blocks of size `d` (lattice dimension)
- Corresponding blocks of `x` are processed together
- Each block can have different decoding depths

## Computation Options

### Option 4.1: No Lookup Table (Standard np.dot)

#### 4.1a: Fixed Depth Decoding
- Decode all columns to the same depth `M`
- Use standard `np.dot` for actual computation
- Simple but may be suboptimal for varying column importance

#### 4.1b: Variable Depth Decoding
- Each column can have different decoding depth `M_j`
- Decode each column to its specific depth
- Use standard `np.dot` for computation
- More flexible but requires individual column decoding

#### 4.1c: Adaptive Depth Based on Sparsity
- Adjust decoding depth based on `x` sparsity pattern
- Higher depth for columns with larger `|x_j|` values
- Lower depth for columns with smaller or zero `x_j` values

### Option 4.2: With Lookup Tables

#### 4.2a: Precomputed Inner Product Tables
- Precompute inner products between quantized vectors and codewords
- Use lookup tables for fast inner product computation
- Trade memory for computation speed

#### 4.2b: Layer-wise Histogram Approach
- Pool identical codewords at each layer
- Compute histograms `s_{m,k} = Σ_{j: m<M_j, b_{j,m}=k} x_j`
- Accumulate: `y = Σ_{m=0}^{M-1} q^m * Σ_k s_{m,k} * λ(k)`
- Most efficient when many columns share code indices

#### 4.2c: Hybrid Approach
- Use lookup tables for common patterns
- Fall back to direct computation for rare patterns
- Balance between memory usage and computation speed

## API Design

### Core Interface

```python
class ColumnwiseMatVecProcessor:
    """Base class for columnwise matrix-vector multiplication."""
    
    def __init__(
        self,
        matrix: np.ndarray,
        lattice_type: str = "D4",
        M: int = 2,
        q: int = 4,
        beta: float = 0.2,
        use_lookup: bool = False,
        quantize_x: bool = False,
        sparsity_threshold: float = 1e-10
    ):
        """Initialize the processor."""
        pass
    
    def compute_matvec(
        self,
        x: np.ndarray,
        decoding_depths: Optional[List[int]] = None,
        sparsity_pattern: Optional[List[int]] = None
    ) -> np.ndarray:
        """Compute y = Wx using columnwise approach."""
        pass
    
    def get_compression_stats(self) -> Dict[str, float]:
        """Get compression and performance statistics."""
        pass
```

### Specialized Implementations

#### 1. StandardDotProcessor
```python
class StandardDotProcessor(ColumnwiseMatVecProcessor):
    """Uses np.dot for computation without lookup tables."""
    
    def __init__(self, fixed_depth: bool = True, **kwargs):
        """
        Args:
            fixed_depth: If True, use same depth for all columns
        """
        pass
```

#### 2. LookupTableProcessor
```python
class LookupTableProcessor(ColumnwiseMatVecProcessor):
    """Uses precomputed lookup tables for fast computation."""
    
    def __init__(self, table_strategy: str = "layer_wise_histogram", **kwargs):
        """
        Args:
            table_strategy: "layer_wise_histogram", "inner_product", or "hybrid"
        """
        pass
```

#### 3. AdaptiveProcessor
```python
class AdaptiveProcessor(ColumnwiseMatVecProcessor):
    """Adapts computation strategy based on input characteristics."""
    
    def __init__(self, adaptation_threshold: float = 0.1, **kwargs):
        """
        Args:
            adaptation_threshold: Sparsity threshold for switching strategies
        """
        pass
```

## Implementation Strategy

### Phase 1: Core Infrastructure
1. **Padding Module**: Handle dimension padding transparently
2. **Sparsity Detection**: Efficient zero detection and pattern analysis
3. **Blocking Strategy**: Organize columns and vectors into lattice-sized blocks

### Phase 2: Standard Dot Implementation
1. **Fixed Depth**: All columns decoded to same depth
2. **Variable Depth**: Per-column decoding depths
3. **Sparsity-Aware**: Skip zero elements efficiently

### Phase 3: Lookup Table Implementation
1. **Layer-wise Histogram**: Pool identical codewords
2. **Inner Product Tables**: Precomputed dot products
3. **Hybrid Approach**: Combine both strategies

### Phase 4: Adaptive Implementation
1. **Strategy Selection**: Choose best approach based on input
2. **Performance Monitoring**: Track computation vs memory tradeoffs
3. **Dynamic Adaptation**: Switch strategies at runtime

## Performance Considerations

### Memory Usage
- **Standard Dot**: Minimal memory overhead
- **Lookup Tables**: Higher memory usage, scales with codebook size
- **Layer-wise Histogram**: Moderate memory, scales with number of layers

### Computation Complexity
- **Standard Dot**: O(m * n * M) for full decoding
- **Lookup Tables**: O(m * K * M) where K is codebook size
- **Layer-wise Histogram**: O(m * K * M) but with better constants

### Sparsity Benefits
- **Zero Skipping**: Skip computation for zero `x_j` values
- **Pattern Optimization**: Optimize for known sparsity patterns
- **Adaptive Depth**: Use lower depth for less important columns

## Testing Strategy

### Unit Tests
1. **Padding Tests**: Verify correct handling of non-multiple dimensions
2. **Sparsity Tests**: Test with various sparsity patterns
3. **Quantization Tests**: Test with quantized and non-quantized `x`

### Performance Tests
1. **Speed Comparison**: Compare different approaches
2. **Memory Usage**: Measure memory overhead
3. **Accuracy Tests**: Verify numerical accuracy

### Integration Tests
1. **End-to-End**: Test complete pipeline
2. **Edge Cases**: Test with extreme sparsity, very large matrices
3. **Adaptive Behavior**: Test strategy switching

## Future Extensions

### Advanced Features
1. **Multi-GPU Support**: Distribute computation across GPUs
2. **Streaming**: Process matrices larger than memory
3. **Compression**: Further compress lookup tables

### Optimization Opportunities
1. **SIMD Instructions**: Vectorize inner product computations
2. **Cache Optimization**: Optimize memory access patterns
3. **Parallel Processing**: Parallelize across columns or layers
