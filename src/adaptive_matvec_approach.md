# Adaptive Matrix-Vector Multiplication with Hierarchical Nested Quantizers

## Overview

This module implements adaptive matrix-vector multiplication using hierarchical nested lattice quantizers. The approach encodes matrix W once with maximum bit rate, then adaptively decodes columns based on bit budget for each input vector x, exploiting the hierarchical levels M for variable precision decoding.

## Key Innovation

### Fixed Matrix Encoding with Adaptive Decoding

Unlike traditional approaches that encode each column with different parameters, this method:

1. **Encodes W once** with maximum bit rate (full precision)
2. **Decodes columns adaptively** based on bit budget for each x
3. **Exploits hierarchical levels M** to decode only to necessary depth
4. **Handles varying sparsity** of x while keeping W fixed

This approach is much more efficient for scenarios where:
- Matrix W is fixed and reused for multiple vectors x
- Different vectors x have different bit budget requirements
- Sparsity patterns of x vary across computations

## Mathematical Framework

### Problem Formulation

Given:
- Fixed matrix W ∈ ℝ^(m×n) (encoded once with max rate)
- Vector x ∈ ℝ^n (with varying sparsity patterns)
- Bit budget allocation B = [B₁, B₂, ..., Bₙ] for each column
- Maximum encoding rate R_max for W
- Hierarchical levels M

Goal: Compute y = Wx efficiently using adaptive column decoding based on bit budget.

### Column-wise Interpretation

Matrix-vector multiplication is expressed as:
```
y = Wx = Σᵢ₌₁ⁿ xᵢ · W[:,i]
```

Where:
- W[:,i] is the i-th column of matrix W
- xᵢ is the i-th element of vector x
- Only columns corresponding to non-zero xᵢ need to be processed

## Adaptive Decoding Strategy

### 1. Fixed Matrix Encoding

Matrix W is encoded once using hierarchical nested quantization with maximum rate:

```
W[:,i] → (b₁⁽ⁱ⁾, b₂⁽ⁱ⁾, ..., b_M⁽ⁱ⁾, T⁽ⁱ⁾)
```

Where:
- b_j⁽ⁱ⁾ are the encoding vectors for level j
- T⁽ⁱ⁾ is the overload scaling factor
- M is the number of hierarchical levels
- All columns use the same quantization parameters (q, β, α)

### 2. Adaptive Column Decoding

For each vector x, columns are decoded based on bit budget allocation:

```
Decode(W[:,i], Bᵢ) = DecodePartial((b₁⁽ⁱ⁾, b₂⁽ⁱ⁾, ..., b_M⁽ⁱ⁾), T⁽ⁱ⁾, levels_to_use)
```

Where:
- `levels_to_use = max(1, int(M * Bᵢ / R_max))`
- Only the first `levels_to_use` encoding vectors are used
- Lower bit budgets result in fewer hierarchical levels used

### 3. Hierarchical Level Exploitation

The key insight is using the hierarchical structure for variable precision:

- **High bit budget**: Use all M levels (full precision)
- **Medium bit budget**: Use subset of levels (reduced precision)
- **Low bit budget**: Use only first few levels (coarse precision)

This provides a natural rate-distortion tradeoff without re-encoding.

## Implementation Approach

### Core Components

#### 1. FixedMatrixQuantizer
```python
class FixedMatrixQuantizer:
    def __init__(self, matrix, lattice_type, max_rate, M):
        # Encode entire matrix W once with max rate
        pass
    
    def decode_column_adaptive(self, col_idx, target_rate):
        # Decode column with variable precision based on target_rate
        pass
    
    def estimate_inner_product_adaptive(self, col_idx, weight, target_rate):
        # Estimate inner product using lookup tables
        pass
```

#### 2. AdaptiveMatVecProcessor
```python
class AdaptiveMatVecProcessor:
    def __init__(self, matrix, lattice_type, max_rate, M):
        # Initialize with fixed matrix quantizer
        pass
    
    def compute_matvec(self, vector, column_rates, use_lookup=False):
        # Compute Wx with adaptive column decoding
        pass
    
    def compute_matvec_sparse(self, sparse_vector, non_zero_indices, column_rates):
        # Efficient computation for sparse vectors
        pass
```

#### 3. Precomputed Lookup Tables
```python
# Precompute lookup tables for different hierarchical levels
lookup_tables = {
    R_max: precompute_hq_lut(G, Q_nn, q, M, eps),      # Full precision
    R_max * (M-1)/M: precompute_hq_lut(G, Q_nn, q, M-1, eps),  # M-1 levels
    R_max * (M-2)/M: precompute_hq_lut(G, Q_nn, q, M-2, eps),  # M-2 levels
    # ... and so on
}
```

### Algorithm Flow

#### Phase 1: One-time Setup
1. **Matrix Encoding**: Encode W with maximum rate R_max and M levels
2. **Lookup Table Generation**: Precompute tables for different hierarchical levels
3. **Memory Allocation**: Store encoded matrix and lookup tables

#### Phase 2: Adaptive Computation (per vector x)
1. **Bit Budget Analysis**: Determine column rates based on x characteristics
2. **Selective Decoding**: Decode only necessary columns with appropriate precision
3. **Sparse Processing**: Skip zero elements in x
4. **Result Assembly**: Combine decoded columns to form y = Wx

### Optimization Strategies

#### 1. Memory Efficiency
- **Single Encoding**: W encoded once, stored efficiently
- **Shared Lookup Tables**: Tables shared across all computations
- **Lazy Decoding**: Decode columns only when needed

#### 2. Computational Efficiency
- **Hierarchical Exploitation**: Use fewer levels for lower bit budgets
- **Sparsity Awareness**: Skip processing of zero elements
- **Lookup Table Acceleration**: Fast inner product estimation

#### 3. Rate-Distortion Optimization
- **Adaptive Precision**: Match decoding precision to bit budget
- **Error Control**: Theoretical bounds on quantization error
- **Progressive Refinement**: Can increase precision if needed

## Key Features

### 1. Fixed Matrix Encoding
- W encoded once with maximum precision
- No need to re-encode for different x
- Efficient storage and memory usage

### 2. Adaptive Column Decoding
- Variable precision based on bit budget
- Exploits hierarchical structure naturally
- No re-encoding required

### 3. Sparsity Exploitation
- Skip processing of zero elements in x
- Memory and computation savings proportional to sparsity
- Efficient handling of varying sparsity patterns

### 4. Hierarchical Precision Control
- Natural rate-distortion tradeoff through M levels
- Progressive refinement capability
- Efficient inner product estimation

### 5. Flexible Bit Budget Allocation
- Different bit budgets for different columns
- Dynamic allocation based on x characteristics
- Support for various allocation strategies

## Performance Characteristics

### Computational Complexity
- **Setup**: O(m × n × M) for one-time encoding
- **Per Vector**: O(|S| × M_avg) where |S| is sparsity, M_avg is average levels used
- **Lookup Table**: O(|S| × M_avg²) for inner product estimation

### Memory Requirements
- **Encoded Matrix**: O(m × n × M × log₂(q)) bits
- **Lookup Tables**: O(Σᵢ q^(2i)) for i = 1 to M levels
- **Working Memory**: O(m) for result vector

### Rate-Distortion Performance
- **Adaptive Precision**: Better overall rate-distortion than fixed precision
- **Sparsity Gain**: Additional savings proportional to sparsity ratio
- **Hierarchical Benefits**: Improved performance with increasing M

## Applications

### 1. Neural Network Inference
- Fixed weight matrices with varying input sparsity
- Adaptive precision based on input characteristics
- Efficient handling of different input types

### 2. Recommendation Systems
- Fixed user-item matrices with sparse user vectors
- Adaptive precision based on user activity level
- Efficient cold-start and active user handling

### 3. Signal Processing
- Fixed filter matrices with varying signal sparsity
- Adaptive precision based on signal characteristics
- Efficient real-time processing

### 4. Scientific Computing
- Fixed coefficient matrices with varying right-hand sides
- Adaptive precision based on solution requirements
- Efficient iterative solvers

## Implementation Considerations

### 1. Parameter Selection
- **Lattice Type**: Choose based on dimension and performance requirements
- **Hierarchical Levels**: Balance accuracy vs. complexity (typically M = 3-5)
- **Maximum Rate**: Set based on accuracy requirements and memory constraints

### 2. Bit Budget Allocation Strategies
- **Uniform**: Same rate for all columns
- **Energy-based**: Higher rates for columns with larger magnitude
- **Sparsity-aware**: Higher rates for columns corresponding to non-zero x elements
- **Adaptive**: Dynamic allocation based on x characteristics

### 3. Error Control
- **Overload Handling**: Robust overload detection and handling
- **Error Bounds**: Theoretical and empirical error bounds
- **Adaptive Refinement**: Increase precision if error is too high

### 4. Integration
- **API Design**: Clean interface for different use cases
- **Compatibility**: Ensure compatibility with existing frameworks
- **Extensibility**: Support for custom quantization strategies

## Comparison with Previous Approach

### Previous Approach (Column-wise Encoding)
- **Encoding**: Each column encoded with different parameters
- **Storage**: Multiple encodings for different rate requirements
- **Computation**: Re-encoding needed for different rate allocations
- **Memory**: Higher memory usage due to multiple encodings

### New Approach (Fixed Encoding, Adaptive Decoding)
- **Encoding**: Matrix encoded once with maximum rate
- **Storage**: Single encoding shared across all computations
- **Computation**: Adaptive decoding without re-encoding
- **Memory**: Lower memory usage with shared encoding

### Advantages of New Approach
1. **Efficiency**: No re-encoding required
2. **Memory**: Lower storage requirements
3. **Flexibility**: Easy adaptation to different bit budgets
4. **Scalability**: Better for large matrices
5. **Simplicity**: Cleaner implementation and API

## Future Extensions

### 1. Dynamic Bit Budget Optimization
- Online optimization of bit budget allocation
- Learning-based rate allocation strategies
- Adaptive bit budget based on error feedback

### 2. Multi-dimensional Sparsity
- Handle block sparsity patterns
- Support for structured sparsity
- Efficient encoding of sparse tensors

### 3. Hardware Acceleration
- GPU-optimized implementations
- FPGA acceleration for lookup table operations
- Specialized hardware for lattice operations

### 4. Advanced Quantization Strategies
- Non-uniform quantization
- Entropy-coded quantization
- Learned quantization parameters

This approach provides a comprehensive framework for efficient matrix-vector multiplication using fixed encoding with adaptive decoding, leveraging hierarchical nested quantization for optimal performance across varying bit budget requirements. 