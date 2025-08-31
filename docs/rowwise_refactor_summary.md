# Rowwise GEMV Refactoring Summary

## Overview

This document summarizes the refactoring of the rowwise matrix-vector multiplication (GEMV) implementation to work with the updated Hierarchical Nested Lattice Quantizer (HNLQ).

## Key Changes Made

### 1. Updated HNLQ Interface Integration

**Before:**
```python
# Old interface
self.quantizers[block_idx] = HNLQ(
    G=self.G,
    Q_nn=self.Q_nn,
    q=self.q,
    beta=self.beta,
    alpha=self.alpha,
    eps=self.eps,
    dither=dither,
    M=self.M,
    decoding=self.decoding,
)
```

**After:**
```python
# New interface with HNLQConfig
config = HNLQConfig(
    lattice_type=self.lattice_type,
    q=self.q,
    M=self.M,
    beta=self.beta,
    alpha=self.alpha,
    eps=self.eps,
    overload=self.overload,
    decoding=self.decoding,
    max_scaling_iterations=self.max_scaling_iterations,
    with_tie_dither=self.with_tie_dither,
    with_dither=self.with_dither,
)

self.quantizers[block_idx] = HNLQ(
    config=config,
    G=self.G,
    Q_nn=self.Q_nn,
)
```

### 2. Enhanced Parameter Support

Added support for new HNLQ parameters:

- `overload`: Whether to handle overload by scaling (default: True)
- `max_scaling_iterations`: Maximum scaling iterations (default: 10)
- `with_tie_dither`: Whether to add dither for tie breaking (default: True)
- `with_dither`: Whether to add dither for randomized quantization (default: False)

### 3. Updated Method Signatures

**Decoding Methods:**
- `get_default_decoding()` → `decode()`
- `decode_coarse_to_fine()` now uses `depth` parameter instead of `max_level`
- Updated parameter ranges: depth now ranges from 1 to M (instead of 0 to M-1)

**Encoding Methods:**
- `encode()` now properly handles the `with_dither` parameter
- Consistent dither handling across all operations

### 4. Improved Parameter Validation

- Decoding depths now validated to be between 1 and M (inclusive)
- Better error messages for parameter validation
- Consistent parameter handling across all methods

### 5. Updated Default Values

- `alpha`: Changed from 1/3 to 1.0
- `beta`: Changed from 0.2 to 1.0
- `decoding_depths`: Now defaults to M instead of M-1

## Benefits of the Refactoring

### 1. **Structured Configuration Management**
- Uses `HNLQConfig` dataclass for organized parameter management
- Built-in validation and default values
- Easier to maintain and extend

### 2. **Enhanced Flexibility**
- Support for all new HNLQ features
- Better control over quantization behavior
- More granular parameter tuning

### 3. **Improved Consistency**
- Consistent interface with other GEMV implementations
- Unified parameter handling across the codebase
- Better alignment with the updated HNLQ design

### 4. **Better Error Handling**
- Comprehensive parameter validation
- Clear error messages for invalid configurations
- Robust handling of edge cases

## Backward Compatibility

The refactoring maintains backward compatibility for basic usage:

```python
# Old usage still works
row_gemv = RowWiseGEMV(matrix, lattice_type='D4', M=2)

# New usage with enhanced parameters
row_gemv = RowWiseGEMV(
    matrix, 
    lattice_type='D4', 
    M=2,
    overload=True,
    max_scaling_iterations=10,
    with_tie_dither=True,
    with_dither=False
)
```

## Testing

The refactored implementation has been thoroughly tested:

1. **Unit Tests**: Updated test suite to verify all functionality
2. **Integration Tests**: Verified compatibility with the updated HNLQ
3. **Performance Tests**: Confirmed no performance regression
4. **Parameter Tests**: Validated all new parameters work correctly

## Migration Guide

### For Existing Code

1. **Basic Usage**: No changes required
2. **Custom Parameters**: Update to use new parameter names
3. **Decoding Methods**: Update method calls to use new signatures
4. **Depth Parameters**: Adjust depth values to new range (1 to M)

### Example Migration

**Before:**
```python
row_gemv = RowWiseGEMV(matrix, M=2, alpha=1/3, beta=0.2)
result = row_gemv.multiply_coarse_to_fine(vector, max_level=0)
```

**After:**
```python
row_gemv = RowWiseGEMV(matrix, M=2, alpha=1.0, beta=1.0)
result = row_gemv.multiply_coarse_to_fine(vector, max_level=1)
```

## Future Enhancements

The refactored implementation provides a solid foundation for future enhancements:

1. **Advanced Quantization Strategies**: Easy to add new quantization methods
2. **Performance Optimizations**: Better structure for optimization
3. **Additional Lattice Types**: Simplified addition of new lattices
4. **Enhanced Statistics**: Better monitoring and analysis capabilities

## Conclusion

The refactoring successfully modernizes the rowwise GEMV implementation to work seamlessly with the updated HNLQ while maintaining backward compatibility and improving overall code quality. The new implementation is more robust, flexible, and maintainable.
