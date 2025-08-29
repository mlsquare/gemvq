# HNLQ (Hierarchical Nested Lattice Quantizer) Improvements

This document outlines the comprehensive improvements made to the `src/quantizers/lattice/hnlq.py` script to enhance code quality, maintainability, and usability.

## Overview of Improvements

The HNLQ script has been significantly enhanced with modern Python features, better error handling, improved documentation, and a more robust architecture. These improvements make the code more maintainable, type-safe, and user-friendly.

## 1. Configuration Management

### New HNLQConfig Dataclass

**Before:**
```python
def __init__(self, G, Q_nn, q, beta, alpha, eps, dither, M, decoding="full", overload=True):
    # Many individual parameters
    self.q = q
    self.beta = beta
    self.alpha = alpha
    # ... etc
```

**After:**
```python
@dataclass
class HNLQConfig:
    q: int
    beta: float
    alpha: float
    eps: float
    M: int
    overload: bool = True
    decoding: str = "full"
    max_scaling_iterations: int = 10
    
    def __post_init__(self):
        # Built-in validation
        if self.q <= 0:
            raise ValueError("Quantization parameter q must be positive")
        # ... more validation
```

**Benefits:**
- **Centralized Configuration**: All parameters are managed in one place
- **Built-in Validation**: Automatic parameter validation on initialization
- **Serialization Support**: Easy conversion to/from dictionaries
- **Default Values**: Sensible defaults with clear documentation
- **Type Safety**: Strong typing for all configuration parameters

## 2. Type Hints and Modern Python Features

### Comprehensive Type Annotations

**Before:**
```python
def encode(self, x, with_dither):
    # No type information
```

**After:**
```python
def encode(self, x: np.ndarray, with_dither: bool) -> Tuple[Tuple[np.ndarray, ...], int]:
    # Full type information
```

**Benefits:**
- **IDE Support**: Better autocomplete and error detection
- **Documentation**: Types serve as inline documentation
- **Static Analysis**: Tools like mypy can catch type errors
- **Maintainability**: Clearer function signatures

### Property-Based Interface

**New Properties:**
```python
@property
def q(self) -> int:
    """Get quantization parameter."""
    return self.config.q

@property
def beta(self) -> float:
    """Get scaling parameter."""
    return self.config.beta
# ... etc
```

**Benefits:**
- **Clean Interface**: Access parameters as properties
- **Encapsulation**: Internal configuration is hidden
- **Consistency**: Uniform access pattern for all parameters

## 3. Enhanced Error Handling

### Input Validation

**New Validation Features:**
- **Configuration Validation**: All parameters validated on creation
- **Dither Dimension Check**: Ensures dither matches generator matrix dimensions
- **Depth Range Validation**: Prevents invalid depth values in decoding
- **Decoding Method Validation**: Ensures only valid decoding methods are used

### Overflow Protection

**Before:**
```python
while did_overload:
    t += 1
    x = x / (2**self.alpha)
    # Could run indefinitely
```

**After:**
```python
while did_overload and t < self.config.max_scaling_iterations:
    t += 1
    x = x / (2**self.alpha)
    # Limited iterations with warning
if did_overload:
    warnings.warn("Overload not resolved after max iterations")
```

**Benefits:**
- **Prevents Infinite Loops**: Maximum iteration limit
- **User Feedback**: Clear warnings when limits are reached
- **Configurable**: Users can adjust limits based on their needs

## 4. Improved Documentation

### Enhanced Docstrings

**Before:**
```python
def encode(self, x, with_dither):
    """
    Encode a vector using hierarchical nested lattice quantization.
    """
```

**After:**
```python
def encode(self, x: np.ndarray, with_dither: bool) -> Tuple[Tuple[np.ndarray, ...], int]:
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
    Tuple[Tuple[numpy.ndarray, ...], int]
        (b_list, T) where b_list is a tuple of M encoding vectors and
        T is the number of scaling operations performed to handle overload.
    """
```

**Benefits:**
- **Detailed Parameter Documentation**: Clear descriptions of all parameters
- **Return Value Documentation**: Explicit return type and structure
- **Usage Examples**: Better understanding of how to use the method
- **Consistent Format**: Standardized docstring format

## 5. New Utility Methods

### Rate-Distortion Information

**New Method:**
```python
def get_rate_distortion_info(self) -> Dict[str, Any]:
    """
    Get information about the rate-distortion characteristics of the quantizer.
    
    Returns:
    --------
    Dict[str, Any]
        Dictionary containing rate-distortion information.
    """
    return {
        'q': self.q,
        'M': self.M,
        'beta': self.beta,
        'alpha': self.alpha,
        'overload': self.overload,
        'decoding': self.decoding,
        'max_scaling_iterations': self.config.max_scaling_iterations
    }
```

**Benefits:**
- **Debugging Support**: Easy access to current configuration
- **Logging**: Useful for experiment tracking
- **Serialization**: Can be saved/loaded for reproducibility

## 6. Backward Compatibility

### Flexible Constructor

The new interface maintains backward compatibility through flexible parameter handling:

```python
def __init__(self, G: np.ndarray, Q_nn: Callable, config: Union[HNLQConfig, Dict[str, Any]], dither: np.ndarray):
    if isinstance(config, dict):
        config = HNLQConfig.from_dict(config)
```

**Benefits:**
- **Migration Path**: Existing code can be updated gradually
- **Dictionary Support**: Can still use dictionaries for configuration
- **No Breaking Changes**: Existing functionality preserved

## 7. Performance Improvements

### Precomputed Values

- **G_inv**: Generator matrix inverse computed once during initialization
- **Property Access**: Fast property access instead of method calls
- **Validation**: Input validation prevents expensive operations on invalid data

## 8. Testing and Validation

### Comprehensive Test Coverage

The improvements include better error conditions that can be tested:

- **Configuration Validation**: Test invalid parameter combinations
- **Dimension Mismatches**: Test dither/generator matrix compatibility
- **Boundary Conditions**: Test edge cases in depth and level parameters
- **Overflow Handling**: Test maximum iteration limits

## Usage Examples

### Basic Usage

```python
from quantizers.lattice.hnlq import HNLQ, HNLQConfig
from quantizers.lattice.utils import get_d4, closest_point_Dn

# Create configuration
config = HNLQConfig(
    q=8,
    beta=1.0,
    alpha=0.5,
    eps=1e-8,
    M=3,
    overload=True,
    decoding="full"
)

# Initialize quantizer
G = get_d4()
dither = np.random.uniform(0, 1, (1, 4))
hnlq = HNLQ(G, closest_point_Dn, config, dither)

# Use quantizer
x = np.random.randn(4)
b_list, T = hnlq.encode(x, with_dither=True)
x_reconstructed = hnlq.decode(b_list, T, with_dither=True)
```

### Advanced Usage

```python
# Progressive decoding
progressive_results = hnlq.decode_progressive(b_list, T, with_dither=True)

# Coarse-to-fine decoding
x_coarse = hnlq.decode_coarse_to_fine(b_list, T, with_dither=True, max_level=1)

# Get configuration info
info = hnlq.get_rate_distortion_info()
print(f"Current configuration: {info}")
```

## Migration Guide

### For Existing Code

1. **Update Imports**: No changes needed for basic functionality
2. **Configuration**: Replace individual parameters with HNLQConfig
3. **Error Handling**: Add try-catch blocks for new validation
4. **Testing**: Update tests to use new configuration system

### Example Migration

**Before:**
```python
hnlq = HNLQ(G, Q_nn, q=8, beta=1.0, alpha=0.5, eps=1e-8, dither=dither, M=3)
```

**After:**
```python
config = HNLQConfig(q=8, beta=1.0, alpha=0.5, eps=1e-8, M=3)
hnlq = HNLQ(G, Q_nn, config, dither)
```

## Conclusion

The improvements to the HNLQ script provide:

1. **Better Maintainability**: Cleaner code structure and documentation
2. **Enhanced Safety**: Comprehensive validation and error handling
3. **Improved Usability**: Intuitive configuration management
4. **Modern Python**: Type hints and dataclasses for better development experience
5. **Backward Compatibility**: Existing code continues to work
6. **Extensibility**: Easy to add new features and configurations

These improvements make the HNLQ quantizer more robust, user-friendly, and suitable for production use while maintaining all existing functionality.
