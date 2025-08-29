# GEMV Module Reorganization Summary

## Overview

The GEMV (General Matrix-Vector) module has been reorganized to provide a clean, extensible architecture that supports multiple matrix-vector multiplication strategies with a unified interface.

## New Structure

```
src/gemv/
├── __init__.py                    # Main module interface
├── base/                          # Base classes and factory
│   ├── __init__.py
│   ├── gemv_processor.py          # Abstract base class
│   └── gemv_factory.py            # Factory pattern
├── columnwise/                    # Columnwise GEMV processors
│   ├── __init__.py
│   └── columnwise_processor.py    # Linear combination of columns
├── rowwise/                       # Rowwise GEMV processors
│   ├── __init__.py
│   └── rowwise_processor.py       # Series of dot products
├── svd/                          # SVD-based GEMV processors
│   ├── __init__.py
│   └── svd_gemv_processor.py      # Wx in SVD domain
├── utils/                        # Shared utilities
│   ├── __init__.py
│   └── padder.py                  # Blocking strategies
└── demo_new_structure.py         # Demo script
```

## Key Features

### 1. Unified Interface
- All processors inherit from `GEMVProcessor` base class
- Common interface: `process(matrix, vector) -> result`
- Consistent configuration and validation

### 2. Factory Pattern
- `GEMVFactory` for creating processors
- Easy registration of new processor types
- Configuration-based processor creation

### 3. Three Processor Types

#### Columnwise Processor
- Implements `y = Wx` as linear combination of matrix columns
- Supports sparse vectors and quantization
- Efficient for sparse input vectors

#### Rowwise Processor  
- Implements `y = Wx` as series of dot products
- Each output element computed independently
- Good for parallel processing

#### SVD Processor
- Implements `y = Wx` in SVD domain: `W = U * S * V^T`
- Computation: `y = U * (S * (V^T * x))`
- Supports truncated SVD and quantization of components

### 4. Shared Utilities
- `BlockingStrategy` for efficient memory access
- Common lattice quantization support
- Reusable preprocessing and postprocessing

## Usage Examples

### Basic Usage
```python
from src.gemv import create_gemv_processor

# Create a columnwise processor
processor = create_gemv_processor(
    'columnwise',
    lattice_type='D4',
    M=2,
    q=4,
    beta=0.2,
    alpha=1/3
)

# Use the processor
result = processor(matrix, vector)
```

### Factory Usage
```python
from src.gemv import GEMVFactory

# List available processors
print(GEMVFactory.get_available_processors())
# ['columnwise', 'rowwise', 'svd']

# Get processor info
info = GEMVFactory.get_processor_info('svd')
print(info)
```

### Direct Class Usage
```python
from src.gemv import ColumnwiseGEMVProcessor, RowwiseGEMVProcessor, SVDGEMVProcessor

# Create processors directly
col_processor = ColumnwiseGEMVProcessor(lattice_type='D4')
row_processor = RowwiseGEMVProcessor(lattice_type='D4')
svd_processor = SVDGEMVProcessor(lattice_type='D4', svd_rank=20)
```

## Migration from Old Structure

### Old Structure
```
src/gemv/
├── columnwise_matvec_processor.py
├── row_wise_gemv.py
├── column_wise_gemv.py
├── lookup_table_processor.py
├── standard_dot_processor.py
├── adaptive_processor.py
└── padder.py
```

### Migration Path
1. **Columnwise**: Use `ColumnwiseGEMVProcessor` instead of `ColumnwiseMatVecProcessor`
2. **Rowwise**: Use `RowwiseGEMVProcessor` instead of `RowWiseGEMV`
3. **SVD**: New `SVDGEMVProcessor` for SVD domain operations
4. **Utilities**: `BlockingStrategy` moved to `utils/padder.py`

## Benefits of New Structure

1. **Consistency**: All processors follow the same interface
2. **Extensibility**: Easy to add new processor types
3. **Maintainability**: Clear separation of concerns
4. **Reusability**: Shared base classes and utilities
5. **Testability**: Isolated components for easier testing
6. **Documentation**: Clear module structure and interfaces

## Future Extensions

The new structure makes it easy to add:
- Block-based processors
- Diagonal processors  
- Structured matrix processors
- GPU-accelerated processors
- Custom quantization strategies

## Testing

Run the demo to verify the new structure:
```bash
python -m src.gemv.demo_new_structure
```

This will test all three processor types and demonstrate the factory capabilities.
