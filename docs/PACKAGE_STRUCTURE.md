# GEMV-Q Package Structure

This document explains the clean, professional package structure of GEMV-Q.

## Overview

GEMV-Q has been restructured to provide a clean, intuitive package structure that's easy to use and maintain. The package is now installable as `gemvq` with professional import patterns.

## Package Structure

```
gemvq/
├── gemvq/                    # Main package (installable)
│   ├── quantizers/           # Quantization core
│   │   ├── nlq.py           # Single-level quantization
│   │   ├── hnlq.py          # Multi-level quantization
│   │   ├── utils.py         # Lattice utilities
│   │   └── __init__.py
│   │
│   ├── gemv/                # Matrix-vector multiplication
│   │   ├── columnwise/      # Column-wise implementations
│   │   ├── rowwise/         # Row-wise implementations
│   │   ├── svd/             # SVD-based implementations
│   │   ├── utils/           # GEMV utilities
│   │   ├── gemv_processor.py
│   │   ├── gemv_factory.py
│   │   ├── adaptive_processor.py
│   │   └── __init__.py
│   │
│   └── __init__.py          # Main package initialization
│
├── tests/                   # Test suite
├── examples/                # Usage examples
├── docs/                    # Documentation
├── setup.py                 # Package setup
└── README.md               # Main documentation
```

## Import Patterns

### Clean Main Imports
```python
# Import main classes directly from the package
from gemvq import NLQ, HNLQ, get_d4, get_a2, get_e8, closest_point_Dn
```

### Specific Module Imports
```python
# Import specific processors
from gemvq.gemv.columnwise import ColumnwiseMatVecProcessor
from gemvq.gemv.rowwise import RowwiseGEMVProcessor
from gemvq.gemv.svd import SVDGEMVProcessor

# Import utilities
from gemvq.gemv.utils import BlockingStrategy, LookupTableProcessor

# Import adaptive processing
from gemvq.gemv.adaptive_processor import AdaptiveProcessor
```

### Quantizer Imports
```python
# Import quantizers directly
from gemvq.quantizers import NLQ, HNLQ, get_d4, get_a2, get_e8

# Or import specific functions
from gemvq.quantizers.utils import closest_point_Dn, get_d4
```

## Installation

### Development Installation
```bash
# Clone and install in development mode
git clone <repository-url>
cd gemvq
pip install -e .
```

### Usage After Installation
```python
# Clean imports work immediately
from gemvq import NLQ, HNLQ, get_d4

# Create a quantizer
G = get_d4()
quantizer = NLQ(G=G, Q_nn=closest_point_Dn, q=4, beta=0.2, 
                alpha=1/3, eps=1e-8, dither=np.zeros(4))
```

## Benefits of New Structure

### 1. **Professional Package**
- No more `src/` prefix
- Clean, installable Python package
- Proper namespace organization

### 2. **Intuitive Imports**
- `from gemvq import NLQ` instead of `from src.quantizers.lattice.nlq import NLQ`
- Easy to remember and use
- Better IDE support and autocomplete

### 3. **Logical Organization**
- `quantizers/` for all quantization functionality
- `gemv/` for all matrix-vector multiplication functionality
- Clear separation of concerns

### 4. **Easy Maintenance**
- Clean module boundaries
- No circular import issues
- Proper `__init__.py` files with clean exports

### 5. **Better Testing**
- Tests can import cleanly from the installed package
- No path manipulation needed
- Consistent import patterns

## Migration from Old Structure

### Old Imports (Deprecated)
```python
# Old structure - no longer supported
from src.quantizers.lattice.nlq import NLQ
from src.gemv.columnwise.columnwise_matvec_processor import ColumnwiseMatvecProcessor
from src.adaptive.adaptive_matvec import AdaptiveMatvecProcessor
```

### New Imports (Current)
```python
# New clean structure
from gemvq import NLQ
from gemvq.gemv.columnwise import ColumnwiseMatVecProcessor
from gemvq.gemv.adaptive_processor import AdaptiveProcessor
```

## Package Configuration

The package is configured in `setup.py` to install as `gemvq`:

```python
setup(
    name="gemvq",
    version="1.0.0",
    packages=find_packages(),
    # ... other configuration
)
```

This allows users to install the package and import it cleanly without any path manipulation.

## Development Workflow

### 1. **Install in Development Mode**
```bash
pip install -e .
```

### 2. **Import and Use**
```python
from gemvq import NLQ, HNLQ, get_d4
```

### 3. **Run Tests**
```bash
python -m tests.run_all_tests
```

### 4. **Run Examples**
```bash
python examples/demo_coarse_to_fine.py
```

The new structure makes GEMV-Q a professional, easy-to-use Python package that follows best practices for Python package organization.
