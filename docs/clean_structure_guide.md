# Clean Structure Guide

This document describes the clean structure and organization of the gemvq library, including the reorganization that was performed to improve maintainability and follow Python best practices.

## Overview

The gemvq library has been reorganized to follow clean Python practices with minimal `__init__.py` files, explicit imports, and logical module organization. This improves maintainability, reduces dependencies, and makes the codebase easier to understand and extend.

## Key Principles

### 🧹 **Clean Module Organization**
- **No imports in `__init__.py`**: All `__init__.py` files contain only module descriptions
- **Explicit imports**: Users must import specific modules directly
- **Logical organization**: Related functionality is grouped in subdirectories
- **Clear separation**: Utilities, demos, and implementations are clearly separated

### 📦 **Import Patterns**
- **Direct module imports**: Import specific classes and functions from their modules
- **No circular dependencies**: Clean import hierarchy
- **Updated paths**: All import paths reflect the reorganized structure
- **Method signature compliance**: All method calls include required parameters

## Directory Structure

### Before Reorganization
```
src/gemv/
├── __init__.py                    # Heavy imports and exports
├── row_wise_gemv.py              # Mixed with other files
├── column_wise_gemv.py           # Mixed with other files
├── lookup_table_processor.py     # Utility mixed with main files
├── padder.py                     # Utility mixed with main files
├── demo_*.py                     # Demos mixed with main files
└── [other mixed files]
```

### After Reorganization
```
src/gemv/
├── __init__.py                    # Minimal, only module description
├── adaptive_processor.py          # Main level file
├── base/                          # Base classes and factory
│   ├── __init__.py               # Minimal
│   ├── gemv_factory.py
│   └── gemv_processor.py
├── columnwise/                    # All column-wise implementations
│   ├── __init__.py               # Minimal
│   ├── columnwise_processor.py
│   ├── column_wise_gemv.py
│   ├── columnwise_matvec_processor.py
│   ├── columnwise_matvec_factory.py
│   ├── simple_columnwise_matvec.py
│   └── standard_dot_processor.py
├── rowwise/                       # All row-wise implementations
│   ├── __init__.py               # Minimal
│   ├── rowwise_processor.py
│   └── row_wise_gemv.py
├── svd/                           # SVD-based implementations
│   ├── __init__.py               # Minimal
│   └── svd_gemv_processor.py
├── utils/                         # Utility classes and functions
│   ├── __init__.py               # Minimal
│   ├── padder.py
│   └── lookup_table_processor.py
└── demos/                         # Demo scripts and examples
    ├── __init__.py               # Minimal
    ├── demo_new_structure.py
    ├── demo_columnwise_options_comprehensive.py
    ├── demo_columnwise_matvec_options.py
    └── columnwise_matvec_options.md
```

## Import Patterns

### ❌ **Old Pattern (Avoid)**
```python
# Heavy imports in __init__.py files
from src.gemv import GEMVProcessor, ColumnwiseGEMVProcessor, RowwiseGEMVProcessor

# Mixed file locations
from src.gemv.lookup_table_processor import LookupTableProcessor
from src.gemv.padder import BlockingStrategy
```

### ✅ **New Pattern (Use)**
```python
# Import specific modules directly
from src.gemv.columnwise.columnwise_matvec_processor import ColumnwiseMatvecProcessor
from src.gemv.rowwise.rowwise_processor import RowwiseGEMVProcessor
from src.gemv.utils.padder import BlockingStrategy
from src.gemv.utils.lookup_table_processor import LookupTableProcessor
from src.lattices.utils import get_d4, closest_point_Dn
from src.lattices.quantizers.nested_lattice_quantizer import NestedLatticeQuantizer
```

## Module Execution

### ✅ **Running Modules**
```bash
# Run modules directly using python -m
python -m src.gemv.columnwise.columnwise_matvec_processor
python -m src.gemv.rowwise.rowwise_processor
python -m src.lattices.utils
python -m src.lattices.quantizers.nested_lattice_quantizer
```

### ✅ **Running Tests**
```bash
# Run test modules directly
python -m tests.test_simple_columnwise_matvec
python -m tests.test_nested_lattice_quantizer
python -m tests.test_decoding_parameter
python -m tests.test_d4_lattice_simulation

# Or use the test runner
python -m tests.run_all_tests
```

## Benefits of Clean Structure

### 🚀 **Performance Benefits**
- **Faster module loading**: Minimal `__init__.py` files load faster
- **Reduced memory usage**: No unnecessary imports loaded
- **Better caching**: Python can cache modules more efficiently

### 🛠️ **Maintenance Benefits**
- **Easier to understand**: Clear module organization
- **Easier to modify**: Changes are localized to specific modules
- **Easier to test**: Explicit imports make testing simpler
- **Easier to debug**: Clear dependency chains

### 🔧 **Development Benefits**
- **No circular dependencies**: Clean import hierarchy
- **Explicit dependencies**: Clear what each module needs
- **Better IDE support**: IDEs can better understand the structure
- **Easier refactoring**: Changes don't affect unrelated modules

## Migration Guide

### For Users
1. **Update import statements**: Use the new import paths
2. **Use explicit imports**: Import specific classes and functions
3. **Use python -m**: Run modules using the `-m` flag

### For Developers
1. **Keep `__init__.py` minimal**: Only add module descriptions
2. **Group related files**: Put related functionality in subdirectories
3. **Use explicit imports**: Don't rely on `__init__.py` imports
4. **Update tests**: Ensure tests use the new import paths

## Examples

### Basic Usage
```python
# Import specific processors
from src.gemv.columnwise.columnwise_matvec_processor import ColumnwiseMatvecProcessor
from src.gemv.rowwise.rowwise_processor import RowwiseGEMVProcessor

# Import utilities
from src.gemv.utils.padder import BlockingStrategy
from src.lattices.utils import get_d4, closest_point_Dn

# Create processors
processor = ColumnwiseMatvecProcessor(matrix, 'D4', M=3)
```

### Advanced Usage
```python
# Import quantizers
from src.lattices.quantizers.nested_lattice_quantizer import NestedLatticeQuantizer
from src.lattices.quantizers.hierarchical_nested_lattice_quantizer import HierarchicalNestedLatticeQuantizer

# Import factory functions
from src.gemv.columnwise.columnwise_matvec_factory import create_processor

# Create quantizers and processors
quantizer = NestedLatticeQuantizer(G=get_d4(), q=4, beta=0.2)
processor = create_processor(matrix, 'standard_dot')
```

## Best Practices

### ✅ **Do**
- Import specific modules directly
- Use the `python -m` approach for running modules
- Keep `__init__.py` files minimal
- Group related functionality in subdirectories
- Use explicit import paths

### ❌ **Don't**
- Add imports to `__init__.py` files
- Mix different types of files in the same directory
- Rely on `__init__.py` imports
- Create circular dependencies
- Use relative imports when absolute imports are clearer

## Conclusion

The clean structure of gemvq provides better maintainability, performance, and developer experience. By following these patterns, the codebase becomes easier to understand, modify, and extend while maintaining high performance and reducing potential issues.
