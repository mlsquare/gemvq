# gemvq Tests

This directory contains all the tests for the gemvq library.

## Clean Test Structure

The test suite follows clean Python practices with explicit imports and minimal dependencies:

### 🧹 **Clean Import Patterns**
- **Explicit imports**: Tests import specific modules directly
- **No circular dependencies**: Clean import hierarchy
- **Updated paths**: All import paths reflect the reorganized structure
- **Method signature compliance**: All method calls include required parameters

### 📦 **Updated Import Examples**
```python
# Import from reorganized structure
from src.gemv.columnwise.columnwise_matvec_processor import ColumnwiseMatvecProcessor
from src.gemv.columnwise.simple_columnwise_matvec import SimpleColumnwiseMatvecProcessor
from src.gemv.rowwise.rowwise_processor import RowwiseGEMVProcessor
from src.gemv.utils.padder import BlockingStrategy
from src.lattices.utils import get_d4, closest_point_Dn
from src.lattices.quantizers.nested_lattice_quantizer import NestedLatticeQuantizer
```

## Test Organization

Tests are organized into the following categories:

### Core Functionality Tests
- `test_closest_point.py` - Tests for closest point algorithms
- `test_nested_lattice_quantizer.py` - Tests for nested lattice quantization
- `test_gemv.py` - Tests for matrix-vector multiplication
- `test_adaptive_matvec.py` - Tests for adaptive matrix-vector multiplication
- `test_layer_wise_histogram.py` - Tests for layer-wise histogram functionality
- `test_standalone_layer_wise_histogram.py` - Standalone histogram tests



### Analysis & Debugging Tests
- `test_error_types.py` - Error type analysis (cumulative vs tile-specific)
- `test_error_trends.py` - Error trend analysis

## Running Tests

### Run All Tests
```bash
# Using the test runner
python -m tests.run_all_tests

# Or run individual test modules
python -m tests.test_simple_columnwise_matvec
python -m tests.test_nested_lattice_quantizer
python -m tests.test_decoding_parameter
python -m tests.test_d4_lattice_simulation
```

### Run Specific Test Categories
```bash
# Run only core functionality tests
python -m tests.run_all_tests --category "Core Functionality"

# Run only analysis tests
python -m tests.run_all_tests --category "Analysis & Debugging"
```

### Run Specific Tests
```bash
# Run a specific test (partial name match)
python -m tests.run_all_tests --test "coarse_to_fine"

# Run individual test modules directly
python -m tests.test_simple_columnwise_matvec
python -m tests.test_nested_lattice_quantizer
python -m tests.test_decoding_parameter
```

### List Available Tests
```bash
python -m tests.run_all_tests --list
```

### Run Individual Tests
```bash
# Run a specific test file directly
python -m tests.test_closest_point
python -m tests.test_nested_lattice_quantizer
```

## Test Dependencies

Most tests require the following packages:
- numpy
- matplotlib (for visualization tests)
- scipy (for some advanced functionality)

Install dependencies with:
```bash
pip install -r requirements-dev.txt
```

## Test Output

Tests generate various outputs:
- **Console output**: Test results and error messages
- **Visualizations**: PNG files saved to the docs/ directory
- **Analysis reports**: Detailed analysis in console output

## Key Test Features



## Debugging Tests

The debugging tests help identify issues with the hierarchical quantization:
- `debug_hierarchical_quantizer.py`: Detailed analysis of quantizer behavior
- `test_error_types.py`: Analysis of cumulative vs tile-specific error
- `test_error_trends.py`: Statistical analysis of error patterns

## Continuous Integration

The test suite is designed to be run in CI/CD pipelines:
- All tests should pass for a successful build
- Tests are categorized for selective running
- Timeout protection prevents hanging tests
- Comprehensive error reporting

## Adding New Tests

When adding new tests:
1. Place them in the appropriate category
2. Update the `categorize_tests()` function in `run_all_tests.py`
3. Ensure tests have clear, descriptive names
4. Add appropriate documentation
5. Test both success and failure cases
