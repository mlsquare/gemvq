# LatticeQuant Tests

This directory contains all the tests for the LatticeQuant library.

## Test Organization

Tests are organized into the following categories:

### Core Functionality Tests
- `test_closest_point.py` - Tests for closest point algorithms
- `test_nested_lattice_quantizer.py` - Tests for nested lattice quantization
- `test_gemv.py` - Tests for matrix-vector multiplication
- `test_adaptive_matvec.py` - Tests for adaptive matrix-vector multiplication
- `test_layer_wise_histogram.py` - Tests for layer-wise histogram functionality
- `test_standalone_layer_wise_histogram.py` - Standalone histogram tests

### Coarse-to-Fine Decoding Tests
- `test_coarse_to_fine.py` - Basic coarse-to-fine decoding tests
- `test_large_matrix_coarse_to_fine.py` - Large matrix coarse-to-fine tests
- `test_scaled_matrix_coarse_to_fine.py` - Scaled matrix coarse-to-fine tests
- `test_uniform_small_matrices.py` - Uniform random variable tests
- `test_fixed_hierarchical.py` - Fixed hierarchical quantizer tests
- `test_final_error_improvement.py` - Error improvement validation tests

### Analysis & Debugging Tests
- `test_error_types.py` - Error type analysis (cumulative vs tile-specific)
- `test_error_trends.py` - Error trend analysis
- `debug_hierarchical_quantizer.py` - Debugging tools for hierarchical quantization

### Other Tests
- `test_reorganization.py` - Reorganization functionality tests
- `test_demo.py` - Demo functionality tests

## Running Tests

### Run All Tests
```bash
python tests/run_all_tests.py
```

### Run Specific Test Categories
```bash
# Run only core functionality tests
python tests/run_all_tests.py --category "Core Functionality"

# Run only coarse-to-fine decoding tests
python tests/run_all_tests.py --category "Coarse-to-Fine Decoding"

# Run only analysis tests
python tests/run_all_tests.py --category "Analysis & Debugging"
```

### Run Specific Tests
```bash
# Run a specific test (partial name match)
python tests/run_all_tests.py --test "coarse_to_fine"

# Run uniform matrix tests
python tests/run_all_tests.py --test "uniform"
```

### List Available Tests
```bash
python tests/run_all_tests.py --list
```

### Run Individual Tests
```bash
# Run a specific test file directly
python tests/test_coarse_to_fine.py
python tests/test_uniform_small_matrices.py
python tests/test_scaled_matrix_coarse_to_fine.py
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

### Coarse-to-Fine Decoding Tests
These tests validate the hierarchical quantization's ability to provide progressive reconstruction:
- Error should decrease as more levels are used
- Monotonic error reduction is expected
- Different matrix sizes and quantization parameters are tested

### Uniform Random Variable Tests
These tests use uniform random variables for more controlled analysis:
- Better scale alignment with quantization levels
- More predictable behavior
- Ideal for algorithm validation

### Scaled Matrix Tests
These tests use matrices scaled by q^M to better reveal hierarchical behavior:
- Proper scale matching with quantization levels
- Better level separation
- Improved monotonicity

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
