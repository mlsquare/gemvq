# LatticeQuant

A comprehensive library for lattice quantization and hierarchical nested lattice quantization with coarse-to-fine decoding capabilities.

## Overview

LatticeQuant provides efficient implementations of:
- **Lattice Quantization**: Quantizing vectors to points on lattices (D4, A2, E8, Z2, Z3)
- **Hierarchical Nested Lattice Quantization**: Multi-level quantization for progressive refinement
- **Coarse-to-Fine Decoding**: Progressive reconstruction from coarse to fine levels
- **Matrix-Vector Multiplication**: Efficient GEMV operations using quantized matrices
- **Adaptive Quantization**: Dynamic quantization based on data characteristics

## Key Features

### 🎯 **Coarse-to-Fine Decoding**
- Progressive reconstruction from coarse to fine levels
- Monotonic error reduction as more levels are used
- Support for variable depth decoding
- Both column-wise and row-wise GEMV approaches

### 🔧 **Multiple Lattice Types**
- **D4**: 4-dimensional checkerboard lattice
- **A2**: 2-dimensional hexagonal lattice  
- **E8**: 8-dimensional Gosset lattice
- **Z2/Z3**: Integer lattices

### 📊 **Comprehensive Testing**
- Extensive test suite with categorized tests
- Uniform random variable testing for controlled analysis
- Scaled matrix testing for better hierarchical behavior
- Error type analysis (cumulative vs tile-specific)

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd LatticeQuant

# Install dependencies
pip install -r requirements.txt

# For development
pip install -r requirements-dev.txt
```

## Quick Start

### Basic Hierarchical Quantization

```python
import numpy as np
from src.quantizers.hierarchical_nested_lattice_quantizer import HierarchicalNestedLatticeQuantizer
from src.utils import get_d4
from src.quantizers.closest_point import closest_point_Dn

# Setup quantizer
G = get_d4()
quantizer = HierarchicalNestedLatticeQuantizer(
    G=G, Q_nn=closest_point_Dn, q=4, beta=0.2,
    alpha=1/3, eps=1e-8, dither=np.zeros(4), M=3
)

# Encode vector
x = np.random.uniform(0, 1, 4) * 64  # Scale by q^M
b_list, T = quantizer.encode(x, with_dither=False)

# Decode at different levels
for level in range(3):
    reconstruction = quantizer.decode_coarse_to_fine(b_list, T, with_dither=False, max_level=level)
    error = np.linalg.norm(reconstruction - x) / np.linalg.norm(x)
    print(f"Level {level}: Error = {error:.6f}")
```

### Matrix-Vector Multiplication

```python
from src.gemv.lattice_quantized_gemv import LatticeQuantizedGEMV

# Create quantized matrix processor
matrix = np.random.uniform(0, 1, (100, 50)) * 256  # Scale by q^M
processor = LatticeQuantizedGEMV(matrix, 'auto', 'D4', M=3)

# Perform coarse-to-fine multiplication
vector = np.random.uniform(0, 1, 50) * 256
result = processor.multiply_coarse_to_fine(vector, max_level=1)  # Use 2 levels
```

## Project Structure

```
LatticeQuant/
├── src/                          # Source code
│   ├── quantizers/              # Quantization algorithms
│   ├── gemv/                    # Matrix-vector multiplication
│   ├── adaptive/                # Adaptive quantization
│   └── utils.py                 # Utility functions
├── tests/                       # Test suite
│   ├── run_all_tests.py         # Comprehensive test runner
│   ├── README.md                # Test documentation
│   └── [test files]             # Categorized test files
├── docs/                        # Documentation and results
│   ├── *.md                     # Analysis documents
│   └── *.png                    # Visualization results
├── requirements.txt             # Basic dependencies
├── requirements-dev.txt         # Development dependencies
└── setup.py                     # Package setup
```

## Running Tests

### Run All Tests
```bash
python tests/run_all_tests.py
```

### Run Specific Categories
```bash
# Core functionality
python tests/run_all_tests.py --category "Core Functionality"

# Coarse-to-fine decoding
python tests/run_all_tests.py --category "Coarse-to-Fine Decoding"

# Analysis and debugging
python tests/run_all_tests.py --category "Analysis & Debugging"
```

### Run Specific Tests
```bash
# Run uniform matrix tests
python tests/run_all_tests.py --test "uniform"

# Run coarse-to-fine tests
python tests/run_all_tests.py --test "coarse_to_fine"
```

### List Available Tests
```bash
python tests/run_all_tests.py --list
```

## Key Concepts

### Hierarchical Quantization
The library implements hierarchical nested lattice quantization with M levels:
- **Level 0**: Coarsest approximation (MSB - Most Significant Bits)
- **Level M-1**: Finest detail (LSB - Least Significant Bits)
- **Progressive Reconstruction**: Can decode from any level 0 to M-1

### Coarse-to-Fine Decoding
- **Cumulative Error**: Error calculated for complete reconstruction using levels 0 to max_level
- **Monotonic Reduction**: Error should decrease as more levels are used
- **Progressive Quality**: Quality improves progressively with each additional level

### Scaling Strategy
For optimal performance, scale input data by q^M:
- **Uniform Random Variables**: Provide controlled, predictable behavior
- **Scale Factor**: q^M where q is quantization parameter, M is number of levels
- **Better Alignment**: Ensures proper alignment with quantization levels

## Documentation

Comprehensive documentation is available in the `docs/` folder:

- **COARSE_TO_FINE_DECODING.md**: Detailed explanation of coarse-to-fine decoding
- **UNIFORM_RANDOM_ANALYSIS.md**: Analysis of uniform random variable testing
- **SCALED_MATRIX_ANALYSIS.md**: Analysis of scaled matrix testing
- **ERROR_TYPE_ANALYSIS.md**: Analysis of cumulative vs tile-specific error
- **MSB_LSB_FIX_SUMMARY.md**: Summary of MSB/LSB ordering fixes

## Performance Characteristics

### Monotonicity Results
- **Uniform Random Variables**: 100% monotonic quantizer performance
- **Scaled Matrices**: 2.6x better monotonicity than unscaled
- **Overall Success Rate**: ~74% with proper scaling and uniform variables

### Compression Ratios
- **M=2**: ~8x compression
- **M=3**: ~5.3x compression  
- **M=4**: ~4x compression

## Contributing

1. Follow the existing code structure
2. Add tests for new functionality
3. Update documentation
4. Ensure all tests pass
5. Use uniform random variables for controlled testing

## License

[License information]

## Citation

If you use this library in your research, please cite:

```bibtex
[Citation information]
```

## Acknowledgments

- Implementation based on hierarchical nested lattice quantization theory
- Coarse-to-fine decoding inspired by progressive transmission techniques
- Testing methodology developed for robust algorithm validation
