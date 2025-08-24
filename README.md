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
from src.lattices.quantizers.hierarchical_nested_lattice_quantizer import HierarchicalNestedLatticeQuantizer
from src.lattices.utils import get_d4
from src.lattices.quantizers.nested_lattice_quantizer import closest_point_Dn

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
from src.gemv.columnwise_matvec_processor import ColumnwiseMatvecProcessor

# Create quantized matrix processor
matrix = np.random.uniform(0, 1, (100, 50)) * 256  # Scale by q^M
processor = ColumnwiseMatvecProcessor(matrix, 'D4', M=3)

# Perform coarse-to-fine multiplication
vector = np.random.uniform(0, 1, 50) * 256
result = processor.multiply_coarse_to_fine(vector, max_level=1)  # Use 2 levels
```

## Project Structure


```
LatticeQuant/
├── 📁 src/                          # Main source code
│   ├── 📁 lattices/                 # Lattice quantization core
│   │   ├── 📁 quantizers/           # Quantization algorithms
│   │   │   ├── hierarchical_nested_lattice_quantizer.py  # Multi-level quantization
│   │   │   └── nested_lattice_quantizer.py              # Single-level quantization
│   │   └── utils.py                 # Lattice utilities (D4, A2, E8, etc.)
│   │
│   ├── 📁 gemv/                     # Matrix-vector multiplication (GEMV)
│   │   ├── 📁 base/                 # Base classes and factory patterns
│   │   │   ├── gemv_factory.py      # Factory for creating GEMV processors
│   │   │   └── gemv_processor.py    # Base processor interface
│   │   │
│   │   ├── 📁 columnwise/           # Column-wise GEMV implementations
│   │   │   ├── columnwise_processor.py      # Column-wise processing logic
│   │   │   └── __init__.py
│   │   │
│   │   ├── 📁 rowwise/              # Row-wise GEMV implementations
│   │   │   ├── rowwise_processor.py         # Row-wise processing logic
│   │   │   └── __init__.py
│   │   │
│   │   ├── 📁 svd/                  # SVD-based GEMV
│   │   │   ├── svd_gemv_processor.py        # SVD decomposition approach
│   │   │   └── __init__.py
│   │   │
│   │   ├── 📁 utils/                # GEMV utilities
│   │   │   ├── padder.py            # Matrix padding utilities
│   │   │   └── __init__.py
│   │   │
│   │   ├── columnwise_matvec_processor.py   # Main column-wise processor
│   │   ├── rowwise_gemv.py                  # Row-wise GEMV implementation
│   │   ├── lookup_table_processor.py        # Lookup table approach
│   │   ├── standard_dot_processor.py        # Standard dot product
│   │   └── adaptive_processor.py            # Adaptive processing
│   │
│   ├── 📁 adaptive/                 # Adaptive quantization
│   │   ├── adaptive_matvec.py       # Adaptive matrix-vector multiplication
│   │   ├── layer_wise_histogram_matvec.py   # Layer-wise histogram approach
│   │   ├── demo_adaptive_matvec.py          # Adaptive GEMV demonstrations
│   │   └── demo_layer_wise_histogram.py     # Histogram demo
│   │
│   └── simple_test.py               # Simple test utilities
│
├── 📁 tests/                        # Comprehensive test suite
│   ├── run_all_tests.py             # Main test runner with categories
│   ├── README.md                    # Test documentation
│   ├── test_nested_lattice_quantizer.py     # Core quantization tests
│   ├── test_hierarchical_*.py       # Hierarchical quantization tests
│   ├── test_columnwise_matvec_options.py    # Column-wise GEMV tests
│   ├── test_adaptive_matvec.py              # Adaptive GEMV tests
│   ├── test_error_*.py              # Error analysis tests
│   └── test_*.py                    # Other specialized tests
│
├── 📁 examples/                     # Usage examples and demonstrations
│   ├── demo_coarse_to_fine.py       # Coarse-to-fine decoding demo
│   ├── example_adaptive_matvec.py   # Adaptive GEMV example
│   ├── example_coarse_to_fine.py    # Coarse-to-fine example
│   ├── analyze_rate_distortion_results.py   # Rate-distortion analysis
│   ├── compare_quantizer_distortion.py      # Quantizer comparison
│   ├── estimate_*.py                # Estimation examples
│   └── plot_*.py                    # Visualization examples
│
├── 📁 docs/                         # Documentation and analysis
│   ├── adaptive_matvec_approach.md  # Adaptive approach documentation
│   └── [other analysis docs]        # Various analysis documents
│
├── requirements.txt                 # Basic dependencies
├── requirements-dev.txt             # Development dependencies
├── setup.py                         # Package setup
└── README.md                        # This file
```

### Module Organization

#### 🏗️ **Core Architecture**
- **`src/lattices/`**: Core lattice quantization algorithms and utilities
- **`src/gemv/`**: Matrix-vector multiplication implementations with multiple strategies
- **`src/adaptive/`**: Adaptive quantization approaches for dynamic scenarios

#### 🔧 **GEMV Processing Strategies**
- **Column-wise**: Process matrix column by column (default approach)
- **Row-wise**: Process matrix row by row (alternative approach)
- **SVD-based**: Use singular value decomposition for efficient processing
- **Lookup Table**: Pre-computed lookup tables for fast computation
- **Adaptive**: Dynamic strategy selection based on data characteristics

#### 🧪 **Testing Framework**
- **Categorized Tests**: Organized by functionality (core, hierarchical, GEMV, etc.)
- **Comprehensive Runner**: `run_all_tests.py` with filtering and categorization
- **Specialized Tests**: Error analysis, distortion comparison, parameter testing

#### 📚 **Examples & Documentation**
- **Usage Examples**: Practical demonstrations of library capabilities
- **Analysis Tools**: Rate-distortion analysis, quantizer comparison
- **Visualization**: Plotting and result analysis utilities

## Running Tests

### Run All Tests
```bash
python -m tests.run_all_tests
```

### Run Specific Categories
```bash
# Core functionality
python -m tests.run_all_tests --category "Core Functionality"

# Coarse-to-fine decoding
python -m tests.run_all_tests --category "Coarse-to-Fine Decoding"

# Analysis and debugging
python -m tests.run_all_tests --category "Analysis & Debugging"
```

### Run Specific Tests
```bash
# Run uniform matrix tests
python -m tests.run_all_tests --test "uniform"

# Run coarse-to-fine tests
python -m tests.run_all_tests --test "coarse_to_fine"
```

### List Available Tests
```bash
python -m tests.run_all_tests --list
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

### 📖 **Technical Documentation**

#### **Adaptive Matrix-Vector Multiplication** (`adaptive_matvec_approach.md`)
- **Mathematical Framework**: Complete mathematical formulation of adaptive GEMV
- **Column-wise Interpretation**: Matrix-vector multiplication as linear combination of columns
- **Adaptive Quantization Strategy**: Dynamic bit rate allocation per column
- **Sparsity Exploitation**: Handling sparse vectors with known patterns
- **Implementation Approach**: Core components and architecture design
- **Performance Characteristics**: Computational complexity and memory requirements
- **Applications**: Neural networks, recommendation systems, signal processing
- **Future Extensions**: Multi-dimensional sparsity, dynamic adaptation, hardware acceleration

### 🔬 **Key Concepts Covered**

#### **Adaptive Bit Allocation**
- Each column can have different target bit rates based on importance
- Dynamic rate allocation using energy-based or importance-based methods
- Rate-distortion optimization for overall performance

#### **Sparsity Handling**
- Skip processing of zero vector elements for computational efficiency
- Pre-compute lookup tables only for active columns
- Memory and computation savings proportional to sparsity ratio

#### **Hierarchical Refinement**
- Multi-level quantization for better rate-distortion performance
- Successive refinement capability with coarse-to-fine decoding
- Efficient inner product estimation using precomputed tables

#### **Flexible Lattice Support**
- Support for different lattice types (Dₙ, A₂, E₈, Zⁿ)
- Optimized closest point algorithms for each lattice type
- Configurable quantization parameters for different use cases

### 📊 **Performance Analysis**

#### **Computational Complexity**
- **Encoding**: O(|S| × M × d) where |S| is sparsity, M is hierarchical levels, d is dimension
- **Decoding**: O(|S| × M × d) for selective column decoding
- **MatVec**: O(|S| × M²) using precomputed lookup tables

#### **Memory Requirements**
- **Encoded Matrix**: O(|S| × M × d × log₂(q)) bits
- **Lookup Tables**: O(Σᵢ q⁽ⁱ⁾^(2M)) entries
- **Working Memory**: O(d) for intermediate computations

#### **Rate-Distortion Performance**
- **Adaptive Allocation**: Better overall rate-distortion than uniform allocation
- **Sparsity Gain**: Additional compression proportional to sparsity ratio
- **Hierarchical Benefits**: Improved performance with increasing M

### 🎯 **Practical Applications**

#### **Neural Network Compression**
- Compress weight matrices with different importance levels
- Handle sparse activations efficiently
- Adaptive quantization based on layer sensitivity

#### **Recommendation Systems**
- Compress user-item matrices with varying sparsity
- Handle cold-start scenarios with known zero patterns
- Adaptive bit allocation based on popularity

#### **Signal Processing**
- Compress correlation matrices with known structure
- Handle sparse frequency domain representations
- Adaptive quantization based on signal characteristics

#### **Scientific Computing**
- Compress sparse matrices from finite element methods
- Handle structured sparsity in PDE discretizations
- Adaptive precision based on physical constraints

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
