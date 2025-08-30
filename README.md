# GEMV-Q

A comprehensive library for matrix-vector multiplication (GEMV) based on different lattice quantizers, including nested lattice quantizers and hierarchical nested lattice quantizers. This library provides efficient implementations of quantized matrix-vector operations with support for multiple processing strategies and adaptive quantization.

## Overview

GEMV-Q provides efficient implementations of:
- **Matrix-Vector Multiplication**: Efficient GEMV operations using quantized matrices with different lattice quantizers
- **Lattice Quantization**: Quantizing vectors to points on lattices (D4, A2, E8, Z2, Z3) using latticequant library
- **Nested Lattice Quantization**: Single-level quantization for matrix compression
- **Hierarchical Nested Lattice Quantization**: Multi-level quantization for progressive refinement
- **Coarse-to-Fine Decoding**: Progressive reconstruction from coarse to fine levels
- **Adaptive Quantization**: Dynamic quantization based on data characteristics
- **Multiple Processing Strategies**: Column-wise, row-wise, SVD-based, and lookup table approaches

## Key Features

### 🎯 **Matrix-Vector Multiplication (GEMV)**
- **Quantized Matrix Operations**: Efficient matrix-vector multiplication using lattice-quantized matrices
- **Multiple Processing Strategies**: Column-wise, row-wise, SVD-based, and lookup table approaches
- **Coarse-to-Fine Decoding**: Progressive reconstruction from coarse to fine levels
- **Monotonic Error Reduction**: Error decreases as more quantization levels are used
- **Variable Depth Decoding**: Support for decoding at different quantization levels
- **Adaptive Processing**: Dynamic strategy selection based on data characteristics

### 🔧 **Lattice Quantizer Support**
- **Nested Lattice Quantizer (NLQ)**: Single-level quantization for matrix compression
- **Hierarchical Nested Lattice Quantizer (HNLQ)**: Multi-level quantization for progressive refinement
- **Multiple Lattice Types**: Support for D4, A2, E8, Z2, Z3 lattices from latticequant library
- **D4**: 4-dimensional checkerboard lattice
- **A2**: 2-dimensional hexagonal lattice  
- **E8**: 8-dimensional Gosset lattice
- **Z2/Z3**: Integer lattices

### 📊 **Comprehensive Testing**
- Extensive test suite with categorized tests
- Uniform random variable testing for controlled analysis
- Scaled matrix testing for better hierarchical behavior
- Error type analysis (cumulative vs tile-specific)
- Performance benchmarking and validation

### 🎛️ **Adaptive Features**
- **Layer-wise Histogram Processing**: Adaptive quantization based on data distribution
- **Dynamic Bit Allocation**: Optimal bit rate allocation per column
- **Sparsity Exploitation**: Efficient handling of sparse vectors
- **Rate-Distortion Optimization**: Adaptive quantization for optimal performance

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd gemvq

# Install the package in development mode
pip install -e .

# Install dependencies
pip install -r requirements.txt

# For development
pip install -r requirements-dev.txt
```

### Package Installation
After installation, you can import the package directly:

```python
# Clean imports from the installed package
from gemvq import NLQ, HNLQ, get_d4
from gemvq.gemv.columnwise import ColumnwiseMatVecProcessor
```

## Quick Start

### Matrix-Vector Multiplication with Lattice Quantizers

```python
import numpy as np
from gemvq.gemv.columnwise import ColumnwiseMatVecProcessor

# Create a matrix and quantize it using D4 lattice
matrix = np.random.uniform(0, 1, (100, 50)) * 256  # Scale by q^M
processor = ColumnwiseMatVecProcessor(matrix, 'D4', M=3)

# Perform matrix-vector multiplication with coarse-to-fine decoding
vector = np.random.uniform(0, 1, 50) * 256
result = processor.multiply_coarse_to_fine(vector, max_level=1)  # Use 2 levels
print(f"Matrix-vector multiplication result shape: {result.shape}")
```

### Using Different Lattice Quantizers

```python
# Using nested lattice quantizer (single-level)
from gemvq import NLQ, get_d4

G = get_d4()
nested_quantizer = NLQ(G=G, Q_nn=closest_point_Dn, q=4, beta=0.2, alpha=1/3, eps=1e-8)

# Using hierarchical nested lattice quantizer (multi-level)
from gemvq import HNLQ, closest_point_Dn

hierarchical_quantizer = HNLQ(
    G=G, Q_nn=closest_point_Dn, q=4, beta=0.2,
    alpha=1/3, eps=1e-8, dither=np.zeros(4), M=3
)

# Encode and decode with progressive refinement
x = np.random.uniform(0, 1, 4) * 64  # Scale by q^M
b_list, T = hierarchical_quantizer.encode(x, with_dither=False)

# Decode at different levels for progressive quality
for level in range(3):
    reconstruction = hierarchical_quantizer.decode_coarse_to_fine(b_list, T, with_dither=False, max_level=level)
    error = np.linalg.norm(reconstruction - x) / np.linalg.norm(x)
    print(f"Level {level}: Error = {error:.6f}")
```

### Adaptive Matrix-Vector Multiplication

```python
from gemvq import get_d4
from gemvq.gemv.adaptive_processor import AdaptiveProcessor

# Create adaptive processor with D4 lattice
G = get_d4()
processor = AdaptiveProcessor(G=G, q=4, beta=0.2, M=3)

# Create matrix and vector
matrix = np.random.uniform(0, 1, (100, 50)) * 256
vector = np.random.uniform(0, 1, 50) * 256

# Perform adaptive matrix-vector multiplication
result = processor.adaptive_matvec(matrix, vector, target_rate=2.0)
print(f"Adaptive GEMV result shape: {result.shape}")
```

## Project Structure

```
gemvq/
├── 📁 gemvq/                        # Main package (installable)
│   ├── 📁 quantizers/               # Quantization core
│   │   ├── nlq.py                   # Single-level quantization
│   │   ├── hnlq.py                  # Multi-level quantization
│   │   ├── utils.py                 # Lattice utilities (D4, A2, E8, etc.)
│   │   └── __init__.py
│   │
│   ├── 📁 gemv/                     # Matrix-vector multiplication
│   │   ├── 📁 columnwise/           # Column-wise GEMV implementations
│   │   │   ├── columnwise_matvec_processor.py # Main column-wise processor
│   │   │   ├── columnwise_matvec_factory.py  # Factory functions
│   │   │   ├── columnwise_processor.py       # Column-wise processing logic
│   │   │   ├── column_wise_gemv.py           # Column-wise GEMV implementation
│   │   │   ├── simple_columnwise_matvec.py   # Simplified column-wise processor
│   │   │   ├── standard_dot_processor.py     # Standard dot product
│   │   │   └── __init__.py
│   │   │
│   │   ├── 📁 rowwise/              # Row-wise GEMV implementations
│   │   │   ├── rowwise_processor.py          # Row-wise processing logic
│   │   │   ├── row_wise_gemv.py              # Row-wise GEMV implementation
│   │   │   └── __init__.py
│   │   │
│   │   ├── 📁 svd/                  # SVD-based GEMV
│   │   │   ├── svd_gemv_processor.py         # SVD decomposition approach
│   │   │   └── __init__.py
│   │   │
│   │   ├── 📁 utils/                # GEMV utilities
│   │   │   ├── padder.py            # Matrix padding utilities
│   │   │   ├── lookup_table_processor.py     # Lookup table approach
│   │   │   └── __init__.py
│   │   │
│   │   ├── gemv_processor.py        # Base processor interface
│   │   ├── gemv_factory.py          # Factory for creating GEMV processors
│   │   ├── adaptive_processor.py    # Adaptive processing
│   │   └── __init__.py
│   │
│   └── __init__.py                  # Main package initialization
│
├── 📁 tests/                        # Comprehensive test suite
│   ├── run_all_tests.py             # Main test runner with categories
│   ├── test_nested_lattice_quantizer.py     # Core quantization tests
│   ├── test_hierarchical_*.py       # Hierarchical quantization tests
│   ├── test_columnwise_matvec_options.py    # Column-wise GEMV tests
│   ├── test_adaptive_matvec.py              # Adaptive GEMV tests
│   ├── test_layer_wise_histogram.py         # Layer-wise histogram tests
│   ├── test_d4_lattice_simulation.py        # D4 lattice simulation tests
│   ├── test_decoding_parameter.py           # Decoding parameter tests
│   └── __init__.py
│
├── 📁 examples/                     # Usage examples and demonstrations
│   ├── demo_coarse_to_fine.py       # Coarse-to-fine decoding demo
│   ├── example_adaptive_matvec.py   # Adaptive GEMV example
│   ├── example_coarse_to_fine.py    # Coarse-to-fine example
│   ├── analyze_rate_distortion_results.py   # Rate-distortion analysis
│   ├── compare_quantizer_distortion.py      # Quantizer comparison
│   ├── estimate_*.py                # Estimation examples
│   ├── plot_*.py                    # Visualization examples
│   ├── demo_hnlq_*.py               # HNLQ demonstrations
│   ├── d4_nested_lattice_manim.py   # D4 lattice visualization
│   └── __init__.py
│
├── 📁 docs/                         # Documentation and analysis
│   ├── 📁 gemv/                     # GEMV documentation
│   ├── 📁 lattices/                 # Lattice documentation
│   ├── 📁 quantizers/               # Quantizer documentation
│   ├── 📁 notebooks/                # Jupyter notebooks
│   └── 📁 papers/                   # Research papers
│
├── requirements.txt                 # Basic dependencies
├── requirements-dev.txt             # Development dependencies
├── setup.py                         # Package setup
└── README.md                        # This file
```

### Module Organization

#### 🏗️ **Core Architecture**

**Quantization Layer** (`gemvq/quantizers/`)
- **`nlq.py`**: Single-level nested lattice quantization for matrix compression
- **`hnlq.py`**: Multi-level hierarchical nested lattice quantization with progressive refinement
- **`utils.py`**: Lattice utilities supporting D4, A2, E8, Z2, Z3 lattices with optimized closest point algorithms

**GEMV Processing Layer** (`gemvq/gemv/`)
- **`columnwise/`**: Column-wise matrix-vector multiplication with multiple strategies
  - `columnwise_matvec_processor.py`: Main processor with coarse-to-fine decoding
  - `columnwise_matvec_factory.py`: Factory pattern for processor creation
  - `simple_columnwise_matvec.py`: Simplified implementation for basic use cases
- **`rowwise/`**: Row-wise processing alternative for specific matrix structures
- **`svd/`**: SVD-based decomposition for efficient processing of structured matrices
- **`utils/`**: Shared utilities including lookup tables and matrix padding
- **`adaptive_processor.py`**: Dynamic strategy selection based on data characteristics

**Adaptive Layer** (`gemvq/adaptive/`)
- **`adaptive_matvec.py`**: Core adaptive matrix-vector multiplication algorithms
- **`layer_wise_histogram_matvec.py`**: Histogram-based adaptive quantization
- **`demo_adaptive_matvec.py`**: Demonstrations of adaptive processing capabilities

**Architecture Benefits**:
- **Modular Design**: Each layer has clear responsibilities and interfaces
- **Strategy Pattern**: Multiple processing strategies can be easily swapped
- **Factory Pattern**: Clean object creation with configurable parameters
- **Progressive Refinement**: Support for coarse-to-fine decoding at multiple levels
- **Extensible Framework**: Easy to add new lattice types and processing strategies

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
# Run specific test modules
python -m tests.test_nested_lattice_quantizer
python -m tests.test_columnwise_matvec_options
python -m tests.test_adaptive_matvec
python -m tests.test_layer_wise_histogram
python -m tests.test_d4_lattice_simulation
python -m tests.test_decoding_parameter

# Run uniform matrix tests
python -m tests.run_all_tests --test "uniform"

# Run coarse-to-fine tests
python -m tests.run_all_tests --test "coarse_to_fine"
```

### List Available Tests
```bash
python -m tests.run_all_tests --list
```

## Clean Structure and Import Patterns

### 📦 **Import Patterns**
```python
# Clean, intuitive imports from the main package
from gemvq import NLQ, HNLQ, get_d4, closest_point_Dn

# Import specific modules directly
from gemvq.gemv.columnwise import ColumnwiseMatVecProcessor
from gemvq.gemv.rowwise import RowwiseGEMVProcessor
from gemvq.gemv.svd import SVDGEMVProcessor
from gemvq.gemv.utils import BlockingStrategy, LookupTableProcessor
from gemvq.gemv.adaptive_processor import AdaptiveProcessor

# Import quantizers directly
from gemvq.quantizers import NLQ, HNLQ, get_d4, get_a2, get_e8

# Run modules directly
python -m gemvq.gemv.columnwise.columnwise_matvec_processor
python -m gemvq.quantizers.utils
python -m tests.test_nested_lattice_quantizer
```

### 🏗️ **Directory Structure Benefits**
- **`columnwise/`**: All column-wise implementations in one place
- **`rowwise/`**: All row-wise implementations in one place  
- **`utils/`**: Shared utilities and helper functions
- **`demos/`**: Demonstration scripts and examples
- **`base/`**: Base classes and factory patterns

## Key Concepts

### Matrix-Vector Multiplication with Lattice Quantizers
The library provides efficient matrix-vector multiplication using different lattice quantizers:
- **Quantized Matrices**: Matrices are quantized using lattice quantizers (nested or hierarchical)
- **Multiple Processing Strategies**: Column-wise, row-wise, SVD-based, and lookup table approaches
- **Progressive Decoding**: Can decode matrix-vector products at different quantization levels
- **Latticequant Integration**: Builds on the latticequant library for core quantization algorithms

### Lattice Quantizer Types
- **Nested Lattice Quantizer (NLQ)**: Single-level quantization for matrix compression
- **Hierarchical Nested Lattice Quantizer (HNLQ)**: Multi-level quantization with M levels:
  - **Level 0**: Coarsest approximation (MSB - Most Significant Bits)
  - **Level M-1**: Finest detail (LSB - Least Significant Bits)
  - **Progressive Reconstruction**: Can decode from any level 0 to M-1

### Coarse-to-Fine Decoding
- **Cumulative Error**: Error calculated for complete reconstruction using levels 0 to max_level
- **Monotonic Reduction**: Error should decrease as more levels are used
- **Progressive Quality**: Quality improves progressively with each additional level

### Adaptive Quantization
- **Dynamic Bit Allocation**: Optimal bit rate allocation per column based on importance
- **Layer-wise Processing**: Adaptive quantization based on data distribution characteristics
- **Sparsity Exploitation**: Efficient handling of sparse vectors with known patterns
- **Rate-Distortion Optimization**: Adaptive quantization for optimal performance

### Scaling Strategy
For optimal performance, scale input data by q^M:
- **Uniform Random Variables**: Provide controlled, predictable behavior
- **Scale Factor**: q^M where q is quantization parameter, M is number of levels
- **Better Alignment**: Ensures proper alignment with quantization levels

## Documentation

Comprehensive documentation is available in the `docs/` folder:

### 📖 **Technical Documentation**

#### **GEMV Module** (`docs/gemv/`)
- **Column-wise Processing**: Complete guide to column-wise matrix-vector multiplication
- **Row-wise Processing**: Alternative row-wise processing strategies
- **SVD-based Processing**: Singular value decomposition approaches
- **Lookup Table Processing**: Pre-computed table strategies

#### **Lattice Documentation** (`docs/lattices/`)
- **D4 Lattice**: 4-dimensional checkerboard lattice implementation
- **A2 Lattice**: 2-dimensional hexagonal lattice
- **E8 Lattice**: 8-dimensional Gosset lattice
- **Z2/Z3 Lattices**: Integer lattice implementations

#### **Quantizer Documentation** (`docs/quantizers/`)
- **Nested Lattice Quantization**: Single-level quantization guide
- **Hierarchical Nested Lattice Quantization**: Multi-level quantization guide
- **Beta Selection**: Parameter optimization for HNLQ
- **Improvements**: Recent enhancements and optimizations

#### **Adaptive Approach** (`docs/gemv/adaptive_matvec_approach.qmd`)
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

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

The MIT License is a permissive license that allows for:
- Commercial use
- Modification
- Distribution
- Private use

While providing liability protection for the authors. See the full license text in the [LICENSE](LICENSE) file.

## Citation

If you use this library in your research, please cite:

```bibtex
@article{gemv-q,
  title={A library for GEMV based on Lattice Quantizers},
  author={Soma S Dhavala},s
}


```

## References

This library is based on the following foundational works:

1. Kaplan and O. Ordentlich, “High-Rate Nested-Lattice Quantized Matrix Multiplication with Small Lookup Tables”, to be presented in ISIT 2025, arXiv:2505.13164, 2025.

2. O. Ordentlich and Y. Polyanskiy, “Optimal quantization for matrix multiplication”, arXiv preprint arXiv:2410.13780, 2024.

3. J. Conway and N. Sloane, "Fast quantizing and decoding and algorithms for lattice quantizers and codes", in IEEE Transactions on Information Theory, vol. 28, no. 2, pp. 227-232, March 1982, doi: 10.1109/TIT.1982.1056484..

## Acknowledgments

**[LatticeQuant Repository](https://github.com/iriskaplan/LatticeQuant)**: The core implementation by Iris Kaplan provides the foundational M-leveled nested-lattice quantizer and Voronoi code quantizer implementations. This repository implements the work described in the papers above and serves as the base for GEMV-Q's quantization capabilities.