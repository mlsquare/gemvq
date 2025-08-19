# LatticeQuant: High-Rate Nested-Lattice Quantized Matrix Multiplication

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)

## Overview

LatticeQuant is a Python library implementing high-rate nested-lattice quantization for matrix multiplication with small lookup tables. This repository presents and demonstrates the work described in the paper "High-Rate Nested-Lattice Quantized Matrix Multiplication with Small Lookup Tables" by Kaplan and Ordentlich (ISIT 2025).

The library provides efficient implementations of both classic nested lattice quantization and a novel hierarchical approach that achieves superior rate-distortion performance while enabling fast inner product estimation using compact lookup tables.

## Key Features

### 🏗️ **Hierarchical Nested Lattice Quantizer**
- **Multi-level quantization**: Implements M-level hierarchical quantization for successive refinement
- **Efficient inner product estimation**: Uses small lookup tables for fast matrix multiplication
- **Superior rate-distortion performance**: Achieves better compression than traditional methods
- **Flexible parameterization**: Supports various lattice types and quantization parameters

### 🔧 **Classic Nested Lattice Quantizer**
- **Reference implementation**: Provides baseline Voronoi code quantization
- **Comparison framework**: Enables performance analysis against hierarchical methods
- **Standard compliance**: Implements traditional nested lattice quantization

### 🎯 **Advanced Lattice Algorithms**
- **Multiple lattice types**: Support for Dₙ, A₂, E₈, and Zⁿ lattices
- **Fast closest point algorithms**: Optimized implementations from Conway & Sloane
- **Efficient codebook generation**: Precomputed lookup tables for acceleration

### 📊 **Comprehensive Analysis Tools**
- **Rate-distortion analysis**: Compare quantization methods against theoretical bounds
- **Correlation analysis**: Study performance with correlated data
- **Visualization tools**: Plot codebooks and distortion-rate curves
- **Performance benchmarking**: Evaluate quantization efficiency

## Installation

### Prerequisites

- Python 3.7 or higher
- NumPy
- Matplotlib
- SciPy

### Quick Install

```bash
# Clone the repository
git clone https://github.com/mlsquare/LatticeQuant.git
cd LatticeQuant

# Install dependencies
pip install numpy matplotlib scipy

# Optional: Install in development mode
pip install -e .
```

### From Source

```bash
# Clone and setup
git clone https://github.com/mlsquare/LatticeQuant.git
cd LatticeQuant

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### Basic Quantization

```python
import numpy as np
from src.hierarchical_nested_lattice_quantizer import HierarchicalNestedLatticeQuantizer
from src.utils import get_d4
from src.closest_point import closest_point_Dn

# Setup parameters
G = get_d4()  # D4 lattice
q = 4         # Quantization parameter
M = 2         # Hierarchical levels
beta = 0.2    # Scaling parameter
alpha = 1/3   # Overload scaling
eps = 1e-8    # Perturbation
dither = np.zeros(4)  # No dither

# Create quantizer
quantizer = HierarchicalNestedLatticeQuantizer(
    G=G, Q_nn=closest_point_Dn, q=q, beta=beta, 
    alpha=alpha, eps=eps, dither=dither, M=M
)

# Quantize a vector
x = np.random.normal(0, 1, size=4)
encoded, T = quantizer.encode(x, with_dither=False)
decoded = quantizer.decode(encoded, T, with_dither=False)

print(f"Original: {x}")
print(f"Quantized: {decoded}")
print(f"MSE: {np.mean((x - decoded)**2):.6f}")
```

### Inner Product Estimation

```python
from src.utils import precompute_hq_lut, calculate_weighted_sum

# Precompute lookup table
lut = precompute_hq_lut(G, closest_point_Dn, q, M, eps)

# Estimate inner product between two vectors
x1 = np.random.normal(0, 1, size=4)
x2 = np.random.normal(0, 1, size=4)

# Encode both vectors
enc1, T1 = quantizer.encode(x1, with_dither=False)
enc2, T2 = quantizer.encode(x2, with_dither=False)

# Calculate scaling factor
c = (2**(quantizer.alpha * (T1 + T2))) * (quantizer.beta**2)

# Estimate inner product using lookup table
estimated_ip = c * calculate_weighted_sum(enc1, enc2, lut, q)
true_ip = np.dot(x1, x2)

print(f"True inner product: {true_ip:.6f}")
print(f"Estimated inner product: {estimated_ip:.6f}")
print(f"Relative error: {abs(estimated_ip - true_ip)/abs(true_ip)*100:.2f}%")
```

### Rate-Distortion Analysis

```python
from src.estimate_inner_product import plot_distortion_rate

# Generate comprehensive rate-distortion comparison
plot_distortion_rate()
```

## Advanced Usage

### Custom Lattice Types

```python
from src.utils import get_a2, get_e8
from src.closest_point import closest_point_A2, closest_point_E8

# A2 lattice (hexagonal)
G_a2 = get_a2()
quantizer_a2 = HierarchicalNestedLatticeQuantizer(
    G=G_a2, Q_nn=closest_point_A2, q=4, beta=0.2,
    alpha=1/3, eps=1e-8, dither=np.zeros(2), M=2
)

# E8 lattice (8D)
G_e8 = get_e8()
quantizer_e8 = HierarchicalNestedLatticeQuantizer(
    G=G_e8, Q_nn=closest_point_E8, q=4, beta=0.2,
    alpha=1/3, eps=1e-8, dither=np.zeros(8), M=2
)
```

### Parameter Optimization

```python
from src.estimate_inner_product import find_best_beta

# Find optimal beta for given parameters
G = get_d4()
q, m = 4, 2
alpha = 1/3
sig_l = np.sqrt(2) * 0.076602  # D4 lattice parameter
eps = 1e-8 * np.random.normal(0, 1, size=4)

optimal_R, optimal_beta = find_best_beta(
    G, closest_point_Dn, q, m, alpha, sig_l, eps
)

print(f"Optimal beta: {optimal_beta:.4f}")
print(f"Optimal rate: {optimal_R:.4f} bits/dimension")
```

### Correlation Analysis

```python
from src.estimate_correlated_inner_product import plot_distortion_rho

# Analyze performance with correlated data
plot_distortion_rho()
```

## API Reference

### Core Classes

#### `HierarchicalNestedLatticeQuantizer`
The main quantizer implementing multi-level hierarchical quantization.

**Methods:**
- `encode(x, with_dither)`: Encode vector with M hierarchical levels
- `decode(b_list, T, with_dither)`: Decode hierarchical encoding
- `quantize(x, with_dither)`: Complete encode-decode process
- `create_q_codebook(with_dither)`: Generate lookup codebook

#### `NestedLatticeQuantizer`
Classic nested lattice quantizer for comparison.

**Methods:**
- `encode(x, with_dither)`: Single-level encoding
- `decode(enc, T, with_dither)`: Single-level decoding
- `quantize(x)`: Complete quantization
- `create_codebook(with_dither)`: Generate codebook

### Utility Functions

#### Lattice Generation
- `get_d4()`: D4 lattice generator matrix
- `get_a2()`: A2 lattice generator matrix
- `get_e8()`: E8 lattice generator matrix
- `get_z2()`, `get_z3()`: Z², Z³ lattice matrices

#### Closest Point Algorithms
- `closest_point_Dn(x)`: Dₙ lattice closest point
- `closest_point_A2(u)`: A₂ lattice closest point
- `closest_point_E8(x)`: E₈ lattice closest point

#### Analysis Tools
- `precompute_hq_lut(G, Q_nn, q, m, eps)`: Precompute lookup table
- `calculate_weighted_sum(a_list, b_list, lut, q)`: Weighted inner product
- `calculate_mse(x, x_hat)`: Mean squared error
- `calculate_t_entropy(T_values, q)`: Entropy calculation

## Potential Applications

### 🚀 **Machine Learning Acceleration**
- **Neural network compression**: Reduce model size while maintaining accuracy
- **Fast similarity search**: Efficient nearest neighbor algorithms
- **Embedding compression**: Compress word/document embeddings
- **Federated learning**: Reduce communication overhead

### 📱 **Edge Computing**
- **Mobile AI**: Enable complex models on resource-constrained devices
- **IoT applications**: Efficient sensor data processing
- **Real-time inference**: Fast matrix operations for streaming data

### 🔬 **Scientific Computing**
- **Signal processing**: Efficient filtering and correlation
- **Image compression**: Advanced quantization for visual data
- **Numerical analysis**: Fast approximation algorithms
- **Optimization**: Efficient gradient computations

### 💾 **Data Compression**
- **Database systems**: Compress high-dimensional data
- **Information retrieval**: Fast similarity computations
- **Recommendation systems**: Efficient user-item similarity
- **Bioinformatics**: Sequence alignment and analysis

### 🎮 **Computer Graphics**
- **Real-time rendering**: Fast lighting calculations
- **Procedural generation**: Efficient noise and pattern generation
- **Animation**: Fast physics simulations
- **VR/AR**: Efficient spatial computations

## Performance Characteristics

### Rate-Distortion Performance
- **Hierarchical quantization** achieves better rate-distortion than traditional methods
- **Theoretical bounds**: Approaches optimal performance for inner product estimation
- **Scalability**: Performance improves with higher dimensional lattices

### Computational Complexity
- **Encoding**: O(M × d) where M is hierarchical levels, d is dimension
- **Decoding**: O(M × d) for hierarchical reconstruction
- **Lookup table**: O(q^(2M)) storage, O(1) lookup time
- **Inner product estimation**: O(M²) using precomputed tables

### Memory Efficiency
- **Compact representations**: M encoding vectors per input vector
- **Small lookup tables**: q^(2M) entries for inner product estimation
- **Scalable storage**: Memory grows polynomially with parameters

## Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Clone and setup development environment
git clone https://github.com/mlsquare/LatticeQuant.git
cd LatticeQuant

# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
python -m pytest src/tests/

# Run linting
flake8 src/
```

## Testing

```bash
# Run all tests
python -m pytest src/tests/

# Run specific test file
python -m pytest src/tests/test_closest_point.py

# Run with coverage
python -m pytest --cov=src src/tests/
```

## References

1. **I. Kaplan and O. Ordentlich**, "High-Rate Nested-Lattice Quantized Matrix Multiplication with Small Lookup Tables", to be presented in ISIT 2025, [arXiv:2505.13164, 2025](https://arxiv.org/abs/2505.13164).

2. **O. Ordentlich and Y. Polyanskiy**, "Optimal quantization for matrix multiplication", [arXiv preprint arXiv:2410.13780, 2024](https://arxiv.org/abs/2410.13780).

3. **J. Conway and N. Sloane**, "Fast quantizing and decoding and algorithms for lattice quantizers and codes", in IEEE Transactions on Information Theory, vol. 28, no. 2, pp. 227-232, March 1982, [doi: 10.1109/TIT.1982.1056484](https://doi.org/10.1109/TIT.1982.1056484).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **Efficient C Implementation**: Check out the [NestedLatticeLut](https://github.com/orimeirgit/NestedLatticeLut) repository for a high-performance C implementation.
- **Research Community**: Thanks to the information theory and quantization research communities for foundational work.

## Support

- **Issues**: Report bugs and request features on [GitHub Issues](https://github.com/mlsquare/LatticeQuant/issues)
- **Discussions**: Join community discussions on [GitHub Discussions](https://github.com/mlsquare/LatticeQuant/discussions)
- **Documentation**: Check the [Wiki](https://github.com/mlsquare/LatticeQuant/wiki) for detailed guides

---

**LatticeQuant**: Enabling efficient matrix multiplication through advanced lattice quantization techniques.





