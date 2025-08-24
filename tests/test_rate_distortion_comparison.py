"""
Test for Rate-Distortion Comparison

This test validates the rate-distortion comparison functionality and ensures
that all quantizers work correctly with the D4 lattice.
"""

import numpy as np
import pytest
import sys
import os

from src.exps.rate_distortion_comparison import (
    QuantizerConfig, 
    calculate_mse_and_overload_for_samples,
    calculate_rate_and_distortion,
    run_rate_distortion_comparison,
    generate_theoretical_bounds,
    analyze_performance_slopes
)
from src.lattices.quantizers.hierarchical_nested_lattice_quantizer import HierarchicalNestedLatticeQuantizer as HQuantizer
from src.lattices.quantizers.nested_lattice_quantizer import NestedLatticeQuantizer as NQuantizer
from src.lattices.utils import get_d4, calculate_mse
from src.lattices.utils import closest_point_Dn


def test_quantizer_config():
    """Test QuantizerConfig class."""
    config = QuantizerConfig(
        name="Test Quantizer",
        quantizer_class=HQuantizer,
        nesting_func=lambda q: q**2,
        color="blue",
        marker="o"
    )
    
    assert config.name == "Test Quantizer"
    assert config.quantizer_class == HQuantizer
    assert config.nesting_func(3) == 9
    assert config.color == "blue"
    assert config.marker == "o"


def test_calculate_mse_and_overload():
    """Test MSE and overload calculation for samples."""
    # Setup simple test data
    G = get_d4()
    q = 3
    M = 2
    beta = 0.1
    eps = 1e-8
    
    # Create hierarchical quantizer
    quantizer = HQuantizer(
        G=G, Q_nn=closest_point_Dn, q=q, beta=beta, alpha=1/3,
        eps=eps, M=M, dither=np.zeros(len(G))
    )
    
    # Generate small test samples
    samples = np.random.normal(0, 0.5, size=(10, len(G)))
    
    mse, T_values = calculate_mse_and_overload_for_samples(samples, quantizer)
    
    assert isinstance(mse, float)
    assert mse >= 0
    assert len(T_values) == len(samples)
    assert all(isinstance(t, int) and t >= 0 for t in T_values)


def test_hierarchical_quantizer_rate_distortion():
    """Test rate-distortion calculation for hierarchical quantizer."""
    G = get_d4()
    samples = np.random.normal(0, 0.5, size=(50, len(G)))
    
    config = QuantizerConfig(
        name="Hierarchical Quantizer",
        quantizer_class=HQuantizer,
        nesting_func=lambda q: q**3,
        color="red",
        marker="s"
    )
    
    q = 3
    M = 3
    beta_min = 0.05
    
    R, mse, optimal_beta = calculate_rate_and_distortion(
        config, samples, q, beta_min, M, G
    )
    
    assert isinstance(R, float)
    assert R > 0
    assert isinstance(mse, float)
    assert mse >= 0
    assert isinstance(optimal_beta, float)
    assert optimal_beta >= beta_min


def test_voronoi_quantizer_rate_distortion():
    """Test rate-distortion calculation for Voronoi quantizer."""
    G = get_d4()
    samples = np.random.normal(0, 0.5, size=(50, len(G)))
    
    config = QuantizerConfig(
        name="q² Voronoi Code",
        quantizer_class=NQuantizer,
        nesting_func=lambda q: q**2,
        color="green",
        marker="x"
    )
    
    q = 3
    M = 3
    beta_min = 0.05
    
    R, mse, optimal_beta = calculate_rate_and_distortion(
        config, samples, q, beta_min, M, G
    )
    
    assert isinstance(R, float)
    assert R > 0
    assert isinstance(mse, float)
    assert mse >= 0
    assert isinstance(optimal_beta, float)
    assert optimal_beta >= beta_min


def test_theoretical_bounds():
    """Test theoretical bounds generation."""
    rates = [1.0, 2.0, 3.0, 4.0]
    bounds = generate_theoretical_bounds(rates)
    
    assert "Rate-Distortion Lower Bound" in bounds
    assert "Gaussian Source Bound" in bounds
    
    rd_bound = bounds["Rate-Distortion Lower Bound"]
    gaussian_bound = bounds["Gaussian Source Bound"]
    
    assert len(rd_bound) == len(rates)
    assert len(gaussian_bound) == len(rates)
    
    # Check that bounds are positive and decreasing with rate
    assert all(d > 0 for d in rd_bound)
    assert all(d > 0 for d in gaussian_bound)
    assert rd_bound == sorted(rd_bound, reverse=True)  # Decreasing
    assert gaussian_bound == sorted(gaussian_bound, reverse=True)  # Decreasing


def test_small_rate_distortion_comparison():
    """Test rate-distortion comparison with small parameters."""
    q_values = np.array([3, 4])  # Small test
    n_samples = 100  # Small sample size
    M = 2  # Smaller M for faster testing
    
    print("\nRunning small rate-distortion comparison test...")
    results, configs = run_rate_distortion_comparison(q_values, n_samples, M)
    
    # Check that results have expected structure
    expected_names = ["q(q-1) Voronoi Code", "Hierarchical Quantizer", "q² Voronoi Code"]
    
    for name in expected_names:
        assert name in results
        assert "R" in results[name]
        assert "min_errors" in results[name]
        assert "optimal_betas" in results[name]
    
    # Check that we got some results
    at_least_one_result = False
    for name in expected_names:
        if len(results[name]["R"]) > 0:
            at_least_one_result = True
            # Check that rates and errors are positive
            assert all(r > 0 for r in results[name]["R"])
            assert all(e >= 0 for e in results[name]["min_errors"])
    
    assert at_least_one_result, "No quantizer produced valid results"


def test_performance_analysis():
    """Test performance slope analysis."""
    # Create mock results
    results = {
        "Test Quantizer": {
            "R": [1.0, 2.0, 3.0, 4.0],
            "min_errors": [0.5, 0.25, 0.125, 0.0625],
            "optimal_betas": [0.1, 0.2, 0.3, 0.4]
        }
    }
    
    # This should run without error
    print("\nTesting performance analysis...")
    analyze_performance_slopes(results)


def test_quantizer_consistency():
    """Test that quantizers produce consistent results."""
    G = get_d4()
    samples = np.random.normal(0, 0.5, size=(20, len(G)))
    
    # Test hierarchical quantizer consistency
    hq = HQuantizer(
        G=G, Q_nn=closest_point_Dn, q=3, beta=0.1, alpha=1/3,
        eps=1e-8, M=2, dither=np.zeros(len(G))
    )
    
    # Run quantization twice with same parameters
    mse1, T1 = calculate_mse_and_overload_for_samples(samples, hq)
    mse2, T2 = calculate_mse_and_overload_for_samples(samples, hq)
    
    # Results should be identical (no randomness in quantization)
    assert np.isclose(mse1, mse2, rtol=1e-10)
    assert T1 == T2


if __name__ == "__main__":
    print("Running Rate-Distortion Comparison Tests...")
    
    test_quantizer_config()
    print("✓ QuantizerConfig test passed")
    
    test_calculate_mse_and_overload()
    print("✓ MSE and overload calculation test passed")
    
    test_hierarchical_quantizer_rate_distortion()
    print("✓ Hierarchical quantizer rate-distortion test passed")
    
    test_voronoi_quantizer_rate_distortion()
    print("✓ Voronoi quantizer rate-distortion test passed")
    
    test_theoretical_bounds()
    print("✓ Theoretical bounds test passed")
    
    test_small_rate_distortion_comparison()
    print("✓ Small rate-distortion comparison test passed")
    
    test_performance_analysis()
    print("✓ Performance analysis test passed")
    
    test_quantizer_consistency()
    print("✓ Quantizer consistency test passed")
    
    print("\nAll tests passed! ✅")
