"""
Test Hierarchical Encoding Analysis

This test analyzes the hierarchical encoding process to understand why
the rate-distortion performance is poor even after fixing the decoding.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from gemvq.quantizers.hnlq import HNLQ, HNLQConfig
from gemvq.quantizers.utils import (
    calculate_mse,
    calculate_t_entropy,
    closest_point_Dn,
    get_d4,
)


def test_hierarchical_encoding_process():
    """
    Test the hierarchical encoding process step by step.
    """
    print("=== Testing Hierarchical Encoding Process ===\n")

    # Setup parameters
    G = get_d4()
    q = 3
    M = 3
    beta = 1.0
    alpha = 1.0
    eps = 1e-8
    dither = np.zeros(4)

    # Create hierarchical quantizer
    config = HNLQConfig(lattice_type="D4", q=q, M=M)
    hq = HNLQ(config)

    # Test with a simple vector (not necessarily a D4 lattice point)
    test_point = np.array([0.5, -0.3, 0.8, -0.2])

    print(f"Testing with vector: {test_point}")

    # Encode
    b_list, T = hq.encode(test_point, with_dither=False)
    print(f"Encoding vectors: {b_list}")
    print(f"Scaling factor T: {T}")

    # Decode
    reconstructed = hq.decode(b_list, T, with_dither=False)
    mse = calculate_mse(test_point, reconstructed)
    print(f"Reconstruction MSE: {mse:.6f}")
    print(f"Reconstructed: {reconstructed}")

    # Analyze encoding process
    print(f"\n--- Encoding Analysis ---")

    # Check if overload occurred
    if T > 0:
        print(f"Overload occurred: T = {T}")
        print(f"Original vector was scaled by 2^(-{alpha * T}) = {2**(-alpha * T):.6f}")
    else:
        print("No overload occurred")

    # Analyze encoding vectors
    for i, b in enumerate(b_list):
        print(f"Level {i}: encoding vector = {b}")
        print(f"  Contribution: q^{i} * G * {b} = {q**i} * G * {b}")
        contribution = (q**i) * np.dot(G, b)
        print(f"  Contribution value: {contribution}")

    print("\n" + "=" * 50)


def test_hierarchical_vs_voronoi_encoding():
    """
    Compare hierarchical encoding with Voronoi encoding for the same input.
    """
    print("=== Comparing Hierarchical vs Voronoi Encoding ===\n")

    from gemvq.quantizers.nlq import NLQ

    # Setup parameters
    G = get_d4()
    q = 3
    M = 3
    beta = 1.0
    alpha = 1.0
    eps = 1e-8
    dither = np.zeros(4)

    # Create quantizers
    config = HNLQConfig(lattice_type="D4", q=q, M=M)
    hq = HNLQ(config)

    # Create Voronoi quantizer with equivalent parameters
    effective_q = q**M  # 3^3 = 27
    nlq_config = {"lattice_type": "D4", "q": effective_q}
    vq = NLQ(nlq_config)

    # Test with random vectors
    np.random.seed(42)  # For reproducibility
    test_vectors = np.random.normal(0, 1, size=(10, 4))

    print("Comparing encoding for 10 random vectors:")
    print("Vector | Hierarchical MSE | Voronoi MSE | Ratio")
    print("-" * 50)

    for i, vector in enumerate(test_vectors):
        # Hierarchical encoding
        h_b_list, h_T = hq.encode(vector, with_dither=False)
        h_reconstructed = hq.decode(h_b_list, h_T, with_dither=False)
        h_mse = calculate_mse(vector, h_reconstructed)

        # Voronoi encoding
        v_encoded, v_T = vq.encode(vector, with_dither=False)
        v_reconstructed = vq.decode(v_encoded, v_T, with_dither=False)
        v_mse = calculate_mse(vector, v_reconstructed)

        ratio = h_mse / v_mse if v_mse > 0 else float("inf")

        print(f"{i:2d}     | {h_mse:12.6f} | {v_mse:10.6f} | {ratio:5.2f}")

    print("\n" + "=" * 50)


def test_rate_calculation():
    """
    Test rate calculation for hierarchical quantizer.
    """
    print("=== Testing Rate Calculation ===\n")

    # Setup parameters
    G = get_d4()
    q = 3
    M = 3
    beta = 1.0
    alpha = 1.0
    eps = 1e-8
    dither = np.zeros(4)

    # Create hierarchical quantizer
    config = HNLQConfig(lattice_type="D4", q=q, M=M)
    hq = HNLQ(config)

    # Generate test samples
    np.random.seed(42)
    samples = np.random.normal(0, 1, size=(100, 4))

    print("Calculating rates for 100 random samples:")

    T_values = []
    for i, sample in enumerate(samples):
        b_list, T = hq.encode(sample, with_dither=False)
        T_values.append(T)

        if i < 5:  # Show details for first 5 samples
            print(f"Sample {i}: T = {T}")

    # Calculate entropy
    H_T, T_counts = calculate_t_entropy(T_values, q)
    d = len(G)

    # Calculate rate
    R = M * np.log2(q) + (H_T / d)

    print(f"\nRate Analysis:")
    print(f"  M * log2(q) = {M} * {np.log2(q):.3f} = {M * np.log2(q):.3f}")
    print(f"  H(T)/d = {H_T:.3f} / {d} = {H_T/d:.3f}")
    print(f"  Total rate R = {R:.3f} bits/dim")
    print(f"  T distribution: {T_counts}")

    print("\n" + "=" * 50)


def test_parameter_sensitivity():
    """
    Test sensitivity to different parameter values.
    """
    print("=== Testing Parameter Sensitivity ===\n")

    # Setup base parameters
    G = get_d4()
    q = 3
    M = 3
    alpha = 1.0
    eps = 1e-8
    dither = np.zeros(4)

    # Test different beta values
    beta_values = [0.1, 0.5, 1.0, 2.0, 5.0]
    test_vector = np.array([0.5, -0.3, 0.8, -0.2])

    print("Testing different beta values:")
    print("Beta  | MSE     | T")
    print("-" * 20)

    for beta in beta_values:
        config = HNLQConfig(lattice_type="D4", q=q, M=M)
        hq = HNLQ(config)

        b_list, T = hq.encode(test_vector, with_dither=False)
        reconstructed = hq.decode(b_list, T, with_dither=False)
        mse = calculate_mse(test_vector, reconstructed)

        print(f"{beta:5.1f} | {mse:7.6f} | {T}")

    print("\n" + "=" * 50)


def test_m_values():
    """
    Test different M values to see if M=3 is the issue.
    """
    print("=== Testing Different M Values ===\n")

    # Setup parameters
    G = get_d4()
    q = 3
    beta = 1.0
    alpha = 1.0
    eps = 1e-8
    dither = np.zeros(4)

    # Test different M values
    M_values = [1, 2, 3, 4]
    test_vector = np.array([0.5, -0.3, 0.8, -0.2])

    print("Testing different M values:")
    print("M | MSE     | T | Rate")
    print("-" * 25)

    for M in M_values:
        config = HNLQConfig(lattice_type="D4", q=q, M=M)
        hq = HNLQ(config)

        b_list, T = hq.encode(test_vector, with_dither=False)
        reconstructed = hq.decode(b_list, T, with_dither=False)
        mse = calculate_mse(test_vector, reconstructed)

        # Calculate rate
        rate = M * np.log2(q)

        print(f"{M} | {mse:7.6f} | {T} | {rate:.2f}")

    print("\n" + "=" * 50)


if __name__ == "__main__":
    print("Testing Hierarchical Encoding Analysis\n")

    test_hierarchical_encoding_process()
    test_hierarchical_vs_voronoi_encoding()
    test_rate_calculation()
    test_parameter_sensitivity()
    test_m_values()

    print("\nAll tests completed!")
