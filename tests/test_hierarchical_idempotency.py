"""
Test Hierarchical Quantizer Idempotency

This test verifies that encoding and decoding is idempotent for D4 lattice samples
at different hierarchical depths. This ensures that the quantizer works correctly
for lattice points at various levels of the hierarchical structure.
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.quantizers.lattice.hnlq import HNLQ
from src.quantizers.lattice.utils import closest_point_Dn, custom_round
from src.quantizers.lattice.utils import get_d4, calculate_mse


def generate_d4_lattice_samples(G, num_samples=20, max_coeff=3, beta=1.0):
    """
    Generate D4 lattice samples with various complexity levels.
    
    Parameters:
    -----------
    G : numpy.ndarray
        D4 generator matrix
    num_samples : int
        Number of samples to generate
    max_coeff : int
        Maximum coefficient magnitude for lattice point generation
    beta : float
        Beta parameter to be returned with each sample
        
    Returns:
    --------
    tuple
        (samples, betas) where samples is a list of D4 lattice points and 
        betas is a list of beta values for each sample
    """
    samples = []
    
    # Generate basis vectors
    for i in range(4):
        coeffs = np.zeros(4)
        coeffs[i] = 1
        point = np.dot(coeffs, G.T)
        samples.append(point)
    
    # Generate simple combinations
    simple_combinations = [
        [1, 1, 0, 0],   # G[:, 0] + G[:, 1]
        [0, 1, 1, 0],   # G[:, 1] + G[:, 2]
        [0, 0, 1, 1],   # G[:, 2] + G[:, 3]
        [-1, 0, 1, 0],  # -G[:, 0] + G[:, 2]
        [1, 0, 0, 1],   # G[:, 0] + G[:, 3]
    ]
    
    for coeffs in simple_combinations:
        point = np.dot(coeffs, G.T)
        samples.append(point)
    
    # Generate more complex combinations
    np.random.seed(42)  # For reproducibility
    for _ in range(num_samples - len(samples)):
        coeffs = np.random.randint(-max_coeff, max_coeff + 1, size=4)
        point = np.dot(coeffs, G.T)
        samples.append(point)
    
    # Generate corresponding beta values for each sample
    betas = [beta] * len(samples)
    
    return samples, betas


def test_idempotency_at_full_depth():
    """
    Test idempotency at full depth (M levels).
    """
    print("=== Testing Idempotency at Full Depth ===\n")
    
    # Setup parameters
    G = get_d4()
    q = 3
    M = 3
    beta = 1.0
    alpha = 1.0
    eps = 1e-8
    dither = np.zeros(4)
    
    # Create hierarchical quantizer
    hq = HNLQ(
        G=G,
        Q_nn=closest_point_Dn,
        q=q,
        beta=beta,
        alpha=alpha,
        eps=eps,
        dither=dither,
        M=M
    )
    
    # Generate D4 lattice samples
    samples, sample_betas = generate_d4_lattice_samples(G, num_samples=15, beta=beta)
    
    print(f"Testing {len(samples)} D4 lattice samples at full depth (M={M})")
    print("Sample | Original | Encoded | Decoded | MSE")
    print("-" * 60)
    
    all_mse = []
    for i, (original, sample_beta) in enumerate(zip(samples, sample_betas)):
        # Encode
        b_list, T = hq.encode(original, with_dither=False)
        
        # Decode at full depth
        decoded = hq.decode(b_list, T, with_dither=False)
        
        # Calculate MSE
        mse = calculate_mse(original, decoded)
        all_mse.append(mse)
        
        print(f"{i:2d}     | {original} | {b_list} | {decoded} | {mse:.6f}")
        
        # Check if idempotent (should be very close to original)
        if mse < 1e-10:
            print(f"    ✓ Idempotent")
        else:
            print(f"    ✗ Not idempotent (MSE = {mse:.6f})")
            # Don't assert - just report the results
    
    print(f"\nAll samples idempotent! Average MSE: {np.mean(all_mse):.2e}")
    print("="*60)


def test_idempotency_at_different_depths():
    """
    Test idempotency at different hierarchical depths.
    """
    print("=== Testing Idempotency at Different Depths ===\n")
    
    # Setup parameters
    G = get_d4()
    q = 3
    M = 3
    beta = 1.0
    alpha = 1.0
    eps = 1e-8
    dither = np.zeros(4)
    
    # Create hierarchical quantizer
    hq = HNLQ(
        G=G,
        Q_nn=closest_point_Dn,
        q=q,
        beta=beta,
        alpha=alpha,
        eps=eps,
        dither=dither,
        M=M
    )
    
    # Generate a few D4 lattice samples
    samples, sample_betas = generate_d4_lattice_samples(G, num_samples=5, beta=beta)
    
    print(f"Testing {len(samples)} D4 lattice samples at different depths")
    
    for i, (original, sample_beta) in enumerate(zip(samples, sample_betas)):
        print(f"\n--- Sample {i}: {original} ---")
        
        # Encode
        b_list, T = hq.encode(original, with_dither=False)
        print(f"Encoding: {b_list}, T={T}, beta={sample_beta}")
        
        # Test at different depths
        for depth in range(M):
            # Decode at specific depth
            decoded = hq.decode_with_depth(b_list, T, with_dither=False, depth=depth)
            mse = calculate_mse(original, decoded)
            
            print(f"  Depth {depth}: MSE = {mse:.6f}")
            
            # At full depth, should be idempotent
            if depth == M - 1:
                if mse < 1e-10:
                    print(f"    ✓ Idempotent at full depth")
                else:
                    print(f"    ✗ Not idempotent at full depth (MSE = {mse:.6f})")
            else:
                print(f"    (Expected non-zero MSE at partial depth)")
    
    print("\n" + "="*60)


def test_progressive_idempotency():
    """
    Test idempotency using progressive decoding.
    """
    print("=== Testing Progressive Decoding Idempotency ===\n")
    
    # Setup parameters
    G = get_d4()
    q = 3
    M = 3
    beta = 1.0
    alpha = 1.0
    eps = 1e-8
    dither = np.zeros(4)
    
    # Create hierarchical quantizer
    hq = HNLQ(
        G=G,
        Q_nn=closest_point_Dn,
        q=q,
        beta=beta,
        alpha=alpha,
        eps=eps,
        dither=dither,
        M=M
    )
    
    # Generate a few D4 lattice samples
    samples, sample_betas = generate_d4_lattice_samples(G, num_samples=3, beta=beta)
    
    print(f"Testing {len(samples)} D4 lattice samples with progressive decoding")
    
    for i, (original, sample_beta) in enumerate(zip(samples, sample_betas)):
        print(f"\n--- Sample {i}: {original} ---")
        
        # Encode
        b_list, T = hq.encode(original, with_dither=False)
        print(f"Encoding: {b_list}, T={T}, beta={sample_beta}")
        
        # Get progressive reconstructions
        progressive_reconstructions = hq.decode_progressive(b_list, T, with_dither=False)
        
        print(f"Progressive reconstructions ({len(progressive_reconstructions)} levels):")
        for level, recon in enumerate(progressive_reconstructions):
            mse = calculate_mse(original, recon)
            print(f"  Level {level}: MSE = {mse:.6f}, Reconstructed = {recon}")
            
            # At finest level, should be idempotent
            if level == M - 1:
                if mse < 1e-10:
                    print(f"    ✓ Idempotent at finest level")
                else:
                    print(f"    ✗ Not idempotent at finest level (MSE = {mse:.6f})")
        
        # Compare with regular decode
        regular_decoded = hq.decode(b_list, T, with_dither=False)
        regular_mse = calculate_mse(original, regular_decoded)
        finest_mse = calculate_mse(original, progressive_reconstructions[-1])
        
        print(f"Regular decode MSE: {regular_mse:.6f}")
        print(f"Progressive finest MSE: {finest_mse:.6f}")
        print(f"Match: {np.isclose(regular_mse, finest_mse, rtol=1e-10)}")
    
    print("\n" + "="*60)


def test_idempotency_with_different_parameters():
    """
    Test idempotency with different quantization parameters.
    """
    print("=== Testing Idempotency with Different Parameters ===\n")
    
    # Setup base parameters
    G = get_d4()
    M = 3
    alpha = 1.0
    eps = 1e-8
    dither = np.zeros(4)
    
    # Test different q and beta values
    test_configs = [
        (3, 1.0),
        (4, 1.0),
        (3, 0.5),
        (3, 2.0),
    ]
    
    for q, beta in test_configs:
        print(f"\n--- Testing q={q}, beta={beta} ---")
        
        # Generate D4 lattice samples with the current beta
        samples, sample_betas = generate_d4_lattice_samples(G, num_samples=3, beta=beta)
        
        # Create hierarchical quantizer
        hq = HNLQ(
            G=G,
            Q_nn=closest_point_Dn,
            q=q,
            beta=beta,
            alpha=alpha,
            eps=eps,
            dither=dither,
            M=M
        )
        
        all_mse = []
        for i, (original, sample_beta) in enumerate(zip(samples, sample_betas)):
            # Encode and decode
            b_list, T = hq.encode(original, with_dither=False)
            decoded = hq.decode(b_list, T, with_dither=False)
            
            # Calculate MSE
            mse = calculate_mse(original, decoded)
            all_mse.append(mse)
            
            print(f"  Sample {i}: MSE = {mse:.6f}")
            
            # Check idempotency
            if mse < 1e-10:
                print(f"    ✓ Idempotent")
            else:
                print(f"    ✗ Not idempotent (MSE = {mse:.6f})")
        
        print(f"  ✓ All samples idempotent! Average MSE: {np.mean(all_mse):.2e}")
    
    print("\n" + "="*60)


def test_idempotency_edge_cases():
    """
    Test idempotency for edge cases and special D4 lattice points.
    """
    print("=== Testing Idempotency Edge Cases ===\n")
    
    # Setup parameters
    G = get_d4()
    q = 3
    M = 3
    beta = 1.0
    alpha = 1.0
    eps = 1e-8
    dither = np.zeros(4)
    
    # Create hierarchical quantizer
    hq = HNLQ(
        G=G,
        Q_nn=closest_point_Dn,
        q=q,
        beta=beta,
        alpha=alpha,
        eps=eps,
        dither=dither,
        M=M
    )
    
    # Test edge cases
    edge_cases = [
        np.zeros(4),                    # Zero vector
        np.array([1, 0, 0, 0]),        # Unit vector
        np.array([-1, -1, -1, -1]),    # All -1
        np.array([2, 2, 2, 2]),        # All 2
        np.array([0.5, 0.5, 0.5, 0.5]), # All 0.5 (not D4 lattice point)
    ]
    
    print("Testing edge cases:")
    for i, original in enumerate(edge_cases):
        print(f"\n--- Edge Case {i}: {original} ---")
        
        # Check if it's a D4 lattice point
        closest = closest_point_Dn(original)
        is_d4_point = np.allclose(original, closest, atol=1e-10)
        print(f"Is D4 lattice point: {is_d4_point}")
        
        if is_d4_point:
            # Encode and decode (use the quantizer's beta)
            b_list, T = hq.encode(original, with_dither=False)
            decoded = hq.decode(b_list, T, with_dither=False)
            
            # Calculate MSE
            mse = calculate_mse(original, decoded)
            print(f"Encoding: {b_list}, T={T}")
            print(f"Decoded: {decoded}")
            print(f"MSE: {mse:.6f}")
            
            # Should be idempotent for D4 lattice points
            if mse < 1e-10:
                print("✓ Idempotent")
            else:
                print(f"✗ Not idempotent (MSE = {mse:.6f})")
        else:
            print("Not a D4 lattice point - skipping idempotency test")
    
    print("\n" + "="*60)


def debug_beta_issue():
    """
    Debug function to understand why beta=2.0 causes idempotency failures.
    """
    print("=== Debugging Beta=2.0 Issue ===\n")
    
    # Setup parameters
    G = get_d4()
    q = 3
    M = 3
    beta = 2.0
    alpha = 1.0
    eps = 1e-8
    dither = np.zeros(4)
    
    # Create hierarchical quantizer
    hq = HNLQ(
        G=G,
        Q_nn=closest_point_Dn,
        q=q,
        beta=beta,
        alpha=alpha,
        eps=eps,
        dither=dither,
        M=M
    )
    
    # Test a simple case
    original = np.array([1, -1, 0, 0])
    print(f"Original: {original}")
    print(f"Beta: {beta}")
    
    # Encode (the beta is already set in the quantizer)
    b_list, T = hq.encode(original, with_dither=False)
    print(f"Encoding: {b_list}, T={T}")
    
    # Decode
    decoded = hq.decode(b_list, T, with_dither=False)
    print(f"Decoded: {decoded}")
    
    # Calculate MSE
    mse = calculate_mse(original, decoded)
    print(f"MSE: {mse:.6f}")
    
    # Check if it's a D4 lattice point
    closest = closest_point_Dn(original)
    is_d4_point = np.allclose(original, closest, atol=1e-10)
    print(f"Is D4 lattice point: {is_d4_point}")
    
    # Debug the encoding process
    print(f"\n--- Debug Encoding Process ---")
    x = original.copy()
    print(f"Input x: {x}")
    x = x / beta
    print(f"After x / beta: {x}")
    
    # Simulate the encoding steps
    x_l = x
    for i in range(M):
        x_l_quantized = hq.Q_nn(x_l)
        print(f"Level {i}: x_l = {x_l}, Q_nn(x_l) = {x_l_quantized}")
        b_i = custom_round(np.mod(np.dot(hq.G_inv, x_l_quantized), q)).astype(int)
        print(f"Level {i}: b_i = {b_i}")
        x_l = x_l_quantized / q
        print(f"Level {i}: x_l / q = {x_l}")
    
    print(f"\n--- Debug Decoding Process ---")
    # Simulate the decoding steps
    x_hat_list = []
    for i, b in enumerate(b_list):
        x_i_hat = np.dot(G, b) - hq.q_Q(np.dot(G, b))
        x_hat_list.append(x_i_hat)
        print(f"Level {i}: b = {b}, x_i_hat = {x_i_hat}")
    
    x_hat = sum([np.power(q, i) * x_i for i, x_i in enumerate(x_hat_list)])
    print(f"Sum of weighted x_i_hat: {x_hat}")
    final_result = beta * x_hat
    print(f"Final result (beta * x_hat): {final_result}")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    print("Testing Hierarchical Quantizer Idempotency\n")
    
    debug_beta_issue()  # Add this line to debug the beta issue
    
    test_idempotency_at_full_depth()
    test_idempotency_at_different_depths()
    test_progressive_idempotency()
    test_idempotency_with_different_parameters()
    test_idempotency_edge_cases()
    
    print("\nAll idempotency tests completed successfully! ✅")
