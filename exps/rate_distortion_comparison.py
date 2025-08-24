"""
Rate-Distortion Comparison for Lattice Quantizers

This module provides a comprehensive comparison of rate-distortion performance
between different quantization methods:
1. Hierarchical Nested Lattice Quantizer 
2. Theoretical bounds
3. Voronoi quantizers with q² and q(q-1) rates

The comparison focuses on D4 lattice quantization with M=3 hierarchical levels.
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Tuple, Callable

from src.lattices.utils import closest_point_Dn
from src.lattices.quantizers.hierarchical_nested_lattice_quantizer import HierarchicalNestedLatticeQuantizer as HQuantizer
from src.lattices.quantizers.nested_lattice_quantizer import NestedLatticeQuantizer as NQuantizer
from src.lattices.utils import calculate_mse, calculate_t_entropy, get_d4, SIG_D4


class QuantizerConfig:
    """Configuration class for different quantization schemes."""
    
    def __init__(self, name: str, quantizer_class, nesting_func: Callable, color: str, marker: str):
        self.name = name
        self.quantizer_class = quantizer_class
        self.nesting_func = nesting_func
        self.color = color
        self.marker = marker


def calculate_mse_and_overload_for_samples(samples: np.ndarray, quantizer) -> Tuple[float, List[int]]:
    """
    Calculate the MSE and overload values for quantized samples.

    Parameters:
    -----------
    samples : numpy.ndarray
        Array of sample vectors to be quantized.
    quantizer : object
        Quantizer object (NestedLatticeQuantizer or HierarchicalNestedLatticeQuantizer).

    Returns:
    --------
    tuple
        (mse, T_count) where mse is the mean squared error and
        T_count is a list of overload counts (T values).
    """
    mse = 0
    T_count = []
    for x in samples:
        try:
            encoding, t = quantizer.encode(x, with_dither=False)
            x_hat = quantizer.decode(encoding, t, with_dither=False)
            mse += calculate_mse(x, x_hat)
            T_count.append(t)
        except Exception as e:
            print(f"Warning: Quantization failed for sample, skipping. Error: {e}")
            T_count.append(0)  # Default T value for failed quantization
    
    return mse / len(samples), T_count


def calculate_rate_and_distortion(config: QuantizerConfig, samples: np.ndarray, 
                                q: int, beta_min: float, M: int = 3, G: np.ndarray = None) -> Tuple[float, float, float]:
    """
    Calculate rate and distortion for a given quantizer configuration.

    This function performs a grid search over beta values to find the
    optimal rate-distortion performance for a given quantizer.

    Parameters:
    -----------
    config : QuantizerConfig
        Configuration object containing quantizer details.
    samples : numpy.ndarray
        Sample vectors for evaluation.
    q : int
        Quantization parameter.
    beta_min : float
        Minimum beta value for the grid search.
    M : int
        Number of hierarchical levels (for hierarchical quantizer).
    G : numpy.ndarray
        Generator matrix for the lattice.

    Returns:
    --------
    tuple
        (optimal_R, optimal_mse, optimal_beta) containing the optimal
        rate, mean squared error, and beta parameter.
    """
    if G is None:
        G = get_d4()
    
    # Create grid of beta values for optimization
    betas = beta_min + 0.05 * beta_min * np.arange(0, 40)
    d = len(G)
    eps = 1e-8

    min_f_beta = float("inf")
    optimal_beta = beta_min
    optimal_mse = None
    optimal_H_T = None
    optimal_R = None

    print(f"  Optimizing {config.name} for q={q}...")

    for beta_idx, beta in enumerate(betas):
        try:
            # Create quantizer based on configuration
            if "Hierarchical" in config.name:
                # For hierarchical quantizer, use q and M directly
                quantizer = config.quantizer_class(
                    G=G, Q_nn=closest_point_Dn, q=q, beta=beta, alpha=1/3, 
                    eps=eps, M=M, dither=np.zeros(d)
                )
            else:
                # For Voronoi quantizers, use nesting function
                effective_q = config.nesting_func(q)
                quantizer = config.quantizer_class(
                    G=G, Q_nn=closest_point_Dn, q=effective_q, beta=beta, alpha=1/3,
                    eps=eps, dither=np.zeros(d)
                )
            
            # Calculate MSE and overload statistics
            mse, T_values = calculate_mse_and_overload_for_samples(samples, quantizer)
            
            if mse == 0:  # Skip if MSE is zero (potential issue)
                continue
                
            H_T, T_counts = calculate_t_entropy(T_values, q)
            
            # Calculate rate based on quantizer type
            if "Hierarchical" in config.name:
                R = M * np.log2(q) + (H_T / d)
            else:
                # For Voronoi quantizers
                if "q²" in config.name or "q^2" in config.name:
                    R = 2 * np.log2(q) + (H_T / d)
                else:  # q(q-1) case
                    R = np.log2(q * (q - 1)) + (H_T / d)

            # Optimization objective: minimize MSE / 2^(-2*R)
            f_beta = mse / (2 ** (-2 * R))

            if f_beta < min_f_beta:
                min_f_beta = f_beta
                optimal_beta = beta
                optimal_mse = mse
                optimal_H_T = H_T
                optimal_R = R
                
        except Exception as e:
            print(f"    Warning: Beta optimization failed at beta={beta:.4f}, skipping. Error: {e}")
            continue

    if optimal_mse is None:
        print(f"    Error: No valid optimization found for {config.name} at q={q}")
        return 0, float('inf'), beta_min

    overload_percentage = 0  # Will calculate if needed
    print(f"    {config.name}: Optimal beta={optimal_beta:.3f}, MSE={optimal_mse:.6f}, R={optimal_R:.3f}")
    
    return optimal_R, optimal_mse, optimal_beta


def generate_theoretical_bounds(rates: List[float]) -> Dict[str, List[float]]:
    """
    Generate theoretical rate-distortion bounds.
    
    Parameters:
    -----------
    rates : List[float]
        Rate values for which to calculate theoretical bounds.
        
    Returns:
    --------
    Dict[str, List[float]]
        Dictionary containing theoretical distortion bounds.
    """
    theoretical_bounds = {
        "Rate-Distortion Lower Bound": [2 ** (-2 * R) for R in rates],
        "Gaussian Source Bound": [2 ** (-2 * R) * np.pi * np.e / 6 for R in rates]
    }
    return theoretical_bounds


def run_rate_distortion_comparison(q_values: np.ndarray, n_samples: int = 5000, 
                                 M: int = 3, sigma_squared: float = 1.0) -> Dict:
    """
    Run a comprehensive rate-distortion comparison between different quantization schemes.

    Parameters:
    -----------
    q_values : numpy.ndarray
        Array of quantization parameters to test.
    n_samples : int
        Number of sample vectors for evaluation.
    M : int
        Number of hierarchical levels.
    sigma_squared : float
        Variance of the input distribution.

    Returns:
    --------
    Dict
        Dictionary containing rate-distortion results for each scheme.
    """
    print("=== Rate-Distortion Comparison for Lattice Quantizers ===\n")
    
    # Setup lattice and parameters
    G = get_d4()
    d = len(G)
    sig_l = SIG_D4  # D4 lattice second moment
    
    # Generate random samples
    print(f"Generating {n_samples} random samples (dimension {d})...")
    x_std = np.sqrt(sigma_squared)
    samples = np.random.normal(0, x_std, size=(n_samples, d))
    
    # Define quantization schemes
    quantizer_configs = [
        QuantizerConfig(
            name="q(q-1) Voronoi Code",
            quantizer_class=NQuantizer,
            nesting_func=lambda q: int(q * (q - 1)),
            color="blue",
            marker="o"
        ),
        QuantizerConfig(
            name="Hierarchical Quantizer",
            quantizer_class=HQuantizer,
            nesting_func=lambda q: int(q**M),  # Not used for hierarchical
            color="red",
            marker="s"
        ),
        QuantizerConfig(
            name="q² Voronoi Code",
            quantizer_class=NQuantizer,
            nesting_func=lambda q: int(q**2),
            color="green",
            marker="x"
        )
    ]
    
    # Initialize results storage
    results = {config.name: {"R": [], "min_errors": [], "optimal_betas": []} 
               for config in quantizer_configs}
    
    print(f"\nTesting quantization parameters: {q_values}")
    print(f"Using M={M} hierarchical levels\n")
    
    # Run experiments for each q value
    for q_idx, q in enumerate(q_values):
        print(f"Processing q={q} ({q_idx + 1}/{len(q_values)})...")
        
        for config in quantizer_configs:
            # Calculate beta_min based on the quantization scheme
            if "Hierarchical" in config.name:
                # For hierarchical quantizer, use q^M scaling
                effective_q = q**M
            else:
                # For other quantizers, use the nesting function
                effective_q = config.nesting_func(q)
            
            beta_min = (1 / effective_q) * np.sqrt(1 / sig_l) * np.sqrt(d / (d + 2))
            
            R, min_error, optimal_beta = calculate_rate_and_distortion(
                config, samples, q, beta_min, M, G
            )
            
            if min_error != float('inf'):
                results[config.name]["R"].append(R)
                results[config.name]["min_errors"].append(min_error)
                results[config.name]["optimal_betas"].append(optimal_beta)
        
        print()  # Add spacing between q values
    
    return results, quantizer_configs


def plot_rate_distortion_results(results: Dict, quantizer_configs: List[QuantizerConfig], 
                                save_path: str = None, show_theoretical: bool = True):
    """
    Plot the rate-distortion comparison results.
    
    Parameters:
    -----------
    results : Dict
        Results dictionary from run_rate_distortion_comparison.
    quantizer_configs : List[QuantizerConfig]
        List of quantizer configurations.
    save_path : str, optional
        Path to save the plot. If None, plot is displayed.
    show_theoretical : bool
        Whether to show theoretical bounds.
    """
    plt.figure(figsize=(12, 8))
    
    # Plot experimental results
    for config in quantizer_configs:
        name = config.name
        if name in results and len(results[name]["R"]) > 0:
            R = results[name]["R"]
            min_errors = results[name]["min_errors"]
            plt.plot(R, min_errors, label=name, marker=config.marker, 
                    color=config.color, linewidth=2, markersize=8)
    
    # Add theoretical bounds
    if show_theoretical:
        # Get rate range for theoretical bounds
        all_rates = []
        for config_name, data in results.items():
            all_rates.extend(data["R"])
        
        if all_rates:
            rate_range = np.linspace(min(all_rates), max(all_rates), 100)
            theoretical_bounds = generate_theoretical_bounds(rate_range)
            
            plt.plot(rate_range, theoretical_bounds["Rate-Distortion Lower Bound"],
                    label="Rate-Distortion Lower Bound", color="black", 
                    linestyle="--", linewidth=2)
            
            plt.plot(rate_range, theoretical_bounds["Gaussian Source Bound"],
                    label="Gaussian Source Bound", color="gray", 
                    linestyle=":", linewidth=2)
    
    # Formatting
    plt.xlabel(r"Rate $R$ (bits per dimension)", fontsize=12)
    plt.ylabel(r"Distortion $D$ (log scale)", fontsize=12)
    plt.title("Rate-Distortion Comparison: D₄ Lattice Quantizers", fontsize=14)
    plt.yscale("log")
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        plt.show()


def analyze_performance_slopes(results: Dict):
    """
    Analyze the slope of rate-distortion curves for performance comparison.
    
    Parameters:
    -----------
    results : Dict
        Results dictionary from run_rate_distortion_comparison.
    """
    print("=== Rate-Distortion Slope Analysis ===\n")
    
    for name, data in results.items():
        if len(data["R"]) >= 2:
            R = np.array(data["R"])
            errors = np.array(data["min_errors"])
            
            # Calculate slope in log-log space
            log_R = np.log2(R)
            log_errors = np.log2(errors)
            
            # Fit linear regression
            slope, intercept = np.polyfit(log_R, log_errors, 1)
            
            print(f"{name}:")
            print(f"  Slope: {slope:.3f}")
            print(f"  R-squared: {np.corrcoef(log_R, log_errors)[0,1]**2:.3f}")
            print(f"  Rate range: {min(R):.3f} - {max(R):.3f}")
            print(f"  Distortion range: {min(errors):.6f} - {max(errors):.6f}")
            print()


def main():
    """
    Main function to run the rate-distortion comparison.
    """
    # Experiment parameters
    q_values = np.arange(3, 8)  # Test q from 3 to 7
    n_samples = 5000
    M = 3  # Number of hierarchical levels
    sigma_squared = 1.0
    
    print("Starting Rate-Distortion Comparison Experiment")
    print(f"Parameters: q_values={q_values}, n_samples={n_samples}, M={M}")
    print("-" * 60)
    
    # Run the comparison
    results, quantizer_configs = run_rate_distortion_comparison(
        q_values, n_samples, M, sigma_squared
    )
    
    # Analyze and plot results
    analyze_performance_slopes(results)
    plot_rate_distortion_results(results, quantizer_configs, 
                                save_path="rate_distortion_comparison.png")
    
    print("Rate-distortion comparison complete!")
    return results


if __name__ == "__main__":
    results = main()
