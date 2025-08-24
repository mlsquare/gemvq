"""
Rate-Distortion Results Analysis

This module provides detailed analysis of the rate-distortion comparison results,
including performance metrics, theoretical comparisons, and insights.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple

from .rate_distortion_comparison import (
    run_rate_distortion_comparison,
    plot_rate_distortion_results,
    analyze_performance_slopes,
    generate_theoretical_bounds
)


def calculate_gap_to_theoretical(results: Dict) -> Dict[str, List[float]]:
    """
    Calculate the gap between experimental results and theoretical bounds.
    
    Parameters:
    -----------
    results : Dict
        Results from rate-distortion comparison.
        
    Returns:
    --------
    Dict[str, List[float]]
        Gap analysis for each quantizer.
    """
    gaps = {}
    
    for name, data in results.items():
        if len(data["R"]) == 0:
            continue
            
        rates = np.array(data["R"])
        errors = np.array(data["min_errors"])
        
        # Calculate theoretical lower bound at same rates
        theoretical_bounds = generate_theoretical_bounds(rates)
        rd_lower_bound = np.array(theoretical_bounds["Rate-Distortion Lower Bound"])
        
        # Calculate gap in dB
        gap_linear = errors / rd_lower_bound
        gap_db = 10 * np.log10(gap_linear)
        
        gaps[name] = {
            "gap_linear": gap_linear.tolist(),
            "gap_db": gap_db.tolist(),
            "mean_gap_db": np.mean(gap_db),
            "std_gap_db": np.std(gap_db)
        }
    
    return gaps


def calculate_rate_efficiency(results: Dict) -> Dict[str, float]:
    """
    Calculate rate efficiency for each quantizer.
    
    Parameters:
    -----------
    results : Dict
        Results from rate-distortion comparison.
        
    Returns:
    --------
    Dict[str, float]
        Rate efficiency metrics.
    """
    efficiency = {}
    
    # Find minimum rate across all quantizers for each distortion level
    all_errors = set()
    for data in results.values():
        all_errors.update(data["min_errors"])
    
    all_errors = sorted(list(all_errors))
    
    for name, data in results.items():
        if len(data["R"]) == 0:
            continue
            
        rates = np.array(data["R"])
        errors = np.array(data["min_errors"])
        
        # Calculate average rate for comparable distortion levels
        avg_rate = np.mean(rates)
        efficiency[name] = {
            "average_rate": avg_rate,
            "rate_range": (np.min(rates), np.max(rates)),
            "distortion_range": (np.min(errors), np.max(errors))
        }
    
    return efficiency


def analyze_convergence_behavior(results: Dict) -> Dict:
    """
    Analyze the convergence behavior of the rate-distortion curves.
    
    Parameters:
    -----------
    results : Dict
        Results from rate-distortion comparison.
        
    Returns:
    --------
    Dict
        Convergence analysis results.
    """
    convergence = {}
    
    for name, data in results.items():
        if len(data["R"]) < 3:
            continue
            
        rates = np.array(data["R"])
        errors = np.array(data["min_errors"])
        
        # Calculate rate differences and error ratios
        rate_diffs = np.diff(rates)
        error_ratios = errors[1:] / errors[:-1]
        
        convergence[name] = {
            "rate_increment_mean": np.mean(rate_diffs),
            "rate_increment_std": np.std(rate_diffs),
            "error_ratio_mean": np.mean(error_ratios),
            "error_ratio_std": np.std(error_ratios),
            "exponential_decay_rate": -np.mean(np.log(error_ratios))
        }
    
    return convergence


def create_detailed_analysis_plots(results: Dict, quantizer_configs: List):
    """
    Create detailed analysis plots for the rate-distortion results.
    
    Parameters:
    -----------
    results : Dict
        Results from rate-distortion comparison.
    quantizer_configs : List
        List of quantizer configurations.
    """
    
    # Create subplot layout
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle("Detailed Rate-Distortion Analysis", fontsize=16)
    
    # Plot 1: Rate-Distortion curves (main plot)
    ax1 = axes[0, 0]
    for config in quantizer_configs:
        name = config.name
        if name in results and len(results[name]["R"]) > 0:
            R = results[name]["R"]
            min_errors = results[name]["min_errors"]
            ax1.plot(R, min_errors, label=name, marker=config.marker, 
                    color=config.color, linewidth=2)
    
    # Add theoretical bounds
    all_rates = []
    for data in results.values():
        all_rates.extend(data["R"])
    
    if all_rates:
        rate_range = np.linspace(min(all_rates), max(all_rates), 100)
        theoretical_bounds = generate_theoretical_bounds(rate_range)
        ax1.plot(rate_range, theoretical_bounds["Rate-Distortion Lower Bound"],
                label="R-D Lower Bound", color="black", linestyle="--")
    
    ax1.set_xlabel("Rate (bits per dimension)")
    ax1.set_ylabel("Distortion")
    ax1.set_yscale("log")
    ax1.set_title("Rate-Distortion Performance")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Gap to theoretical bound
    ax2 = axes[0, 1]
    gaps = calculate_gap_to_theoretical(results)
    
    for config in quantizer_configs:
        name = config.name
        if name in gaps:
            rates = results[name]["R"]
            gap_db = gaps[name]["gap_db"]
            ax2.plot(rates, gap_db, label=name, marker=config.marker,
                    color=config.color, linewidth=2)
    
    ax2.set_xlabel("Rate (bits per dimension)")
    ax2.set_ylabel("Gap to R-D Bound (dB)")
    ax2.set_title("Performance Gap Analysis")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Rate efficiency comparison
    ax3 = axes[1, 0]
    efficiency = calculate_rate_efficiency(results)
    
    names = []
    avg_rates = []
    colors = []
    
    for config in quantizer_configs:
        name = config.name
        if name in efficiency:
            names.append(name.replace(" ", "\n"))  # Break long names
            avg_rates.append(efficiency[name]["average_rate"])
            colors.append(config.color)
    
    bars = ax3.bar(names, avg_rates, color=colors, alpha=0.7)
    ax3.set_ylabel("Average Rate (bits per dimension)")
    ax3.set_title("Rate Efficiency Comparison")
    ax3.tick_params(axis='x', rotation=45)
    
    # Add value labels on bars
    for bar, rate in zip(bars, avg_rates):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{rate:.2f}', ha='center', va='bottom')
    
    # Plot 4: Slope comparison
    ax4 = axes[1, 1]
    
    slope_data = {}
    for name, data in results.items():
        if len(data["R"]) >= 2:
            R = np.array(data["R"])
            errors = np.array(data["min_errors"])
            log_R = np.log2(R)
            log_errors = np.log2(errors)
            slope, _ = np.polyfit(log_R, log_errors, 1)
            slope_data[name] = slope
    
    names = []
    slopes = []
    colors = []
    
    for config in quantizer_configs:
        name = config.name
        if name in slope_data:
            names.append(name.replace(" ", "\n"))
            slopes.append(slope_data[name])
            colors.append(config.color)
    
    bars = ax4.bar(names, slopes, color=colors, alpha=0.7)
    ax4.set_ylabel("R-D Curve Slope")
    ax4.set_title("Rate-Distortion Slope Comparison")
    ax4.tick_params(axis='x', rotation=45)
    ax4.axhline(y=-2, color='red', linestyle='--', alpha=0.5, label='Ideal Slope (-2)')
    ax4.legend()
    
    # Add value labels on bars
    for bar, slope in zip(bars, slopes):
        height = bar.get_height()
        y_pos = height + 0.1 if height >= 0 else height - 0.3
        ax4.text(bar.get_x() + bar.get_width()/2., y_pos,
                f'{slope:.2f}', ha='center', va='bottom' if height >= 0 else 'top')
    
    plt.tight_layout()
    plt.savefig("detailed_rate_distortion_analysis.png", dpi=300, bbox_inches='tight')
    plt.show()


def generate_performance_report(results: Dict, quantizer_configs: List) -> str:
    """
    Generate a comprehensive performance report.
    
    Parameters:
    -----------
    results : Dict
        Results from rate-distortion comparison.
    quantizer_configs : List
        List of quantizer configurations.
        
    Returns:
    --------
    str
        Formatted performance report.
    """
    report = []
    report.append("="*80)
    report.append("RATE-DISTORTION COMPARISON PERFORMANCE REPORT")
    report.append("="*80)
    report.append("")
    
    # Summary statistics
    report.append("SUMMARY STATISTICS")
    report.append("-" * 50)
    
    for config in quantizer_configs:
        name = config.name
        if name in results and len(results[name]["R"]) > 0:
            data = results[name]
            rates = np.array(data["R"])
            errors = np.array(data["min_errors"])
            
            report.append(f"\n{name}:")
            report.append(f"  Rate range: {np.min(rates):.3f} - {np.max(rates):.3f} bits/dim")
            report.append(f"  Distortion range: {np.min(errors):.6f} - {np.max(errors):.6f}")
            report.append(f"  Average rate: {np.mean(rates):.3f} bits/dim")
            report.append(f"  Average distortion: {np.mean(errors):.6f}")
    
    # Gap analysis
    gaps = calculate_gap_to_theoretical(results)
    if gaps:
        report.append("\n\nGAP TO THEORETICAL BOUNDS")
        report.append("-" * 50)
        
        for name, gap_data in gaps.items():
            report.append(f"\n{name}:")
            report.append(f"  Mean gap: {gap_data['mean_gap_db']:.2f} dB")
            report.append(f"  Gap std: {gap_data['std_gap_db']:.2f} dB")
    
    # Convergence analysis
    convergence = analyze_convergence_behavior(results)
    if convergence:
        report.append("\n\nCONVERGENCE BEHAVIOR")
        report.append("-" * 50)
        
        for name, conv_data in convergence.items():
            report.append(f"\n{name}:")
            report.append(f"  Rate increment: {conv_data['rate_increment_mean']:.3f} ± {conv_data['rate_increment_std']:.3f}")
            report.append(f"  Error decay rate: {conv_data['exponential_decay_rate']:.3f}")
    
    # Performance ranking
    report.append("\n\nPERFORMANCE RANKING")
    report.append("-" * 50)
    
    # Rank by minimum distortion achieved
    min_distortions = {}
    for name, data in results.items():
        if len(data["min_errors"]) > 0:
            min_distortions[name] = min(data["min_errors"])
    
    ranked_by_distortion = sorted(min_distortions.items(), key=lambda x: x[1])
    
    report.append("\nBy Minimum Distortion Achieved:")
    for i, (name, min_dist) in enumerate(ranked_by_distortion, 1):
        report.append(f"  {i}. {name}: {min_dist:.6f}")
    
    # Rank by rate efficiency (lowest average rate)
    efficiency = calculate_rate_efficiency(results)
    ranked_by_rate = sorted(efficiency.items(), key=lambda x: x[1]["average_rate"])
    
    report.append("\nBy Rate Efficiency (Lower is Better):")
    for i, (name, eff_data) in enumerate(ranked_by_rate, 1):
        report.append(f"  {i}. {name}: {eff_data['average_rate']:.3f} bits/dim")
    
    report.append("\n" + "="*80)
    
    return "\n".join(report)


def main():
    """
    Main function to run comprehensive rate-distortion analysis.
    """
    print("Running Comprehensive Rate-Distortion Analysis...")
    
    # Run comparison with more q values for better analysis
    q_values = np.arange(3, 8)
    n_samples = 3000  # Slightly smaller for faster execution
    M = 3
    
    results, quantizer_configs = run_rate_distortion_comparison(
        q_values, n_samples, M, sigma_squared=1.0
    )
    
    # Generate detailed analysis
    print("\nGenerating detailed analysis plots...")
    create_detailed_analysis_plots(results, quantizer_configs)
    
    # Generate performance report
    report = generate_performance_report(results, quantizer_configs)
    print(report)
    
    # Save report to file
    with open("rate_distortion_performance_report.txt", "w") as f:
        f.write(report)
    
    print("\nAnalysis complete!")
    print("- Main plot saved as: rate_distortion_comparison.png")
    print("- Detailed analysis saved as: detailed_rate_distortion_analysis.png")
    print("- Performance report saved as: rate_distortion_performance_report.txt")


if __name__ == "__main__":
    main()
