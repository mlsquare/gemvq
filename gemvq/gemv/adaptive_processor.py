"""
Adaptive Processor for Columnwise Matrix-Vector Multiplication

This module implements an adaptive processor that can dynamically choose between
different computation strategies based on input characteristics like sparsity,
matrix properties, and performance requirements.
"""

import time
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from .columnwise.columnwise_matvec_processor import ColumnwiseMatVecProcessor
from .columnwise.standard_dot_processor import StandardDotProcessor
from .utils.lookup_table_processor import LookupTableProcessor


class AdaptiveProcessor(ColumnwiseMatVecProcessor):
    """
    Adaptive processor that dynamically chooses computation strategy.

    This implementation can switch between different computation strategies
    based on input characteristics:
    - Standard dot product for simple cases
    - Layer-wise histogram for efficient pooling
    - Hybrid approaches for optimal performance
    """

    def __init__(
        self,
        matrix: np.ndarray,
        lattice_type: str = "D4",
        M: int = 2,
        q: int = 4,
        beta: float = 0.2,
        alpha: float = 1 / 3,
        eps: float = 1e-8,
        adaptation_threshold: float = 0.1,
        sparsity_threshold: float = 1e-10,
        decoding: str = "full",
        enable_standard_dot: bool = True,
        enable_lookup_tables: bool = True,
        enable_adaptive_depth: bool = True,
    ):
        """
        Initialize the adaptive processor.

        Parameters:
        -----------
        matrix : np.ndarray
            Input matrix W (m x n).
        lattice_type : str
            Type of lattice to use.
        M : int
            Number of hierarchical levels.
        q : int
            Quantization parameter.
        beta : float
            Scaling parameter for quantization.
        alpha : float
            Scaling parameter for overload handling.
        eps : float
            Small perturbation parameter.
        adaptation_threshold : float
            Sparsity threshold for switching strategies.
        sparsity_threshold : float
            Threshold for considering elements as zero.
        decoding : str
            Default decoding method.
        enable_standard_dot : bool
            Whether to enable standard dot product strategy.
        enable_lookup_tables : bool
            Whether to enable lookup table strategies.
        enable_adaptive_depth : bool
            Whether to enable adaptive depth decoding.
        """
        super().__init__(
            matrix=matrix,
            lattice_type=lattice_type,
            M=M,
            q=q,
            beta=beta,
            alpha=alpha,
            eps=eps,
            use_lookup=True,
            quantize_x=False,
            sparsity_threshold=sparsity_threshold,
            decoding=decoding,
        )

        self.adaptation_threshold = adaptation_threshold
        self.enable_standard_dot = enable_standard_dot
        self.enable_lookup_tables = enable_lookup_tables
        self.enable_adaptive_depth = enable_adaptive_depth

        # Initialize sub-processors
        self.sub_processors = {}
        self._initialize_sub_processors()

        # Strategy selection history
        self.strategy_history = []
        self.performance_history = []

    def _initialize_sub_processors(self):
        """Initialize sub-processors for different strategies."""
        if self.enable_standard_dot:
            # Standard dot processor with adaptive depth
            self.sub_processors["standard_dot"] = StandardDotProcessor(
                matrix=self.original_matrix,
                lattice_type=self.lattice_type,
                M=self.M,
                q=self.q,
                beta=self.beta,
                alpha=self.alpha,
                eps=self.eps,
                fixed_depth=False,
                adaptive_depth=self.enable_adaptive_depth,
                sparsity_threshold=self.sparsity_threshold,
                decoding=self.decoding,
            )

        if self.enable_lookup_tables:
            # Layer-wise histogram processor
            self.sub_processors["layer_wise_histogram"] = LookupTableProcessor(
                matrix=self.original_matrix,
                lattice_type=self.lattice_type,
                M=self.M,
                q=self.q,
                beta=self.beta,
                alpha=self.alpha,
                eps=self.eps,
                table_strategy="layer_wise_histogram",
                precompute_tables=True,
                sparsity_threshold=self.sparsity_threshold,
                decoding=self.decoding,
            )

            # Inner product tables processor
            self.sub_processors["inner_product"] = LookupTableProcessor(
                matrix=self.original_matrix,
                lattice_type=self.lattice_type,
                M=self.M,
                q=self.q,
                beta=self.beta,
                alpha=self.alpha,
                eps=self.eps,
                table_strategy="inner_product",
                precompute_tables=True,
                sparsity_threshold=self.sparsity_threshold,
                decoding=self.decoding,
            )

    def compute_matvec(
        self,
        x: np.ndarray,
        decoding_depths: Optional[List[int]] = None,
        sparsity_pattern: Optional[List[int]] = None,
        force_strategy: Optional[str] = None,
    ) -> np.ndarray:
        """
        Compute y = Wx using adaptive strategy selection.

        Parameters:
        -----------
        x : np.ndarray
            Input vector x.
        decoding_depths : List[int], optional
            Decoding depth for each column block (0 to M-1).
        sparsity_pattern : List[int], optional
            Indices of non-zero elements in x.
        force_strategy : str, optional
            Force a specific strategy to be used.

        Returns:
        --------
        np.ndarray
            Result vector y.
        """
        start_time = time.time()

        # Pad vector to match matrix dimensions
        padded_x = self._pad_vector(x)

        # Detect sparsity if not provided
        if sparsity_pattern is None:
            sparsity_pattern, sparsity_ratio = self._detect_sparsity(padded_x)

        # Select strategy
        if force_strategy is not None:
            strategy = force_strategy
        else:
            strategy = self._select_strategy(padded_x, sparsity_pattern, sparsity_ratio)

        # Validate strategy
        if strategy not in self.sub_processors:
            raise ValueError(
                f"Strategy '{strategy}' not available. Available: {list(self.sub_processors.keys())}"
            )

        # Execute computation using selected strategy
        processor = self.sub_processors[strategy]
        result = processor.compute_matvec(x, decoding_depths, sparsity_pattern)

        # Record strategy selection and performance
        computation_time = time.time() - start_time
        self._record_strategy_usage(strategy, computation_time, sparsity_ratio)

        # Update performance stats
        self.stats["computation_time"] = computation_time
        self.stats["selected_strategy"] = strategy

        return result

    def _select_strategy(
        self, x: np.ndarray, sparsity_pattern: List[int], sparsity_ratio: float
    ) -> str:
        """
        Select the best computation strategy based on input characteristics.

        Strategy selection criteria:
        1. High sparsity (> threshold): Use standard dot with adaptive depth
        2. Low sparsity, small matrix: Use standard dot
        3. Low sparsity, large matrix: Use layer-wise histogram
        4. Very low sparsity: Use inner product tables
        """
        matrix_size = self.original_m * self.original_n
        n_blocks = self.n // self.dimension

        # Criterion 1: High sparsity - use standard dot with adaptive depth
        if sparsity_ratio > self.adaptation_threshold:
            if self.enable_standard_dot and self.enable_adaptive_depth:
                return "standard_dot"
            elif self.enable_standard_dot:
                return "standard_dot"

        # Criterion 2: Small matrix - use standard dot
        if matrix_size < 10000:  # Small matrix threshold
            if self.enable_standard_dot:
                return "standard_dot"

        # Criterion 3: Large matrix, low sparsity - use layer-wise histogram
        if matrix_size >= 10000 and sparsity_ratio < 0.5:
            if self.enable_lookup_tables:
                return "layer_wise_histogram"

        # Criterion 4: Very low sparsity - use inner product tables
        if sparsity_ratio < 0.1:
            if self.enable_lookup_tables:
                return "inner_product"

        # Default: use layer-wise histogram if available, otherwise standard dot
        if self.enable_lookup_tables:
            return "layer_wise_histogram"
        elif self.enable_standard_dot:
            return "standard_dot"
        else:
            raise ValueError("No computation strategies available")

    def _record_strategy_usage(
        self, strategy: str, computation_time: float, sparsity_ratio: float
    ):
        """Record strategy usage for performance analysis."""
        self.strategy_history.append(
            {
                "strategy": strategy,
                "computation_time": computation_time,
                "sparsity_ratio": sparsity_ratio,
                "timestamp": time.time(),
            }
        )

        # Keep only recent history (last 100 entries)
        if len(self.strategy_history) > 100:
            self.strategy_history = self.strategy_history[-100:]

    def get_strategy_recommendations(self) -> Dict[str, Union[str, float]]:
        """Get strategy recommendations based on historical performance."""
        if not self.strategy_history:
            return {"recommendation": "insufficient_data", "confidence": 0.0}

        # Analyze performance by strategy
        strategy_performance = {}
        for entry in self.strategy_history:
            strategy = entry["strategy"]
            if strategy not in strategy_performance:
                strategy_performance[strategy] = []
            strategy_performance[strategy].append(entry["computation_time"])

        # Find best performing strategy
        best_strategy = None
        best_avg_time = float("inf")

        for strategy, times in strategy_performance.items():
            avg_time = np.mean(times)
            if avg_time < best_avg_time:
                best_avg_time = avg_time
                best_strategy = strategy

        # Calculate confidence based on consistency
        if best_strategy:
            best_times = strategy_performance[best_strategy]
            confidence = (
                1.0 - (np.std(best_times) / np.mean(best_times))
                if np.mean(best_times) > 0
                else 0.0
            )
            confidence = max(0.0, min(1.0, confidence))
        else:
            confidence = 0.0

        return {
            "recommendation": best_strategy or "no_recommendation",
            "confidence": confidence,
            "best_avg_time": best_avg_time,
            "strategy_performance": strategy_performance,
        }

    def get_adaptive_stats(self) -> Dict[str, Union[str, float, List]]:
        """Get adaptive processor statistics."""
        stats = self.get_compression_stats()

        # Add adaptive-specific stats
        stats.update(
            {
                "adaptation_threshold": self.adaptation_threshold,
                "enable_standard_dot": self.enable_standard_dot,
                "enable_lookup_tables": self.enable_lookup_tables,
                "enable_adaptive_depth": self.enable_adaptive_depth,
                "available_strategies": list(self.sub_processors.keys()),
                "strategy_history_length": len(self.strategy_history),
            }
        )

        # Add strategy recommendations
        recommendations = self.get_strategy_recommendations()
        stats.update(recommendations)

        return stats

    def compute_matvec_with_strategy(
        self,
        x: np.ndarray,
        strategy: str,
        decoding_depths: Optional[List[int]] = None,
        sparsity_pattern: Optional[List[int]] = None,
    ) -> np.ndarray:
        """
        Compute matvec using a specific strategy.

        Parameters:
        -----------
        x : np.ndarray
            Input vector x.
        strategy : str
            Strategy to use ('standard_dot', 'layer_wise_histogram', 'inner_product').
        decoding_depths : List[int], optional
            Decoding depth for each column block.
        sparsity_pattern : List[int], optional
            Indices of non-zero elements in x.

        Returns:
        --------
        np.ndarray
            Result vector y.
        """
        return self.compute_matvec(
            x, decoding_depths, sparsity_pattern, force_strategy=strategy
        )

    def benchmark_strategies(
        self, x: np.ndarray, num_runs: int = 5
    ) -> Dict[str, Dict[str, float]]:
        """
        Benchmark all available strategies.

        Parameters:
        -----------
        x : np.ndarray
            Input vector x.
        num_runs : int
            Number of runs for averaging.

        Returns:
        --------
        Dict[str, Dict[str, float]]
            Performance statistics for each strategy.
        """
        benchmark_results = {}

        for strategy_name, processor in self.sub_processors.items():
            times = []

            for _ in range(num_runs):
                start_time = time.time()
                result = processor.compute_matvec(x)
                times.append(time.time() - start_time)

            benchmark_results[strategy_name] = {
                "mean_time": np.mean(times),
                "std_time": np.std(times),
                "min_time": np.min(times),
                "max_time": np.max(times),
            }

        return benchmark_results

    def get_performance_stats(self) -> Dict[str, Union[str, float, List]]:
        """Get detailed performance statistics."""
        return self.get_adaptive_stats()
