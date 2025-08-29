"""
Base GEMV processor abstract class.

This module provides the base abstract class for all GEMV (General Matrix-Vector) 
processors, defining the common interface that all implementations must follow.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np


class GEMVProcessor(ABC):
    """
    Abstract base class for GEMV processors.
    
    All GEMV processors (columnwise, rowwise, SVD, etc.) must inherit from this
    class and implement the required methods.
    """
    
    def __init__(self, **kwargs):
        """Initialize the GEMV processor with configuration options."""
        self.config = kwargs
        self._validate_config()
    
    @abstractmethod
    def _validate_config(self) -> None:
        """Validate the configuration parameters."""
        pass
    
    @abstractmethod
    def process(self, matrix: np.ndarray, vector: np.ndarray) -> np.ndarray:
        """
        Process the matrix-vector multiplication.
        
        Args:
            matrix: Input matrix
            vector: Input vector
            
        Returns:
            Result vector
        """
        pass
    
    @abstractmethod
    def get_processor_info(self) -> Dict[str, Any]:
        """
        Get information about the processor configuration.
        
        Returns:
            Dictionary containing processor information
        """
        pass
    
    def preprocess_matrix(self, matrix: np.ndarray) -> np.ndarray:
        """
        Preprocess the matrix if needed.
        
        Args:
            matrix: Input matrix
            
        Returns:
            Preprocessed matrix
        """
        return matrix
    
    def preprocess_vector(self, vector: np.ndarray) -> np.ndarray:
        """
        Preprocess the vector if needed.
        
        Args:
            vector: Input vector
            
        Returns:
            Preprocessed vector
        """
        return vector
    
    def postprocess_result(self, result: np.ndarray) -> np.ndarray:
        """
        Postprocess the result if needed.
        
        Args:
            result: Computed result
            
        Returns:
            Postprocessed result
        """
        return result
    
    def __call__(self, matrix: np.ndarray, vector: np.ndarray) -> np.ndarray:
        """
        Convenience method to call the processor directly.
        
        Args:
            matrix: Input matrix
            vector: Input vector
            
        Returns:
            Result vector
        """
        matrix = self.preprocess_matrix(matrix)
        vector = self.preprocess_vector(vector)
        result = self.process(matrix, vector)
        return self.postprocess_result(result)
