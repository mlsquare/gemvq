"""
GEMV processor factory.

This module provides a factory pattern for creating different types of GEMV processors
(columnwise, rowwise, SVD, etc.) based on configuration parameters.
"""

from typing import Any, Dict, Optional, Type
import numpy as np

from .gemv_processor import GEMVProcessor


class GEMVFactory:
    """
    Factory class for creating GEMV processors.
    
    This factory can create different types of GEMV processors based on the
    provided configuration parameters.
    """
    
    _processors: Dict[str, Type[GEMVProcessor]] = {}
    
    @classmethod
    def register_processor(cls, name: str, processor_class: Type[GEMVProcessor]) -> None:
        """
        Register a processor class with the factory.
        
        Args:
            name: Name to register the processor under
            processor_class: The processor class to register
        """
        cls._processors[name] = processor_class
    
    @classmethod
    def create_processor(cls, processor_type: str, **kwargs) -> GEMVProcessor:
        """
        Create a GEMV processor of the specified type.
        
        Args:
            processor_type: Type of processor to create
            **kwargs: Configuration parameters for the processor
            
        Returns:
            Configured GEMV processor
            
        Raises:
            ValueError: If the processor type is not registered
        """
        if processor_type not in cls._processors:
            available = list(cls._processors.keys())
            raise ValueError(
                f"Unknown processor type '{processor_type}'. "
                f"Available types: {available}"
            )
        
        processor_class = cls._processors[processor_type]
        return processor_class(**kwargs)
    
    @classmethod
    def get_available_processors(cls) -> list[str]:
        """
        Get list of available processor types.
        
        Returns:
            List of registered processor type names
        """
        return list(cls._processors.keys())
    
    @classmethod
    def get_processor_info(cls, processor_type: str) -> Dict[str, Any]:
        """
        Get information about a specific processor type.
        
        Args:
            processor_type: Type of processor
            
        Returns:
            Dictionary containing processor information
            
        Raises:
            ValueError: If the processor type is not registered
        """
        if processor_type not in cls._processors:
            available = list(cls._processors.keys())
            raise ValueError(
                f"Unknown processor type '{processor_type}'. "
                f"Available types: {available}"
            )
        
        processor_class = cls._processors[processor_type]
        return {
            'name': processor_type,
            'class': processor_class,
            'docstring': processor_class.__doc__,
            'module': processor_class.__module__
        }


# Convenience function for creating processors
def create_gemv_processor(processor_type: str, **kwargs) -> GEMVProcessor:
    """
    Convenience function to create a GEMV processor.
    
    Args:
        processor_type: Type of processor to create
        **kwargs: Configuration parameters for the processor
        
    Returns:
        Configured GEMV processor
    """
    return GEMVFactory.create_processor(processor_type, **kwargs)
