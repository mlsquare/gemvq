"""
GEMV utilities module.

This module provides utility functions and classes for GEMV processors.
"""

from .padder import BlockingStrategy

__all__ = ['BlockingStrategy']
