#!/usr/bin/env python3
"""
Setup script for gemvq package.
"""

import os

from setuptools import find_packages, setup


# Read the README file
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()


# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [
            line.strip() for line in fh if line.strip() and not line.startswith("#")
        ]


setup(
    name="gemvq",
    version="1.0.0",
    author="gemvq Contributors",
    author_email="soma@mlsquare.org",
    description="High-Rate Nested-Lattice Quantized Matrix Multiplication with Small Lookup Tables",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/mlsquare/gemvq",
    project_urls={
        "Bug Tracker": "https://github.com/mlsquare/gemvq/issues",
        "Documentation": "https://github.com/mlsquare/gemvq/wiki",
        "Source Code": "https://github.com/mlsquare/gemvq",
    },
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.7",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "flake8>=3.8",
            "black>=21.0",
            "isort>=5.0",
        ],
        "docs": [
            "sphinx>=3.0",
            "sphinx-rtd-theme>=0.5",
            "nbsphinx>=0.8",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    keywords=[
        "lattice",
        "quantization",
        "matrix-multiplication",
        "information-theory",
        "compression",
        "machine-learning",
        "signal-processing",
    ],
)
