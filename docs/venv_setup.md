# Virtual Environment Setup with uv

This project uses `uv` as the package manager and has a virtual environment set up for development.

## Environment Setup

The virtual environment is located at `.venv/` and includes all necessary dependencies for the project.

### Available Kernels

The following Jupyter kernels are available:

- **gemvq**: The main project kernel with all dependencies installed
- **python3**: Default Python 3 kernel from the virtual environment

### Using the Environment

#### Option 1: Use the convenience script
```bash
./start_jupyter.sh
```

#### Option 2: Manual activation
```bash
# Activate the virtual environment
source .venv/bin/activate

# Start Jupyter notebook
jupyter notebook
```

#### Option 3: Use uv run
```bash
uv run jupyter notebook
```

### Importing Project Modules

Since the project is installed in development mode, you can import modules directly:

```python
# Instead of sys.path.append('../')
from src.lattices.quantizers.nested_lattice_quantizer import NestedLatticeQuantizer
from src.gemv.adaptive.adaptive_matvec import AdaptiveMatvecProcessor
# ... etc
```

### Kernel Selection in Jupyter

When opening a notebook, make sure to select the **"Python (gemvq)"** kernel from the kernel menu to ensure you're using the correct environment with all dependencies.

### Installing Additional Packages

To install additional packages in the virtual environment:

```bash
# Activate the environment
source .venv/bin/activate

# Install with uv
uv pip install package_name

# Or install with pip
pip install package_name
```

### Rebuilding the Environment

If you need to rebuild the environment:

```bash
# Remove existing environment
rm -rf .venv

# Create new environment
uv venv

# Activate and install dependencies
source .venv/bin/activate
uv pip install -r requirements.txt
uv pip install jupyter ipykernel
uv pip install -e .

# Reinstall the kernel
python -m ipykernel install --user --name=gemvq --display-name="Python (gemvq)"
```
