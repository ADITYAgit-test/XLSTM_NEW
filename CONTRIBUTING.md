# Contributing to xLSTM-UNet-PyTorch

We welcome contributions to xLSTM-UNet-PyTorch! This document provides guidelines for contributing to the project.

## Development Setup

1. Fork the repository
2. Clone your fork: `git clone https://github.com/your-username/xLSTM-UNet-PyTorch.git`
3. Create a virtual environment: `python -m venv venv`
4. Activate the environment: `source venv/bin/activate` (Linux/Mac) or `venv\Scripts\activate` (Windows)
5. Install dependencies: `pip install -r requirements.txt`
6. Install pre-commit hooks: `pip install pre-commit && pre-commit install`

## Development Workflow

1. Create a feature branch: `git checkout -b feature/your-feature-name`
2. Make your changes
3. Run tests and linting: `pre-commit run --all-files`
4. Commit your changes: `git commit -m "Add your commit message"`
5. Push to your fork: `git push origin feature/your-feature-name`
6. Create a Pull Request

## Code Style

- Follow PEP 8 style guidelines
- Use type hints where possible
- Write docstrings for all functions and classes
- Keep line length under 88 characters (Black formatter default)

## Testing

- Add unit tests for new functionality
- Ensure all existing tests pass
- Test on multiple Python versions if possible

## Documentation

- Update README.md for significant changes
- Add docstrings to new functions
- Update type hints

## Reporting Issues

When reporting bugs, please include:
- Python version
- PyTorch version
- CUDA version (if applicable)
- Operating system
- Steps to reproduce
- Expected vs. actual behavior

## Questions?

Feel free to open an issue for questions or discussions!