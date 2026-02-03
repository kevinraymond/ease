# Contributing to EASE

Thank you for your interest in contributing to EASE! This document provides guidelines for contributing to the project.

## Reporting Bugs

If you find a bug, please open an issue on GitHub with:

1. A clear, descriptive title
2. Steps to reproduce the issue
3. Expected behavior vs. actual behavior
4. Your environment (OS, GPU, CUDA version, browser)
5. Any relevant error messages or logs

## Suggesting Features

Feature requests are welcome! Please open an issue with:

1. A clear description of the feature
2. The problem it solves or use case it enables
3. Any implementation ideas you have

## Submitting Pull Requests

1. Fork the repository
2. Create a feature branch from `main`
3. Make your changes
4. Ensure tests pass and code style checks pass
5. Submit a pull request with a clear description of your changes

### Development Setup

See the [Quick Start Guide](docs/QUICKSTART.md) for setting up your development environment.

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/ease.git
cd ease

# Run the setup script
./scripts/setup.sh

# Install dev dependencies
cd server && uv sync --all-extras
cd ../frontend && npm install
```

### Code Style

**Python (server)**:
- Format with `black` (line length 100)
- Lint with `ruff`
- Type check with `mypy`

```bash
cd server
uv run black src/
uv run ruff check src/
uv run mypy src/
```

**TypeScript (frontend)**:
- Use the project's ESLint configuration
- Follow existing code patterns

```bash
cd frontend
npm run lint
```

### Running Tests

```bash
# Python tests
cd server
uv run pytest

# TypeScript tests
cd frontend
npm test
```

## Code of Conduct

Be respectful and constructive in all interactions. We're all here to build something cool together.

## Questions?

Feel free to open an issue for any questions about contributing.
