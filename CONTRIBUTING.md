# Contributing to CrewMaster

Thank you for your interest in contributing to CrewMaster! We welcome contributions from the community and are grateful for your support.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Contributing Guidelines](#contributing-guidelines)
- [Pull Request Process](#pull-request-process)
- [Issue Reporting](#issue-reporting)
- [Development Workflow](#development-workflow)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Documentation](#documentation)

## Code of Conduct

This project and everyone participating in it is governed by our Code of Conduct. By participating, you are expected to uphold this code. Please report unacceptable behavior to [vishnuprasadapp@gmail.com](mailto:vishnuprasadapp@gmail.com).

## Getting Started

1. Fork the repository on GitHub
2. Clone your fork locally
3. Set up the development environment
4. Create a feature branch
5. Make your changes
6. Test your changes
7. Submit a pull request

## Development Setup

### Prerequisites

- Python 3.9 or higher
- Git
- UV package manager (recommended) or pip

### Setup Instructions

```bash
# Clone your fork
git clone https://github.com/VishApp/crewmaster.git
cd crewmaster

# Install development dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Verify installation
crewmaster --help
```

### Alternative Setup with UV

```bash
# Clone your fork
git clone https://github.com/VishApp/crewmaster.git
cd crewmaster

# Install with UV
uv sync --extra dev

# Install pre-commit hooks
pre-commit install
```

## Contributing Guidelines

### Types of Contributions

We welcome several types of contributions:

- **Bug fixes**: Fix existing issues or bugs
- **Feature enhancements**: Improve existing functionality
- **New features**: Add new capabilities to CrewMaster
- **Documentation**: Improve or add documentation
- **Tests**: Add or improve test coverage
- **Performance improvements**: Optimize existing code

### Before Contributing

1. Check existing issues and pull requests to avoid duplicates
2. For major changes, please open an issue first to discuss your proposal
3. Ensure your changes align with the project's goals and architecture

## Pull Request Process

1. **Create a feature branch** from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** following our coding standards

3. **Test your changes**:
   ```bash
   # Run tests
   pytest tests/
   
   # Run linting
   ruff check crewmaster/
   ruff format crewmaster/
   
   # Run type checking
   mypy crewmaster/
   ```

4. **Update documentation** if needed

5. **Commit your changes** with a clear message:
   ```bash
   git commit -m "feat: add new feature for better crew orchestration"
   ```

6. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

7. **Create a pull request** on GitHub

### Pull Request Requirements

- Clear description of the changes
- Reference to related issues (if applicable)
- Tests for new functionality
- Documentation updates (if applicable)
- Changelog entry (if applicable)
- All CI checks passing

## Issue Reporting

### Bug Reports

When reporting bugs, please include:

- **Clear title** and description
- **Steps to reproduce** the issue
- **Expected vs. actual behavior**
- **Environment details** (OS, Python version, CrewMaster version)
- **Error messages** or logs (if applicable)
- **Minimal reproducible example** (if possible)

### Feature Requests

For feature requests, please include:

- **Clear description** of the proposed feature
- **Use case** and motivation
- **Possible implementation** approach (if you have ideas)
- **Alternative solutions** you've considered

## Development Workflow

### Branch Naming

- `feature/description` - New features
- `fix/description` - Bug fixes
- `docs/description` - Documentation changes
- `refactor/description` - Code refactoring
- `test/description` - Test improvements

### Commit Messages

Follow conventional commit format:

- `feat:` - New features
- `fix:` - Bug fixes
- `docs:` - Documentation changes
- `style:` - Code style changes
- `refactor:` - Code refactoring
- `test:` - Test changes
- `chore:` - Maintenance tasks

Examples:
```
feat: add support for custom tool creation
fix: resolve memory leak in agent orchestration
docs: update installation instructions
```

## Coding Standards

### Python Style

- Follow PEP 8 guidelines
- Use Black for code formatting (line length: 88)
- Use Ruff for linting
- Use type hints for all functions and methods
- Write docstrings for all public functions and classes

### Code Organization

- Keep functions and classes focused and small
- Use descriptive variable and function names
- Add comments for complex logic
- Follow existing project structure and patterns

### Error Handling

- Use appropriate exception types
- Provide clear error messages
- Log errors appropriately
- Handle edge cases gracefully

## Testing

### Test Requirements

- All new features must include tests
- Maintain or improve test coverage
- Tests should be clear and focused
- Use descriptive test names

### Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_specific_file.py

# Run with coverage
pytest tests/ --cov=crewmaster

# Run tests in parallel
pytest tests/ -n auto
```

### Test Categories

- **Unit tests**: Test individual functions and classes
- **Integration tests**: Test component interactions
- **End-to-end tests**: Test complete workflows
- **Performance tests**: Test performance characteristics

## Documentation

### Documentation Standards

- Use clear, concise language
- Include code examples
- Update relevant documentation for changes
- Follow existing documentation style

### Documentation Types

- **API documentation**: Docstrings in code
- **User guides**: README and usage documentation
- **Developer guides**: This contributing guide
- **Examples**: Real-world usage examples

## Release Process

### Version Management

We follow [Semantic Versioning](https://semver.org/):

- **MAJOR**: Incompatible API changes
- **MINOR**: New functionality (backward compatible)
- **PATCH**: Bug fixes (backward compatible)

### Changelog

- Update `CHANGELOG.md` for significant changes
- Follow [Keep a Changelog](https://keepachangelog.com/) format
- Include migration notes for breaking changes

## Community

### Getting Help

- **GitHub Issues**: For bug reports and feature requests
- **GitHub Discussions**: For questions and community discussions
- **Email**: [vishnuprasadapp@gmail.com](mailto:vishnuprasadapp@gmail.com) for private matters

### Stay Updated

- Watch the repository for updates
- Follow release notes and changelogs
- Participate in community discussions

## Recognition

Contributors are recognized in:

- GitHub contributors list
- Release notes (for significant contributions)
- Special mentions in documentation

## Questions?

If you have questions about contributing, please:

1. Check this guide and existing documentation
2. Search existing issues and discussions
3. Open a new discussion or issue
4. Contact the maintainers directly

Thank you for contributing to CrewMaster! 🚀