# Contributing to Gap Trading System

We love your input! We want to make contributing to this project as easy and transparent as possible, whether it's:

- Reporting a bug
- Discussing the current state of the code
- Submitting a fix
- Proposing new features
- Becoming a maintainer

## 🚀 Development Process

We use GitHub to host code, track issues and feature requests, as well as accept pull requests.

### Pull Requests
Pull requests are the best way to propose changes to the codebase. We actively welcome your pull requests:

1. Fork the repo and create your branch from `main`.
2. If you've added code that should be tested, add tests.
3. If you've changed APIs, update the documentation.
4. Ensure the test suite passes.
5. Make sure your code lints.
6. Issue that pull request!

## 🐛 Bug Reports

We use GitHub issues to track public bugs. Report a bug by [opening a new issue](https://github.com/yourusername/gappers-trader/issues).

**Great Bug Reports** tend to have:

- A quick summary and/or background
- Steps to reproduce
  - Be specific!
  - Give sample code if you can
- What you expected would happen
- What actually happens
- Notes (possibly including why you think this might be happening, or stuff you tried that didn't work)

## 💻 Development Setup

### Prerequisites

- Python 3.12+
- Poetry
- Git
- Docker (optional, for containerized development)

### Setup Instructions

1. **Fork and clone the repository**
   ```bash
   git clone https://github.com/yourusername/gappers-trader.git
   cd gappers-trader
   ```

2. **Install Poetry** (if not already installed)
   ```bash
   curl -sSL https://install.python-poetry.org | python3 -
   ```

3. **Install dependencies**
   ```bash
   poetry install --with dev
   ```

4. **Activate virtual environment**
   ```bash
   poetry shell
   ```

5. **Install pre-commit hooks**
   ```bash
   pre-commit install
   ```

6. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

7. **Verify installation**
   ```bash
   # Run tests
   pytest

   # Start development server
   streamlit run app.py
   ```

## 🧪 Testing

We maintain high test coverage (80%+) and use pytest for testing.

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=gappers --cov-report=html

# Run specific test file
pytest tests/test_backtest.py -v

# Run tests with specific markers
pytest -m "not slow"  # Skip slow integration tests
```

### Writing Tests

- Place tests in the `tests/` directory
- Follow the naming convention `test_*.py`
- Use descriptive test names that explain what is being tested
- Include both positive and negative test cases
- Mock external dependencies (APIs, file system, etc.)

**Example test structure:**
```python
def test_calculate_gap_basic_functionality():
    """Test basic gap calculation with valid data."""
    # Arrange
    signal_gen = SignalGenerator(mock_data_feed)
    test_data = create_test_ohlcv_data()
    
    # Act
    result = signal_gen._calculate_symbol_gap('AAPL', test_data, test_date)
    
    # Assert
    assert result is not None
    assert abs(result['gap_pct'] - 0.05) < 0.001
```

### Test Categories

- **Unit Tests**: Test individual functions and methods
- **Integration Tests**: Test component interactions
- **End-to-End Tests**: Test complete workflows
- **Performance Tests**: Test speed and memory usage

## 📝 Code Style

We use several tools to maintain consistent code style:

### Formatting and Linting

- **Black**: Code formatting
- **Ruff**: Fast Python linter
- **MyPy**: Static type checking
- **Bandit**: Security vulnerability scanning

### Running Code Quality Checks

```bash
# Run all checks (via pre-commit)
pre-commit run --all-files

# Individual tools
black gappers/ tests/
ruff check gappers/ tests/
mypy gappers/
bandit -r gappers/
```

### Code Style Guidelines

1. **Follow PEP 8** - Python style guide
2. **Use type hints** - All functions should have type annotations
3. **Write docstrings** - Use Google-style docstrings
4. **Keep functions small** - Prefer smaller, focused functions
5. **Use descriptive names** - Variable and function names should be clear
6. **Handle errors gracefully** - Always include appropriate error handling

**Example function with proper style:**
```python
def calculate_sharpe_ratio(
    returns: pd.Series, 
    risk_free_rate: float = 0.02
) -> float:
    """
    Calculate the Sharpe ratio for a series of returns.
    
    Args:
        returns: Series of daily returns
        risk_free_rate: Annual risk-free rate (default: 2%)
        
    Returns:
        Sharpe ratio as float
        
    Raises:
        ValueError: If returns series is empty
    """
    if returns.empty:
        raise ValueError("Returns series cannot be empty")
        
    excess_returns = returns - (risk_free_rate / 252)
    return excess_returns.mean() / returns.std() * np.sqrt(252)
```

## 📚 Documentation

### Code Documentation

- Use Google-style docstrings for all functions and classes
- Include type hints for all parameters and return values
- Document complex algorithms and business logic
- Add inline comments for non-obvious code

### README and Guides

- Update README.md for any user-facing changes
- Add new configuration options to documentation
- Include examples for new features
- Update API documentation for interface changes

## 🏗️ Architecture Guidelines

### Design Principles

1. **Separation of Concerns** - Each module has a single responsibility
2. **Dependency Injection** - Use dependency injection for testability
3. **Configuration Management** - Centralize configuration
4. **Error Handling** - Graceful error handling with logging
5. **Performance** - Optimize for speed and memory usage

### Module Structure

```
gappers/
├── __init__.py          # Public API exports
├── config.py            # Configuration management
├── datafeed.py          # Data ingestion
├── universe.py          # Symbol universe construction
├── signals.py           # Signal generation
├── backtest.py          # Backtesting engine
├── live.py              # Live trading
├── analytics.py         # Performance analysis
└── cli.py               # Command-line interface
```

### Adding New Features

When adding new features:

1. **Design first** - Create an issue to discuss the design
2. **Start with tests** - Write tests before implementation
3. **Keep it modular** - New features should fit the existing architecture
4. **Document thoroughly** - Include docstrings and usage examples
5. **Consider performance** - Profile code for performance bottlenecks

## 🔄 Release Process

### Versioning

We use [Semantic Versioning](https://semver.org/):

- **MAJOR** version when you make incompatible API changes
- **MINOR** version when you add functionality in a backwards compatible manner
- **PATCH** version when you make backwards compatible bug fixes

### Release Steps

1. Update version in `pyproject.toml`
2. Update `CHANGELOG.md` with release notes
3. Create a pull request with changes
4. After merge, create a GitHub release with tag
5. CI/CD pipeline will automatically build and publish

## 🔒 Security Guidelines

### Security Best Practices

1. **Never commit secrets** - Use environment variables
2. **Validate all inputs** - Check user inputs and API responses
3. **Use secure dependencies** - Regularly update dependencies
4. **Log security events** - Log authentication and authorization events
5. **Follow least privilege** - Minimize permissions and access

### Reporting Security Issues

Please do NOT report security vulnerabilities in public issues. Instead:

1. Email security issues to [security@yourproject.com]
2. Include a detailed description of the vulnerability
3. Provide steps to reproduce if possible
4. We will respond within 48 hours

## 🤝 Community Guidelines

### Code of Conduct

This project follows the [Contributor Covenant Code of Conduct](https://www.contributor-covenant.org/). By participating, you are expected to uphold this code.

### Getting Help

- 📖 **Documentation**: Check the [project wiki](https://github.com/yourusername/gappers-trader/wiki)
- 💬 **Discussions**: Use [GitHub Discussions](https://github.com/yourusername/gappers-trader/discussions)
- 🐛 **Issues**: Report bugs via [GitHub Issues](https://github.com/yourusername/gappers-trader/issues)

### Recognition

Contributors are recognized in:
- `CONTRIBUTORS.md` file
- Release notes for significant contributions
- GitHub contributor statistics

## 📋 Pull Request Template

When creating a pull request, please use this template:

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)
- [ ] Documentation update

## Testing
- [ ] Tests pass locally
- [ ] New tests added for new functionality
- [ ] Code coverage maintained

## Checklist
- [ ] Code follows project style guidelines
- [ ] Self-review of code completed
- [ ] Documentation updated
- [ ] No new warnings introduced
```

## 🏷️ Issue Labels

We use these labels to categorize issues:

- `bug`: Something isn't working
- `enhancement`: New feature or request
- `documentation`: Improvements or additions to documentation
- `good first issue`: Good for newcomers
- `help wanted`: Extra attention is needed
- `performance`: Performance improvements
- `security`: Security-related issues
- `testing`: Related to testing

## 🎯 Development Roadmap

Check our [project roadmap](https://github.com/yourusername/gappers-trader/projects) for:

- Planned features
- Current development priorities
- Long-term goals
- Community requests

## 📞 Contact

- **Project Maintainer**: [Your Name](mailto:your.email@example.com)
- **Project Repository**: https://github.com/yourusername/gappers-trader
- **Documentation**: https://gappers-trader.readthedocs.io

---

Thank you for contributing to the Gap Trading System! 🚀