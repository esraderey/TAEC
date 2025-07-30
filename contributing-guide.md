# Contributing to TAEC v3 🤝

Thank you for your interest in contributing to TAEC (Advanced Cognitive Evolution System)! This document provides guidelines and instructions for contributing to the project.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [How to Contribute](#how-to-contribute)
- [Development Setup](#development-setup)
- [Coding Standards](#coding-standards)
- [Testing Guidelines](#testing-guidelines)
- [Documentation](#documentation)
- [Pull Request Process](#pull-request-process)
- [Community](#community)

## Code of Conduct

### Our Pledge

We are committed to providing a friendly, safe, and welcoming environment for all contributors, regardless of experience level, gender identity and expression, sexual orientation, disability, personal appearance, body size, race, ethnicity, age, religion, nationality, or other similar characteristics.

### Expected Behavior

- Be respectful and inclusive
- Welcome newcomers and help them get started
- Focus on what is best for the community
- Show empathy towards other community members
- Be constructive in your feedback

### Unacceptable Behavior

- Harassment, discrimination, or offensive comments
- Personal attacks or trolling
- Publishing others' private information
- Other conduct that could reasonably be considered inappropriate

## Getting Started

1. **Fork the Repository**
   ```bash
   # Fork on GitHub, then:
   git clone https://github.com/YOUR_USERNAME/taec-v3.git
   cd taec-v3
   ```

2. **Set Up Development Environment**
   ```bash
   # Run the setup script
   ./setup.sh
   
   # Or manually:
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   pip install -r requirements-dev.txt
   ```

3. **Create a Branch**
   ```bash
   git checkout -b feature/your-feature-name
   # or
   git checkout -b fix/your-bug-fix
   ```

## How to Contribute

### Types of Contributions

#### 🐛 Bug Reports
- Use the issue tracker to report bugs
- Include detailed steps to reproduce
- Provide system information and error messages
- Add labels: `bug`, `priority`, etc.

#### ✨ Feature Requests
- Discuss new features in issues first
- Provide use cases and examples
- Consider backward compatibility
- Add label: `enhancement`

#### 📝 Documentation
- Improve README and docs
- Add code examples
- Write tutorials
- Translate documentation

#### 🔧 Code Contributions
- Fix bugs
- Implement new features
- Improve performance
- Refactor code

#### 🧪 Tests
- Add unit tests
- Write integration tests
- Improve test coverage
- Add benchmarks

### Contribution Workflow

1. **Check Existing Issues**
   - Look for existing issues or discussions
   - Comment on issues you want to work on

2. **Create an Issue** (if needed)
   - Describe the problem or feature
   - Wait for feedback before starting major work

3. **Develop Your Contribution**
   - Write clean, documented code
   - Add tests for new functionality
   - Update documentation

4. **Submit a Pull Request**
   - Reference the related issue
   - Describe your changes
   - Ensure all tests pass

## Development Setup

### Prerequisites

- Python 3.8 or higher
- Git
- Docker (optional)
- GPU support (optional, for ML features)

### Environment Setup

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/taec-v3.git
cd taec-v3

# Add upstream remote
git remote add upstream https://github.com/taec-team/taec-v3.git

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install
```

### Running the Development Server

```bash
# Start the API server
python api/main.py

# In another terminal, start the frontend
cd frontend
npm install
npm run dev
```

## Coding Standards

### Python Style Guide

We follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) with some modifications:

```python
# Good example
class QuantumState:
    """
    Represents a quantum state in the TAEC system.
    
    Args:
        dimensions (int): Number of quantum dimensions
        normalize (bool): Whether to normalize the state
    
    Attributes:
        amplitudes (np.ndarray): Complex amplitudes
        coherence (float): Coherence measure [0, 1]
    """
    
    def __init__(self, dimensions: int = 2, normalize: bool = True):
        self.dimensions = dimensions
        self.amplitudes = self._initialize_amplitudes()
        if normalize:
            self.normalize()
    
    def normalize(self) -> None:
        """Normalize the quantum state to unit norm."""
        norm = np.linalg.norm(self.amplitudes)
        if norm > 0:
            self.amplitudes /= norm
    
    def calculate_entropy(self) -> float:
        """
        Calculate von Neumann entropy.
        
        Returns:
            float: Entropy value
        """
        # Implementation here
        pass
```

### Code Quality Tools

```bash
# Format code with Black
black .

# Sort imports
isort .

# Check style with Flake8
flake8 .

# Type checking with MyPy
mypy .

# Security check with Bandit
bandit -r .
```

### MSC-Lang Style Guide

```mscl
# Good MSC-Lang example
synth example_synthesis {
    # Clear variable names
    high_value_nodes = filter_nodes(graph, threshold=0.8);
    
    # Use pattern matching effectively
    for node in high_value_nodes {
        match node.type {
            case "quantum" => apply_quantum_transform(node);
            case "classical" => apply_classical_transform(node);
            case _ => log_warning("Unknown node type");
        }
    }
    
    # Document complex operations
    # Create superposition of high-value states
    quantum_state = create_superposition(high_value_nodes);
    
    return quantum_state;
}
```

## Testing Guidelines

### Writing Tests

```python
# test_quantum_memory.py
import pytest
import numpy as np
from taec_v3.quantum import QuantumState

class TestQuantumState:
    """Test cases for QuantumState class."""
    
    @pytest.fixture
    def quantum_state(self):
        """Create a quantum state for testing."""
        return QuantumState(dimensions=4)
    
    def test_initialization(self, quantum_state):
        """Test quantum state initialization."""
        assert quantum_state.dimensions == 4
        assert quantum_state.amplitudes.shape == (4,)
        assert np.isclose(np.linalg.norm(quantum_state.amplitudes), 1.0)
    
    def test_normalization(self):
        """Test state normalization."""
        state = QuantumState(dimensions=2, normalize=False)
        state.amplitudes = np.array([1.0, 1.0])
        state.normalize()
        
        expected = np.array([1/np.sqrt(2), 1/np.sqrt(2)])
        np.testing.assert_array_almost_equal(state.amplitudes, expected)
    
    @pytest.mark.parametrize("dimensions,expected_max_entropy", [
        (2, 1.0),
        (4, 2.0),
        (8, 3.0),
    ])
    def test_entropy_bounds(self, dimensions, expected_max_entropy):
        """Test entropy calculation bounds."""
        state = QuantumState(dimensions=dimensions)
        entropy = state.calculate_entropy()
        
        assert 0 <= entropy <= expected_max_entropy
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=taec_v3 --cov-report=html

# Run specific test file
pytest tests/test_quantum_memory.py

# Run with verbose output
pytest -v

# Run only marked tests
pytest -m "not slow"
```

### Test Categories

- **Unit Tests**: Test individual components
- **Integration Tests**: Test component interactions
- **Performance Tests**: Benchmark critical paths
- **End-to-End Tests**: Test complete workflows

## Documentation

### Docstring Format

We use Google-style docstrings:

```python
def evolve_population(
    population: List[Individual],
    fitness_function: Callable[[Individual], float],
    generations: int = 100
) -> Tuple[List[Individual], EvolutionMetrics]:
    """
    Evolve a population using genetic algorithms.
    
    Args:
        population: Initial population of individuals
        fitness_function: Function to evaluate individual fitness
        generations: Number of generations to evolve
    
    Returns:
        Tuple containing:
            - Evolved population
            - Evolution metrics and statistics
    
    Raises:
        ValueError: If population is empty
        EvolutionError: If evolution fails
    
    Example:
        >>> pop = create_random_population(50)
        >>> evolved, metrics = evolve_population(pop, my_fitness, 100)
        >>> print(f"Best fitness: {metrics.best_fitness}")
    """
    # Implementation
    pass
```

### Documentation Types

1. **Code Documentation**
   - Inline comments for complex logic
   - Docstrings for all public functions/classes
   - Type hints for all parameters

2. **API Documentation**
   - OpenAPI/Swagger specs
   - Example requests/responses
   - Authentication details

3. **User Documentation**
   - Installation guides
   - Tutorials and examples
   - FAQ section

4. **Developer Documentation**
   - Architecture overview
   - Design decisions
   - Performance considerations

## Pull Request Process

### Before Submitting

1. **Update your fork**
   ```bash
   git fetch upstream
   git checkout main
   git merge upstream/main
   ```

2. **Rebase your branch**
   ```bash
   git checkout your-feature-branch
   git rebase main
   ```

3. **Run quality checks**
   ```bash
   # Format code
   black .
   isort .
   
   # Run tests
   pytest
   
   # Check style
   flake8 .
   mypy .
   ```

### PR Guidelines

1. **Title Format**
   ```
   [TYPE] Brief description
   
   Types: FEAT, FIX, DOCS, STYLE, REFACTOR, TEST, CHORE
   Example: [FEAT] Add quantum entanglement visualization
   ```

2. **Description Template**
   ```markdown
   ## Description
   Brief description of changes
   
   ## Related Issue
   Fixes #123
   
   ## Type of Change
   - [ ] Bug fix
   - [ ] New feature
   - [ ] Documentation update
   - [ ] Performance improvement
   
   ## Testing
   - [ ] Unit tests pass
   - [ ] Integration tests pass
   - [ ] Manual testing completed
   
   ## Screenshots (if applicable)
   
   ## Checklist
   - [ ] Code follows style guidelines
   - [ ] Self-review completed
   - [ ] Documentation updated
   - [ ] Tests added/updated
   ```

3. **Review Process**
   - At least one approval required
   - CI/CD must pass
   - No merge conflicts
   - Documentation updated

### After Merge

- Delete your feature branch
- Update your local main branch
- Consider working on another issue!

## Community

### Communication Channels

- **GitHub Issues**: Bug reports and feature requests
- **Discussions**: General questions and ideas
- **Discord**: Real-time chat (link in README)
- **Email**: taec-team@example.com

### Getting Help

- Check the documentation first
- Search existing issues
- Ask in discussions
- Join our Discord server

### Recognition

We value all contributions! Contributors are:
- Listed in CONTRIBUTORS.md
- Mentioned in release notes
- Given credit in documentation

## Development Tips

### Debugging

```python
# Use structured logging
import structlog
logger = structlog.get_logger()

logger.info("Processing node", node_id=node.id, state=node.state)

# Debug MSC-Lang compilation
from taec_v3.compiler import MSCLCompiler

compiler = MSCLCompiler(debug=True)
compiled, errors, warnings = compiler.compile(source_code)
```

### Performance

```python
# Profile code execution
import cProfile
import pstats

profiler = cProfile.Profile()
profiler.enable()

# Your code here
result = expensive_operation()

profiler.disable()
stats = pstats.Stats(profiler).sort_stats('cumulative')
stats.print_stats(10)  # Top 10 functions
```

### Common Issues

1. **Import Errors**
   - Ensure virtual environment is activated
   - Check PYTHONPATH
   - Verify all dependencies installed

2. **Test Failures**
   - Update test fixtures
   - Check for race conditions
   - Verify mock configurations

3. **Performance Issues**
   - Profile before optimizing
   - Use caching appropriately
   - Consider async operations

## Thank You! 🙏

Your contributions make TAEC better for everyone. We appreciate your time and effort in improving this project.

Happy coding! 🚀