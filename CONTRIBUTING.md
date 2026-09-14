# Contributing to AARAG

Thank you for your interest in contributing to AARAG! This document provides guidelines and instructions for contributing.

---

## Table of Contents

- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Code Style](#code-style)
- [Making Changes](#making-changes)
- [Testing](#testing)
- [Pull Request Process](#pull-request-process)
- [Reporting Issues](#reporting-issues)

---

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/your-username/AARAG.git`
3. Create a branch: `git checkout -b feature/your-feature-name`
4. Make your changes
5. Push and create a Pull Request

---

## Development Setup

### Prerequisites

- Python 3.10+
- pip or conda
- GPU recommended (RTX 3060+ or A100)

### Installation

```bash
# Create virtual environment
python -m venv aarag-env
source aarag-env/bin/activate  # Linux/Mac
# aarag-env\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Install development dependencies
pip install pytest pytest-cov black ruff mypy

# Verify installation
python -m pytest tests/ -v
```

### Pre-commit Hooks

```bash
pip install pre-commit
pre-commit install
```

This will run linting and formatting checks before each commit.

---

## Code Style

### Python Style

- Follow [PEP 8](https://peps.python.org/pep-0008/) conventions
- Use type hints for all function signatures
- Maximum line length: 100 characters
- Use docstrings for all public classes and methods (Google style)

### Docstring Format

```python
def my_function(param1: str, param2: int) -> dict:
    """Short description of the function.

    Longer description if needed, explaining the purpose,
    algorithm, or important details.

    Args:
        param1: Description of param1
        param2: Description of param2

    Returns:
        Dict with keys 'key1' and 'key2'

    Raises:
        ValueError: If param1 is empty
    """
    pass
```

### Naming Conventions

- **Classes:** PascalCase (`AdaptiveRouter`, `VectorStore`)
- **Functions/methods:** snake_case (`route_query`, `add_documents`)
- **Constants:** UPPER_SNAKE_CASE (`MAX_RETRIES`, `DEFAULT_EMBEDDING_MODEL`)
- **Private methods:** Leading underscore (`_load_config`, `_get_llm_fn`)

### Type Hints

```python
from typing import Optional, Dict, List, Any

def process_query(
    query: str,
    top_k: int = 5,
    enable_web: bool = True,
) -> Dict[str, Any]:
    """Process a query and return results."""
    pass
```

---

## Making Changes

### Branch Naming

- `feature/description` — New features
- `fix/description` — Bug fixes
- `docs/description` — Documentation changes
- `refactor/description` — Code refactoring
- `test/description` — Test additions/changes

### Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
type(scope): description

[optional body]

[optional footer]
```

**Types:**
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation
- `style`: Formatting, missing semicolons, etc.
- `refactor`: Code refactoring
- `test`: Adding tests
- `chore`: Maintenance tasks

**Examples:**
```
feat(router): add multi-label classification support
fix(crag): handle empty document list in evaluator
docs(api): update VectorStore API reference
test(reflection): add edge case tests for claim verifier
```

---

## Testing

### Running Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run with coverage
python -m pytest tests/ -v --cov=src --cov-report=html

# Run specific test file
python -m pytest tests/test_router.py -v

# Run specific test
python -m pytest tests/test_router.py::TestFeatureExtractor::test_simple_features -v
```

### Writing Tests

- Place tests in the `tests/` directory
- Name test files `test_*.py`
- Use descriptive test names that explain what is being tested
- Use pytest fixtures for common setup
- Mock external dependencies (models, APIs)

**Example:**

```python
import pytest
from src.layers.adaptive_router import AdaptiveRouter

class TestAdaptiveRouter:
    """Tests for the AdaptiveRouter."""

    @pytest.fixture
    def router(self):
        """Create a router with rule-based fallback."""
        return AdaptiveRouter()

    def test_simple_query_returns_low_complexity(self, router):
        """Simple factual queries should be classified as low complexity."""
        decision = router.route("What is the capital of France?")
        assert decision.level <= 1
        assert decision.confidence > 0

    def test_multi_hop_query_returns_high_complexity(self, router):
        """Multi-hop queries should be classified as higher complexity."""
        decision = router.route("Compare Python and Java programming languages")
        assert decision.level >= 2
```

### Test Coverage

Aim for:
- **Unit tests:** 80%+ coverage for core modules
- **Integration tests:** All layer interactions
- **Edge cases:** Empty inputs, large inputs, error conditions

---

## Pull Request Process

### Before Submitting

1. **Run tests:** `python -m pytest tests/ -v`
2. **Check linting:** `ruff check src/ tests/`
3. **Format code:** `black src/ tests/`
4. **Type check:** `mypy src/`
5. **Update docs:** If adding/changing features, update relevant documentation
6. **Update CHANGELOG.md:** Add entry for your changes

### PR Template

```markdown
## Description
Brief description of changes.

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Refactoring
- [ ] Test addition

## Testing
- [ ] All existing tests pass
- [ ] New tests added for new functionality
- [ ] Manual testing performed

## Checklist
- [ ] Code follows project style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] CHANGELOG.md updated
```

### Review Process

1. All PRs require at least one review
2. CI must pass (tests, linting, type checking)
3. Documentation must be updated for user-facing changes
4. Changes must be backward-compatible unless explicitly discussed

---

## Reporting Issues

### Bug Reports

Include:
- Python version
- OS and architecture
- GPU type (if relevant)
- Steps to reproduce
- Expected behavior
- Actual behavior
- Error messages/tracebacks
- Relevant configuration

### Feature Requests

Include:
- Use case description
- Proposed solution
- Alternatives considered
- Impact on existing functionality

---

## Code of Conduct

- Be respectful and inclusive
- Focus on constructive feedback
- Assume good intentions
- Help others learn and grow

---

## Questions?

Open an issue with the `question` label or reach out to the maintainers.
