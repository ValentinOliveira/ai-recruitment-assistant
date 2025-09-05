# Contributing to AI Recruitment Assistant

We love your input! We want to make contributing to this project as easy and transparent as possible, whether it's:

- Reporting a bug
- Discussing the current state of the code
- Submitting a fix
- Proposing new features
- Becoming a maintainer

## Development Process

We use GitHub to host code, to track issues and feature requests, as well as accept pull requests.

1. Fork the repo and create your branch from `main`.
2. If you've added code that should be tested, add tests.
3. If you've changed APIs, update the documentation.
4. Ensure the test suite passes.
5. Make sure your code lints.
6. Issue that pull request!

## Code Style

* We use [Black](https://github.com/psf/black) for code formatting
* We use [isort](https://github.com/PyCQA/isort) for import sorting
* We use [flake8](https://flake8.pycqa.org/) for linting
* We use [mypy](http://mypy-lang.org/) for type checking

Before submitting your code, please run:

```bash
# Format code
black .

# Sort imports
isort .

# Lint code
flake8 src/ tests/

# Type check
mypy src/

# Run tests
pytest tests/
```

## Setting Up Development Environment

1. Clone the repository:
```bash
git clone https://github.com/your-username/ai-recruitment-assistant.git
cd ai-recruitment-assistant
```

2. Create and activate a virtual environment:
```bash
conda env create -f environment.yml
conda activate recruitment-assistant
```

3. Install development dependencies:
```bash
pip install -r requirements.txt
pip install pytest pytest-cov black isort flake8 mypy
```

## Testing

We use pytest for testing. Please ensure your changes include appropriate tests:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_dataset_builder.py
```

## Submitting Changes

1. **Create an Issue**: For major changes, please create an issue first to discuss what you would like to change.

2. **Branch Naming**: Use descriptive branch names:
   - `feature/add-new-agent`
   - `bugfix/fix-memory-leak`
   - `docs/update-readme`

3. **Commit Messages**: Use clear and meaningful commit messages:
   - `feat: add new email agent functionality`
   - `fix: resolve GPU memory allocation issue`
   - `docs: update installation instructions`

4. **Pull Request**: 
   - Fill out the PR template completely
   - Link to any related issues
   - Include screenshots for UI changes
   - Ensure all checks pass

## Pull Request Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Code refactoring

## Testing
- [ ] Tests pass locally
- [ ] Added new tests for new functionality
- [ ] Updated documentation

## Checklist
- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Changes generate no new warnings
- [ ] Any dependent changes merged
```

## Code of Conduct

### Our Pledge
We pledge to make participation in our project a harassment-free experience for everyone, regardless of age, body size, disability, ethnicity, sex characteristics, gender identity and expression, level of experience, education, socio-economic status, nationality, personal appearance, race, religion, or sexual identity and orientation.

### Our Standards
Examples of behavior that contributes to creating a positive environment include:
* Using welcoming and inclusive language
* Being respectful of differing viewpoints and experiences
* Gracefully accepting constructive criticism
* Focusing on what is best for the community
* Showing empathy towards other community members

## Questions?

Don't hesitate to ask! You can:
1. Open an issue with the `question` label
2. Start a discussion in the GitHub Discussions tab
3. Contact the maintainers directly

Thank you for contributing! 🚀
