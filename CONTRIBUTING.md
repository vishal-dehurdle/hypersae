# Contributing to HyperSAE

Thank you for your interest in contributing to `hypersae`! We welcome bug reports, documentation updates, theoretical enhancements, and performance optimizations.

---

## Development Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/vishal-dehurdle/hypersae.git
   cd hypersae
   ```

2. **Create a virtual environment & install in editable mode:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -e ".[dev]"
   ```

3. **Run the unit test suite:**
   ```bash
   pytest
   ```

---

## Submitting Pull Requests

1. **Fork & Branch:** Create a feature branch off `main`:
   ```bash
   git checkout -b feat/your-feature-name
   ```
2. **Code Style:** Ensure clean PyTorch code formatting with type annotations.
3. **Tests:** Add unit tests under `tests/` for any new geometry, engine, or hook functions. Verify all 19 existing tests pass cleanly (`pytest`).
4. **Pull Request:** Push your branch to your fork and submit a PR to `vishal-dehurdle/hypersae`.

---

## Reporting Issues

- **Bug Reports:** Provide a minimal reproducible example (including PyTorch version and hardware environment).
- **Feature Requests:** Outline the proposed API signature and theoretical motivation.
