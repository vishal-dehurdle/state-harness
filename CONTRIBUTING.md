# Contributing to state-harness

Thanks for considering a contribution. This document covers setup, style, and process.

## Development Environment

### Prerequisites

- Python ≥ 3.10
- Rust toolchain ([rustup.rs](https://rustup.rs/))
- [maturin](https://github.com/PyO3/maturin) ≥ 1.5

### Setup

```bash
git clone https://github.com/vishal-dehurdle/state-harness.git
cd state-harness

python -m venv .venv && source .venv/bin/activate

pip install maturin pytest
maturin develop --release
```

### Running Tests

```bash
pytest tests/
```

Tests cover the full Rust↔Python interface: Lyapunov monitor, RG decimator, holographic engine, growth ratio guard, and failure diagnostics.

## Code Style

### Rust (`src/`)

- Follow standard `rustfmt` formatting
- Use `///` doc comments on all public items
- Section dividers (`// ─── Section ───`) are fine for organizing large files
- Keep comments focused on *why*, not *what* — the code should be self-explanatory

### Python (`python/`, `tests/`)

- Follow [PEP 8](https://peps.python.org/pep-0008/)
- Type hints on all public functions
- Docstrings on all public classes and methods
- Section dividers (`# ─── Section ───`) are fine for organizing large files

## Pull Requests

1. Fork the repo and create a branch from `main`
2. Make your changes with tests
3. Ensure `pytest tests/` passes
4. Ensure `cargo clippy` reports no warnings
5. Open a PR with a clear description of what and why

## License

By contributing, you agree that:

- **Rust contributions** (`src/`) are licensed under BSL 1.1 (converts to Apache 2.0 on May 26, 2030)
- **Python contributions** (`python/`, `tests/`, `examples/`, `benchmarks/`) are licensed under Apache 2.0

## Questions?

Open a [Discussion](https://github.com/vishal-dehurdle/state-harness/discussions) or file an [Issue](https://github.com/vishal-dehurdle/state-harness/issues).
