## Description

Please include a summary of the changes and the motivation behind them. Include relevant motivation and context.

List any dependencies that are required for this change.

Fixes/Closes # (issue number)

## Type of Change

Please mark the relevant options:

- [ ] **Bug fix** (non-breaking change which fixes an issue)
- [ ] **New feature** (non-breaking change which adds functionality)
- [ ] **Breaking change** (fix or feature that would cause existing functionality to not work as expected)
- [ ] **Documentation update** (non-breaking change to docs, README, or guides)
- [ ] **Refactoring / Cleanup** (non-breaking change to improve code quality or structure)

## How Has This Been Tested?

Please describe the tests that you ran to verify your changes. Provide instructions so we can reproduce.

```bash
# Example test command:
pytest tests/
# Or if running Cargo tests:
# cargo test
```

## Technical Checklist

Before submitting this PR, please ensure the following:

- [ ] My code follows the style guidelines of this project (see [CONTRIBUTING.md](file:///CONTRIBUTING.md)).
- [ ] For **Rust** code:
  - [ ] Code is formatted using `cargo fmt`.
  - [ ] `cargo clippy` runs cleanly with no warnings.
- [ ] For **Python** code:
  - [ ] Code is typed and uses type hints for all public functions/methods.
  - [ ] Code is formatted and PEP 8 compliant.
- [ ] I have performed a self-review of my own code.
- [ ] I have commented my code, particularly in hard-to-understand areas.
- [ ] I have made corresponding changes to the documentation (like `README.md` or `SECURITY.md`).
- [ ] My changes generate no new warnings or build errors.
- [ ] I have added tests that prove my fix is effective or that my feature works.
- [ ] New and existing unit tests pass locally with my changes (`pytest tests/`).

## Contributions Licensing

By checking this box, I confirm that:
- [ ] My contributions to **Rust** source code (`src/`) are licensed under BSL 1.1 (Business Source License).
- My contributions to **Python** source code (`python/`, `tests/`, `examples/`, `benchmarks/`) are licensed under the Apache License 2.0.
