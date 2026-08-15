---
name: Bug report
about: Report a bug in state-harness
title: "[BUG] "
labels: bug
---

**Describe the bug**
A clear description of what the bug is.

**To reproduce**
```python
# Minimal code to reproduce the issue
from state_harness import GrowthRatioGuard

guard = GrowthRatioGuard(token_budget=50_000)
# ...
```

**Expected behavior**
What you expected to happen.

**Actual behavior**
What actually happened. Include the full error traceback if applicable.

**Environment**
- OS: [e.g., macOS 15, Ubuntu 24.04]
- Python: [e.g., 3.12]
- state-harness version: [e.g., 0.2.0]
- Install method: [pip / maturin develop]
