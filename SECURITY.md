# Security Policy

## Reporting a Vulnerability

If you discover a security vulnerability in state-harness, please report it responsibly.

**Do not open a public GitHub issue.**

Instead, email: **vishal.verma@dehurdle.com** with:

- A description of the vulnerability
- Steps to reproduce
- Potential impact assessment
- Any suggested fixes (optional)

You will receive acknowledgment within 48 hours and a detailed response within 5 business days.

## Scope

Security issues include:

- Vulnerabilities in the Rust compute engine (`src/`) that could lead to memory safety issues
- Bypasses of the circuit breaker / stability monitor that could allow uncontrolled token spend
- Injection or escalation vectors through the Python SDK
- Dependency vulnerabilities in `Cargo.toml` or `pyproject.toml` dependencies

Out of scope:

- Issues in benchmark scripts (these are development tools, not production code)
- Performance regressions (file a regular issue instead)
- Feature requests

## Supported Versions

| Version | Supported |
|:--------|:---------:|
| 0.2.x   | ✅        |
| < 0.2   | ❌        |

## Disclosure Policy

We follow coordinated disclosure. We will:

1. Acknowledge your report within 48 hours
2. Work with you to understand and validate the issue
3. Develop and test a fix
4. Release a patched version
5. Credit you in the release notes (unless you prefer anonymity)
