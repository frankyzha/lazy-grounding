# Contributing

1. Create a focused branch from `main`.
2. Install the development environment with `uv sync --all-extras`.
3. Run `make check` before opening a pull request.
4. Keep benchmark data, traces, model outputs, and credentials out of Git.
5. Document any behavioral change to the paper protocol in the pull request.

Bug fixes should include a regression test. Changes to scoring or metrics must
include a fixture demonstrating the intended label or estimate.
