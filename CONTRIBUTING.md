# Contributing to TriLibGo

## Getting Started

1. Fork the repository and clone it.
2. Build the C++ core and run tests:
   ```bash
   cmake -S . -B build -DTRILIBGO_BUILD_APP=OFF -DTRILIBGO_BUILD_TESTS=ON
   cmake --build build --target trilibgo_tests
   ctest --test-dir build --output-on-failure
   ```
3. Run the Python RL smoke test:
   ```bash
   pip install -r python/requirements-train.txt
   python -m python.rl.train --config python/rl/configs/tiny_cpu_smoke.json
   ```

## Areas for Contribution

- **Game engine** (`src/core/`, `include/trilibgo/core/`): Bug fixes, rule
  refinements, better dead-stone analysis, test coverage.
- **RL training** (`python/rl/`): Architecture experiments, hyperparameter
  tuning, new auxiliary heads, better endgame resolution, alternative search
  algorithms.
- **Desktop app** (`src/app/`): UI polish, analysis overlays, replay features,
  ONNX model integration.
- **Documentation**: Docstrings, tutorials, architecture guides, config
  reference.

## Coding Conventions

- **C++**: C++20, `snake_case` functions/variables, `PascalCase` classes,
  4-space indent, `#pragma once` for headers. Namespaces `trilibgo::core`
  and `trilibgo::ai`. No Qt dependencies in core/ai libraries.
- **Python**: Python 3.10+, `snake_case` files, type hints where useful,
  `unittest` for tests, config-driven behavior.
- **Tests**: C++ tests go in `tests/core_tests.cpp` as `test_<name>()` functions
  registered in `main()`. Python tests go in `python/rl/tests/` or
  `python/tools/tests/` as `test_*.py` files with `unittest.TestCase`.

## Pull Request Process

1. Run `ctest` for C++ changes.
2. Run `python -m unittest discover python/rl/tests` and
   `python -m unittest discover python/tools/tests` for Python changes.
3. Run `python python/tools/verify_fixtures.py` if rule logic changed.
4. Keep generated artifacts (checkpoints, ONNX files, self-play traces) out of
   commits.

## Project Status

The C++ game engine and Qt desktop app are functional. The Python RL training
pipeline is **experimental** and has not yet produced strong play. See
[docs/design.md](docs/design.md) for known issues and architectural rationale.
