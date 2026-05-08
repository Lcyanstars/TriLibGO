# Repository Guidelines

## Project Structure & Module Organization

TriLibGo is a C++20 Go-like engine with a Qt Widgets demo and a Python RL stack. Public C++ headers live in `include/trilibgo`, with matching implementations in `src`. Core rules, state, topology, scoring, and records are under `core`; agent interfaces, feature encoding, ONNX integration, and random play are under `ai`; the desktop UI is under `app`. C++ regression coverage is in `tests/core_tests.cpp`. Python training, evaluation, export, and self-play code lives in `python/rl`, with unit tests in `python/rl/tests` and `python/tools/tests`. Rule fixtures and sample records are in `records`; generated checkpoints, reports, exports, and self-play traces belong in `artifacts`. Treat `build*`, `.conda`, `__pycache__`, and generated artifacts as local outputs.

## Build, Test, and Development Commands

- `cmake -S . -B build -DTRILIBGO_BUILD_APP=OFF -DTRILIBGO_BUILD_TESTS=ON`: configure core libraries and C++ tests without Qt.
- `cmake --build build --config Debug --target trilibgo_tests`: build the C++ test executable.
- `ctest --test-dir build -C Debug --output-on-failure`: run registered C++ tests.
- `cmake -S . -B build-app -DTRILIBGO_BUILD_APP=ON -DTRILIBGO_BUILD_TESTS=ON -DCMAKE_PREFIX_PATH=<Qt6-root>`: configure the Qt app when Qt 6 is installed.
- `python -m unittest discover python/rl/tests`: run RL unit tests.
- `python -m unittest discover python/tools/tests`: run tool/report tests.
- `python python/tools/verify_fixtures.py` and `python python/tools/verify_position_fixtures.py`: validate fixture parity.
- `python -m python.rl.train --config python/rl/configs/tiny_cpu_smoke.json`: run the smallest end-to-end training smoke test.

## Coding Style & Naming Conventions

Use C++20, four-space indentation, `snake_case` functions and variables, `PascalCase` classes, and namespaces such as `trilibgo::core`. Keep headers in `include/trilibgo/<module>` and implementations in `src/<module>`. Python files use `snake_case.py`, standard `unittest`, type hints where useful, and config-driven behavior under `python/rl/configs`.

## Testing Guidelines

Add C++ rule or encoder regressions to `tests/core_tests.cpp` using `test_<behavior>()` helpers and `expect(...)`. Add Python tests as `test_*.py` with `unittest.TestCase`. Prefer deterministic fixture tests for rules, records, encoders, reports, and training bookkeeping before running longer RL experiments.

## Commit & Pull Request Guidelines

This snapshot has no `.git` metadata, so no historical commit convention is observable. Use short imperative commit subjects such as `Add endgame fixture validation`, and keep generated artifacts out unless they are intentional fixtures or reports. Pull requests should describe behavior changes, list commands run, link related issues, and include screenshots for Qt UI changes or report output.

## Security & Configuration Tips

Do not commit local Conda environments, Qt installations, ONNX Runtime SDKs, or large training checkpoints. Pass local paths through CMake variables such as `CMAKE_PREFIX_PATH` and `ONNXRUNTIME_ROOT`.
