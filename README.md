# TriLibGo

**TriLibGo** is a reinforcement learning research project for **Three-Liberty Go**
(三气围棋) — an original Go variant played on the vertices of a hexagonal board.
Interior stones have three liberties instead of four, creating a faster, sharper
capture dynamic. The project includes a **C++ game engine**, a **Qt desktop
application** for human play and replay analysis, and a **PyTorch RL training
pipeline** (AlphaZero-lite with PUCT MCTS) designed for CPU-only experimentation.

## Project Status

The core game engine and Qt desktop app are functional and stable. The Python RL
training pipeline is an **active research experiment** and has not yet achieved
strong play. It is published for transparency and collaboration; expect rough
edges. See [docs/design.md](docs/design.md) for architectural rationale and known
issues.

## Project Layout

- `include/trilibgo/core`, `src/core`: board topology, game state, move legality, capture, ko, scoring, and record serialization
- `include/trilibgo/ai`, `src/ai`: stable interfaces for local agents and feature encoding
- `include/trilibgo/app`, `src/app`: Qt Widgets demo for local two-player play
- `tests`: core regression tests

## Build

**Windows (PowerShell):**

```powershell
# Core library and tests (no Qt required)
cmake -S . -B build -DTRILIBGO_BUILD_APP=OFF -DTRILIBGO_BUILD_TESTS=ON
cmake --build build --config Debug --target trilibgo_tests
ctest --test-dir build -C Debug --output-on-failure

# Qt desktop app (requires Qt 6)
cmake -S . -B build-app -DTRILIBGO_BUILD_APP=ON -DTRILIBGO_BUILD_TESTS=ON -DCMAKE_PREFIX_PATH=<Qt6-root>
cmake --build build-app --config Debug --target trilibgo_app
```

**Linux / WSL (bash):**

```bash
# Core library and tests (no Qt required)
cmake -S . -B build -DTRILIBGO_BUILD_APP=OFF -DTRILIBGO_BUILD_TESTS=ON
cmake --build build --target trilibgo_tests
ctest --test-dir build --output-on-failure

# Qt desktop app (requires Qt 6)
cmake -S . -B build-app -DTRILIBGO_BUILD_APP=ON -DTRILIBGO_BUILD_TESTS=ON -DCMAKE_PREFIX_PATH=<Qt6-root>
cmake --build build-app --target trilibgo_app
```

If CMake cannot find Qt, set `CMAKE_PREFIX_PATH` or `Qt6_DIR` to the Qt 6
installation that contains `Qt6Config.cmake`.

**ONNX Runtime (optional):** Set `ONNXRUNTIME_ROOT` to enable C++ model
inference. The library path is auto-detected per platform (`.lib` on Windows,
`.so` on Linux, `.dylib` on macOS).

## Current Rules

- Board shape: hexagonal honeycomb region
- Play points: board vertices
- Liberties: interior stones have 3 liberties
- Turns: one stone per turn
- End: two consecutive passes
- Scoring: Chinese-style area scoring with configurable komi
- Ko: simple ko via immediate board repetition check
- Suicide: disabled by default

See [docs/board-rules.md](docs/board-rules.md) for a detailed rules reference.

##
## AI Hooks

- `trilibgo::ai::IAgent`: pluggable move selection API
- `trilibgo::ai::FeatureEncoder`: converts the board state into stable plane features
- `trilibgo::ai::ActionCodec`: stable action-index mapping for policy heads and exported models
- `trilibgo::ai::IAnalysisModel`: future C++ inference interface for winrate and move-probability analysis
- `trilibgo::core::GameRecord`: serializes move history for replay or dataset export

These APIs are intentionally independent of Qt so future local self-play, MCTS, or neural training code can run headless.

## RL Research Stack

The research pipeline lives under `python/rl` and is designed for CPU-first
experimentation. See [python/rl/README.md](python/rl/README.md) for the subsystem
overview.

### Environment Setup

```bash
# Option 1: Conda (recommended)
conda env create --prefix .conda/trilibgo-rl --file python/environment-train.yml
# Then use: .conda/trilibgo-rl/bin/python (Linux) or .conda\trilibgo-rl\python.exe (Windows)

# Option 2: pip
pip install -r python/requirements-train.txt
```

### Running Training

```bash
# Smoke test (validates the full pipeline quickly)
python -m python.rl.train --config python/rl/configs/tiny_cpu_smoke.json

# Baseline training
python -m python.rl.train --config python/rl/configs/tiny_cpu_baseline.json

# Resume from checkpoint
python -m python.rl.train --config python/rl/configs/tiny_cpu_baseline.json --resume artifacts/checkpoints/tiny_cpu_baseline_iter3.pt
```

Stage-specific launch scripts are provided at the repo root:
`run_stage1_rl.sh` / `run_stage2_rl.sh` (and `.cmd` equivalents for Windows).
Each accepts an optional checkpoint path for resume.

### Validation

```bash
python python/tools/verify_record.py records/r1.tgo
python python/tools/verify_fixtures.py
python python/tools/verify_position_fixtures.py
```

### Running Tests

```bash
python -m unittest discover python/rl/tests
python -m unittest discover python/tools/tests
```
