# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Test Commands

**Windows (PowerShell):**
```powershell
# C++ core library + tests (no Qt required)
cmake -S . -B build -DTRILIBGO_BUILD_APP=OFF -DTRILIBGO_BUILD_TESTS=ON
cmake --build build --config Debug --target trilibgo_tests
ctest --test-dir build -C Debug --output-on-failure

# C++ app (requires Qt 6)
cmake -S . -B build-app -DTRILIBGO_BUILD_APP=ON -DTRILIBGO_BUILD_TESTS=ON -DCMAKE_PREFIX_PATH=<Qt6-root>
cmake --build build-app --config Debug --target trilibgo_app

# ONNX Runtime — set ONNXRUNTIME_ROOT to enable C++ inference
cmake -S . -B build -DTRILIBGO_BUILD_APP=OFF -DTRILIBGO_BUILD_TESTS=ON -DONNXRUNTIME_ROOT=<path>
```

**Linux (bash):**
```bash
# C++ core library + tests (no Qt required)
cmake -S . -B build -DTRILIBGO_BUILD_APP=OFF -DTRILIBGO_BUILD_TESTS=ON
cmake --build build --target trilibgo_tests
ctest --test-dir build --output-on-failure

# C++ app (requires Qt 6)
cmake -S . -B build-app -DTRILIBGO_BUILD_APP=ON -DTRILIBGO_BUILD_TESTS=ON -DCMAKE_PREFIX_PATH=<Qt6-root>
cmake --build build-app --target trilibgo_app

# ONNX Runtime
cmake -S . -B build -DTRILIBGO_BUILD_APP=OFF -DTRILIBGO_BUILD_TESTS=ON -DONNXRUNTIME_ROOT=<path>
```

CMake options: `TRILIBGO_BUILD_APP` (default ON), `TRILIBGO_BUILD_TESTS` (default ON), `TRILIBGO_ENABLE_ONNXRUNTIME` (default ON, requires `ONNXRUNTIME_ROOT`).

C++ tests are registered manually in `tests/core_tests.cpp` `main()` — there is no test framework with `--gtest_filter`-style filtering. To run a single test, comment out the other `test_*()` calls in `main()` and rebuild, or add command-line argument parsing to that file.

### Python environment

```bash
# Create conda environment (preferred, works on both platforms)
conda env create --prefix .conda/trilibgo-rl --file python/environment-train.yml

# Windows fallback
.\.conda\trilibgo-rl\python.exe -m pip install numpy onnx onnxruntime torch

# Linux fallback
.conda/trilibgo-rl/bin/python -m pip install numpy onnx onnxruntime torch
```

### Python RL

```bash
# Run training — use Conda python if available, else system python
.conda/trilibgo-rl/bin/python -m python.rl.train --config python/rl/configs/tiny_cpu_smoke.json

# Resume from checkpoint
.conda/trilibgo-rl/bin/python -m python.rl.train --config python/rl/configs/stage1_cpu_i5_12500h.json --resume artifacts/checkpoints/stage1_iter3.pt

# Run a single test file
.conda/trilibgo-rl/bin/python -m unittest python.rl.tests.test_endgame
.conda/trilibgo-rl/bin/python -m unittest python.rl.tests.test_curriculum

# Run all Python tests
.conda/trilibgo-rl/bin/python -m unittest discover python/rl/tests
.conda/trilibgo-rl/bin/python -m unittest discover python/tools/tests

# Fixture and record validation
.conda/trilibgo-rl/bin/python python/tools/verify_fixtures.py
.conda/trilibgo-rl/bin/python python/tools/verify_position_fixtures.py
.conda/trilibgo-rl/bin/python python/tools/verify_record.py records/r1.tgo
```

On Windows, replace `.conda/trilibgo-rl/bin/python` with `.conda\trilibgo-rl\python.exe`.

Stage-specific training is launched via scripts at repo root:
- `.cmd` for Windows, `.sh` for Linux
- `run_stage1_rl` / `run_stage2_rl` / `run_stage2_curriculum_rl` / `run_stage3_rl` / `run_stage4_rl` / `run_stage5_side3_rl`
- Each accepts an optional checkpoint path for resume.

## Architecture

### C++ side

Two static libraries, public API under `include/trilibgo/`, implementations under `src/`:

- **`trilibgo_core`** (`src/core/`) — `BoardTopology` (hexagonal vertex graph, `neighbors()`, `adjacency`), `GameState` (board, captures, passes, ko hash, phase machine), `RulesEngine` (move validation, capture resolution, ko detection via immediate board repetition, Chinese area scoring, endgame review), `GameRecord` (text serialization of move history + metadata).
- **`trilibgo_ai`** (`src/ai/`) — `IAgent` (pluggable `select_move` interface), `FeatureEncoder` (3-plane encoding: black/empty/white), `ActionCodec` (vertex → policy-index; policy_size = vertices + 1 for pass), `IAnalysisModel` / `OnnxAnalysisModel` (inference wrapper gated by `TRILIBGO_HAS_ONNXRUNTIME`), `RandomAgent`.

Source layout mirrors headers: `src/core/` for core implementations, `src/ai/` for AI, `src/app/` for the Qt application.

The Qt app (`trilibgo_app`, `src/app/`) links both libraries. `GameController` orchestrates the rules engine + optional AI agents; `BoardWidget` renders the hexagonal board; `MainWindow` wires UI actions. `replay_compat.cpp` maps self-play trace JSON (from the Python stack) into app-side replay overlays including policy heatmaps.

### Python RL stack (`python/rl`)

The training loop (`train.py`, ~62KB) is the orchestrator. Each iteration:
1. **Self-play** (`selfplay.py`): multi-process MCTS with the current model, producing game trajectories with per-move policy vectors and terminal outcomes.
2. **Replay buffer** (`replay_buffer.py`): ring buffer of recent positions, gated by `sample_weight` so abnormal games can be down-weighted rather than discarded.
3. **Training**: supervised policy loss + auxiliary value/ownership/score_margin heads, curriculum-based reward shaping, optional LR warmup+cosine decay.
4. **Evaluation** (`eval.py`): periodic promotion matches against the incumbent best model, gated by `evaluation.interval` and `evaluation.interval_schedule`.
5. **Checkpointing**: saves candidate model + optimizer state; promoted best model is exported to ONNX.

Key config mechanisms (in `config.py` and JSON configs):
- `mcts.simulations` for self-play, `evaluation.simulations` for cheaper promotion gates
- `rules.opening_no_pass_moves` disables pass during opening
- `rules.cleanup_dead_stones` + `cleanup_dead_stones_mode` controls endgame dead-stone removal
- `selfplay.sampling_decay_moves` anneals move temperature from opening to `sampling_final_temperature`
- `mcts.selfplay_exploration_scale` / `selfplay_exploration_decay_iterations` broaden early search
- `mcts.pass_prior_scale` soft-suppresses pass during active board play
- `training.curriculum` block controls phased reward shaping (capture bonus, response weighting, false-eye penalty)
- `training.endgame_confidence_weighting` down-weights noisy terminal labels while keeping policy weight full
- `komi_adjust_*` keys control auto komi adjustment to maintain first-player win-rate balance

Training stages represent progressive refinements:
- **Stage 1**: Baseline — small MCTS, basic policy/value, smoke-test convergence
- **Stage 2**: Graph backbone, multi-head model (policy/value/ownership/score), curriculum, komi auto-adjust
- **Stage 3**: Training strategy overhaul — metric-driven curriculum stop, endgame confidence weighting, better teacher quality; same model architecture as Stage 2
- **Stage 4**: 2D conv residual backbone (replaces graph), fixes to terminal supervision weights, simplified curriculum value bonus
- **Stage 5**: `board_side=3` experiments with Stage 4 architecture

Stage N must never reuse Stage N-1 checkpoints or replay buffers. Each stage has its own config and artifact directories.

### Cross-stack data flow

Self-play traces → JSONL → `artifacts/selfplay/` → Qt app replays via `replay_compat.cpp` for visual review. Models → ONNX export → C++ `OnnxAnalysisModel` for in-app inference. Position fixtures in `records/` are validated by both C++ tests and Python tools to ensure rule parity.

## Configuration management

Training configs live in `python/rl/configs/`. Configs are standalone JSON files with all parameters inlined. Each stage has its own config and artifact directories. See [docs/design.md](docs/design.md) for training strategy rationale and known issues.

## Naming conventions

- C++: `snake_case` functions/variables, `PascalCase` classes, namespace `trilibgo::core` / `trilibgo::ai`
- Python: `snake_case` files, `unittest` for tests, config-driven behavior
- C++20, 4-space indent; no Qt dependency in core/ai libraries
- C++ tests: `test_<behavior>()` functions registered manually in `tests/core_tests.cpp` `main()`
- Python tests: `test_*.py` files with `unittest.TestCase`

## Things to avoid

- Don't commit conda environments, Qt/ONNX SDKs, or training checkpoints — these are local outputs
- Generated artifacts belong in `artifacts/`, `build*`, or `.conda/`
- Don't introduce Qt dependencies into `trilibgo_core` or `trilibgo_ai`
- Stage N training must never reuse Stage N-1 checkpoints or replay buffers
- C++ build configuration uses CMake variables (`CMAKE_PREFIX_PATH`, `ONNXRUNTIME_ROOT`), not hardcoded paths
