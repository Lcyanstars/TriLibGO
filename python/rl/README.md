# TriLibGo RL Stack

This directory contains the CPU-first research pipeline for TriLibGo:

- `config.py`: experiment and model configuration
- `topology.py`, `game.py`: pure Python game environment used for self-play
- `encoder.py`: shared state-to-plane encoding for policy/value learning
- `model.py`: small residual CNN policy-value network
- `model.py`: topology-aware graph residual policy-value network
- `mcts.py`: PUCT MCTS for self-play and evaluation
- `selfplay.py`: self-play game generation
- `replay_buffer.py`: in-memory replay buffer
- `train.py`: training loop and checkpointing
- `eval.py`: model-vs-model evaluation
- `export.py`: ONNX export
- `analysis.py`: per-position policy/winrate/ownership/score analysis

Recommended progression:

1. Train on a small side length with the `tiny` configuration.
2. Verify the model beats the random baseline and earlier checkpoints.
3. Export ONNX and bind it into the desktop app.
4. Scale to a larger board and a `small` network only after throughput is acceptable.

Training artifacts:

- checkpoints are written to `artifacts/checkpoints`
- promoted best snapshots are copied to `artifacts/best`
- promoted best models attempt ONNX export to `artifacts/export`
- metrics include evaluation wins/losses/draws, promotion status, and export status
- telemetry can emit per-game self-play JSONL and an HTML report with loss/winrate/first-player-bias curves
- training progress now reports per-stage terminal progress bars for self-play, SGD steps, and evaluation matches

CLI usage:

```powershell
python -m python.rl.train --config python/rl/configs/tiny_cpu_baseline.json
python -m python.rl.train --config python/rl/configs/tiny_cpu_baseline.json --resume artifacts/checkpoints/tiny_cpu_baseline_iter3.pt
python -m python.rl.train --config python/rl/configs/tiny_cpu_smoke.json
python -m python.rl.train --config python/rl/configs/tiny_cpu_visual_smoke.json
python -m python.rl.train --config python/rl/configs/stage1_cpu_i5_12500h.json
python -m python.rl.train --config python/rl/configs/stage2_priority1_cpu_i5_12500h.json
python -m python.rl.train --config python/rl/configs/stage2_graph_cpu_i5_12500h.json
```

Environment setup:

```powershell
conda env create --prefix .conda/trilibgo-rl --file python/environment-train.yml
conda run --prefix .conda/trilibgo-rl python -m python.rl.train --config python/rl/configs/tiny_cpu_smoke.json
```

Windows fallback if `conda env create` is too slow or runs out of memory during solve:

```powershell
conda create --prefix .conda/trilibgo-rl python=3.10 pip -y
.\.conda\trilibgo-rl\python.exe -m pip install numpy onnx onnxruntime torch
```

Notes:

- use the `tiny_cpu_smoke` config first to validate the full loop on a CPU-only machine
- use `tiny_cpu_visual_smoke` when you want the fastest end-to-end check of telemetry and HTML reporting
- action-space size and ONNX export input shapes are derived from the board topology, so board-size changes do not require manual shape edits
- MCTS, analysis, training, and export all use the configured `input_history`
- encoder history planes now use real board history instead of zero padding
- the policy-value backbone now aggregates over `BoardTopology.adjacency` instead of convolving over vertex index order
- model outputs now include `policy_logits`, `value`, `ownership`, and `score_margin`
- training now optimizes auxiliary `ownership` and `score_margin` heads alongside policy/value
- analysis can emit conservative ownership-based dead-group suggestions for endgame review
- `rules.komi` sets fixed komi; `rules.selfplay_komi` can sample from a small komi set for bias-measurement experiments
- `rules.opening_no_pass_moves` disables `pass` during the opening for both self-play and evaluation
- `rules.cleanup_dead_stones` runs a conservative dead-stone cleanup pass before terminal scoring
- `training.steps_per_iteration` overrides the SGD update count; when left at `0`, the loop uses `epochs_per_iteration * ceil(buffer_size / batch_size)`
- `training.min_steps_per_iteration` adds a floor to the auto-computed SGD budget so a cheap CPU training step can still do meaningful work each iteration
- `training.gradient_clip_norm` clips optimizer gradients for better stability on the larger graph network
- `training.lr_schedule`, `training.lr_warmup_steps`, `training.lr_decay_steps`, and `training.lr_min_ratio` control optional warmup + cosine decay without changing the self-play loop
- `evaluation.simulations` can be set below `mcts.simulations` to make promotion matches cheaper than self-play
- `evaluation.interval` skips promotion matches on most iterations so long overnight runs do not spend most of their wall clock in eval
- `evaluation.interval_schedule` can define a piecewise schedule such as "first 16 iters every 16, then every 8, then every 6"
- `evaluation.full_games_every`, `evaluation.full_games`, and `evaluation.full_simulations` enable a periodic stronger evaluation pass while keeping the default per-iteration gate cheap
- `telemetry.report_path` points to a self-contained HTML report generated after training
- `telemetry.selfplay_dir` stores per-game records including move trace, root value, and top policy actions
- `telemetry.show_progress` controls the terminal progress bars without affecting JSON metrics output
- new selfplay traces also store the full policy vector for each move, which the Qt app can replay as a dense heatmap
- self-play traces also record first pass turn, total pass count, cleaned dead stones, and the game end reason
- self-play move sampling now anneals from the opening temperature down to `selfplay.sampling_final_temperature` over `selfplay.sampling_decay_moves`
- self-play traces now flag `abnormal_tags` such as `early_pass` and `short_game`, and store the per-game `sample_weight`
- replay keeps only samples with positive `sample_weight`, and training losses are weighted so abnormal games can be down-weighted instead of fully discarded
- checkpoints now store the current incumbent model as well as optimizer state, so resume keeps the promotion baseline instead of collapsing to the latest candidate
- the official best model is now a sparse evaluation gate rather than a per-iteration requirement; training and self-play continue with the latest candidate even when eval is skipped
- HTML reports now chart auxiliary losses, learning rate, replay effective size, and recent game cleanup/pass metadata
- `selfplay.workers` is now active: values above `1` use multi-process CPU self-play with one Torch thread per worker
- `mcts.selfplay_exploration_scale` and `mcts.selfplay_exploration_decay_iterations` make early self-play iterations search more broadly, then decay back to the base `c_puct`
- `mcts.pass_prior_scale` down-weights `pass` inside search whenever board plays still exist
- `mcts.consecutive_pass_min_value` and `mcts.consecutive_pass_min_score_margin` stop the search from ending immediately with a reply `pass` while the current player is still predicted to be behind
- evaluation matches now disable root Dirichlet noise so promotion gates are less stochastic than self-play
- `run_stage1_rl.cmd` is the Windows entry point for the formal first-stage CPU training run and accepts an optional checkpoint path for resume
