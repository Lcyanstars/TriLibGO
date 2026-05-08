# Design Notes

## Why AlphaZero-lite on CPU

TriLibGo's RL pipeline targets CPU-only training (e.g., a laptop i5-12500H). This
constrains every design choice:

- **Model size**: channels ≤ 128, residual blocks ≤ 12. GPU-scale architectures
  with 256+ channels and 20+ blocks are off the table.
- **MCTS simulations**: self-play at 64–96 sims/move, evaluation at 24–48. Too few
  for strong play, but enough to produce a training signal.
- **Board size**: `board_side=4` (96 vertices) is the primary target. `side=3`
  (54 vertices) is used for faster iteration.

The AlphaZero-lite approach (policy-value net + PUCT MCTS + self-play) was chosen
over MuZero or offline RL because it is the simplest architecture that still
produces a self-improving loop. The goal is research infrastructure first, strong
play later.

## Stage Progression

Training is organized into stages, each learning from the previous one's failures:

| Stage | Backbone | Key Change | Status |
|-------|----------|------------|--------|
| 1 | Conv1d over vertex index | Baseline: small MCTS, basic policy/value heads | Did not converge well |
| 2 | Graph residual | Multi-head (policy/value/ownership/score/liberty), curriculum, komi auto-adjust | Improved but pass and life-death still broken |
| 3 | Graph residual (same as S2) | Training strategy overhaul: metric-driven curriculum stop, endgame confidence weighting | Marginal improvement |
| 4 | Conv2d residual (4×4 grid) | 2D convolution backbone, fixed terminal supervision weights, simpler value bonus | Current mainline |
| 5 | Conv2d (same as S4) | `board_side=3` experiments | Exploratory |

Stages never reuse checkpoints or replay buffers from previous stages — each is a
clean restart.

## Model Architecture Evolution

### Stage 1–3: Graph Residual Network

The hex board is not a regular grid, so the natural representation is a graph
where vertices are nodes and adjacency edges come from `BoardTopology`. Each graph
residual block does neighbor aggregation + residual update:

```
x' = ReLU(BN(W * x + x @ adjacency))
```

This preserves topology but has downsides:
- Message-passing depth needed to propagate information across the board
- Graph convolution is harder to optimize than 2D convolution

### Stage 4+: 2D Conv Residual Network

The insight: side=4 boards can be reshaped to (4, 4) 2D grids despite the
hexagonal topology. This allows standard Conv2d + residual blocks:

- Backbone: Conv2d + 6× ResidualBlock2d, channels=96
- Policy head: 1×1 conv → board logits + independent global pass logit
- Value head: 1×1 conv → global average pool → MLP → tanh
- Ownership head: per-point Conv2d
- Score head: 1×1 conv → global average pool → MLP

The trade-off: loses exact topology information but gains parameter efficiency
and better generalization across board positions.

## Known Failure Modes

The root cause of most training failures is the **Flatten → Linear policy/value
head** used in Stage 1–3. After the graph backbone learns spatial patterns (e.g.,
"one liberty = danger"), `nn.Flatten()` destroys the spatial layout, so the same
life-death pattern at different board positions must be learned independently.
The ownership head (which keeps Conv1d → per-vertex processing) does not suffer
from this, confirming the backbone works but the head architecture is broken.

Other issues:

1. **Tanh value clipping + curriculum bonus** → gradient vanishing when targets
   saturate at ±1.
2. **Terminal weight down-scaling** on `max_moves` games (weight=0.25) creates a
   feedback loop: bad play → less supervision → worse play.
3. **No entropy regularization** in early stages → policy collapses to
   deterministic moves, oscillating between extremes (all-pass vs. all-capture).
4. **Simple dead-stone cleanup** (≤1 liberty → dead) is inaccurate in capturing
   races, producing noisy ownership/score targets.
5. **Graph normalization** uses asymmetric degree normalization
   (`1/(deg(v)+1)`) instead of the standard symmetric form
   (`1/sqrt(deg(v)*deg(u))`).

## Training Pipeline

```
Self-play (multi-process MCTS)
    → Replay buffer (ring buffer, sample-weighted)
    → SGD training (policy loss + value/ownership/score/liberty aux losses)
    → Periodic evaluation (promotion gate vs. incumbent best)
    → Checkpoint + ONNX export
```

Key mechanisms:
- **Curriculum learning**: Early iterations get capture bonuses and response
  weighting to bootstrap tactical understanding. Stops when metrics (capture
  rate, eye-fill rate) meet thresholds.
- **Endgame confidence weighting**: Noisy terminal labels (max_moves,
  high dead-stone count, extreme score margins) get reduced weight for
  value/ownership/score heads, while policy weight stays full.
- **Auto komi adjustment**: If first-player win rate drifts outside [0.45, 0.55],
  komi is nudged to restore balance.

## Pass Handling

The single hardest behavioral problem. The approach evolved through stages:

1. **Stage 1**: No pass restriction → models learn to pass immediately
2. **Stage 2**: `opening_no_pass_moves=12` + `pass_prior_scale=0.15` → better
   but still passes too early in midgame
3. **Stage 3+**: Opening pass ban, soft pass prior penalty, and consecutive
   pass guard (only blocks reply-pass when model is clearly behind). No hard
   midgame pass prohibition — the goal is to let the model learn when to pass
   rather than coding it by hand.

The current sweet spot: `pass_prior_scale` around 0.15–0.25 gives enough
suppression to prevent pathological passing without making pass a rare event.
