from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_jsonl(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    rows: list[dict[str, object]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


def load_jsonl_dir(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    rows: list[dict[str, object]] = []
    for child in sorted(path.glob("*.jsonl")):
        rows.extend(load_jsonl(child))
    return rows


def render_html(metrics: list[dict[str, object]], games: list[dict[str, object]], output: Path, recent_games: int = 50) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "metrics": metrics,
        "games": games[-max(1, int(recent_games)):],
    }
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>TriLibGo Training Report</title>
  <style>
    :root {{
      --bg: #f4efe6;
      --panel: #fffaf0;
      --ink: #20201c;
      --accent: #146c43;
      --accent2: #8d3b12;
      --grid: #d8cdb7;
    }}
    body {{
      margin: 0;
      font-family: "Segoe UI", "Noto Sans SC", sans-serif;
      background: linear-gradient(160deg, #f8f3ea, #efe4cf);
      color: var(--ink);
    }}
    .wrap {{
      max-width: 1200px;
      margin: 0 auto;
      padding: 24px;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
      gap: 16px;
    }}
    .panel {{
      background: var(--panel);
      border: 1px solid #e7dcc7;
      border-radius: 16px;
      padding: 16px;
      box-shadow: 0 10px 30px rgba(70, 50, 20, 0.08);
    }}
    h1, h2 {{
      margin: 0 0 12px;
    }}
    canvas {{
      width: 100%;
      height: 240px;
      display: block;
      background: #fffdfa;
      border-radius: 12px;
      border: 1px solid #eadfcf;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 14px;
    }}
    th, td {{
      padding: 8px 6px;
      border-bottom: 1px solid #eee2cf;
      text-align: left;
      vertical-align: top;
    }}
    .muted {{ color: #6d675f; }}
    .badge {{
      display: inline-block;
      padding: 4px 8px;
      border-radius: 999px;
      background: #efe5d3;
      margin-right: 8px;
      margin-bottom: 8px;
      font-size: 13px;
    }}
    .moves {{
      white-space: pre-wrap;
      line-height: 1.5;
      max-height: 220px;
      overflow: auto;
      font-family: Consolas, monospace;
      font-size: 12px;
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <h1>TriLibGo Training Report</h1>
    <div class="panel">
      <div id="summary"></div>
    </div>
    <div class="grid">
      <div class="panel">
        <h2>Loss</h2>
        <canvas id="lossChart" width="520" height="240"></canvas>
      </div>
      <div class="panel">
        <h2>Aux Loss</h2>
        <canvas id="auxLossChart" width="520" height="240"></canvas>
      </div>
      <div class="panel">
        <h2>Win Rate</h2>
        <canvas id="winChart" width="520" height="240"></canvas>
      </div>
      <div class="panel">
        <h2>First Player Bias</h2>
        <canvas id="biasChart" width="520" height="240"></canvas>
      </div>
      <div class="panel">
        <h2>Score Margin</h2>
        <canvas id="marginChart" width="520" height="240"></canvas>
      </div>
      <div class="panel">
        <h2>Score Error</h2>
        <canvas id="scoreErrorChart" width="520" height="240"></canvas>
      </div>
      <div class="panel">
        <h2>Learning Rate</h2>
        <canvas id="lrChart" width="520" height="240"></canvas>
      </div>
      <div class="panel">
        <h2>Replay Effective Size</h2>
        <canvas id="bufferChart" width="520" height="240"></canvas>
      </div>
    </div>
    <div class="panel">
      <h2>Recent Self-Play Games</h2>
      <table>
        <thead>
          <tr><th>Iter</th><th>Game</th><th>Komi</th><th>Winner</th><th>Score</th><th>Moves</th><th>1st Pass</th><th>Cleaned</th><th>Avg |Err|</th><th>Weight</th><th>Flags</th><th>End</th><th>Trace</th></tr>
        </thead>
        <tbody id="gamesBody"></tbody>
      </table>
    </div>
  </div>
  <script>
    const data = {json.dumps(data, ensure_ascii=False)};

    function lineChart(canvasId, rows, field, color, baseline = null) {{
      const canvas = document.getElementById(canvasId);
      const ctx = canvas.getContext('2d');
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      const pad = 28;
      const values = rows.map(r => Number(r[field] ?? 0));
      const xs = rows.map(r => Number(r.iteration ?? 0));
      if (!values.length) return;
      let minV = Math.min(...values);
      let maxV = Math.max(...values);
      if (baseline !== null) {{
        minV = Math.min(minV, baseline);
        maxV = Math.max(maxV, baseline);
      }}
      if (minV === maxV) {{
        minV -= 1;
        maxV += 1;
      }}
      const minX = Math.min(...xs);
      const maxX = Math.max(...xs);
      const xScale = (x) => pad + ((x - minX) / Math.max(maxX - minX, 1)) * (canvas.width - pad * 2);
      const yScale = (v) => canvas.height - pad - ((v - minV) / (maxV - minV)) * (canvas.height - pad * 2);

      ctx.strokeStyle = '#ccbda2';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(pad, pad);
      ctx.lineTo(pad, canvas.height - pad);
      ctx.lineTo(canvas.width - pad, canvas.height - pad);
      ctx.stroke();

      if (baseline !== null) {{
        ctx.strokeStyle = '#b8aa8f';
        ctx.setLineDash([6, 6]);
        ctx.beginPath();
        ctx.moveTo(pad, yScale(baseline));
        ctx.lineTo(canvas.width - pad, yScale(baseline));
        ctx.stroke();
        ctx.setLineDash([]);
      }}

      ctx.strokeStyle = color;
      ctx.lineWidth = 2.5;
      ctx.beginPath();
      values.forEach((v, i) => {{
        const x = xScale(xs[i]);
        const y = yScale(v);
        if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
      }});
      ctx.stroke();

      ctx.fillStyle = '#5e584f';
      ctx.font = '12px Segoe UI';
      ctx.fillText(minV.toFixed(2), 2, canvas.height - pad + 4);
      ctx.fillText(maxV.toFixed(2), 2, pad + 4);
      ctx.fillText(String(minX), pad, canvas.height - 6);
      ctx.fillText(String(maxX), canvas.width - pad - 20, canvas.height - 6);
    }}

    function multiLineChart(canvasId, rows, series, baseline = null) {{
      const canvas = document.getElementById(canvasId);
      const ctx = canvas.getContext('2d');
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      const pad = 28;
      const xs = rows.map(r => Number(r.iteration ?? 0));
      if (!xs.length) return;
      const values = series.flatMap(s => rows.map(r => Number(r[s.field] ?? 0)));
      let minV = Math.min(...values);
      let maxV = Math.max(...values);
      if (baseline !== null) {{
        minV = Math.min(minV, baseline);
        maxV = Math.max(maxV, baseline);
      }}
      if (minV === maxV) {{
        minV -= 1;
        maxV += 1;
      }}
      const minX = Math.min(...xs);
      const maxX = Math.max(...xs);
      const xScale = (x) => pad + ((x - minX) / Math.max(maxX - minX, 1)) * (canvas.width - pad * 2);
      const yScale = (v) => canvas.height - pad - ((v - minV) / (maxV - minV)) * (canvas.height - pad * 2);

      ctx.strokeStyle = '#ccbda2';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(pad, pad);
      ctx.lineTo(pad, canvas.height - pad);
      ctx.lineTo(canvas.width - pad, canvas.height - pad);
      ctx.stroke();

      if (baseline !== null) {{
        ctx.strokeStyle = '#b8aa8f';
        ctx.setLineDash([6, 6]);
        ctx.beginPath();
        ctx.moveTo(pad, yScale(baseline));
        ctx.lineTo(canvas.width - pad, yScale(baseline));
        ctx.stroke();
        ctx.setLineDash([]);
      }}

      series.forEach((item, index) => {{
        ctx.strokeStyle = item.color;
        ctx.lineWidth = 2.5;
        ctx.beginPath();
        rows.forEach((row, rowIndex) => {{
          const x = xScale(xs[rowIndex]);
          const y = yScale(Number(row[item.field] ?? 0));
          if (rowIndex === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
        }});
        ctx.stroke();
        ctx.fillStyle = item.color;
        ctx.fillRect(canvas.width - pad - 120, pad + index * 16, 10, 10);
        ctx.fillStyle = '#5e584f';
        ctx.font = '12px Segoe UI';
        ctx.fillText(item.label, canvas.width - pad - 104, pad + 9 + index * 16);
      }});

      ctx.fillStyle = '#5e584f';
      ctx.font = '12px Segoe UI';
      ctx.fillText(minV.toFixed(2), 2, canvas.height - pad + 4);
      ctx.fillText(maxV.toFixed(2), 2, pad + 4);
      ctx.fillText(String(minX), pad, canvas.height - 6);
      ctx.fillText(String(maxX), canvas.width - pad - 20, canvas.height - 6);
    }}

    function renderSummary(rows) {{
      const target = document.getElementById('summary');
      if (!rows.length) {{
        target.innerHTML = '<div class="muted">No metrics found.</div>';
        return;
      }}
      const last = rows[rows.length - 1];
      const lastEval = [...rows].reverse().find(row => row.eval_ran !== false && row.eval_win_rate !== null && row.eval_win_rate !== undefined) || last;
      const badges = [
        ['iterations', last.iteration],
        ['policy_loss', Number(last.policy_loss ?? 0).toFixed(3)],
        ['value_loss', Number(last.value_loss ?? 0).toFixed(3)],
        ['ownership_loss', Number(last.ownership_loss ?? 0).toFixed(3)],
        ['score_loss', Number(last.score_loss ?? 0).toFixed(3)],
        ['last_eval_win_rate', Number(lastEval.eval_win_rate ?? 0).toFixed(3)],
        ['eval_interval', Number(last.eval_interval ?? 1).toFixed(0)],
        ['best_updated', String(Boolean(last.best_updated ?? false))],
        ['first_player_win_rate', Number(last.first_player_win_rate ?? 0).toFixed(3)],
        ['avg_margin_b-w', Number(last.avg_score_margin_black_minus_white ?? 0).toFixed(3)],
        ['avg_abs_score_err', Number(last.avg_abs_predicted_score_error ?? 0).toFixed(3)],
        ['avg_sample_weight', Number(last.avg_sample_weight ?? 0).toFixed(3)],
        ['abnormal_game_rate', Number(last.abnormal_game_rate ?? 0).toFixed(3)],
        ['buffer_effective_size', Number(last.buffer_effective_size ?? 0).toFixed(1)],
        ['buffer_avg_weight', Number(last.buffer_avg_sample_weight ?? 0).toFixed(3)],
        ['buffer_downweighted_rate', Number(last.buffer_downweighted_rate ?? 0).toFixed(3)],
        ['avg_effective_batch', Number(last.effective_batch_size ?? 0).toFixed(2)],
        ['avg_batch_weight', Number(last.avg_batch_weight ?? 0).toFixed(3)],
        ['learning_rate', Number(last.learning_rate ?? 0).toFixed(6)],
        ['avg_komi', Number(last.avg_komi ?? 0).toFixed(3)],
        ['avg_first_pass', Number(last.avg_first_pass_turn ?? 0).toFixed(1)],
        ['avg_cleaned', Number(last.avg_cleaned_dead_stones ?? 0).toFixed(2)]
      ];
      target.innerHTML = badges.map(([k, v]) => `<span class="badge">${{k}}: ${{v}}</span>`).join('');
    }}

    function renderGames(rows) {{
      const body = document.getElementById('gamesBody');
      body.innerHTML = rows.map(row => {{
        const trace = (row.moves || []).map(m => `${{m.turn}}.${{m.player}} ${{m.move}} v=${{Number(m.root_value).toFixed(2)}} s=${{Number(m.root_score_margin_black_minus_white ?? m.root_score_margin ?? 0).toFixed(2)}} err=${{Number(m.score_margin_error_black_minus_white ?? 0).toFixed(2)}}`).join('\\n');
        const abnormalTags = (row.abnormal_tags || []).join(', ');
        return `<tr>
          <td>${{row.iteration}}</td>
          <td>${{row.game_index}}</td>
          <td>${{Number(row.komi).toFixed(2)}}</td>
          <td>${{row.winner}}</td>
          <td>${{Number(row.black_score).toFixed(1)}} : ${{Number(row.white_score).toFixed(1)}}</td>
          <td>${{row.move_count}}</td>
          <td>${{row.first_pass_turn || '-'}}</td>
          <td>${{row.cleaned_dead_stones || 0}}</td>
          <td>${{Number(row.avg_abs_predicted_score_error ?? 0).toFixed(2)}}</td>
          <td>${{Number(row.sample_weight ?? 1).toFixed(2)}}</td>
          <td>${{abnormalTags || '-'}}</td>
          <td>${{row.end_reason || '-'}}</td>
          <td><div class="moves">${{trace}}</div></td>
        </tr>`;
      }}).join('');
    }}

    const metrics = data.metrics.filter(row => row.iteration !== undefined && row.event === undefined);
    const evalMetrics = metrics.filter(row => row.eval_ran !== false && row.eval_win_rate !== null && row.eval_win_rate !== undefined);
    renderSummary(metrics);
    lineChart('lossChart', metrics, 'total_loss', '#8d3b12');
    multiLineChart('auxLossChart', metrics, [
      {{ field: 'ownership_loss', color: '#146c43', label: 'ownership_loss' }},
      {{ field: 'score_loss', color: '#174e9a', label: 'score_loss' }}
    ]);
    lineChart('winChart', evalMetrics, 'eval_win_rate', '#146c43', 0.5);
    lineChart('biasChart', metrics, 'first_player_win_rate', '#174e9a', 0.5);
    lineChart('marginChart', metrics, 'avg_score_margin_black_minus_white', '#6c2c8d', 0.0);
    lineChart('scoreErrorChart', metrics, 'avg_abs_predicted_score_error', '#b45414', 0.0);
    lineChart('lrChart', metrics, 'learning_rate', '#0f766e', 0.0);
    lineChart('bufferChart', metrics, 'buffer_effective_size', '#9a3412', 0.0);
    renderGames(data.games);
  </script>
</body>
</html>
"""
    output.write_text(html, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a local HTML training report from TriLibGo JSONL logs.")
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--games", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--recent-games", type=int, default=50)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    game_rows = load_jsonl_dir(args.games) if args.games.is_dir() else load_jsonl(args.games)
    render_html(load_jsonl(args.metrics), game_rows, args.output, recent_games=args.recent_games)
