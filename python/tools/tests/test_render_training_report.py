from __future__ import annotations

import shutil
import unittest
from pathlib import Path

from python.tools.render_training_report import load_jsonl_dir, render_html


class RenderTrainingReportTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp_root = Path("artifacts/test_tmp/render_training_report")
        if self._tmp_root.exists():
            shutil.rmtree(self._tmp_root)
        self._tmp_root.mkdir(parents=True, exist_ok=True)
        self.addCleanup(self._cleanup_tmp_root)

    def _cleanup_tmp_root(self) -> None:
        if self._tmp_root.exists():
            shutil.rmtree(self._tmp_root)

    def test_report_renders_auxiliary_loss_panels_and_game_fields(self) -> None:
        metrics = [
            {
                "iteration": 1,
                "policy_loss": 4.0,
                "value_loss": 1.0,
                "ownership_loss": 0.5,
                "score_loss": 0.25,
                "total_loss": 5.0,
                "eval_ran": False,
                "eval_interval": 4,
                "best_updated": False,
            },
            {
                "iteration": 4,
                "policy_loss": 3.5,
                "value_loss": 0.8,
                "ownership_loss": 0.4,
                "score_loss": 0.2,
                "total_loss": 4.4,
                "eval_ran": True,
                "eval_win_rate": 0.5,
                "first_player_win_rate": 0.6,
                "avg_score_margin_black_minus_white": 1.5,
                "avg_abs_predicted_score_error": 0.75,
                "avg_sample_weight": 0.6,
                "abnormal_game_rate": 0.5,
                "buffer_effective_size": 128.5,
                "buffer_avg_sample_weight": 0.8,
                "buffer_downweighted_rate": 0.25,
                "effective_batch_size": 14.5,
                "avg_batch_weight": 0.9,
                "learning_rate": 0.00075,
                "eval_interval": 4,
                "best_updated": True,
                "avg_komi": 3.0,
                "avg_first_pass_turn": 20.0,
                "avg_cleaned_dead_stones": 0.5,
            }
        ]
        games = [
            {
                "iteration": 1,
                "game_index": 1,
                "komi": 3.0,
                "winner": "B",
                "black_score": 12.0,
                "white_score": 8.0,
                "move_count": 30,
                "first_pass_turn": 21,
                "cleaned_dead_stones": 2,
                "avg_abs_predicted_score_error": 0.75,
                "sample_weight": 0.35,
                "abnormal_tags": ["early_pass", "short_game"],
                "end_reason": "double_pass",
                "moves": [
                    {
                        "turn": 1,
                        "player": "B",
                        "move": "A1",
                        "root_value": 0.1,
                        "root_score_margin_black_minus_white": 1.25,
                        "score_margin_error_black_minus_white": -0.75,
                    }
                ],
            }
        ]
        output = self._tmp_root / "report.html"
        render_html(metrics, games, output)
        html = output.read_text(encoding="utf-8")

        self.assertIn("Aux Loss", html)
        self.assertIn("ownership_loss", html)
        self.assertIn("score_loss", html)
        self.assertIn("first_pass_turn", html)
        self.assertIn("cleaned_dead_stones", html)
        self.assertIn("avg_abs_predicted_score_error", html)
        self.assertIn("avg_sample_weight", html)
        self.assertIn("abnormal_game_rate", html)
        self.assertIn("buffer_effective_size", html)
        self.assertIn("buffer_avg_weight", html)
        self.assertIn("buffer_downweighted_rate", html)
        self.assertIn("avg_effective_batch", html)
        self.assertIn("learning_rate", html)
        self.assertIn("last_eval_win_rate", html)
        self.assertIn("eval_interval", html)
        self.assertIn("best_updated", html)
        self.assertIn("Replay Effective Size", html)
        self.assertIn("Learning Rate", html)
        self.assertIn("early_pass", html)
        self.assertIn("short_game", html)
        self.assertIn("root_score_margin_black_minus_white", html)
        self.assertIn("double_pass", html)

    def test_load_jsonl_dir_merges_sharded_logs(self) -> None:
        shard_dir = self._tmp_root / "games"
        shard_dir.mkdir(parents=True, exist_ok=True)
        (shard_dir / "a.jsonl").write_text('{"iteration":1}\n', encoding="utf-8")
        (shard_dir / "b.jsonl").write_text('{"iteration":2}\n', encoding="utf-8")

        rows = load_jsonl_dir(shard_dir)

        self.assertEqual(rows, [{"iteration": 1}, {"iteration": 2}])


if __name__ == "__main__":
    unittest.main()
