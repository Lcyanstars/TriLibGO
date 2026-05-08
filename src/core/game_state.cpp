#include "trilibgo/core/game_state.h"

namespace trilibgo::core {

GameState::GameState(GameConfig config)
    : config_(config), topology_(config.side_length), board_(static_cast<std::size_t>(topology_.vertex_count()), Stone::Empty) {}

const GameConfig& GameState::config() const { return config_; }
const BoardTopology& GameState::topology() const { return topology_; }
Stone GameState::at(int vertex) const { return board_[static_cast<std::size_t>(vertex)]; }
Stone GameState::current_player() const { return current_player_; }
int GameState::consecutive_passes() const { return consecutive_passes_; }
bool GameState::is_finished() const { return result_.has_value(); }
Phase GameState::phase() const { return phase_; }
const std::optional<GameResult>& GameState::result() const { return result_; }
const std::optional<EndgameReviewState>& GameState::review_state() const { return review_state_; }
const std::vector<Move>& GameState::move_history() const { return move_history_; }
const std::vector<std::uint64_t>& GameState::board_hash_history() const { return board_hash_history_; }
int GameState::move_number() const { return static_cast<int>(move_history_.size()); }
const std::vector<Stone>& GameState::board() const { return board_; }

int GameState::captures(Stone stone) const {
    if (stone == Stone::Black) {
        return black_captures_;
    }
    if (stone == Stone::White) {
        return white_captures_;
    }
    return 0;
}

void GameState::set_board(std::vector<Stone> board) { board_ = std::move(board); }
void GameState::set_current_player(Stone stone) { current_player_ = stone; }
void GameState::set_consecutive_passes(int value) { consecutive_passes_ = value; }
void GameState::set_phase(Phase phase) { phase_ = phase; }
void GameState::set_result(std::optional<GameResult> result) { result_ = std::move(result); }
void GameState::set_review_state(std::optional<EndgameReviewState> review_state) { review_state_ = std::move(review_state); }
void GameState::push_move(Move move) { move_history_.push_back(move); }
void GameState::push_board_hash(std::uint64_t hash) { board_hash_history_.push_back(hash); }

void GameState::clear_and_seed_board_hash(std::uint64_t hash) {
    board_hash_history_.clear();
    board_hash_history_.push_back(hash);
}

void GameState::add_captures(Stone stone, int amount) {
    if (stone == Stone::Black) {
        black_captures_ += amount;
    } else if (stone == Stone::White) {
        white_captures_ += amount;
    }
}

}  // namespace trilibgo::core
