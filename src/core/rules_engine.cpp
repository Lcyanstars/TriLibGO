#include "trilibgo/core/rules_engine.h"

#include <cmath>
#include <queue>
#include <set>

namespace trilibgo::core {
namespace {

int stone_hash_value(Stone stone) {
    if (stone == Stone::Black) {
        return 1;
    }
    if (stone == Stone::White) {
        return 2;
    }
    return 0;
}

std::string stone_name(Stone stone) {
    if (stone == Stone::Black) {
        return "Black";
    }
    if (stone == Stone::White) {
        return "White";
    }
    return "Draw";
}

}  // namespace

MoveOutcome RulesEngine::apply_move(GameState& state, Move move) const {
    if (state.phase() != Phase::Playing) {
        return {false, "Moves are disabled outside the playing phase.", {}};
    }
    if (state.is_finished()) {
        return {false, "Game is already finished.", {}};
    }
    if (!is_legal(state, move)) {
        return {false, "Illegal move.", {}};
    }

    if (move.kind == MoveKind::Resign) {
        const Stone winner = opposite(state.current_player());
        GameResult result;
        result.reason = EndReason::Resignation;
        result.winner = winner;
        result.summary = stone_name(winner) + " wins by resignation";
        state.push_move(move);
        state.set_phase(Phase::Finished);
        state.set_result(result);
        return {true, result.summary, {}};
    }

    if (move.kind == MoveKind::Pass) {
        state.push_move(move);
        state.set_consecutive_passes(state.consecutive_passes() + 1);
        state.set_current_player(opposite(state.current_player()));
        state.push_board_hash(compute_hash(state.board(), state.current_player()));
        finalize_if_finished(state);
        return {true, "Pass", {}};
    }

    std::vector<Stone> next_board = state.board();
    next_board[static_cast<std::size_t>(move.coord.index)] = state.current_player();
    std::vector<int> captured;

    for (int neighbor : state.topology().neighbors(move.coord.index)) {
        if (next_board[static_cast<std::size_t>(neighbor)] != opposite(state.current_player())) {
            continue;
        }
        const auto group = collect_group(state, next_board, neighbor);
        if (group.liberties == 0) {
            captured.insert(captured.end(), group.stones.begin(), group.stones.end());
            for (int stone : group.stones) {
                next_board[static_cast<std::size_t>(stone)] = Stone::Empty;
            }
        }
    }

    state.set_board(std::move(next_board));
    state.add_captures(state.current_player(), static_cast<int>(captured.size()));
    state.push_move(move);
    state.set_consecutive_passes(0);
    state.set_review_state(std::nullopt);
    state.set_current_player(opposite(state.current_player()));
    state.push_board_hash(compute_hash(state.board(), state.current_player()));
    return {true, "Placed at " + state.topology().label_for_vertex(move.coord.index), captured};
}

bool RulesEngine::is_legal(const GameState& state, Move move) const {
    if (state.phase() != Phase::Playing || state.is_finished()) {
        return false;
    }
    if (move.kind == MoveKind::Pass || move.kind == MoveKind::Resign) {
        return true;
    }
    if (!state.topology().is_valid_vertex(move.coord.index) || state.at(move.coord.index) != Stone::Empty) {
        return false;
    }

    std::vector<Stone> next_board = state.board();
    next_board[static_cast<std::size_t>(move.coord.index)] = state.current_player();

    for (int neighbor : state.topology().neighbors(move.coord.index)) {
        if (next_board[static_cast<std::size_t>(neighbor)] != opposite(state.current_player())) {
            continue;
        }
        const auto enemy_group = collect_group(state, next_board, neighbor);
        if (enemy_group.liberties == 0) {
            for (int stone : enemy_group.stones) {
                next_board[static_cast<std::size_t>(stone)] = Stone::Empty;
            }
        }
    }

    const auto own_group = collect_group(state, next_board, move.coord.index);
    if (!state.config().allow_suicide && own_group.liberties == 0) {
        return false;
    }
    return !is_simple_ko_violation(state, next_board);
}

std::vector<Move> RulesEngine::legal_moves(const GameState& state) const {
    std::vector<Move> moves;
    if (state.phase() != Phase::Playing || state.is_finished()) {
        return moves;
    }
    for (int i = 0; i < state.topology().vertex_count(); ++i) {
        if (is_legal(state, Move::place(i))) {
            moves.push_back(Move::place(i));
        }
    }
    moves.push_back(Move::pass());
    moves.push_back(Move::resign());
    return moves;
}

ScoreBreakdown RulesEngine::score(const GameState& state) const {
    if (state.review_state().has_value()) {
        return score(state, state.review_state()->user_statuses);
    }
    std::vector<ChainLifeStatus> statuses(static_cast<std::size_t>(state.topology().vertex_count()), ChainLifeStatus::Alive);
    return score(state, statuses);
}

ScoreBreakdown RulesEngine::score(const GameState& state, const std::vector<ChainLifeStatus>& statuses) const {
    ScoreBreakdown score;
    score.komi = state.config().komi;
    const auto board = scoring_board(state, statuses, score);
    std::vector<bool> visited(static_cast<std::size_t>(state.topology().vertex_count()), false);

    for (int vertex = 0; vertex < state.topology().vertex_count(); ++vertex) {
        const Stone stone = board[static_cast<std::size_t>(vertex)];
        if (stone == Stone::Black) {
            ++score.black_stones;
            continue;
        }
        if (stone == Stone::White) {
            ++score.white_stones;
            continue;
        }
        if (visited[static_cast<std::size_t>(vertex)]) {
            continue;
        }

        std::queue<int> queue;
        queue.push(vertex);
        visited[static_cast<std::size_t>(vertex)] = true;
        int region_size = 0;
        std::set<Stone> border_colors;

        while (!queue.empty()) {
            const int current = queue.front();
            queue.pop();
            ++region_size;
            for (int neighbor : state.topology().neighbors(current)) {
                const Stone neighbor_stone = board[static_cast<std::size_t>(neighbor)];
                if (neighbor_stone == Stone::Empty) {
                    if (!visited[static_cast<std::size_t>(neighbor)]) {
                        visited[static_cast<std::size_t>(neighbor)] = true;
                        queue.push(neighbor);
                    }
                } else {
                    border_colors.insert(neighbor_stone);
                }
            }
        }

        if (border_colors.size() == 1) {
            if (*border_colors.begin() == Stone::Black) {
                score.black_territory += region_size;
            } else {
                score.white_territory += region_size;
            }
        } else {
            score.unsettled_points += region_size;
        }
    }
    return score;
}

ScoreEstimate RulesEngine::estimate_position(const GameState& state) const {
    const auto statuses = suggest_chain_statuses(state);
    const bool recommend_review = state.consecutive_passes() > 0 || state.move_number() > state.topology().vertex_count() / 2;
    return make_estimate(state, statuses, recommend_review);
}

bool RulesEngine::begin_endgame_review(GameState& state) const {
    if (state.phase() == Phase::Finished) {
        return false;
    }
    EndgameReviewState review;
    review.suggested_statuses = suggest_chain_statuses(state);
    review.user_statuses = review.suggested_statuses;
    review.estimate = make_estimate(state, review.user_statuses, true);
    state.set_review_state(review);
    state.set_phase(Phase::ReviewingEndgame);
    return true;
}

bool RulesEngine::toggle_group_status(GameState& state, int vertex) const {
    if (state.phase() != Phase::ReviewingEndgame || !state.review_state().has_value()) {
        return false;
    }
    if (!state.topology().is_valid_vertex(vertex) || state.at(vertex) == Stone::Empty) {
        return false;
    }

    auto review = *state.review_state();
    const auto group = collect_group(state, state.board(), vertex);
    ChainLifeStatus current = review.user_statuses[static_cast<std::size_t>(vertex)];
    const ChainLifeStatus next = current == ChainLifeStatus::Dead ? ChainLifeStatus::Alive : ChainLifeStatus::Dead;
    for (int stone : group.stones) {
        review.user_statuses[static_cast<std::size_t>(stone)] = next;
    }
    review.estimate = make_estimate(state, review.user_statuses, true);
    state.set_review_state(review);
    return true;
}

bool RulesEngine::accept_review_suggestion(GameState& state) const {
    if (state.phase() != Phase::ReviewingEndgame || !state.review_state().has_value()) {
        return false;
    }
    auto review = *state.review_state();
    review.user_statuses = review.suggested_statuses;
    review.estimate = make_estimate(state, review.user_statuses, true);
    state.set_review_state(review);
    return true;
}

bool RulesEngine::clear_review_overrides(GameState& state) const {
    return accept_review_suggestion(state);
}

bool RulesEngine::finalize_reviewed_result(GameState& state) const {
    if (state.phase() != Phase::ReviewingEndgame || !state.review_state().has_value()) {
        return false;
    }

    GameResult result;
    result.reason = EndReason::ConsecutivePasses;
    result.score = score(state, state.review_state()->user_statuses);
    if (std::abs(result.score.black_total() - result.score.white_total()) < 1e-6) {
        result.winner = Stone::Empty;
        result.margin = 0.0;
        result.summary = "Draw";
    } else if (result.score.black_total() > result.score.white_total()) {
        result.winner = Stone::Black;
        result.margin = result.score.black_total() - result.score.white_total();
        result.summary = "Black wins by " + std::to_string(result.margin);
    } else {
        result.winner = Stone::White;
        result.margin = result.score.white_total() - result.score.black_total();
        result.summary = "White wins by " + std::to_string(result.margin);
    }
    state.set_result(result);
    state.set_phase(Phase::Finished);
    return true;
}

std::uint64_t RulesEngine::compute_hash(const std::vector<Stone>& board, Stone player) const {
    std::uint64_t hash = 1469598103934665603ULL;
    for (Stone stone : board) {
        hash ^= static_cast<std::uint64_t>(stone_hash_value(stone) + 1);
        hash *= 1099511628211ULL;
    }
    hash ^= static_cast<std::uint64_t>(stone_hash_value(player) + 11);
    hash *= 1099511628211ULL;
    return hash;
}

RulesEngine::GroupInfo RulesEngine::collect_group(const GameState& state, const std::vector<Stone>& board, int start) const {
    GroupInfo group;
    const Stone color = board[static_cast<std::size_t>(start)];
    if (color == Stone::Empty) {
        return group;
    }

    std::queue<int> queue;
    std::vector<bool> visited(static_cast<std::size_t>(board.size()), false);
    std::set<int> liberties;

    queue.push(start);
    visited[static_cast<std::size_t>(start)] = true;
    while (!queue.empty()) {
        const int current = queue.front();
        queue.pop();
        group.stones.push_back(current);
        for (int neighbor : state.topology().neighbors(current)) {
            const Stone neighbor_stone = board[static_cast<std::size_t>(neighbor)];
            if (neighbor_stone == Stone::Empty) {
                liberties.insert(neighbor);
            } else if (neighbor_stone == color && !visited[static_cast<std::size_t>(neighbor)]) {
                visited[static_cast<std::size_t>(neighbor)] = true;
                queue.push(neighbor);
            }
        }
    }
    group.liberties = static_cast<int>(liberties.size());
    return group;
}

bool RulesEngine::is_simple_ko_violation(const GameState& state, const std::vector<Stone>& next_board) const {
    const auto& history = state.board_hash_history();
    if (history.size() < 2) {
        return false;
    }
    return compute_hash(next_board, opposite(state.current_player())) == history[history.size() - 2];
}

std::vector<ChainLifeStatus> RulesEngine::suggest_chain_statuses(const GameState& state) const {
    std::vector<ChainLifeStatus> statuses(static_cast<std::size_t>(state.topology().vertex_count()), ChainLifeStatus::Unsettled);
    std::vector<bool> visited(static_cast<std::size_t>(state.topology().vertex_count()), false);

    for (int vertex = 0; vertex < state.topology().vertex_count(); ++vertex) {
        if (state.at(vertex) == Stone::Empty || visited[static_cast<std::size_t>(vertex)]) {
            continue;
        }
        const auto group = collect_group(state, state.board(), vertex);
        for (int stone : group.stones) {
            visited[static_cast<std::size_t>(stone)] = true;
        }

        ChainLifeStatus status = ChainLifeStatus::Alive;
        if (group.liberties <= 1) {
            status = ChainLifeStatus::Dead;
        } else if (group.liberties == 2) {
            status = ChainLifeStatus::Unsettled;
        }

        for (int stone : group.stones) {
            statuses[static_cast<std::size_t>(stone)] = status;
        }
    }
    return statuses;
}

ScoreEstimate RulesEngine::make_estimate(const GameState& state, const std::vector<ChainLifeStatus>& statuses, bool recommend_review) const {
    ScoreEstimate estimate;
    estimate.score = score(state, statuses);
    estimate.recommended_endgame_review = recommend_review;
    if (std::abs(estimate.score.black_total() - estimate.score.white_total()) < 1e-6) {
        estimate.leader = Stone::Empty;
        estimate.lead = 0.0;
        estimate.summary = "Estimated draw";
    } else if (estimate.score.black_total() > estimate.score.white_total()) {
        estimate.leader = Stone::Black;
        estimate.lead = estimate.score.black_total() - estimate.score.white_total();
        estimate.summary = "Estimated Black +" + std::to_string(estimate.lead);
    } else {
        estimate.leader = Stone::White;
        estimate.lead = estimate.score.white_total() - estimate.score.black_total();
        estimate.summary = "Estimated White +" + std::to_string(estimate.lead);
    }
    estimate.confidence = estimate.score.unsettled_points <= 4 ? "high" : (estimate.score.unsettled_points <= 10 ? "medium" : "low");
    return estimate;
}

std::vector<Stone> RulesEngine::scoring_board(const GameState& state, const std::vector<ChainLifeStatus>& statuses, ScoreBreakdown& score) const {
    auto board = state.board();
    for (int vertex = 0; vertex < state.topology().vertex_count(); ++vertex) {
        const Stone stone = board[static_cast<std::size_t>(vertex)];
        if (stone == Stone::Empty || statuses[static_cast<std::size_t>(vertex)] != ChainLifeStatus::Dead) {
            continue;
        }
        if (stone == Stone::Black) {
            ++score.black_dead;
        } else {
            ++score.white_dead;
        }
        board[static_cast<std::size_t>(vertex)] = Stone::Empty;
    }
    return board;
}

void RulesEngine::finalize_if_finished(GameState& state) const {
    if (state.consecutive_passes() < 2 || state.phase() != Phase::Playing || state.is_finished()) {
        return;
    }
    (void)begin_endgame_review(state);
}

}  // namespace trilibgo::core
