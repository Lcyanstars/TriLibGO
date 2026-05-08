#include "replay_compat.h"

#include <queue>
#include <set>
#include <utility>
#include <vector>

namespace trilibgo::app {
namespace {

struct GroupInfo {
    std::vector<int> stones;
    int liberties = 0;
};

GroupInfo collect_group_on_board(const trilibgo::core::GameState& state, const std::vector<trilibgo::core::Stone>& board, int start) {
    GroupInfo group;
    if (!state.topology().is_valid_vertex(start)) {
        return group;
    }
    const auto color = board[static_cast<std::size_t>(start)];
    if (color == trilibgo::core::Stone::Empty) {
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
            const auto neighbor_stone = board[static_cast<std::size_t>(neighbor)];
            if (neighbor_stone == trilibgo::core::Stone::Empty) {
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

bool same_move(const trilibgo::core::Move& lhs, const trilibgo::core::Move& rhs) {
    return lhs.kind == rhs.kind && lhs.coord.index == rhs.coord.index;
}

}  // namespace

void normalize_replay_state_after_move(trilibgo::core::GameState& state, bool has_more_moves) {
    if (!has_more_moves || state.is_finished()) {
        return;
    }
    if (state.phase() != trilibgo::core::Phase::ReviewingEndgame) {
        return;
    }
    state.set_phase(trilibgo::core::Phase::Playing);
    state.set_review_state(std::nullopt);
    state.set_consecutive_passes(0);
}

std::optional<trilibgo::core::Move> parse_replay_move_token(const trilibgo::core::GameState& state, std::string_view token) {
    if (token.empty()) {
        return std::nullopt;
    }
    if (token == "pass") {
        return trilibgo::core::Move::pass();
    }
    if (token == "resign") {
        return trilibgo::core::Move::resign();
    }
    for (int vertex = 0; vertex < state.topology().vertex_count(); ++vertex) {
        if (state.topology().label_for_vertex(vertex) == token) {
            return trilibgo::core::Move::place(vertex);
        }
    }
    return std::nullopt;
}

bool apply_replay_move_compat(
    trilibgo::core::GameState& state,
    const trilibgo::core::RulesEngine& rules,
    trilibgo::core::Move move,
    ReplayApplyMode* mode_used
) {
    if (mode_used != nullptr) {
        *mode_used = ReplayApplyMode::Standard;
    }
    if (rules.apply_move(state, move).success) {
        return true;
    }

    if (move.kind != trilibgo::core::MoveKind::Place || state.phase() != trilibgo::core::Phase::Playing || state.is_finished()) {
        return false;
    }
    if (!state.topology().is_valid_vertex(move.coord.index) || state.at(move.coord.index) != trilibgo::core::Stone::Empty) {
        return false;
    }

    std::vector<trilibgo::core::Stone> next_board = state.board();
    next_board[static_cast<std::size_t>(move.coord.index)] = state.current_player();
    std::vector<int> captured;

    for (int neighbor : state.topology().neighbors(move.coord.index)) {
        if (next_board[static_cast<std::size_t>(neighbor)] != trilibgo::core::opposite(state.current_player())) {
            continue;
        }
        const auto group = collect_group_on_board(state, next_board, neighbor);
        if (group.liberties != 0) {
            continue;
        }
        captured.insert(captured.end(), group.stones.begin(), group.stones.end());
        for (int stone : group.stones) {
            next_board[static_cast<std::size_t>(stone)] = trilibgo::core::Stone::Empty;
        }
    }

    const auto own_group = collect_group_on_board(state, next_board, move.coord.index);
    if (!state.config().allow_suicide && own_group.liberties == 0) {
        return false;
    }

    state.set_board(std::move(next_board));
    state.add_captures(state.current_player(), static_cast<int>(captured.size()));
    state.push_move(move);
    state.set_consecutive_passes(0);
    state.set_review_state(std::nullopt);
    state.set_current_player(trilibgo::core::opposite(state.current_player()));
    state.push_board_hash(rules.compute_hash(state.board(), state.current_player()));
    if (mode_used != nullptr) {
        *mode_used = ReplayApplyMode::LegacyNoKo;
    }
    return true;
}

}  // namespace trilibgo::app
