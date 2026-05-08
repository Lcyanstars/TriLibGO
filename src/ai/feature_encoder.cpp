#include "trilibgo/ai/feature_encoder.h"

#include "trilibgo/core/rules_engine.h"

#include <algorithm>
#include <array>
#include <iterator>
#include <queue>
#include <set>

namespace trilibgo::ai {
namespace {

int liberty_bucket(int liberties) {
    if (liberties <= 1) {
        return 0;
    }
    if (liberties == 2) {
        return 1;
    }
    return 2;
}

struct GroupInfo {
    std::vector<int> stones;
    std::set<int> liberties;
};

GroupInfo collect_group(
    const trilibgo::core::GameState& state,
    const std::vector<trilibgo::core::Stone>& board,
    int start
) {
    GroupInfo group;
    const auto color = board[static_cast<std::size_t>(start)];
    if (color == trilibgo::core::Stone::Empty) {
        return group;
    }

    std::queue<int> queue;
    std::vector<bool> visited(static_cast<std::size_t>(state.topology().vertex_count()), false);
    queue.push(start);
    visited[static_cast<std::size_t>(start)] = true;

    while (!queue.empty()) {
        const int current = queue.front();
        queue.pop();
        group.stones.push_back(current);
        for (int neighbor : state.topology().neighbors(current)) {
            const auto stone = board[static_cast<std::size_t>(neighbor)];
            if (stone == trilibgo::core::Stone::Empty) {
                group.liberties.insert(neighbor);
            } else if (stone == color && !visited[static_cast<std::size_t>(neighbor)]) {
                visited[static_cast<std::size_t>(neighbor)] = true;
                queue.push(neighbor);
            }
        }
    }
    return group;
}

std::vector<trilibgo::core::Stone> replay_board_after_moves(
    const trilibgo::core::GameState& source,
    std::size_t move_count
) {
    trilibgo::core::RulesEngine rules;
    trilibgo::core::GameState replay(source.config());
    replay.clear_and_seed_board_hash(rules.compute_hash(replay.board(), replay.current_player()));
    const auto& history = source.move_history();
    for (std::size_t i = 0; i < move_count && i < history.size(); ++i) {
        if (!rules.apply_move(replay, history[i]).success) {
            return source.board();
        }
    }
    return replay.board();
}

void add_board_planes(
    std::vector<trilibgo::core::Plane>& planes,
    const std::vector<trilibgo::core::Stone>& board,
    trilibgo::core::Stone current_player
) {
    trilibgo::core::Plane current(board.size(), 0.0f);
    trilibgo::core::Plane opponent(board.size(), 0.0f);
    const auto enemy = trilibgo::core::opposite(current_player);
    for (std::size_t i = 0; i < board.size(); ++i) {
        if (board[i] == current_player) {
            current[i] = 1.0f;
        } else if (board[i] == enemy) {
            opponent[i] = 1.0f;
        }
    }
    planes.push_back(std::move(current));
    planes.push_back(std::move(opponent));
}

}  // namespace

FeatureEncoder::FeatureEncoder(int input_history) : input_history_(std::max(input_history, 1)) {}

std::vector<trilibgo::core::Plane> FeatureEncoder::encode(const trilibgo::core::GameState& state) const {
    const std::size_t count = static_cast<std::size_t>(state.topology().vertex_count());
    trilibgo::core::Plane current(count, 0.0f);
    trilibgo::core::Plane opponent(count, 0.0f);
    trilibgo::core::Plane legal(count, 0.0f);
    trilibgo::core::Plane last_move(count, 0.0f);

    for (std::size_t i = 0; i < count; ++i) {
        const auto stone = state.board()[i];
        if (stone == state.current_player()) {
            current[i] = 1.0f;
        } else if (stone == trilibgo::core::opposite(state.current_player())) {
            opponent[i] = 1.0f;
        }
    }

    trilibgo::core::RulesEngine rules;
    for (const auto& move : rules.legal_moves(state)) {
        if (move.kind == trilibgo::core::MoveKind::Place && move.coord.is_valid()) {
            legal[static_cast<std::size_t>(move.coord.index)] = 1.0f;
        }
    }

    if (!state.move_history().empty()) {
        const auto& move = state.move_history().back();
        if (move.kind == trilibgo::core::MoveKind::Place && move.coord.is_valid()) {
            last_move[static_cast<std::size_t>(move.coord.index)] = 1.0f;
        }
    }

    std::vector<trilibgo::core::Plane> planes;
    planes.reserve(static_cast<std::size_t>(2 * input_history_ + 4));
    planes.push_back(std::move(current));
    planes.push_back(std::move(opponent));
    planes.push_back(std::move(legal));
    planes.push_back(std::move(last_move));

    const int history_pairs = std::max(input_history_ - 1, 0);
    const int move_count = static_cast<int>(state.move_history().size());
    for (int offset = 0; offset < history_pairs; ++offset) {
        const int replay_moves = move_count - 1 - offset;
        if (replay_moves >= 0) {
            add_board_planes(
                planes,
                replay_board_after_moves(state, static_cast<std::size_t>(replay_moves)),
                state.current_player()
            );
        } else {
            planes.emplace_back(count, 0.0f);
            planes.emplace_back(count, 0.0f);
        }
    }
    return planes;
}

std::vector<float> FeatureEncoder::encode_global_features(const trilibgo::core::GameState& state) const {
    std::array<int, 6> liberty_counts{};
    std::vector<bool> visited(static_cast<std::size_t>(state.topology().vertex_count()), false);
    const auto enemy = trilibgo::core::opposite(state.current_player());
    for (int vertex = 0; vertex < state.topology().vertex_count(); ++vertex) {
        const auto stone = state.board()[static_cast<std::size_t>(vertex)];
        if (stone == trilibgo::core::Stone::Empty || visited[static_cast<std::size_t>(vertex)]) {
            continue;
        }
        const auto group = collect_group(state, state.board(), vertex);
        for (int stone_vertex : group.stones) {
            visited[static_cast<std::size_t>(stone_vertex)] = true;
        }
        const int bucket = liberty_bucket(static_cast<int>(group.liberties.size()));
        int feature_index = -1;
        if (stone == state.current_player()) {
            feature_index = bucket;
        } else if (stone == enemy) {
            feature_index = 3 + bucket;
        }
        if (feature_index >= 0) {
            ++liberty_counts[static_cast<std::size_t>(feature_index)];
        }
    }
    const float scale = state.topology().vertex_count() > 0 ? static_cast<float>(state.topology().vertex_count()) : 1.0f;
    std::vector<float> features(8, 0.0f);
    for (std::size_t i = 0; i < liberty_counts.size(); ++i) {
        features[i] = static_cast<float>(liberty_counts[i]) / scale;
    }
    features[6] = state.current_player() == trilibgo::core::Stone::Black ? 1.0f : 0.0f;
    features[7] = static_cast<float>(std::min(state.consecutive_passes(), 2)) / 2.0f;
    return features;
}

}  // namespace trilibgo::ai
