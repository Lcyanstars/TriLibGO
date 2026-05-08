#include "trilibgo/ai/analysis_model.h"

#include "trilibgo/core/rules_engine.h"

#include <algorithm>

namespace trilibgo::ai {

std::optional<AnalysisResult> NullAnalysisModel::analyze(const trilibgo::core::GameState& state) const {
    trilibgo::core::RulesEngine rules;
    const auto legal_moves = rules.legal_moves(state);
    if (legal_moves.empty()) {
        return std::nullopt;
    }

    AnalysisResult result;
    const double uniform = 1.0 / static_cast<double>(legal_moves.size());
    for (const auto& move : legal_moves) {
        int action_index = -1;
        if (move.kind == trilibgo::core::MoveKind::Pass) {
            action_index = ActionCodec::pass_index(state);
        } else if (move.kind == trilibgo::core::MoveKind::Place) {
            action_index = ActionCodec::vertex_to_action_index(state, move.coord.index);
        } else {
            continue;
        }
        result.policy.push_back({action_index, uniform});
    }
    result.top_policy = result.policy;
    std::sort(result.top_policy.begin(), result.top_policy.end(), [](const PolicyLogit& lhs, const PolicyLogit& rhs) {
        return lhs.probability > rhs.probability;
    });
    if (result.top_policy.size() > 8) {
        result.top_policy.resize(8);
    }
    return result;
}

int ActionCodec::policy_size(const trilibgo::core::GameState& state) {
    return state.topology().vertex_count() + 1;
}

int ActionCodec::pass_index(const trilibgo::core::GameState& state) {
    return state.topology().vertex_count();
}

int ActionCodec::vertex_to_action_index(const trilibgo::core::GameState& state, int vertex) {
    return state.topology().is_valid_vertex(vertex) ? vertex : -1;
}

std::optional<trilibgo::core::Move> ActionCodec::action_index_to_move(const trilibgo::core::GameState& state, int action_index) {
    if (action_index == pass_index(state)) {
        return trilibgo::core::Move::pass();
    }
    if (state.topology().is_valid_vertex(action_index)) {
        return trilibgo::core::Move::place(action_index);
    }
    return std::nullopt;
}

}  // namespace trilibgo::ai
