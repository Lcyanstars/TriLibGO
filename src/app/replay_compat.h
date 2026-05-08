#pragma once

#include "trilibgo/core/rules_engine.h"

#include <optional>
#include <string_view>

namespace trilibgo::app {

enum class ReplayApplyMode {
    Standard,
    LegacyNoKo,
};

void normalize_replay_state_after_move(trilibgo::core::GameState& state, bool has_more_moves);
std::optional<trilibgo::core::Move> parse_replay_move_token(const trilibgo::core::GameState& state, std::string_view token);
bool apply_replay_move_compat(
    trilibgo::core::GameState& state,
    const trilibgo::core::RulesEngine& rules,
    trilibgo::core::Move move,
    ReplayApplyMode* mode_used = nullptr
);

}  // namespace trilibgo::app
