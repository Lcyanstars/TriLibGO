#pragma once

#include "trilibgo/core/game_state.h"

namespace trilibgo::ai {

/** \brief Pluggable agent interface for move selection.

    Implementations include RandomAgent and (when ONNX Runtime is available)
    neural-network-driven agents loaded from exported models.
*/
class IAgent {
public:
    virtual ~IAgent() = default;
    /** \brief Select a move given the current game state. */
    [[nodiscard]] virtual trilibgo::core::Move select_move(const trilibgo::core::GameState& state) = 0;
};

}  // namespace trilibgo::ai
