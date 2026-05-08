#include "trilibgo/ai/random_agent.h"

#include <chrono>

namespace trilibgo::ai {

RandomAgent::RandomAgent() : rng_(static_cast<std::mt19937::result_type>(std::chrono::steady_clock::now().time_since_epoch().count())) {}

trilibgo::core::Move RandomAgent::select_move(const trilibgo::core::GameState& state) {
    auto moves = rules_.legal_moves(state);
    if (moves.empty()) {
        return trilibgo::core::Move::pass();
    }
    std::uniform_int_distribution<std::size_t> distribution(0, moves.size() - 1);
    return moves[distribution(rng_)];
}

}  // namespace trilibgo::ai
