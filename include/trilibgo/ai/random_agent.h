#pragma once

#include "trilibgo/ai/agent.h"
#include "trilibgo/core/rules_engine.h"

#include <random>

namespace trilibgo::ai {

/** \brief Agent that selects a uniformly random legal move. */
class RandomAgent final : public IAgent {
public:
    RandomAgent();
    /** \brief Pick a random legal move (including pass when legal). */
    [[nodiscard]] trilibgo::core::Move select_move(const trilibgo::core::GameState& state) override;

private:
    trilibgo::core::RulesEngine rules_;
    std::mt19937 rng_;
};

}  // namespace trilibgo::ai
