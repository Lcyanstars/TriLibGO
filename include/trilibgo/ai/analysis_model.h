#pragma once

#include "trilibgo/core/game_state.h"

#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace trilibgo::ai {

/** \brief A single action with its probability from the policy head. */
struct PolicyLogit {
    int action_index = -1;
    double probability = 0.0;
};

/** \brief Analysis result: winrate estimate and policy distribution. */
struct AnalysisResult {
    double winrate = 0.5;
    std::vector<PolicyLogit> policy;
    std::vector<PolicyLogit> top_policy;
};

/** \brief Abstract model that analyzes a position (winrate + policy).

    The primary implementation is OnnxAnalysisModel (gated by TRILIBGO_HAS_ONNXRUNTIME).
    NullAnalysisModel provides a no-op fallback.
*/
class IAnalysisModel {
public:
    virtual ~IAnalysisModel() = default;
    /** \brief Analyze a position. Returns nullopt if analysis is unavailable. */
    [[nodiscard]] virtual std::optional<AnalysisResult> analyze(const trilibgo::core::GameState& state) const = 0;
};

/** \brief No-op analysis model that always returns nullopt. */
class NullAnalysisModel final : public IAnalysisModel {
public:
    [[nodiscard]] std::optional<AnalysisResult> analyze(const trilibgo::core::GameState& state) const override;
};

/** \brief Maps between board vertices and policy-index space.

    Policy size = vertex_count + 1 (for pass). Vertex 0..N-1 maps to policy[0..N-1];
    pass maps to policy[N].
*/
class ActionCodec {
public:
    /** \brief Total number of policy logits (vertices + pass). */
    [[nodiscard]] static int policy_size(const trilibgo::core::GameState& state);
    /** \brief Policy index for the pass action. */
    [[nodiscard]] static int pass_index(const trilibgo::core::GameState& state);
    /** \brief Convert a vertex index to its policy index. */
    [[nodiscard]] static int vertex_to_action_index(const trilibgo::core::GameState& state, int vertex);
    /** \brief Convert a policy index back to a Move. */
    [[nodiscard]] static std::optional<trilibgo::core::Move> action_index_to_move(const trilibgo::core::GameState& state, int action_index);
};

}  // namespace trilibgo::ai
