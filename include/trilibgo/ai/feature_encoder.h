#pragma once

#include "trilibgo/core/game_state.h"

#include <vector>

namespace trilibgo::ai {

/** \brief Converts a GameState into fixed-size feature planes for neural network input.

    Produces per-vertex feature planes (black stones, white stones, history, etc.)
    and global scalar features (consecutive passes, current player, captures).
    The encoding is identical between C++ and Python for ONNX model compatibility.
*/
class FeatureEncoder {
public:
    /** \brief Create an encoder with the given number of history frames. */
    explicit FeatureEncoder(int input_history = 4);

    /** \brief Encode the game state as per-vertex feature planes. */
    [[nodiscard]] std::vector<trilibgo::core::Plane> encode(const trilibgo::core::GameState& state) const;
    /** \brief Encode global (non-spatial) features as a flat float vector. */
    [[nodiscard]] std::vector<float> encode_global_features(const trilibgo::core::GameState& state) const;

private:
    int input_history_ = 4;
};

}  // namespace trilibgo::ai
