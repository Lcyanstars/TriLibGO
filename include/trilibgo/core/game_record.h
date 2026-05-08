#pragma once

#include "trilibgo/core/game_state.h"

#include <optional>
#include <string>
#include <vector>

namespace trilibgo::core {

/** \brief Parsed game: config + move sequence. */
struct ParsedGameRecord {
    GameConfig config{};
    std::vector<Move> moves;
};

/** \brief Text serialization and deserialization of game records. */
class GameRecord {
public:
    /** \brief Serialize the full game state (config + moves + result) to text. */
    static std::string serialize(const GameState& state);
    /** \brief Serialize just the move list with line wrapping. */
    static std::string serialize_moves_wrapped(const GameState& state, int moves_per_line = 5);
    /** \brief Parse a text record back into config + moves. */
    static std::optional<ParsedGameRecord> parse(const std::string& text);
};

}  // namespace trilibgo::core
