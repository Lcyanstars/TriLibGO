#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace trilibgo::core {

/** \brief Stone color — Empty, Black, or White. */
enum class Stone : std::uint8_t { Empty = 0, Black = 1, White = 2 };

/** \brief Return the opposite stone color. Empty stays Empty. */
inline Stone opposite(Stone stone) {
    if (stone == Stone::Black) {
        return Stone::White;
    }
    if (stone == Stone::White) {
        return Stone::Black;
    }
    return Stone::Empty;
}

/** \brief 2D pixel position of a vertex for rendering. */
struct VertexPosition {
    double x = 0.0;
    double y = 0.0;
};

/** \brief Vertex index on the board. index >= 0 means valid. */
struct BoardCoord {
    int index = -1;
    [[nodiscard]] bool is_valid() const { return index >= 0; }
};

/** \brief Kind of move: placing a stone, passing, or resigning. */
enum class MoveKind : std::uint8_t { Place, Pass, Resign };

/** \brief A move with its kind and board coordinate. */
struct Move {
    MoveKind kind = MoveKind::Pass;
    BoardCoord coord{};

    /** \brief Create a placement move at the given vertex index. */
    static Move place(int index) { return Move{MoveKind::Place, BoardCoord{index}}; }
    /** \brief Create a pass move. */
    static Move pass() { return Move{MoveKind::Pass, BoardCoord{-1}}; }
    /** \brief Create a resign move. */
    static Move resign() { return Move{MoveKind::Resign, BoardCoord{-1}}; }
};

/** \brief Detailed score breakdown for Chinese area scoring. */
struct ScoreBreakdown {
    int black_stones = 0;
    int white_stones = 0;
    int black_territory = 0;
    int white_territory = 0;
    int black_dead = 0;
    int white_dead = 0;
    int unsettled_points = 0;
    double komi = 0.0;

    /** \brief Black total = stones + territory. */
    [[nodiscard]] double black_total() const { return static_cast<double>(black_stones + black_territory); }
    /** \brief White total = stones + territory + komi. */
    [[nodiscard]] double white_total() const { return static_cast<double>(white_stones + white_territory) + komi; }
};

/** \brief Reason the game ended. */
enum class EndReason : std::uint8_t { None, ConsecutivePasses, Resignation };

/** \brief Current phase of the game lifecycle. */
enum class Phase : std::uint8_t { Playing, ReviewingEndgame, Finished };

/** \brief Life-death status of a stone chain during endgame review. */
enum class ChainLifeStatus : std::uint8_t { Alive, Dead, Unsettled };

/** \brief Final game result with winner, margin, and score breakdown. */
struct GameResult {
    EndReason reason = EndReason::None;
    Stone winner = Stone::Empty;
    double margin = 0.0;
    ScoreBreakdown score{};
    std::string summary;
};

/** \brief Estimated position evaluation with leader and confidence. */
struct ScoreEstimate {
    ScoreBreakdown score{};
    Stone leader = Stone::Empty;
    double lead = 0.0;
    bool recommended_endgame_review = false;
    std::string confidence;
    std::string summary;
};

/** \brief Endgame review state tracking suggested vs. user-overridden chain statuses. */
struct EndgameReviewState {
    std::vector<ChainLifeStatus> suggested_statuses;
    std::vector<ChainLifeStatus> user_statuses;
    ScoreEstimate estimate{};
};

/** \brief Configuration for a new game. */
struct GameConfig {
    int side_length = 4;     ///< Board side length in hex cells.
    double komi = 0.0;       ///< Compensation points for White.
    bool allow_suicide = false;
};

/** \brief A single feature plane (1D vector of float values, one per vertex). */
using Plane = std::vector<float>;

}  // namespace trilibgo::core
