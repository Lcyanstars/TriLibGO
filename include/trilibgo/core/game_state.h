#pragma once

#include "trilibgo/core/board_topology.h"

#include <optional>
#include <vector>

namespace trilibgo::core {

/** \brief Mutable game state: board, player turn, move history, and phase tracking.

    Owns the board topology, stone placement, captures counters, consecutive passes,
    board hash history (for ko detection), and the endgame review state machine.
*/
class GameState {
public:
    /** \brief Create a fresh game with the given config. */
    explicit GameState(GameConfig config);

    /** \brief Game configuration (side length, komi, suicide rule). */
    [[nodiscard]] const GameConfig& config() const;
    /** \brief Board topology for this game's side length. */
    [[nodiscard]] const BoardTopology& topology() const;
    /** \brief Stone at a vertex (Empty/Black/White). */
    [[nodiscard]] Stone at(int vertex) const;
    /** \brief Player whose turn it is. */
    [[nodiscard]] Stone current_player() const;
    /** \brief Number of consecutive passes so far. */
    [[nodiscard]] int consecutive_passes() const;
    /** \brief Whether the game has ended. */
    [[nodiscard]] bool is_finished() const;
    /** \brief Current game phase (Playing/ReviewingEndgame/Finished). */
    [[nodiscard]] Phase phase() const;
    /** \brief Final result if the game has ended. */
    [[nodiscard]] const std::optional<GameResult>& result() const;
    /** \brief Endgame review state (chain status suggestions + user overrides). */
    [[nodiscard]] const std::optional<EndgameReviewState>& review_state() const;
    /** \brief History of all moves played. */
    [[nodiscard]] const std::vector<Move>& move_history() const;
    /** \brief History of board hashes (for ko detection). */
    [[nodiscard]] const std::vector<std::uint64_t>& board_hash_history() const;
    /** \brief Current move number (1-based). */
    [[nodiscard]] int move_number() const;
    /** \brief Number of stones captured by a player. */
    [[nodiscard]] int captures(Stone stone) const;
    /** \brief The raw board array (vertex index → Stone). */
    [[nodiscard]] const std::vector<Stone>& board() const;

    // --- Mutators (used by RulesEngine) ---

    void set_board(std::vector<Stone> board);
    void set_current_player(Stone stone);
    void set_consecutive_passes(int value);
    void set_phase(Phase phase);
    void set_result(std::optional<GameResult> result);
    void set_review_state(std::optional<EndgameReviewState> review_state);
    void push_move(Move move);
    void push_board_hash(std::uint64_t hash);
    void clear_and_seed_board_hash(std::uint64_t hash);
    void add_captures(Stone stone, int amount);

private:
    GameConfig config_;
    BoardTopology topology_;
    std::vector<Stone> board_;
    Stone current_player_ = Stone::Black;
    int consecutive_passes_ = 0;
    int black_captures_ = 0;
    int white_captures_ = 0;
    Phase phase_ = Phase::Playing;
    std::optional<GameResult> result_;
    std::optional<EndgameReviewState> review_state_;
    std::vector<Move> move_history_;
    std::vector<std::uint64_t> board_hash_history_;
};

}  // namespace trilibgo::core
