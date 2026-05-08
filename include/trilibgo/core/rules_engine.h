#pragma once

#include "trilibgo/core/game_state.h"

#include <string>
#include <vector>

namespace trilibgo::core {

/** \brief Result of a move attempt: success flag, error message, captured vertices. */
struct MoveOutcome {
    bool success = false;
    std::string message;
    std::vector<int> captured_vertices;
};

/** \brief Game rules: move legality, capture, ko, scoring, and endgame review. */
class RulesEngine {
public:
    /** \brief Attempt to apply a move to the given state. Returns outcome with captured vertices. */
    [[nodiscard]] MoveOutcome apply_move(GameState& state, Move move) const;
    /** \brief Check if a move is legal in the current state. */
    [[nodiscard]] bool is_legal(const GameState& state, Move move) const;
    /** \brief Return all legal moves for the current player. */
    [[nodiscard]] std::vector<Move> legal_moves(const GameState& state) const;
    /** \brief Compute Chinese area scoring for the current board (no dead-stone removal). */
    [[nodiscard]] ScoreBreakdown score(const GameState& state) const;
    /** \brief Compute scoring with explicit chain life-death statuses. */
    [[nodiscard]] ScoreBreakdown score(const GameState& state, const std::vector<ChainLifeStatus>& statuses) const;
    /** \brief Estimate position with leader, lead margin, and review recommendation. */
    [[nodiscard]] ScoreEstimate estimate_position(const GameState& state) const;
    /** \brief Enter endgame review phase, auto-suggesting chain statuses. */
    [[nodiscard]] bool begin_endgame_review(GameState& state) const;
    /** \brief Toggle a chain's life status (Alive ↔ Dead) during review. */
    [[nodiscard]] bool toggle_group_status(GameState& state, int vertex) const;
    /** \brief Accept all suggested chain statuses. */
    [[nodiscard]] bool accept_review_suggestion(GameState& state) const;
    /** \brief Clear user overrides, reverting to suggested statuses. */
    [[nodiscard]] bool clear_review_overrides(GameState& state) const;
    /** \brief Finalize the reviewed result and end the game. */
    [[nodiscard]] bool finalize_reviewed_result(GameState& state) const;
    /** \brief Compute a Zobrist-style hash of the board for the given player. */
    [[nodiscard]] std::uint64_t compute_hash(const std::vector<Stone>& board, Stone player) const;

private:
    struct GroupInfo {
        std::vector<int> stones;
        int liberties = 0;
    };

    [[nodiscard]] GroupInfo collect_group(const GameState& state, const std::vector<Stone>& board, int start) const;
    [[nodiscard]] bool is_simple_ko_violation(const GameState& state, const std::vector<Stone>& next_board) const;
    [[nodiscard]] std::vector<ChainLifeStatus> suggest_chain_statuses(const GameState& state) const;
    [[nodiscard]] ScoreEstimate make_estimate(const GameState& state, const std::vector<ChainLifeStatus>& statuses, bool recommend_review) const;
    [[nodiscard]] std::vector<Stone> scoring_board(const GameState& state, const std::vector<ChainLifeStatus>& statuses, ScoreBreakdown& score) const;
    void finalize_if_finished(GameState& state) const;
};

}  // namespace trilibgo::core
