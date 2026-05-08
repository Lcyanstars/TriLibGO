#pragma once

#include "trilibgo/ai/analysis_model.h"
#include "trilibgo/core/game_record.h"
#include "trilibgo/core/rules_engine.h"

#include <optional>
#include <memory>
#include <string>
#include <vector>

namespace trilibgo::app {

/** \brief Orchestrates the game: rules engine, move history, undo, replay, and AI analysis.

    Acts as the bridge between the UI (BoardWidget/MainWindow) and the core engine.
    Supports live play, record/text replay, and self-play trace visualization with
    policy heatmap overlays.
*/
class GameController {
public:
    /** \brief One frame of replay analysis: optional AI analysis + metadata. */
    struct ReplayAnalysisFrame {
        std::optional<trilibgo::ai::AnalysisResult> analysis;
        std::string move_label;
        double root_value = 0.0;
        int turn = 0;
        bool has_full_policy = false;
    };

    /** \brief Create a controller for a new game with the given config. */
    explicit GameController(trilibgo::core::GameConfig config);

    /** \brief Current game state. */
    [[nodiscard]] const trilibgo::core::GameState& state() const;
    /** \brief Last error message, if any. */
    [[nodiscard]] std::optional<std::string> error_message() const;
    /** \brief Human-readable status line. */
    [[nodiscard]] std::string status_text() const;
    /** \brief Serialize the current game to text. */
    [[nodiscard]] std::string serialized_record() const;
    /** \brief Position estimate (leader, margin, review recommendation). */
    [[nodiscard]] trilibgo::core::ScoreEstimate estimate() const;
    /** \brief AI analysis of the current position, if a model is loaded. */
    [[nodiscard]] std::optional<trilibgo::ai::AnalysisResult> analysis() const;
    /** \brief Whether the controller is replaying a saved game. */
    [[nodiscard]] bool is_replay_mode() const;
    /** \brief Current replay position index. */
    [[nodiscard]] int replay_index() const;
    /** \brief Total number of replay positions. */
    [[nodiscard]] int replay_total() const;
    /** \brief Whether analysis heatmap overlay is enabled. */
    [[nodiscard]] bool analysis_overlay_enabled() const;
    /** \brief Whether replay data includes per-frame analysis. */
    [[nodiscard]] bool has_replay_analysis() const;
    /** \brief Current replay frame analysis data. */
    [[nodiscard]] std::optional<ReplayAnalysisFrame> replay_analysis_frame() const;
    /** \brief Name of the loaded replay source. */
    [[nodiscard]] std::string replay_source_name() const;
    /** \brief Result summary of the replayed game. */
    [[nodiscard]] std::string replay_result_summary() const;

    // --- Actions ---

    /** \brief Play a stone at the given vertex. Returns true if successful. */
    bool play(int vertex);
    /** \brief Pass the current turn. */
    void pass_turn();
    /** \brief Toggle a chain's life-death status during endgame review. */
    bool toggle_review_group(int vertex);
    /** \brief Accept all suggested chain statuses. */
    void accept_review_suggestion();
    /** \brief Clear user overrides during review. */
    void clear_review_overrides();
    /** \brief Finalize the reviewed result. */
    void finalize_review();
    /** \brief Undo the last move. */
    void undo();
    /** \brief Reset to a new game with the same config. */
    void reset();
    /** \brief Start a new game with the given config. */
    void new_game(trilibgo::core::GameConfig config);
    /** \brief Load a game from text record for replay. */
    bool load_record_text(const std::string& text);
    /** \brief Load a self-play trace (JSONL) for replay with analysis. */
    bool load_selfplay_trace_text(const std::string& text, int game_number = 1);
    /** \brief Load an ONNX analysis model. */
    bool load_analysis_model(const std::string& model_path);
    /** \brief Unload the current analysis model. */
    void clear_analysis_model();
    /** \brief Toggle analysis heatmap overlay on/off. */
    void toggle_analysis_overlay();
    /** \brief Jump to a specific replay position. */
    void set_replay_index(int index);
    /** \brief Go back one move in replay. */
    void step_replay_backward();
    /** \brief Go forward one move in replay. */
    void step_replay_forward();

private:
    void initialize_state();

    trilibgo::core::GameConfig config_;
    trilibgo::core::RulesEngine rules_;
    trilibgo::core::GameState state_;
    std::vector<trilibgo::core::GameState> replay_states_;
    std::vector<std::optional<ReplayAnalysisFrame>> replay_analysis_;
    bool replay_mode_ = false;
    int replay_index_ = 0;
    std::string replay_source_name_;
    std::string replay_result_summary_;
    std::vector<trilibgo::core::GameState> undo_stack_;
    std::optional<std::string> error_message_;
    std::shared_ptr<trilibgo::ai::IAnalysisModel> analysis_model_;
    bool analysis_overlay_enabled_ = false;
};

}  // namespace trilibgo::app
