#include "trilibgo/app/game_controller.h"

#include "replay_compat.h"
#include "trilibgo/ai/onnx_analysis_model.h"
#include "trilibgo/core/board_topology.h"

#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QJsonValue>

#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>

namespace trilibgo::app {
namespace {

int infer_side_length_from_actions(const QJsonArray& moves) {
    int max_action = -1;
    for (const auto& move_value : moves) {
        const QJsonObject move_object = move_value.toObject();
        max_action = std::max(max_action, move_object.value("action").toInt(-1));
        const QJsonArray policy_top = move_object.value("policy_top").toArray();
        for (const auto& item_value : policy_top) {
            const QJsonArray pair = item_value.toArray();
            if (!pair.isEmpty()) {
                max_action = std::max(max_action, pair.at(0).toInt(-1));
            }
        }
    }
    if (max_action < 0) {
        return 4;
    }
    for (int side = 2; side <= 12; ++side) {
        trilibgo::core::BoardTopology topology(side);
        if (topology.vertex_count() == max_action || topology.vertex_count() == max_action + 1) {
            return side;
        }
    }
    return 4;
}

QJsonObject build_replay_object(const trilibgo::core::GameState& state, std::string_view source) {
    QJsonObject root;
    root.insert("format", "trilibgo-replay-v2");
    root.insert("source", QString::fromStdString(std::string(source)));
    root.insert("side_length", state.config().side_length);
    root.insert("komi", state.config().komi);
    root.insert("allow_suicide", state.config().allow_suicide);
    root.insert("move_count", state.move_number());

    QJsonArray moves;
    for (std::size_t i = 0; i < state.move_history().size(); ++i) {
        const auto& move = state.move_history()[i];
        QJsonObject entry;
        entry.insert("turn", static_cast<int>(i + 1));
        entry.insert("player", ((i % 2) == 0) ? "B" : "W");
        entry.insert("action", move.kind == trilibgo::core::MoveKind::Place
                                   ? move.coord.index
                                   : (move.kind == trilibgo::core::MoveKind::Pass ? state.topology().vertex_count() : -1));
        if (move.kind == trilibgo::core::MoveKind::Pass) {
            entry.insert("move", "pass");
        } else if (move.kind == trilibgo::core::MoveKind::Resign) {
            entry.insert("move", "resign");
        } else {
            entry.insert("move", QString::fromStdString(state.topology().label_for_vertex(move.coord.index)));
        }
        moves.push_back(entry);
    }
    root.insert("moves", moves);
    if (state.result().has_value()) {
        root.insert("result", QString::fromStdString(state.result()->summary));
    } else if (state.review_state().has_value()) {
        root.insert("review", QString::fromStdString(state.review_state()->estimate.summary));
    }
    return root;
}

std::vector<trilibgo::ai::PolicyLogit> parse_full_policy(const QJsonArray& policy_array) {
    std::vector<trilibgo::ai::PolicyLogit> policy;
    policy.reserve(static_cast<std::size_t>(policy_array.size()));
    for (int i = 0; i < policy_array.size(); ++i) {
        const double probability = policy_array.at(i).toDouble(std::numeric_limits<double>::quiet_NaN());
        if (!std::isfinite(probability) || probability <= 0.0) {
            continue;
        }
        policy.push_back({i, probability});
    }
    return policy;
}

std::string format_replay_result(const QJsonObject& root) {
    const bool has_black_score = root.contains("black_score");
    const bool has_white_score = root.contains("white_score");
    if (!has_black_score && !has_white_score) {
        const QString explicit_result = root.value("result").toString();
        if (!explicit_result.isEmpty()) {
            return explicit_result.toStdString();
        }
        const QString review = root.value("review").toString();
        if (!review.isEmpty()) {
            return review.toStdString();
        }
        return {};
    }

    const double black_score = root.value("black_score").toDouble(0.0);
    const double white_score = root.value("white_score").toDouble(0.0);
    const double margin = std::abs(black_score - white_score);
    if (margin < 1e-6) {
        return "Draw";
    }
    std::ostringstream out;
    out << (black_score > white_score ? "Black +" : "White +") << margin;
    return out.str();
}

std::vector<int> parse_vertex_array(const QJsonArray& values, int vertex_count) {
    std::vector<int> vertices;
    vertices.reserve(static_cast<std::size_t>(values.size()));
    for (const auto& value : values) {
        const int vertex = value.toInt(-1);
        if (vertex >= 0 && vertex < vertex_count) {
            vertices.push_back(vertex);
        }
    }
    return vertices;
}

void apply_selfplay_cleanup_overlay(
    trilibgo::core::GameState& state,
    const trilibgo::core::RulesEngine& rules,
    const QJsonObject& root
) {
    const auto cleaned_vertices = parse_vertex_array(root.value("cleaned_dead_vertices").toArray(), state.topology().vertex_count());
    if (cleaned_vertices.empty()) {
        return;
    }

    trilibgo::core::EndgameReviewState review;
    review.suggested_statuses.assign(
        static_cast<std::size_t>(state.topology().vertex_count()),
        trilibgo::core::ChainLifeStatus::Alive
    );
    review.user_statuses = review.suggested_statuses;
    for (int vertex : cleaned_vertices) {
        if (state.at(vertex) == trilibgo::core::Stone::Empty) {
            continue;
        }
        review.suggested_statuses[static_cast<std::size_t>(vertex)] = trilibgo::core::ChainLifeStatus::Dead;
        review.user_statuses[static_cast<std::size_t>(vertex)] = trilibgo::core::ChainLifeStatus::Dead;
    }
    review.estimate.score = rules.score(state, review.user_statuses);
    review.estimate.recommended_endgame_review = true;
    review.estimate.summary = "Self-play cleanup";
    state.set_review_state(std::move(review));
    state.set_phase(trilibgo::core::Phase::ReviewingEndgame);
}

}  // namespace

GameController::GameController(trilibgo::core::GameConfig config)
    : config_(config), state_(config_), analysis_model_(std::make_shared<trilibgo::ai::NullAnalysisModel>()) {
    initialize_state();
}

void GameController::initialize_state() {
    state_ = trilibgo::core::GameState(config_);
    state_.clear_and_seed_board_hash(rules_.compute_hash(state_.board(), state_.current_player()));
    undo_stack_.clear();
    replay_states_.clear();
    replay_analysis_.clear();
    replay_mode_ = false;
    replay_index_ = 0;
    replay_source_name_.clear();
    replay_result_summary_.clear();
    error_message_.reset();
}

const trilibgo::core::GameState& GameController::state() const { return state_; }
std::optional<std::string> GameController::error_message() const { return error_message_; }
trilibgo::core::ScoreEstimate GameController::estimate() const { return rules_.estimate_position(state_); }
std::optional<trilibgo::ai::AnalysisResult> GameController::analysis() const {
    const auto replay_frame = replay_analysis_frame();
    if (replay_frame.has_value() && replay_frame->analysis.has_value()) {
        return replay_frame->analysis;
    }
    return analysis_model_ ? analysis_model_->analyze(state_) : std::nullopt;
}
bool GameController::is_replay_mode() const { return replay_mode_; }
int GameController::replay_index() const { return replay_index_; }
int GameController::replay_total() const { return static_cast<int>(replay_states_.size()); }
bool GameController::analysis_overlay_enabled() const { return analysis_overlay_enabled_; }
bool GameController::has_replay_analysis() const { return replay_mode_ && !replay_analysis_.empty(); }
std::string GameController::replay_source_name() const { return replay_source_name_; }
std::string GameController::replay_result_summary() const { return replay_result_summary_; }

std::optional<GameController::ReplayAnalysisFrame> GameController::replay_analysis_frame() const {
    if (!has_replay_analysis() || replay_index_ < 0 || replay_index_ >= static_cast<int>(replay_analysis_.size())) {
        return std::nullopt;
    }
    return replay_analysis_[static_cast<std::size_t>(replay_index_)];
}

std::string GameController::status_text() const {
    if (replay_mode_) {
        std::ostringstream out;
        out << "Replay mode";
        out << " | Move " << replay_index_ << "/" << std::max(static_cast<int>(replay_states_.size()) - 1, 0);
        if (!replay_source_name_.empty()) {
            out << " | " << replay_source_name_;
        }
        if (error_message_.has_value()) {
            out << " | " << *error_message_;
        }
        return out.str();
    }
    if (state_.result().has_value()) {
        return state_.result()->summary;
    }
    if (state_.phase() == trilibgo::core::Phase::ReviewingEndgame) {
        return "Endgame review: click groups to toggle alive/dead, then confirm scoring.";
    }

    std::ostringstream out;
    out << (state_.current_player() == trilibgo::core::Stone::Black ? "Black" : "White") << " to play";
    out << " | Move " << (state_.move_number() + 1);
    if (error_message_.has_value()) {
        out << " | " << *error_message_;
    }
    return out.str();
}

std::string GameController::serialized_record() const {
    return QJsonDocument(build_replay_object(state_, "record")).toJson(QJsonDocument::Compact).toStdString();
}

bool GameController::play(int vertex) {
    if (replay_mode_) {
        error_message_ = "Replay mode is read-only.";
        return false;
    }
    if (vertex < 0) {
        error_message_ = "No vertex selected.";
        return false;
    }
    undo_stack_.push_back(state_);
    const auto outcome = rules_.apply_move(state_, trilibgo::core::Move::place(vertex));
    if (!outcome.success) {
        undo_stack_.pop_back();
        error_message_ = outcome.message;
        return false;
    }
    error_message_.reset();
    return true;
}

void GameController::pass_turn() {
    if (replay_mode_) {
        error_message_ = "Replay mode is read-only.";
        return;
    }
    undo_stack_.push_back(state_);
    const auto outcome = rules_.apply_move(state_, trilibgo::core::Move::pass());
    if (!outcome.success) {
        undo_stack_.pop_back();
        error_message_ = outcome.message;
        return;
    }
    error_message_.reset();
}

bool GameController::toggle_review_group(int vertex) {
    if (replay_mode_) {
        error_message_ = "Replay mode is read-only.";
        return false;
    }
    if (!rules_.toggle_group_status(state_, vertex)) {
        error_message_ = "Unable to toggle group status.";
        return false;
    }
    error_message_.reset();
    return true;
}

void GameController::accept_review_suggestion() {
    if (replay_mode_) {
        error_message_ = "Replay mode is read-only.";
        return;
    }
    if (!rules_.accept_review_suggestion(state_)) {
        error_message_ = "Unable to accept review suggestion.";
        return;
    }
    error_message_.reset();
}

void GameController::clear_review_overrides() {
    if (replay_mode_) {
        error_message_ = "Replay mode is read-only.";
        return;
    }
    if (!rules_.clear_review_overrides(state_)) {
        error_message_ = "Unable to clear review overrides.";
        return;
    }
    error_message_.reset();
}

void GameController::finalize_review() {
    if (replay_mode_) {
        error_message_ = "Replay mode is read-only.";
        return;
    }
    if (!rules_.finalize_reviewed_result(state_)) {
        error_message_ = "Unable to finalize review.";
        return;
    }
    error_message_.reset();
}

void GameController::undo() {
    if (replay_mode_) {
        error_message_ = "Replay mode is read-only.";
        return;
    }
    if (undo_stack_.empty()) {
        error_message_ = "Nothing to undo.";
        return;
    }
    state_ = undo_stack_.back();
    undo_stack_.pop_back();
    error_message_.reset();
}

void GameController::reset() {
    initialize_state();
}

void GameController::new_game(trilibgo::core::GameConfig config) {
    config_ = config;
    initialize_state();
}

bool GameController::load_record_text(const std::string& text) {
    const QJsonDocument doc = QJsonDocument::fromJson(QByteArray::fromStdString(text));
    if (doc.isObject() && doc.object().contains("moves")) {
        return load_selfplay_trace_text(text, 1);
    }

    const auto parsed = trilibgo::core::GameRecord::parse(text);
    if (!parsed.has_value()) {
        error_message_ = "Unable to parse record.";
        return false;
    }

    std::vector<trilibgo::core::GameState> next_replay_states;
    std::vector<std::optional<ReplayAnalysisFrame>> next_replay_analysis;
    trilibgo::core::GameState replay_state(parsed->config);
    replay_state.clear_and_seed_board_hash(rules_.compute_hash(replay_state.board(), replay_state.current_player()));
    next_replay_states.push_back(replay_state);

    for (std::size_t move_index = 0; move_index < parsed->moves.size(); ++move_index) {
        const auto& move = parsed->moves[move_index];
        if (!apply_replay_move_compat(replay_state, rules_, move)) {
            error_message_ = "Record contains an illegal move at turn " + std::to_string(move_index + 1) + ".";
            return false;
        }
        normalize_replay_state_after_move(replay_state, move_index + 1 < parsed->moves.size());
        next_replay_states.push_back(replay_state);
    }

    config_ = parsed->config;
    replay_states_ = std::move(next_replay_states);
    replay_analysis_ = std::move(next_replay_analysis);
    replay_mode_ = true;
    replay_index_ = 0;
    replay_source_name_ = "record";
    replay_result_summary_.clear();
    state_ = replay_states_.front();
    undo_stack_.clear();
    error_message_.reset();
    return true;
}

bool GameController::load_selfplay_trace_text(const std::string& text, int game_number) {
    const QJsonDocument doc = QJsonDocument::fromJson(QByteArray::fromStdString(text));
    if (!doc.isObject()) {
        error_message_ = "Unable to parse selfplay trace JSON.";
        return false;
    }

    const QJsonObject root = doc.object();
    const QJsonArray moves = root.value("moves").toArray();
    if (moves.isEmpty()) {
        error_message_ = "Selfplay trace contains no moves.";
        return false;
    }

    trilibgo::core::GameConfig next_config = config_;
    const int side_length = root.value("side_length").toInt(0);
    next_config.side_length = side_length > 0 ? side_length : infer_side_length_from_actions(moves);
    next_config.komi = root.value("komi").toDouble(0.0);
    next_config.allow_suicide = root.value("allow_suicide").toBool(false);

    std::vector<trilibgo::core::GameState> next_replay_states;
    std::vector<std::optional<ReplayAnalysisFrame>> next_replay_analysis;
    bool has_analysis_payload = false;
    bool used_legacy_replay = false;
    trilibgo::core::GameState replay_state(next_config);
    replay_state.clear_and_seed_board_hash(rules_.compute_hash(replay_state.board(), replay_state.current_player()));
    next_replay_states.push_back(replay_state);
    next_replay_analysis.push_back(std::nullopt);

    for (int move_index = 0; move_index < moves.size(); ++move_index) {
        const QJsonObject move_object = moves.at(move_index).toObject();
        const int action = move_object.value("action").toInt(-1);
        ReplayAnalysisFrame frame;
        frame.turn = move_object.value("turn").toInt(0);
        frame.move_label = move_object.value("move").toString().toStdString();
        frame.root_value = move_object.value("root_value").toDouble(0.0);

        trilibgo::ai::AnalysisResult analysis;
        analysis.winrate = std::clamp((frame.root_value + 1.0) * 0.5, 0.0, 1.0);
        const QJsonArray policy_array = move_object.value("policy").toArray();
        analysis.policy = parse_full_policy(policy_array);
        frame.has_full_policy = !analysis.policy.empty();
        if (analysis.policy.empty()) {
            const QJsonArray policy_top = move_object.value("policy_top").toArray();
            for (const auto& item_value : policy_top) {
                const QJsonArray pair = item_value.toArray();
                if (pair.size() < 2) {
                    continue;
                }
                analysis.policy.push_back({pair.at(0).toInt(-1), pair.at(1).toDouble(0.0)});
            }
        }
        has_analysis_payload = has_analysis_payload || !analysis.policy.empty();
        analysis.top_policy = analysis.policy;
        std::sort(analysis.top_policy.begin(), analysis.top_policy.end(), [](const auto& lhs, const auto& rhs) {
            return lhs.probability > rhs.probability;
        });
        if (analysis.top_policy.size() > 8) {
            analysis.top_policy.resize(8);
        }
        next_replay_analysis.back() = frame;
        next_replay_analysis.back()->analysis = analysis;

        const auto action_move = trilibgo::ai::ActionCodec::action_index_to_move(replay_state, action);
        const auto token_move = parse_replay_move_token(replay_state, frame.move_label);
        auto move = action_move.has_value() ? action_move : token_move;
        if (!move.has_value()) {
            error_message_ = "Selfplay trace contains an invalid action at turn " + std::to_string(move_index + 1) + ".";
            return false;
        }

        ReplayApplyMode apply_mode = ReplayApplyMode::Standard;
        bool applied = apply_replay_move_compat(replay_state, rules_, *move, &apply_mode);
        if (
            !applied && token_move.has_value() &&
            (move->kind != token_move->kind || move->coord.index != token_move->coord.index)
        ) {
            applied = apply_replay_move_compat(replay_state, rules_, *token_move, &apply_mode);
            if (applied) {
                move = token_move;
            }
        }
        if (!applied) {
            error_message_ =
                "Selfplay trace contains an illegal move at turn " + std::to_string(move_index + 1) + ": Illegal move.";
            return false;
        }
        used_legacy_replay = used_legacy_replay || apply_mode == ReplayApplyMode::LegacyNoKo;

        normalize_replay_state_after_move(replay_state, move_index + 1 < moves.size());
        next_replay_states.push_back(replay_state);
        next_replay_analysis.push_back(std::nullopt);
    }

    if (!next_replay_states.empty()) {
        apply_selfplay_cleanup_overlay(next_replay_states.back(), rules_, root);
    }

    config_ = next_config;
    replay_states_ = std::move(next_replay_states);
    replay_analysis_ = std::move(next_replay_analysis);
    replay_mode_ = true;
    replay_index_ = 0;
    const std::string source = root.value("source").toString("selfplay").toStdString();
    replay_source_name_ = source + " #" + std::to_string(game_number) + (used_legacy_replay ? " [legacy-ko]" : "");
    replay_result_summary_ = format_replay_result(root);
    state_ = replay_states_.front();
    undo_stack_.clear();
    analysis_overlay_enabled_ = has_analysis_payload;
    error_message_.reset();
    return true;
}

bool GameController::load_analysis_model(const std::string& model_path) {
    std::string error;
    auto model = trilibgo::ai::load_onnx_analysis_model(model_path, error);
    if (!model) {
        error_message_ = error;
        return false;
    }
    analysis_model_ = std::move(model);
    analysis_overlay_enabled_ = true;
    error_message_.reset();
    return true;
}

void GameController::clear_analysis_model() {
    analysis_model_ = std::make_shared<trilibgo::ai::NullAnalysisModel>();
    analysis_overlay_enabled_ = false;
    error_message_.reset();
}

void GameController::toggle_analysis_overlay() {
    analysis_overlay_enabled_ = !analysis_overlay_enabled_;
}

void GameController::set_replay_index(int index) {
    if (!replay_mode_ || replay_states_.empty()) {
        return;
    }
    replay_index_ = std::clamp(index, 0, static_cast<int>(replay_states_.size()) - 1);
    state_ = replay_states_[static_cast<std::size_t>(replay_index_)];
}

void GameController::step_replay_backward() {
    if (!replay_mode_ || replay_states_.empty() || replay_index_ <= 0) {
        return;
    }
    --replay_index_;
    state_ = replay_states_[static_cast<std::size_t>(replay_index_)];
}

void GameController::step_replay_forward() {
    if (!replay_mode_ || replay_states_.empty() || replay_index_ + 1 >= static_cast<int>(replay_states_.size())) {
        return;
    }
    ++replay_index_;
    state_ = replay_states_[static_cast<std::size_t>(replay_index_)];
}

}  // namespace trilibgo::app
