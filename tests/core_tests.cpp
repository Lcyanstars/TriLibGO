#include "trilibgo/ai/analysis_model.h"
#include "trilibgo/ai/feature_encoder.h"
#include "trilibgo/core/game_record.h"
#include "trilibgo/core/rules_engine.h"

#include <cstdlib>
#include <iostream>
#include <stdexcept>

namespace {

void expect(bool condition, const char* message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

trilibgo::core::GameState fresh_state(int side_length, double komi = 0.0) {
    trilibgo::core::RulesEngine rules;
    trilibgo::core::GameState state({.side_length = side_length, .komi = komi, .allow_suicide = false});
    state.clear_and_seed_board_hash(rules.compute_hash(state.board(), state.current_player()));
    return state;
}

void test_topology() {
    trilibgo::core::BoardTopology topology(4);
    expect(topology.vertex_count() == 96, "Unexpected vertex count for side_length=4");

    int degree_three = 0;
    for (int i = 0; i < topology.vertex_count(); ++i) {
        const int degree = static_cast<int>(topology.neighbors(i).size());
        expect(degree >= 2 && degree <= 3, "Vertex degree out of range");
        if (degree == 3) {
            ++degree_three;
        }
    }
    expect(degree_three > 0, "Expected interior vertices with 3 liberties");
}

void test_capture() {
    trilibgo::core::RulesEngine rules;
    auto state = fresh_state(2, 0.5);
    const auto& topology = state.topology();

    int target = -1;
    for (int i = 0; i < topology.vertex_count(); ++i) {
        if (static_cast<int>(topology.neighbors(i).size()) == 3) {
            target = i;
            break;
        }
    }
    expect(target >= 0, "Need an interior vertex");
    const auto neighbors = topology.neighbors(target);

    expect(rules.apply_move(state, trilibgo::core::Move::place(target)).success, "Black target");
    expect(rules.apply_move(state, trilibgo::core::Move::place(neighbors[0])).success, "White surround 1");

    int filler = -1;
    for (int i = 0; i < topology.vertex_count(); ++i) {
        if (state.at(i) == trilibgo::core::Stone::Empty && i != neighbors[1] && i != neighbors[2]) {
            filler = i;
            break;
        }
    }
    expect(filler >= 0, "Need filler 1");
    expect(rules.apply_move(state, trilibgo::core::Move::place(filler)).success, "Black filler 1");
    expect(rules.apply_move(state, trilibgo::core::Move::place(neighbors[1])).success, "White surround 2");

    int filler2 = -1;
    for (int i = 0; i < topology.vertex_count(); ++i) {
        if (state.at(i) == trilibgo::core::Stone::Empty && i != neighbors[2]) {
            filler2 = i;
            break;
        }
    }
    expect(filler2 >= 0, "Need filler 2");
    expect(rules.apply_move(state, trilibgo::core::Move::place(filler2)).success, "Black filler 2");
    expect(rules.apply_move(state, trilibgo::core::Move::place(neighbors[2])).success, "White capture");

    expect(state.at(target) == trilibgo::core::Stone::Empty, "Captured stone should be removed");
    expect(state.captures(trilibgo::core::Stone::White) == 1, "White capture count mismatch");
}

void test_game_end() {
    trilibgo::core::RulesEngine rules;
    auto state = fresh_state(2, 0.0);
    expect(rules.apply_move(state, trilibgo::core::Move::pass()).success, "Black pass");
    expect(rules.apply_move(state, trilibgo::core::Move::pass()).success, "White pass");
    expect(state.phase() == trilibgo::core::Phase::ReviewingEndgame, "Game should enter review after two passes");
    expect(state.review_state().has_value(), "Review phase should have review state");
    expect(rules.finalize_reviewed_result(state), "Review should finalize");
    expect(state.is_finished(), "Finalized review should finish the game");
    expect(state.result().has_value(), "Finished game should have result");
    expect(state.result()->winner == trilibgo::core::Stone::Empty, "Empty board should now be a draw");
}

void test_estimate() {
    trilibgo::core::RulesEngine rules;
    auto state = fresh_state(2, 0.0);
    const auto estimate = rules.estimate_position(state);
    expect(estimate.score.komi == 0.0, "Default komi should be zero");
    expect(!estimate.summary.empty(), "Estimate should provide a summary");
}

void test_toggle_review_dead_group() {
    trilibgo::core::RulesEngine rules;
    auto state = fresh_state(2, 0.0);
    expect(rules.apply_move(state, trilibgo::core::Move::pass()).success, "Black pass");
    expect(rules.apply_move(state, trilibgo::core::Move::pass()).success, "White pass");
    expect(state.review_state().has_value(), "Review state must exist");
    int stone_vertex = -1;
    for (int i = 0; i < state.topology().vertex_count(); ++i) {
        if (state.at(i) != trilibgo::core::Stone::Empty) {
            stone_vertex = i;
            break;
        }
    }
    expect(stone_vertex == -1, "No stones should exist on empty-board review");
    expect(!rules.toggle_group_status(state, 0), "Cannot toggle an empty vertex");
}

void test_game_record_roundtrip() {
    trilibgo::core::RulesEngine rules;
    auto state = fresh_state(2, 0.0);
    expect(rules.apply_move(state, trilibgo::core::Move::place(0)).success, "Move 1");
    expect(rules.apply_move(state, trilibgo::core::Move::pass()).success, "Move 2");

    const auto text = trilibgo::core::GameRecord::serialize(state);
    const auto parsed = trilibgo::core::GameRecord::parse(text);
    expect(parsed.has_value(), "Serialized record should parse");
    expect(parsed->moves.size() == 2, "Parsed move count mismatch");
    expect(parsed->config.komi == 0.0, "Parsed komi mismatch");

    const auto wrapped = trilibgo::core::GameRecord::serialize_moves_wrapped(state, 1);
    expect(wrapped.find('\n') != std::string::npos, "Wrapped moves should contain line breaks");
}

void test_action_codec() {
    auto state = fresh_state(2, 0.0);
    expect(trilibgo::ai::ActionCodec::policy_size(state) == state.topology().vertex_count() + 1, "Policy size mismatch");
    expect(trilibgo::ai::ActionCodec::pass_index(state) == state.topology().vertex_count(), "Pass index mismatch");
    const auto move = trilibgo::ai::ActionCodec::action_index_to_move(state, 0);
    expect(move.has_value() && move->kind == trilibgo::core::MoveKind::Place && move->coord.index == 0, "Vertex action decode mismatch");
    const auto pass_move = trilibgo::ai::ActionCodec::action_index_to_move(state, trilibgo::ai::ActionCodec::pass_index(state));
    expect(pass_move.has_value() && pass_move->kind == trilibgo::core::MoveKind::Pass, "Pass action decode mismatch");
}

void test_feature_planes() {
    auto state = fresh_state(2, 0.5);
    trilibgo::ai::FeatureEncoder encoder;
    const auto planes = encoder.encode(state);
    expect(planes.size() == 10, "Expected 10 spatial feature planes for input_history=4");
    expect(planes[0].size() == static_cast<std::size_t>(state.topology().vertex_count()), "Plane size mismatch");
    expect(planes[2].size() == static_cast<std::size_t>(state.topology().vertex_count()), "Legal plane size mismatch");
    expect(planes[4].size() == static_cast<std::size_t>(state.topology().vertex_count()), "History plane size mismatch");
    const auto global_features = encoder.encode_global_features(state);
    expect(global_features.size() == 8, "Expected 8 global features");
}

void test_global_features_count_shared_eye_once() {
    auto state = fresh_state(2, 0.5);
    constexpr int eye = 11;
    std::vector<trilibgo::core::Stone> board(
        static_cast<std::size_t>(state.topology().vertex_count()),
        trilibgo::core::Stone::Black
    );
    board[static_cast<std::size_t>(eye)] = trilibgo::core::Stone::Empty;
    state.set_board(board);
    state.set_current_player(trilibgo::core::Stone::Black);

    trilibgo::ai::FeatureEncoder encoder;
    const auto global_features = encoder.encode_global_features(state);
    const float expected_count_plane_value = 1.0f / static_cast<float>(state.topology().vertex_count());

    expect(global_features[0] == expected_count_plane_value, "Shared eye should be counted once globally");
    expect(global_features[1] == 0.0f, "Unexpected two-liberty block count");
    expect(global_features[2] == 0.0f, "Unexpected three-plus-liberty block count");
}

}  // namespace

int main() {
    try {
        test_topology();
        test_capture();
        test_game_end();
        test_estimate();
        test_toggle_review_dead_group();
        test_game_record_roundtrip();
        test_action_codec();
        test_feature_planes();
        test_global_features_count_shared_eye_once();
        std::cout << "All tests passed.\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
