#include "trilibgo/core/game_record.h"

#include <sstream>
#include <unordered_map>

namespace trilibgo::core {
namespace {

std::string move_token(const GameState& state, const Move& move) {
    if (move.kind == MoveKind::Pass) {
        return "pass";
    }
    if (move.kind == MoveKind::Resign) {
        return "resign";
    }
    return state.topology().label_for_vertex(move.coord.index);
}

std::string trim(const std::string& input) {
    const auto start = input.find_first_not_of(" \t\r\n");
    if (start == std::string::npos) {
        return {};
    }
    const auto end = input.find_last_not_of(" \t\r\n");
    return input.substr(start, end - start + 1);
}

}  // namespace

std::string GameRecord::serialize(const GameState& state) {
    std::ostringstream out;
    out << "format=trilibgo-record-v1\n";
    out << "side_length=" << state.config().side_length << "\n";
    out << "komi=" << state.config().komi << "\n";
    out << "allow_suicide=" << (state.config().allow_suicide ? 1 : 0) << "\n";
    out << "phase=" << static_cast<int>(state.phase()) << "\n";
    out << "moves=";
    bool first = true;
    for (const auto& move : state.move_history()) {
        if (!first) {
            out << ",";
        }
        first = false;
        out << move_token(state, move);
    }
    out << "\n";
    if (state.result().has_value()) {
        out << "result=" << state.result()->summary << "\n";
    } else if (state.review_state().has_value()) {
        out << "review=" << state.review_state()->estimate.summary << "\n";
    }
    return out.str();
}

std::string GameRecord::serialize_moves_wrapped(const GameState& state, int moves_per_line) {
    std::ostringstream out;
    out << "Moves\n";
    const auto& history = state.move_history();
    int placements_on_line = 0;
    for (std::size_t i = 0; i < history.size(); ++i) {
        const auto& move = history[i];
        out << (i + 1) << ". \"" << move_token(state, move) << "\"";
        if (i + 1 != history.size()) {
            ++placements_on_line;
            if (placements_on_line >= moves_per_line) {
                out << "\n";
                placements_on_line = 0;
            } else {
                out << "  ";
            }
        }
    }
    if (history.empty()) {
        out << "No moves";
    }
    return out.str();
}

std::optional<ParsedGameRecord> GameRecord::parse(const std::string& text) {
    ParsedGameRecord record;
    record.config.side_length = 4;
    record.config.komi = 0.0;
    record.config.allow_suicide = false;

    std::istringstream in(text);
    std::string line;
    std::string moves_text;
    while (std::getline(in, line)) {
        const auto pos = line.find('=');
        if (pos == std::string::npos) {
            continue;
        }
        const std::string key = trim(line.substr(0, pos));
        const std::string value = trim(line.substr(pos + 1));
        if (key == "side_length") {
            record.config.side_length = std::stoi(value);
        } else if (key == "komi") {
            record.config.komi = std::stod(value);
        } else if (key == "allow_suicide") {
            record.config.allow_suicide = value == "1";
        } else if (key == "moves") {
            moves_text = value;
        }
    }

    const BoardTopology topology(record.config.side_length);
    std::unordered_map<std::string, int> label_to_vertex;
    for (int i = 0; i < topology.vertex_count(); ++i) {
        label_to_vertex[topology.label_for_vertex(i)] = i;
    }

    std::istringstream move_stream(moves_text);
    std::string token;
    while (std::getline(move_stream, token, ',')) {
        token = trim(token);
        if (token.empty()) {
            continue;
        }
        if (token == "pass") {
            record.moves.push_back(Move::pass());
        } else if (token == "resign") {
            record.moves.push_back(Move::resign());
        } else {
            const auto it = label_to_vertex.find(token);
            if (it == label_to_vertex.end()) {
                return std::nullopt;
            }
            record.moves.push_back(Move::place(it->second));
        }
    }
    return record;
}

}  // namespace trilibgo::core
