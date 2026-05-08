#include "trilibgo/core/board_topology.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <map>
#include <set>
#include <sstream>

namespace trilibgo::core {
namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr double kScale = 1'000'000.0;

struct Cell {
    int q = 0;
    int r = 0;
};

VertexPosition hex_center(int q, int r) {
    const double x = std::sqrt(3.0) * (static_cast<double>(q) + static_cast<double>(r) / 2.0);
    const double y = 1.5 * static_cast<double>(r);
    return {x, y};
}

BoardTopology::PointKey make_key(const VertexPosition& position) {
    return {
        static_cast<long long>(std::llround(position.x * kScale)),
        static_cast<long long>(std::llround(position.y * kScale)),
    };
}

std::vector<Cell> build_cells(int side_length) {
    const int radius = side_length - 1;
    std::vector<Cell> cells;
    for (int q = -radius; q <= radius; ++q) {
        const int r_min = std::max(-radius, -q - radius);
        const int r_max = std::min(radius, -q + radius);
        for (int r = r_min; r <= r_max; ++r) {
            cells.push_back({q, r});
        }
    }
    return cells;
}

std::vector<VertexPosition> hex_corners(const VertexPosition& center) {
    std::vector<VertexPosition> corners;
    corners.reserve(6);
    for (int i = 0; i < 6; ++i) {
        const double angle = (60.0 * static_cast<double>(i) - 30.0) * kPi / 180.0;
        corners.push_back({center.x + std::cos(angle), center.y + std::sin(angle)});
    }
    return corners;
}

std::string column_label(int value) {
    std::string result;
    int current = value;
    do {
        result.insert(result.begin(), static_cast<char>('A' + (current % 26)));
        current = current / 26 - 1;
    } while (current >= 0);
    return result;
}

}  // namespace

BoardTopology::BoardTopology(int side_length) : side_length_(side_length) {
    std::map<PointKey, int> vertex_lookup;
    std::vector<std::set<int>> adjacency_sets;
    const auto cells = build_cells(side_length);

    for (const auto& cell : cells) {
        const auto center = hex_center(cell.q, cell.r);
        const auto corners = hex_corners(center);
        std::array<int, 6> ids{};

        for (int i = 0; i < 6; ++i) {
            const auto [it, inserted] = vertex_lookup.emplace(make_key(corners[static_cast<std::size_t>(i)]), static_cast<int>(positions_.size()));
            if (inserted) {
                positions_.push_back(corners[static_cast<std::size_t>(i)]);
                adjacency_sets.emplace_back();
            }
            ids[static_cast<std::size_t>(i)] = it->second;
        }

        for (int i = 0; i < 6; ++i) {
            const int a = ids[static_cast<std::size_t>(i)];
            const int b = ids[static_cast<std::size_t>((i + 1) % 6)];
            adjacency_sets[static_cast<std::size_t>(a)].insert(b);
            adjacency_sets[static_cast<std::size_t>(b)].insert(a);
        }
    }

    adjacency_.reserve(adjacency_sets.size());
    for (const auto& neighbors : adjacency_sets) {
        adjacency_.emplace_back(neighbors.begin(), neighbors.end());
    }

    display_order_.resize(positions_.size());
    for (int i = 0; i < static_cast<int>(positions_.size()); ++i) {
        display_order_[static_cast<std::size_t>(i)] = i;
    }

    std::sort(display_order_.begin(), display_order_.end(), [&](int lhs, int rhs) {
        const auto& a = positions_[static_cast<std::size_t>(lhs)];
        const auto& b = positions_[static_cast<std::size_t>(rhs)];
        if (std::abs(a.y - b.y) > 1e-6) {
            return a.y < b.y;
        }
        return a.x < b.x;
    });

    int current_row = 0;
    double row_anchor = positions_[static_cast<std::size_t>(display_order_.front())].y;
    int column_in_row = 0;
    for (int vertex : display_order_) {
        const auto& pos = positions_[static_cast<std::size_t>(vertex)];
        if (std::abs(pos.y - row_anchor) > 1e-6) {
            ++current_row;
            row_anchor = pos.y;
            column_in_row = 0;
        }
        std::ostringstream label;
        label << column_label(column_in_row) << (current_row + 1);
        labels_[vertex] = label.str();
        ++column_in_row;
    }
}

int BoardTopology::side_length() const { return side_length_; }
int BoardTopology::vertex_count() const { return static_cast<int>(positions_.size()); }
const std::vector<int>& BoardTopology::neighbors(int vertex) const { return adjacency_[static_cast<std::size_t>(vertex)]; }
const VertexPosition& BoardTopology::position(int vertex) const { return positions_[static_cast<std::size_t>(vertex)]; }
const std::vector<int>& BoardTopology::vertices_in_display_order() const { return display_order_; }
bool BoardTopology::is_valid_vertex(int vertex) const { return vertex >= 0 && vertex < static_cast<int>(positions_.size()); }

std::string BoardTopology::label_for_vertex(int vertex) const {
    const auto it = labels_.find(vertex);
    return it == labels_.end() ? "?" : it->second;
}

}  // namespace trilibgo::core
