#pragma once

#include "trilibgo/core/types.h"

#include <compare>
#include <unordered_map>
#include <vector>

namespace trilibgo::core {

/** \brief Hexagonal vertex grid for a given side length. Provides vertex positions and adjacency.

    For side_length N, the board contains 3*N*(N+1)+1 vertices at the corners of a
    hexagonal honeycomb. Interior vertices have 3 neighbors; boundary vertices have 2.
*/
class BoardTopology {
public:
    /** \brief 2D integer key in the axial coordinate system. */
    struct PointKey {
        long long x = 0;
        long long y = 0;
        auto operator<=>(const PointKey&) const = default;
    };

    /** \brief Construct topology for a board with the given side length. */
    explicit BoardTopology(int side_length);

    /** \brief Number of hex cells per side. */
    [[nodiscard]] int side_length() const;
    /** \brief Total number of vertices on the board. */
    [[nodiscard]] int vertex_count() const;
    /** \brief Adjacent vertex indices for a given vertex. */
    [[nodiscard]] const std::vector<int>& neighbors(int vertex) const;
    /** \brief 2D rendering position of a vertex. */
    [[nodiscard]] const VertexPosition& position(int vertex) const;
    /** \brief Vertices in UI display order (left-to-right, top-to-bottom). */
    [[nodiscard]] const std::vector<int>& vertices_in_display_order() const;
    /** \brief Check if a vertex index is valid. */
    [[nodiscard]] bool is_valid_vertex(int vertex) const;
    /** \brief Human-readable label for a vertex (e.g., "A1"). */
    [[nodiscard]] std::string label_for_vertex(int vertex) const;

private:
    int side_length_ = 0;
    std::vector<VertexPosition> positions_;
    std::vector<std::vector<int>> adjacency_;
    std::vector<int> display_order_;
    std::unordered_map<int, std::string> labels_;
};

}  // namespace trilibgo::core
