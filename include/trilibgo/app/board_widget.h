#pragma once

#include "trilibgo/app/game_controller.h"

#include <QWidget>

namespace trilibgo::app {

/** \brief Qt widget that renders the hexagonal board and handles click input.

    Paints the board topology, stones, territory shading, last-move markers,
    move numbers, endgame review overlays, and AI policy heatmaps.
*/
class BoardWidget final : public QWidget {
    Q_OBJECT

public:
    /** \brief How stones are displayed. */
    enum class StoneDisplayMode {
        Plain,       ///< Plain black/white stones.
        LastMove,    ///< Stones with last-move marker.
        MoveNumbers, ///< Stones numbered by move order.
    };

    explicit BoardWidget(QWidget* parent = nullptr);

    /** \brief Access the game controller. */
    [[nodiscard]] GameController& controller();
    /** \brief Const access to the game controller. */
    [[nodiscard]] const GameController& controller() const;
    /** \brief Current stone display mode. */
    [[nodiscard]] StoneDisplayMode stone_display_mode() const;
    /** \brief Cycle to the next stone display mode. */
    void cycle_stone_display_mode();

signals:
    /** \brief Emitted after any state-changing action. */
    void state_changed();

protected:
    void paintEvent(QPaintEvent* event) override;
    void mousePressEvent(QMouseEvent* event) override;
    QSize minimumSizeHint() const override;

private:
    [[nodiscard]] int vertex_at(const QPointF& point) const;
    [[nodiscard]] const trilibgo::core::Move* last_placement_move() const;
    [[nodiscard]] int last_placement_move_number() const;
    [[nodiscard]] std::vector<int> current_stone_move_numbers() const;
    void paint_review_overlay(QPainter& painter, const QRectF& bounds, double scale) const;
    void paint_last_move_marker(QPainter& painter, const QRectF& bounds, double scale) const;
    void paint_move_numbers(QPainter& painter, const QRectF& bounds, double scale) const;
    void paint_analysis_overlay(QPainter& painter, const QRectF& bounds, double scale) const;

    GameController controller_;
    StoneDisplayMode stone_display_mode_ = StoneDisplayMode::LastMove;
};

}  // namespace trilibgo::app
