#include "trilibgo/app/board_widget.h"

#include "replay_compat.h"
#include "trilibgo/core/rules_engine.h"

#include <QMouseEvent>
#include <QPainter>

#include <algorithm>
#include <cmath>
#include <limits>

namespace trilibgo::app {
namespace {

QRectF board_rect(const QSize& size) {
    return QRectF(30.0, 30.0, size.width() - 60.0, size.height() - 60.0);
}

QPointF to_screen(const trilibgo::core::VertexPosition& position, const QRectF& bounds, double scale) {
    return {bounds.center().x() + position.x * scale, bounds.center().y() + position.y * scale};
}

double board_scale(const trilibgo::core::BoardTopology& topology, const QRectF& bounds) {
    double min_x = std::numeric_limits<double>::max();
    double max_x = std::numeric_limits<double>::lowest();
    double min_y = std::numeric_limits<double>::max();
    double max_y = std::numeric_limits<double>::lowest();
    for (int i = 0; i < topology.vertex_count(); ++i) {
        const auto& pos = topology.position(i);
        min_x = std::min(min_x, pos.x);
        max_x = std::max(max_x, pos.x);
        min_y = std::min(min_y, pos.y);
        max_y = std::max(max_y, pos.y);
    }
    return std::min(bounds.width() / (max_x - min_x + 2.5), bounds.height() / (max_y - min_y + 2.5));
}

}  // namespace

BoardWidget::BoardWidget(QWidget* parent)
    : QWidget(parent),
      controller_(trilibgo::core::GameConfig{.side_length = 4, .komi = 0.0, .allow_suicide = false}) {}

GameController& BoardWidget::controller() { return controller_; }
const GameController& BoardWidget::controller() const { return controller_; }
BoardWidget::StoneDisplayMode BoardWidget::stone_display_mode() const { return stone_display_mode_; }

void BoardWidget::cycle_stone_display_mode() {
    switch (stone_display_mode_) {
        case StoneDisplayMode::Plain:
            stone_display_mode_ = StoneDisplayMode::LastMove;
            break;
        case StoneDisplayMode::LastMove:
            stone_display_mode_ = StoneDisplayMode::MoveNumbers;
            break;
        case StoneDisplayMode::MoveNumbers:
            stone_display_mode_ = StoneDisplayMode::Plain;
            break;
    }
}

void BoardWidget::paintEvent(QPaintEvent* event) {
    QWidget::paintEvent(event);
    Q_UNUSED(event);

    QPainter painter(this);
    painter.setRenderHint(QPainter::Antialiasing, true);
    painter.fillRect(rect(), QColor("#f3ecda"));

    const auto& state = controller_.state();
    const auto& topology = state.topology();
    const QRectF bounds = board_rect(size());
    const double scale = board_scale(topology, bounds);

    painter.setPen(QPen(QColor("#6e5330"), 2.0));
    for (int v = 0; v < topology.vertex_count(); ++v) {
        const QPointF from = to_screen(topology.position(v), bounds, scale);
        for (int neighbor : topology.neighbors(v)) {
            if (neighbor < v) {
                continue;
            }
            painter.drawLine(from, to_screen(topology.position(neighbor), bounds, scale));
        }
    }

    for (int v = 0; v < topology.vertex_count(); ++v) {
        const QPointF point = to_screen(topology.position(v), bounds, scale);
        painter.setPen(Qt::NoPen);
        painter.setBrush(QColor("#6e5330"));
        painter.drawEllipse(point, 4.0, 4.0);

        const auto stone = state.at(v);
        if (stone == trilibgo::core::Stone::Empty) {
            continue;
        }
        painter.setPen(QPen(QColor("#303030"), 1.2));
        painter.setBrush(stone == trilibgo::core::Stone::Black ? QColor("#111111") : QColor("#fefefe"));
        painter.drawEllipse(point, 13.0, 13.0);
    }

    if (state.phase() == trilibgo::core::Phase::ReviewingEndgame) {
        paint_review_overlay(painter, bounds, scale);
    }
    if (controller_.analysis_overlay_enabled()) {
        paint_analysis_overlay(painter, bounds, scale);
    }
    if (stone_display_mode_ == StoneDisplayMode::LastMove) {
        paint_last_move_marker(painter, bounds, scale);
    } else if (stone_display_mode_ == StoneDisplayMode::MoveNumbers) {
        paint_move_numbers(painter, bounds, scale);
    }
}

void BoardWidget::mousePressEvent(QMouseEvent* event) {
    const int vertex = vertex_at(event->position());
    if (controller_.state().phase() == trilibgo::core::Phase::ReviewingEndgame) {
        controller_.toggle_review_group(vertex);
    } else {
        controller_.play(vertex);
    }
    update();
    emit state_changed();
}

QSize BoardWidget::minimumSizeHint() const {
    return {700, 700};
}

int BoardWidget::vertex_at(const QPointF& point) const {
    const auto& topology = controller_.state().topology();
    const QRectF bounds = board_rect(size());
    const double scale = board_scale(topology, bounds);

    int best_vertex = -1;
    double best_distance = 20.0;
    for (int v = 0; v < topology.vertex_count(); ++v) {
        const QPointF screen = to_screen(topology.position(v), bounds, scale);
        const double dx = screen.x() - point.x();
        const double dy = screen.y() - point.y();
        const double distance = std::sqrt(dx * dx + dy * dy);
        if (distance < best_distance) {
            best_distance = distance;
            best_vertex = v;
        }
    }
    return best_vertex;
}

const trilibgo::core::Move* BoardWidget::last_placement_move() const {
    const auto& history = controller_.state().move_history();
    for (auto it = history.rbegin(); it != history.rend(); ++it) {
        if (it->kind == trilibgo::core::MoveKind::Place) {
            return &(*it);
        }
    }
    return nullptr;
}

int BoardWidget::last_placement_move_number() const {
    const auto& history = controller_.state().move_history();
    for (int i = static_cast<int>(history.size()) - 1; i >= 0; --i) {
        if (history[static_cast<std::size_t>(i)].kind == trilibgo::core::MoveKind::Place) {
            return i + 1;
        }
    }
    return -1;
}

std::vector<int> BoardWidget::current_stone_move_numbers() const {
    const auto& final_state = controller_.state();
    std::vector<int> numbers(static_cast<std::size_t>(final_state.topology().vertex_count()), -1);
    trilibgo::core::RulesEngine rules;
    trilibgo::core::GameState replay(final_state.config());
    replay.clear_and_seed_board_hash(rules.compute_hash(replay.board(), replay.current_player()));

    const auto& history = final_state.move_history();
    for (std::size_t i = 0; i < history.size(); ++i) {
        const auto& move = history[i];
        if (!apply_replay_move_compat(replay, rules, move)) {
            break;
        }
        normalize_replay_state_after_move(replay, i + 1 < history.size());
        if (move.kind == trilibgo::core::MoveKind::Place && move.coord.is_valid() &&
            replay.at(move.coord.index) != trilibgo::core::Stone::Empty) {
            numbers[static_cast<std::size_t>(move.coord.index)] = static_cast<int>(i + 1);
        }
        for (int v = 0; v < replay.topology().vertex_count(); ++v) {
            if (replay.at(v) == trilibgo::core::Stone::Empty) {
                numbers[static_cast<std::size_t>(v)] = -1;
            }
        }
    }
    return numbers;
}

void BoardWidget::paint_review_overlay(QPainter& painter, const QRectF& bounds, double scale) const {
    const auto& state = controller_.state();
    if (!state.review_state().has_value()) {
        return;
    }
    const auto& statuses = state.review_state()->user_statuses;
    const auto& topology = state.topology();

    for (int v = 0; v < topology.vertex_count(); ++v) {
        if (state.at(v) == trilibgo::core::Stone::Empty) {
            continue;
        }
        if (statuses[static_cast<std::size_t>(v)] == trilibgo::core::ChainLifeStatus::Dead) {
            const QPointF point = to_screen(topology.position(v), bounds, scale);
            painter.setPen(QPen(QColor(192, 57, 43, 220), 2.2));
            painter.drawLine(point + QPointF(-9.0, -9.0), point + QPointF(9.0, 9.0));
            painter.drawLine(point + QPointF(-9.0, 9.0), point + QPointF(9.0, -9.0));
        }
    }
}

void BoardWidget::paint_last_move_marker(QPainter& painter, const QRectF& bounds, double scale) const {
    const auto* move = last_placement_move();
    if (move == nullptr || !move->coord.is_valid()) {
        return;
    }

    const auto& state = controller_.state();
    const auto stone = state.at(move->coord.index);
    if (stone == trilibgo::core::Stone::Empty) {
        return;
    }

    const QPointF point = to_screen(state.topology().position(move->coord.index), bounds, scale);
    painter.setPen(QPen(QColor("#0f766e"), 2.2));
    painter.setBrush(stone == trilibgo::core::Stone::Black ? QColor("#f0fdfa") : QColor("#115e59"));
    painter.drawEllipse(point, 4.0, 4.0);
}

void BoardWidget::paint_move_numbers(QPainter& painter, const QRectF& bounds, double scale) const {
    const auto& state = controller_.state();
    const auto numbers = current_stone_move_numbers();
    QFont font = painter.font();
    font.setBold(true);
    font.setPointSize(7);
    painter.setFont(font);

    for (int v = 0; v < state.topology().vertex_count(); ++v) {
        const auto stone = state.at(v);
        const int move_number = numbers[static_cast<std::size_t>(v)];
        if (stone == trilibgo::core::Stone::Empty || move_number <= 0) {
            continue;
        }

        const QPointF point = to_screen(state.topology().position(v), bounds, scale);
        painter.setPen(stone == trilibgo::core::Stone::Black ? QColor("#f8fafc") : QColor("#111827"));
        painter.drawText(QRectF(point.x() - 10.0, point.y() - 8.0, 20.0, 16.0), Qt::AlignCenter, QString::number(move_number));
    }
}

void BoardWidget::paint_analysis_overlay(QPainter& painter, const QRectF& bounds, double scale) const {
    const auto analysis = controller_.analysis();
    if (!analysis.has_value()) {
        return;
    }

    const auto& state = controller_.state();
    const int pass_index = state.topology().vertex_count();
    for (const auto& item : analysis->policy) {
        if (item.action_index < 0 || item.action_index >= pass_index) {
            continue;
        }
        if (state.at(item.action_index) != trilibgo::core::Stone::Empty) {
            continue;
        }
        const QPointF point = to_screen(state.topology().position(item.action_index), bounds, scale);
        const int alpha = std::clamp(static_cast<int>(40 + item.probability * 180.0), 40, 220);
        painter.setPen(Qt::NoPen);
        painter.setBrush(QColor(15, 118, 110, alpha));
        const double radius = 6.0 + item.probability * 12.0;
        painter.drawEllipse(point, radius, radius);
    }
}

}  // namespace trilibgo::app
