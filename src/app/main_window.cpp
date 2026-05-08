#include "trilibgo/app/main_window.h"

#include "trilibgo/core/game_record.h"

#include <QFile>
#include <QFileDialog>
#include <QFontMetrics>
#include <QHBoxLayout>
#include <QPushButton>
#include <QSizePolicy>
#include <QSignalBlocker>
#include <QSpinBox>
#include <QTextStream>
#include <QVBoxLayout>
#include <QWidget>

namespace trilibgo::app {
namespace {

constexpr int kAnalysisSlots = 3;

}

MainWindow::MainWindow() {
    auto* central = new QWidget(this);
    auto* root = new QHBoxLayout(central);

    board_widget_ = new BoardWidget(central);
    root->addWidget(board_widget_, 1);

    auto* side = new QVBoxLayout();
    auto* side_widget = new QWidget(central);
    side_widget->setLayout(side);
    side_widget->setFixedWidth(280);
    status_label_ = new QLabel(central);
    capture_label_ = new QLabel(central);
    estimate_label_ = new QLabel(central);
    result_label_ = new QLabel(central);
    analysis_label_ = new QLabel(central);
    auto* pass_button = new QPushButton("Pass", central);
    auto* undo_button = new QPushButton("Undo", central);
    auto* new_button = new QPushButton("New Game", central);
    board_side_spin_ = new QSpinBox(central);
    board_side_spin_->setMinimum(2);
    board_side_spin_->setMaximum(12);
    board_side_spin_->setPrefix("Side ");
    board_side_spin_->setValue(board_widget_->controller().state().config().side_length);
    display_mode_button_ = new QPushButton("Display: Last Mark", central);
    load_model_button_ = new QPushButton("Load Model", central);
    reload_selfplay_button_ = new QPushButton("Reload Game", central);
    selfplay_game_spin_ = new QSpinBox(central);
    selfplay_game_spin_->setMinimum(1);
    selfplay_game_spin_->setMaximum(1);
    selfplay_game_spin_->setEnabled(false);
    analysis_overlay_button_ = new QPushButton("Analysis Overlay: Off", central);
    save_button_ = new QPushButton("Save Replay", central);
    load_button_ = new QPushButton("Load Replay", central);
    replay_position_spin_ = new QSpinBox(central);
    replay_position_spin_->setMinimum(0);
    replay_position_spin_->setMaximum(0);
    replay_position_spin_->setPrefix("Move ");
    replay_position_spin_->setEnabled(false);
    finalize_button_ = new QPushButton("Confirm Score", central);
    record_box_ = new QPlainTextEdit(central);
    record_box_->setReadOnly(true);
    status_label_->setWordWrap(true);
    capture_label_->setWordWrap(true);
    estimate_label_->setWordWrap(true);
    result_label_->setWordWrap(true);
    analysis_label_->setWordWrap(true);
    status_label_->setTextFormat(Qt::PlainText);
    capture_label_->setTextFormat(Qt::PlainText);
    estimate_label_->setTextFormat(Qt::PlainText);
    result_label_->setTextFormat(Qt::PlainText);
    analysis_label_->setTextFormat(Qt::PlainText);
    status_label_->setAlignment(Qt::AlignTop | Qt::AlignLeft);
    capture_label_->setAlignment(Qt::AlignTop | Qt::AlignLeft);
    estimate_label_->setAlignment(Qt::AlignTop | Qt::AlignLeft);
    result_label_->setAlignment(Qt::AlignTop | Qt::AlignLeft);
    analysis_label_->setAlignment(Qt::AlignTop | Qt::AlignLeft);
    status_label_->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Minimum);
    capture_label_->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Minimum);
    estimate_label_->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Minimum);
    result_label_->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Minimum);
    analysis_label_->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Minimum);
    const QFontMetrics metrics(analysis_label_->font());
    analysis_label_->setMinimumHeight(metrics.lineSpacing() * 7);
    record_box_->setLineWrapMode(QPlainTextEdit::NoWrap);

    side->addWidget(status_label_);
    side->addWidget(capture_label_);
    side->addWidget(estimate_label_);
    side->addWidget(result_label_);
    side->addWidget(analysis_label_);
    side->addWidget(pass_button);
    side->addWidget(undo_button);
    side->addWidget(board_side_spin_);
    side->addWidget(new_button);
    side->addWidget(display_mode_button_);
    side->addWidget(load_model_button_);
    auto* selfplay_row = new QHBoxLayout();
    selfplay_row->addWidget(selfplay_game_spin_, 1);
    selfplay_row->addWidget(reload_selfplay_button_, 1);
    side->addLayout(selfplay_row);
    side->addWidget(analysis_overlay_button_);
    side->addWidget(save_button_);
    side->addWidget(load_button_);
    side->addWidget(replay_position_spin_);
    side->addWidget(finalize_button_);
    side->addWidget(record_box_, 1);
    root->addWidget(side_widget);

    setCentralWidget(central);
    resize(1100, 760);
    setWindowTitle("TriLibGo Demo");

    connect(board_widget_, &BoardWidget::state_changed, this, &MainWindow::refresh_ui);
    connect(pass_button, &QPushButton::clicked, this, [this]() {
        board_widget_->controller().pass_turn();
        board_widget_->update();
        refresh_ui();
    });
    connect(undo_button, &QPushButton::clicked, this, [this]() {
        board_widget_->controller().undo();
        board_widget_->update();
        refresh_ui();
    });
    connect(new_button, &QPushButton::clicked, this, [this]() {
        auto config = board_widget_->controller().state().config();
        config.side_length = board_side_spin_->value();
        board_widget_->controller().new_game(config);
        board_widget_->update();
        refresh_ui();
    });
    connect(board_side_spin_, qOverload<int>(&QSpinBox::valueChanged), this, [this](int side_length) {
        if (side_length == board_widget_->controller().state().config().side_length &&
            !board_widget_->controller().is_replay_mode()) {
            return;
        }
        auto config = board_widget_->controller().state().config();
        config.side_length = side_length;
        board_widget_->controller().new_game(config);
        board_widget_->update();
        refresh_ui();
    });
    connect(display_mode_button_, &QPushButton::clicked, this, [this]() {
        board_widget_->cycle_stone_display_mode();
        board_widget_->update();
        refresh_ui();
    });
    connect(load_model_button_, &QPushButton::clicked, this, [this]() {
        const QString path = QFileDialog::getOpenFileName(this, "Load Model", QString(), "ONNX Model (*.onnx)");
        if (path.isEmpty()) {
            return;
        }
        board_widget_->controller().load_analysis_model(path.toStdString());
        board_widget_->update();
        refresh_ui();
    });
    connect(load_button_, &QPushButton::clicked, this, [this]() {
        const QString path = QFileDialog::getOpenFileName(this, "Load Replay", QString(), "Replay Files (*.jsonl *.json *.tgo *.txt)");
        if (path.isEmpty()) {
            return;
        }
        QFile file(path);
        if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
            return;
        }
        QTextStream stream(&file);
        const QString content = stream.readAll();
        const QString trimmed = content.trimmed();
        const bool is_jsonl = trimmed.startsWith("{");
        int game_count = 0;
        if (is_jsonl) {
            const auto lines = content.split('\n');
            for (const auto& line : lines) {
                if (!line.trimmed().isEmpty()) {
                    ++game_count;
                }
            }
        } else {
            game_count = trimmed.isEmpty() ? 0 : 1;
        }
        if (game_count <= 0) {
            return;
        }
        selfplay_trace_path_ = path;
        selfplay_game_count_ = game_count;
        selfplay_trace_is_jsonl_ = is_jsonl;
        {
            const QSignalBlocker blocker(selfplay_game_spin_);
            selfplay_game_spin_->setMaximum(game_count);
            selfplay_game_spin_->setValue(1);
        }
        selfplay_game_spin_->setEnabled(true);
        reload_selfplay_button_->setEnabled(true);
        (void)load_selfplay_game(1);
    });
    connect(reload_selfplay_button_, &QPushButton::clicked, this, [this]() {
        (void)load_selfplay_game(selfplay_game_spin_->value());
    });
    connect(analysis_overlay_button_, &QPushButton::clicked, this, [this]() {
        board_widget_->controller().toggle_analysis_overlay();
        board_widget_->update();
        refresh_ui();
    });
    connect(save_button_, &QPushButton::clicked, this, [this]() {
        const QString path = QFileDialog::getSaveFileName(this, "Save Replay", QString(), "Replay Files (*.jsonl);;JSON Files (*.json)");
        if (path.isEmpty()) {
            return;
        }
        QFile file(path);
        if (!file.open(QIODevice::WriteOnly | QIODevice::Text)) {
            return;
        }
        QTextStream stream(&file);
        stream << QString::fromStdString(board_widget_->controller().serialized_record());
    });
    connect(replay_position_spin_, qOverload<int>(&QSpinBox::valueChanged), this, [this](int value) {
        board_widget_->controller().set_replay_index(value);
        board_widget_->update();
        refresh_ui();
    });
    connect(finalize_button_, &QPushButton::clicked, this, [this]() {
        board_widget_->controller().finalize_review();
        board_widget_->update();
        refresh_ui();
    });

    refresh_ui();
}

void MainWindow::refresh_ui() {
    const auto& state = board_widget_->controller().state();
    const auto estimate = board_widget_->controller().estimate();
    int last_move_number = -1;
    QString last_move_label = "None";
    const auto& history = state.move_history();
    for (int i = static_cast<int>(history.size()) - 1; i >= 0; --i) {
        const auto& move = history[static_cast<std::size_t>(i)];
        if (move.kind == trilibgo::core::MoveKind::Place && move.coord.is_valid()) {
            last_move_number = i + 1;
            last_move_label = QString::fromStdString(state.topology().label_for_vertex(move.coord.index));
            break;
        }
    }

    status_label_->setText(
        QString("Status\n%1\nPhase: %2\nMove: %3\nLast stone: %4")
            .arg(QString::fromStdString(board_widget_->controller().status_text()))
            .arg(state.phase() == trilibgo::core::Phase::Playing
                     ? "Playing"
                     : (state.phase() == trilibgo::core::Phase::ReviewingEndgame ? "Endgame review" : "Finished"))
            .arg(state.move_number())
            .arg(last_move_number > 0 ? QString("%1 (#%2)").arg(last_move_label).arg(last_move_number) : last_move_label));
    capture_label_->setText(
        QString("Captures\nBlack: %1\nWhite: %2")
            .arg(state.captures(trilibgo::core::Stone::Black))
            .arg(state.captures(trilibgo::core::Stone::White)));
    estimate_label_->setText(
        QString("Estimate\n%1").arg(QString::fromStdString(estimate.summary)));
    QString result_text = "Result\nPending";
    if (board_widget_->controller().is_replay_mode()) {
        const auto replay_result = QString::fromStdString(board_widget_->controller().replay_result_summary());
        result_text = replay_result.isEmpty() ? "Result\nUnavailable" : QString("Result\n%1").arg(replay_result);
    } else if (state.result().has_value()) {
        if (state.result()->winner == trilibgo::core::Stone::Black) {
            result_text = QString("Result\nBlack +%1").arg(state.result()->margin, 0, 'f', 1);
        } else if (state.result()->winner == trilibgo::core::Stone::White) {
            result_text = QString("Result\nWhite +%1").arg(state.result()->margin, 0, 'f', 1);
        } else {
            result_text = "Result\nDraw";
        }
    } else if (state.phase() == trilibgo::core::Phase::ReviewingEndgame) {
        result_text = "Result\nAwaiting score confirmation";
    }
    result_label_->setText(result_text);
    const auto analysis = board_widget_->controller().analysis();
    QString analysis_text = "Analysis\nUnavailable";
    if (analysis.has_value()) {
        const auto replay_frame = board_widget_->controller().replay_analysis_frame();
        analysis_text = QString("Analysis\nWinrate %1%").arg(analysis->winrate * 100.0, 0, 'f', 1);
        if (replay_frame.has_value()) {
            analysis_text = QString("Analysis\nReplay turn %1\nWinrate %2%\nPolicy MCTS visits")
                                .arg(replay_frame->turn)
                                .arg(analysis->winrate * 100.0, 0, 'f', 1)
                                ;
        } else {
            analysis_text += "\nPolicy model";
        }
        analysis_text += "\nTop moves";
        for (int slot = 0; slot < kAnalysisSlots; ++slot) {
            if (slot >= static_cast<int>(analysis->top_policy.size())) {
                break;
            }
            const auto& item = analysis->top_policy[static_cast<std::size_t>(slot)];
            const auto move = trilibgo::ai::ActionCodec::action_index_to_move(state, item.action_index);
            QString label = "pass";
            if (move.has_value() && move->kind == trilibgo::core::MoveKind::Place) {
                label = QString::fromStdString(state.topology().label_for_vertex(move->coord.index));
            }
            analysis_text += QString("\n%1. %2  %3%")
                                 .arg(slot + 1)
                                 .arg(label)
                                 .arg(item.probability * 100.0, 0, 'f', 1);
        }
    }
    analysis_label_->setText(analysis_text);
    const bool reviewing = state.phase() == trilibgo::core::Phase::ReviewingEndgame;
    const auto display_mode = board_widget_->stone_display_mode();
    display_mode_button_->setText(
        display_mode == BoardWidget::StoneDisplayMode::Plain
            ? "Display: Plain"
            : (display_mode == BoardWidget::StoneDisplayMode::LastMove ? "Display: Last Mark" : "Display: Move Numbers"));
    analysis_overlay_button_->setText(board_widget_->controller().analysis_overlay_enabled() ? "Analysis Overlay: On" : "Analysis Overlay: Off");
    reload_selfplay_button_->setEnabled(!selfplay_trace_path_.isEmpty());
    selfplay_game_spin_->setSuffix(selfplay_game_count_ > 0 ? QString(" / %1").arg(selfplay_game_count_) : QString());
    {
        const QSignalBlocker blocker(board_side_spin_);
        board_side_spin_->setValue(state.config().side_length);
    }
    {
        const QSignalBlocker blocker(replay_position_spin_);
        const bool replay_mode = board_widget_->controller().is_replay_mode();
        const int move_total = std::max(board_widget_->controller().replay_total() - 1, 0);
        replay_position_spin_->setMinimum(0);
        replay_position_spin_->setMaximum(move_total);
        replay_position_spin_->setValue(std::clamp(board_widget_->controller().replay_index(), 0, move_total));
        replay_position_spin_->setSuffix(QString(" / %1").arg(move_total));
        replay_position_spin_->setEnabled(replay_mode);
    }
    finalize_button_->setEnabled(reviewing);
    record_box_->setPlainText(QString::fromStdString(trilibgo::core::GameRecord::serialize_moves_wrapped(state)));
}

bool MainWindow::load_selfplay_game(int game_number) {
    if (selfplay_trace_path_.isEmpty() || game_number <= 0) {
        return false;
    }
    QFile file(selfplay_trace_path_);
    if (!file.open(QIODevice::ReadOnly | QIODevice::Text)) {
        return false;
    }
    QTextStream stream(&file);
    QString selected_game;
    if (selfplay_trace_is_jsonl_) {
        int current_game = 0;
        while (!stream.atEnd()) {
            const QString line = stream.readLine().trimmed();
            if (line.isEmpty()) {
                continue;
            }
            ++current_game;
            if (current_game == game_number) {
                selected_game = line;
                break;
            }
        }
    } else {
        selected_game = stream.readAll().trimmed();
    }
    if (selected_game.isEmpty()) {
        return false;
    }
    bool loaded = false;
    if (selfplay_trace_is_jsonl_) {
        loaded = board_widget_->controller().load_selfplay_trace_text(selected_game.toStdString(), game_number);
    } else {
        loaded = board_widget_->controller().load_record_text(selected_game.toStdString());
    }
    board_widget_->update();
    refresh_ui();
    return loaded;
}

}  // namespace trilibgo::app
