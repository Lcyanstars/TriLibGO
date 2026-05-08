#pragma once

#include "trilibgo/app/board_widget.h"

#include <QLabel>
#include <QMainWindow>
#include <QPlainTextEdit>
#include <QPushButton>
#include <QSpinBox>

namespace trilibgo::app {

/** \brief Main application window: board widget + status labels + control buttons.

    Wires up the BoardWidget with status display, record I/O, model loading,
    replay navigation, and endgame review controls.
*/
class MainWindow final : public QMainWindow {
    Q_OBJECT

public:
    MainWindow();

private slots:
    /** \brief Refresh all UI elements from current controller state. */
    void refresh_ui();

private:
    bool load_selfplay_game(int game_number);

    BoardWidget* board_widget_ = nullptr;
    QLabel* status_label_ = nullptr;
    QLabel* capture_label_ = nullptr;
    QLabel* estimate_label_ = nullptr;
    QLabel* result_label_ = nullptr;
    QLabel* analysis_label_ = nullptr;
    QPlainTextEdit* record_box_ = nullptr;
    QPushButton* display_mode_button_ = nullptr;
    QPushButton* load_model_button_ = nullptr;
    QPushButton* reload_selfplay_button_ = nullptr;
    QPushButton* analysis_overlay_button_ = nullptr;
    QPushButton* save_button_ = nullptr;
    QPushButton* load_button_ = nullptr;
    QPushButton* finalize_button_ = nullptr;
    QSpinBox* board_side_spin_ = nullptr;
    QSpinBox* selfplay_game_spin_ = nullptr;
    QSpinBox* replay_position_spin_ = nullptr;
    QString selfplay_trace_path_;
    int selfplay_game_count_ = 0;
    bool selfplay_trace_is_jsonl_ = true;
};

}  // namespace trilibgo::app
