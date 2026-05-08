#include "trilibgo/app/main_window.h"

#include <QApplication>

int main(int argc, char* argv[]) {
    QApplication app(argc, argv);
    trilibgo::app::MainWindow window;
    window.show();
    return app.exec();
}
