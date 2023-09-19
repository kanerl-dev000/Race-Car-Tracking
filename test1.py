import sys
import os
from PyQt6.QtWidgets import QApplication, QMainWindow, QFileDialog, QPushButton
from PyQt6.QtCore import QDir


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Open File Dialog")
        self.setGeometry(100, 100, 300, 200)

        self.btn = QPushButton("Open PNG File", self)
        self.btn.clicked.connect(self.open_file_dialog)
        self.btn.resize(self.btn.sizeHint())
        self.btn.move(50, 80)

    def open_file_dialog(self):
        filePath, _ = QFileDialog.getOpenFileName(
            self,
            "Open PNG File",
            QDir.currentPath(),
            "PNG Files (*.png);;All Files (*)",
        )

        if filePath:
            filename = os.path.basename(filePath)
            print(filename)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
