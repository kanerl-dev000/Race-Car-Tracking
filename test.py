import sys
from PyQt6.QtWidgets import QApplication, QMainWindow, QPushButton


class MyWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.initUI()

    def initUI(self):
        self.setWindowTitle("Close Window Example")

        # Create a button
        close_button = QPushButton("Close", self)
        close_button.clicked.connect(
            self.close
        )  # Connect the button's click event to the close method

        self.setGeometry(100, 100, 300, 200)  # Set window geometry
        self.show()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MyWindow()
    sys.exit(app.exec())
