from PyQt6.QtWidgets import (
    QApplication,
    QWidget,
    QFileDialog,
    QLabel,
    QLineEdit,
    QDialog,
    QVBoxLayout,
    QDialogButtonBox,
)
from ui_main import Ui_Dialog
from PyQt6.QtGui import QImage, QPixmap
from qt_material import apply_stylesheet
import sys, cv2, json
from PyQt6 import QtCore, QtGui

from PyQt6.QtCore import Qt, QThread, pyqtSignal

from track import CarTrack

import threading

import ndi

import time


class ConfirmationDialog(QDialog):
    data_updated = QtCore.pyqtSignal(str, str)

    def __init__(self, car_num, driver_name, index, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Edit Label Text")
        layout = QVBoxLayout(self)
        self.car_num_edit = QLineEdit(self)
        self.car_num_edit.setText(car_num)
        layout.addWidget(self.car_num_edit)
        self.driver_name_edit = QLineEdit(self)
        self.driver_name_edit.setText(driver_name)
        layout.addWidget(self.driver_name_edit)

        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
            self,
        )

        layout.addWidget(button_box)

        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)

        # Index of the QLabel to delete
        self.index_to_delete = None

        # Set the current index of the QLabel
        self.current_index = index

        self.scaleX = 1
        self.scaleY = 1

    def accept(self):
        updated_car_num = self.car_num_edit.text()
        updated_driver_name = self.driver_name_edit.text()
        self.data_updated.emit(updated_car_num, updated_driver_name)
        super().accept()

    def reject(self):
        super().reject()


class VideoThread(QThread):
    changePixmap = pyqtSignal(QImage)

    updateVariable = pyqtSignal(bool)

    # updateCursor = pyqtSignal()

    def __init__(
        self,
        ui,
        tracker,
        width_source,
        height_source,
        arr_filename,
        mouse,
        cap,
        isMouseOver,
    ):
        super().__init__()
        self.ui = ui
        self.tracker = tracker
        self.width_source = width_source
        self.height_source = height_source
        self.arr_filename = arr_filename
        self.isVideo = True
        self.isMouseOver = isMouseOver
        self.cap = cap
        self.mouse = mouse

    def run(self):
        self.cap = cv2.VideoCapture(self.arr_filename)
        while self.isVideo:
            ret, frame = self.cap.read()
            if not ret:
                break

            height, width, channel = frame.shape
            scaleX = width / self.width_source
            scaleY = height / self.height_source

            self.tracker.mouse = [
                int(self.mouse[0] * scaleX),
                int(self.mouse[1] * scaleY),
            ]

            frame, self.isMouseOver = self.tracker.run(frame=frame)

            self.updateVariable.emit(self.isMouseOver)

            src_img = cv2.resize(frame, (self.width_source, self.height_source))
            src_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2RGB)
            temp_img = QImage(
                src_img,
                self.width_source,
                self.height_source,
                src_img.strides[0],
                QImage.Format.Format_RGB888,
            )
            self.changePixmap.emit(temp_img)

            cv2.waitKey(1)

        self.cap.release()

    def update_mouse(self, mouse):
        self.mouse = mouse

    def stop(self):
        self.isVideo = False


class Main_Window(QWidget):
    def __init__(self):
        super().__init__()
        self.ui = Ui_Dialog()
        self.ui.setupUi(self)
        self.ui.import_btn.clicked.connect(self.main)
        self.ui.close_btn.clicked.connect(self.handle_close)
        self.index = -1
        self.ui.edit_btn.clicked.connect(self.open_confirmation_dialog)
        self.width_source, self.height_source = (
            self.ui.src.width(),
            self.ui.src.height(),
        )
        self.labels = []
        self.sort_cars()
        self.show()
        self.saved_car_num = None
        self.saved_driver_name = None
        self.saved_index = None

        self.isClick = False
        self.mouse = [-1000, -1000]

        self.ui.src.mousePressEvent = lambda event: self.get_cursor_coordinates(event)
        self.cap = None
        self.setMouseTracking(True)

        self.carids = []

        self.bg_im = cv2.imread("./img/background.PNG")

        self.tracker = CarTrack()
        self.video_thread = None

        self.isMouseOver = False

    def handle_close(self):
        if self.cap:
            self.cap.release()
        if self.video_thread != None:
            if self.video_thread.isRunning():
                self.video_thread.stop()
                self.video_thread.wait()
        self.close()

    def change_car_data(self):
        json_object = json.dumps(self.car_dict, indent=4)
        with open("car.json", "w") as outfile:
            outfile.write(json_object)

    def get_cursor_coordinates(self, event):
        self.x_pos, self.y_pos = event.pos().x(), event.pos().y()

        if event.button() == Qt.MouseButton.LeftButton:
            self.mouse = [self.x_pos - 30, self.y_pos - 30]
            self.isClick = True
            self.tracker.isClick = True

    def mouseMoveEvent(self, event):
        self.x_pos, self.y_pos = event.pos().x(), event.pos().y()
        self.mouse = [self.x_pos - 30, self.y_pos - 30]
        if self.video_thread != None:
            self.video_thread.update_mouse(self.mouse)

    # def updateCursor(self):

    def updateVariable(self, isMouseover):
        # self.tracker.isClick = True
        self.isMouseOver = isMouseover
        if self.isMouseOver:
            self.ui.src.setCursor(
                QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor)
            )
        else:
            self.ui.src.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.ArrowCursor))
        self.isClick = False

    def sort_cars(self):
        with open("car.json", "r") as file:
            # Load the JSON data
            self.car_dict = json.load(file)
            for i, car in enumerate(self.car_dict):
                label = QLabel(self)
                label.setGeometry(
                    QtCore.QRect(40 + (i % 9) * 130, 760 + int(i / 9) * 55, 110, 40)
                )
                font = QtGui.QFont()
                font.setBold(True)
                font.setWeight(75)
                label.setFont(font)
                label.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor))
                label.setStyleSheet(
                    "border: 2px solid green;\n"
                    "background-color: black;\n"
                    "color: white;\n"
                    "border-radius: 5px;\n"
                    "font-weight: bold;\n"
                    ""
                )
                label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
                label.setObjectName(f"label{i}")
                label.setText(car["number"] + "\n" + car["name"])
                label.mousePressEvent = lambda event, index=i: self.select_label_text(
                    index
                )  # Connect the clicked event to edit_label_text
                self.labels.append(label)

    def main(self):
        self.arr_filename = QFileDialog.getOpenFileName(
            self, "Select Video", "", "Image Files(*.mp4 *avi)"
        )[0]
        if self.arr_filename != "":
            self.video_thread = VideoThread(
                self.ui,
                self.tracker,
                self.width_source,
                self.height_source,
                self.arr_filename,
                self.mouse,
                self.cap,
                self.isMouseOver,
            )
            self.video_thread.changePixmap.connect(self.set_image)
            self.video_thread.updateVariable.connect(self.updateVariable)
            self.video_thread.start()

    def set_image(self, image):
        self.ui.src.setPixmap(QPixmap.fromImage(image))

    def save_label_data(self, index):
        # Save the data from the QLabel that was clicked
        if len(self.tracker.carids) > 0:
            self.saved_car_num = self.car_dict[self.tracker.carids[0]]["number"]
            self.saved_driver_name = self.car_dict[self.tracker.carids[0]]["name"]
            self.saved_index = self.tracker.carids[0]

    def set_label_checked(self, index):
        self.labels[index].setStyleSheet(
            "border: 2px solid black;\n"
            "background-color: green;\n"
            "color: white;\n"
            "border-radius: 5px;\n"
            "font-weight: bold;\n"
            ""
        )

    def set_label_unchecked(self, index):
        self.labels[index].setStyleSheet(
            "border: 2px solid green;\n"
            "background-color: black;\n"
            "color: white;\n"
            "border-radius: 5px;\n"
            "font-weight: bold;\n"
            ""
        )

    def select_label_text(self, index):
        if index in self.tracker.carids:
            self.set_label_unchecked(index)
            self.tracker.carids.remove(index)
            index_to_remove = next(
                (
                    i
                    for i, obj in enumerate(self.tracker.titles)
                    if obj["index"] == index
                ),
                None,
            )
            if index_to_remove is not None:
                targetID = self.tracker.titles[index_to_remove]["trackid"]
                if targetID in self.tracker.targetID:
                    self.tracker.targetID.remove(targetID)
                self.tracker.titles.pop(index_to_remove)

        else:
            if len(self.tracker.titles) < 3:
                self.set_label_checked(index)
                self.tracker.carids.append(index)
                title = {
                    "index": index,
                    "trackid": 0,
                    "number": self.car_dict[index]["number"],
                    "name": self.car_dict[index]["name"],
                }
                self.tracker.titles.append(title)
            # else:
            #     print("Select the title!")
        self.save_label_data(index)

        if len(self.tracker.carids) == 1:
            self.ui.edit_btn.setEnabled(True)
            self.ui.edit_btn.setStyleSheet(
                "background-color: green; color: white; border: 2px solid green;"
            )
        else:
            self.ui.edit_btn.setEnabled(False)
            self.ui.edit_btn.setStyleSheet(
                "background-color: transparent; color: green; border: 2px solid green;"
            )

    def open_confirmation_dialog(self):
        # Open the confirmation dialog with the saved data
        if self.saved_car_num is not None:
            confirm_dialog = ConfirmationDialog(
                self.saved_car_num,
                self.saved_driver_name,
                self.saved_index,
                self,
            )
            confirm_dialog.data_updated.connect(
                self.on_data_updated
            )  # Connect the signal to the slot
            confirm_dialog.exec()

    def on_data_updated(self, updated_car_num, updated_driver_name):
        index = (
            self.sender().current_index
        )  # Get the index from the sender (ConfirmationDialog)
        # if updated_car_num != list(self.car_dict.keys())[index]:
        #     self.car_dict[updated_car_num] = self.car_dict.pop(
        #         list(self.car_dict.keys())[index]
        #     )
        ind = next(
            (i for i, d in enumerate(self.tracker.titles) if d["index"] == index),
            None,
        )
        if ind != None:
            self.tracker.titles[ind]["number"] = updated_car_num
            self.tracker.titles[ind]["name"] = updated_driver_name

        self.car_dict[index]["number"] = updated_car_num
        self.car_dict[index]["name"] = updated_driver_name

        self.change_car_data()
        self.labels[index].setText(updated_car_num + "\n" + updated_driver_name)
        self.sort_cars()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = Main_Window()
    apply_stylesheet(app, theme="dark_teal.xml")
    # sys.exit(app.exec())
    sys.exit(app.exec())
