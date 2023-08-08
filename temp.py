from PyQt6.QtWidgets import QApplication, QWidget, QFileDialog, QLabel, QLineEdit, QDialog, QVBoxLayout, QDialogButtonBox, QPushButton
from ui_main import Ui_Dialog
from PyQt6.QtGui import QImage, QPixmap
from qt_material import apply_stylesheet
import sys, cv2, json
from PyQt6 import QtCore, QtGui
car_num = ''
driver_name = ''

class ConfirmationDialog(QDialog):
    def __init__(self, index, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Edit Label Text")
        layout = QVBoxLayout(self)
        self.car_num = QLineEdit(self)
        self.car_num.setText(car_num)
        layout.addWidget(self.car_num)
        self.driver_name = QLineEdit(self)
        self.driver_name.setText(driver_name)
        layout.addWidget(self.driver_name)

        button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel, self)
        delete_button = QPushButton("Delete", self)
        delete_button.clicked.connect(self.delete_label)
        button_box.addButton(delete_button, QDialogButtonBox.ButtonRole.RejectRole)
        layout.addWidget(button_box)

        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)

        # Index of the QLabel to delete
        self.index_to_delete = None

        # Set the current index of the QLabel
        self.current_index = index

    def delete_label(self):
        print(self.current_index)

class Main_Window(QWidget):
    def __init__(self):
        super().__init__()
        self.ui = Ui_Dialog()       
        self.ui.setupUi(self)   
        self.ui.import_btn.clicked.connect(self.main)
        self.width_source, self.height_source = self.ui.src.width(), self.ui.src.height()  
        self.labels = []
        self.sort_cars() 
        self.show()

    def change_car_data(self):
        json_object = json.dumps(self.car_dict, indent=4)
        with open("car.json", "w") as outfile:
            outfile.write(json_object)

    def sort_cars(self):
        with open('car.json') as fp:
            elements = json.loads(fp.read())
            sorted_data = sorted(elements.items(), key=lambda x: x[0])
            self.car_dict = dict(sorted_data)
            for i, key in enumerate(self.car_dict):
                label = QLabel(self)
                label.setGeometry(QtCore.QRect(40 + (i%8) * 130, 750+int(i/8)*55, 110, 40))
                font = QtGui.QFont()
                font.setBold(True)
                font.setWeight(75)
                label.setFont(font)
                label.setCursor(QtGui.QCursor(QtCore.Qt.CursorShape.PointingHandCursor))
                label.setStyleSheet("border: 2px solid green;\n"
                                    "background-color: black;\n"
                                    "color: white;\n"
                                    "border-radius: 5px;\n"
                                    "font-weight: bold;\n"
                                    "")
                label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
                label.setObjectName(f"label{i}")
                label.setText(key+'\n'+self.car_dict[key])
                label.mousePressEvent = lambda event, index=i: self.edit_label_text(index)  # Connect the clicked event to edit_label_text
                self.labels.append(label)

    def main(self):
        self.arr_filename = QFileDialog.getOpenFileName(self,"Select Video","","Image Files(*.mp4 *avi)")[0]
        if self.arr_filename != '':
            cap = cv2.VideoCapture(self.arr_filename)
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                src_img = cv2.resize(frame, (self.width_source, self.height_source))
                src_img = cv2.cvtColor(src_img, cv2.COLOR_BGR2RGB)
                temp_img = QImage(src_img, self.width_source, self.height_source, src_img.strides[0], QImage.Format.Format_RGB888) 
                self.ui.src.setPixmap(QPixmap.fromImage(temp_img))
                cv2.waitKey(10)

    def edit_label_text(self, index):
        global car_num, driver_name
        car_num = list(self.car_dict.keys())[index]
        driver_name = self.car_dict[car_num]
        confirm_dialog = ConfirmationDialog(self)
        confirm_dialog.current_index = index  # Pass the index of the QLabel to the confirmation dialog
        if confirm_dialog.exec() == QDialog.DialogCode.Accepted:
            updated_car_num = confirm_dialog.car_num.text()
            if updated_car_num != car_num:
                self.car_dict[updated_car_num] = self.car_dict.pop(list(self.car_dict.keys())[index])
                self.change_car_data()
            updated_driver_name = confirm_dialog.driver_name.text()
            self.labels[index].setText(updated_car_num+'\n'+updated_driver_name)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_window = Main_Window()
    apply_stylesheet(app, theme='dark_teal.xml')
    sys.exit(app.exec())