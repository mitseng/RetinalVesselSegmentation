import sys
from PySide2.QtWidgets import QAction, QApplication, QFileDialog, QGraphicsScene, QMainWindow
from PySide2.QtUiTools import QUiLoader
from PySide2.QtCore import QFile, Slot, Qt
from ui_mainwindow import Ui_MainWindow
from PySide2.QtGui import QImage, QPicture, QPixmap
from PIL import Image
import numpy as np
from my_utils import Util

class MainWindow(QMainWindow):
    def __init__(self):
        QMainWindow.__init__(self)
        # load util
        self.util = Util()
        self.org_file = None
        self.pred = None
        # set ui
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.ui.retranslateUi(self)
        # set actions
        self.ui.actionOpen.triggered.connect(self.open)
        self.ui.runButton.clicked.connect(self.run)
        self.ui.actionSave.triggered.connect(self.save)

    @Slot()
    def open(self):
        self.org_file = QFileDialog.getOpenFileName(
            self,
            "Open File Dialog",
            "C:"
        )[0]
        # open image file then show
        pixmap = QPixmap(self.org_file)
        self.ui.label_2.setPixmap(pixmap)

    @Slot()
    def run(self):
        if self.org_file is None:
            return
        self.pred = self.util.predict(self.org_file)
        if self.pred.flags['C_CONTIGUOUS'] == False:
            self.pred = np.ascontiguousarray(self.pred)
        rows, cols = self.pred.shape
        pred_image = QImage(self.pred, cols, rows, cols, QImage.Format_Grayscale8)
        pixmap = QPixmap.fromImage(pred_image)
        self.ui.label.setPixmap(pixmap)

    @Slot()
    def save(self):
        if self.pred is None:
            return
        save_name = QFileDialog.getSaveFileName(
            self,
            'Save File Dialog',
            'C:'
        )[0]
        save_img = Image.fromarray(self.pred)
        try:
            save_img.save(save_name)
        except ValueError:
            save_img.save(save_name+'.jpg')


if __name__ == "__main__":
    app = QApplication([])

    window = MainWindow()
    window.show()

    sys.exit(app.exec_())