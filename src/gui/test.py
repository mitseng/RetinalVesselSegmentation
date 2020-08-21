from ui_mainwindow import Ui_MainWindow
from PySide2.QtWidgets import QApplication, QMainWindow
import sys


class MainWindow(QMainWindow):
    def __init__(self):
        QMainWindow.__init__(self)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.ui.retranslateUi(self)
        

if __name__ == '__main__':
    app = QApplication([])

    window = MainWindow()
    window.show()

    sys.exit(app.exec_())