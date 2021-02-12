import sys
from PyQt5 import QtWidgets, QtGui, QtCore
import design
import Main

class App(QtWidgets.QMainWindow, design.Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        self.setWindowIcon(QtGui.QIcon('logo.png'))
        self.pushButton.clicked.connect(self.show_simulate)

    def show_simulate(self):
        print()
        Main.Simuate(L=self.plainTextEdit_1.toPlainText(), T=self.plainTextEdit_2.toPlainText(), h=self.plainTextEdit_3.toPlainText(), tau=self.plainTextEdit_4.toPlainText(),
            mu_in=self.plainTextEdit_5.toPlainText(), mu=self.plainTextEdit_24.toPlainText(), H0=self.plainTextEdit_6.toPlainText(), H1=self.plainTextEdit_7.toPlainText(), H2=self.plainTextEdit_8.toPlainText(),
            lam=self.plainTextEdit_9.toPlainText(), p=self.plainTextEdit_10.toPlainText(), Cp=self.plainTextEdit_11.toPlainText(), Ct=self.plainTextEdit_12.toPlainText(), T0=self.plainTextEdit_13.toPlainText(), T1=self.plainTextEdit_14.toPlainText(), T2=self.plainTextEdit_15.toPlainText(),
            D=self.plainTextEdit_16.toPlainText(), y=self.plainTextEdit_17.toPlainText(), n=self.plainTextEdit_18.toPlainText(), C0=self.plainTextEdit_19.toPlainText(), C1=self.plainTextEdit_20.toPlainText(), C2=self.plainTextEdit_21.toPlainText(), m=self.plainTextEdit_22.toPlainText(), i1=self.plainTextEdit_23.toPlainText())

def main():
    app = QtWidgets.QApplication(sys.argv)
    window = App()
    window.show()
    sys.exit(app.exec_())
    
if __name__ == '__main__':
    main()