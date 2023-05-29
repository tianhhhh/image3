import sys
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *

class Win(QWidget):
    def __init__(self):
        super().__init__()
        self.setGeometry(200, 200, 400, 400)
        self.setWindowTitle('QSlider的使用')

        self.lb1 = QLabel('Hello PyQt5')
        self.lb1.setAlignment(Qt.AlignCenter)

        self.s = QSlider(Qt.Horizontal)#水平方向
        self.s.setMinimum(10)#设置最小值
        self.s.setMaximum(50)#设置最大值
        self.s.setSingleStep(3)#设置步长值
        self.s.setValue(30)#设置当前值
        self.s.setTickPosition(QSlider.TicksBelow)#设置刻度位置，在下方
        self.s.setTickInterval(5)#设置刻度间隔
        self.s.valueChanged.connect(self.valueChange)

        layout = QVBoxLayout()
        layout.addWidget(self.lb1)
        layout.addWidget(self.s)
        self.setLayout(layout)

    def valueChange(self):
        print("current slider value:"+str(self.s.value()))
        size = self.s.value()
        self.lb1.setFont(QFont("Arial",size))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    form = Win()
    form.show()
    sys.exit(app.exec_())

