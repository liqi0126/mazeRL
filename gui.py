from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *

from RL import *


class MazeGUI(QWidget):
    def __init__(self, env, Q, args):
        super(MazeGUI, self).__init__()

        self.algo_combo = QComboBox()
        self.algo_combo.addItem("Q_Learning")
        self.algo_combo.addItem("Sarsa")
        self.algo_combo.addItem("Sarsa_Lambda")
        self.algo_combo.setCurrentText(args.method)

        self.episode_edit = QLineEdit()
        self.episode_edit.setText(str(args.episode))
        self.gamma_edit = QLineEdit()
        self.gamma_edit.setText(str(args.gamma))
        self.lr_edit = QLineEdit()
        self.lr_edit.setText(str(args.lr))
        self.e_edit = QLineEdit()
        self.e_edit.setText(str(args.e))
        self.decay_edit = QLineEdit()
        self.decay_edit.setText(str(args.decay))
        self.l_edit = QLineEdit()
        self.l_edit.setText(str(args.l))

        self.learn_button = QPushButton("learn")
        self.reset_button = QPushButton("reset")
        self.restart_button = QPushButton("restart")
        self.next_button = QPushButton("next")
        self.learn_button.clicked.connect(self.learn)
        self.reset_button.clicked.connect(self.reset)
        self.next_button.clicked.connect(self.next_step)
        self.restart_button.clicked.connect(self.restart)

        self.env = env
        self.Q = Q
        self.args = args
        self.rectSize = 80
        self.done = False
        self.state = self.env.reset()
        self.initWindow()

    def initWindow(self):
        self.setWindowTitle("Maze")

        hyperbox = QHBoxLayout()
        hyperbox.addWidget(self.algo_combo)
        hyperbox.addWidget(QLabel("episode"))
        hyperbox.addWidget(self.episode_edit)
        hyperbox.addWidget(QLabel("gamma"))
        hyperbox.addWidget(self.gamma_edit)
        hyperbox.addWidget(QLabel("lr"))
        hyperbox.addWidget(self.lr_edit)
        hyperbox.addWidget(QLabel("e"))
        hyperbox.addWidget(self.e_edit)
        hyperbox.addWidget(QLabel("decay"))
        hyperbox.addWidget(self.decay_edit)
        hyperbox.addWidget(QLabel("l"))
        hyperbox.addWidget(self.l_edit)

        buttonbox = QHBoxLayout()
        buttonbox.addWidget(self.learn_button)
        buttonbox.addWidget(self.reset_button)
        buttonbox.addWidget(self.next_button)
        buttonbox.addWidget(self.restart_button)

        vbox = QVBoxLayout()
        vbox.addStretch(1)
        vbox.addLayout(hyperbox)
        vbox.addLayout(buttonbox)
        self.setLayout(vbox)

        self.setGeometry(300, 300, self.rectSize * self.env.wall.shape[1], self.rectSize * self.env.wall.shape[0] + 20)
        # self.setFixedSize(self.rectSize * self.env.maze.shape[1], self.rectSize * self.env.maze.shape[0] + 20)
        self.show()

    def learn(self):
        episode = int(self.episode_edit.text())
        gamma = float(self.gamma_edit.text())
        lr = float(self.lr_edit.text())
        e = float(self.e_edit.text())
        decay = float(self.decay_edit.text())
        l = float(self.l_edit.text())

        if self.algo_combo.currentText() == 'Q_Learning':
            QLearning(self.env, self.Q, episode, gamma, lr, e, decay)
        elif self.algo_combo.currentText() == 'Sarsa':
            Sarsa(self.env, self.Q, episode, gamma, lr, e, decay)
        else:
            Sarsa_lambda(self.env, self.Q, episode, gamma, lr, e, decay, l)
        self.done = False
        self.env.reset()
        self.repaint()

    def reset(self):
        self.Q[:] = 0
        self.repaint()

    def next_step(self):
        if not self.done:
            action = np.argmax(self.Q[self.state])
            self.state, _, self.done, _ = self.env.step(action)
        self.repaint()

    def restart(self):
        self.env.reset()
        self.done = False
        self.repaint()

    def paintEvent(self, event):
        super(MazeGUI, self).paintEvent(event)

        palette = QPalette()
        palette.setColor(self.backgroundRole(), Qt.black)
        self.setPalette(palette)

        painter = QPainter(self)
        painter.setPen(QPen(Qt.black, 3, Qt.SolidLine))
        font = QFont()
        font.setPixelSize(20)
        painter.setFont(font)

        for i in range(self.env.wall.shape[0]):
            for j in range(self.env.wall.shape[1]):
                if self.env.wall[i, j]:
                    painter.setBrush(QBrush(Qt.black))
                elif self.env.end[i, j]:
                    painter.setBrush(QBrush(Qt.cyan))
                else:
                    painter.setBrush(QBrush(Qt.white))

                painter.drawRect(j * self.rectSize, i * self.rectSize, self.rectSize, self.rectSize)

                if (np.array([i, j]) == self.env.pos).all():
                    painter.setBrush(QBrush(Qt.red))
                    painter.setPen(QPen(Qt.red, 0))
                    painter.drawEllipse(j * self.rectSize + 3, i * self.rectSize + 3, self.rectSize - 5,
                                        self.rectSize - 5)
                    painter.setPen(QPen(Qt.black, 3, Qt.SolidLine))

                if not self.env.wall[i, j] and not self.env.end[i, j]:
                    painter.drawText(j * self.rectSize, i * self.rectSize, self.rectSize, self.rectSize, Qt.AlignCenter,
                                     "%.2f" % np.mean(self.Q[i * self.env.wall.shape[1] + j]))

                if self.env.end[i, j]:
                    painter.drawText(j * self.rectSize, i * self.rectSize, self.rectSize, self.rectSize, Qt.AlignCenter,
                                     str(self.env.end[i, j]))

        painter.end()
