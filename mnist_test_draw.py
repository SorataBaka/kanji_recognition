import sys
import numpy as np
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QPushButton
from PyQt5.QtGui import QPainter, QPen, QPainterPath, QImage
from PyQt5.QtCore import Qt, QPoint
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load the pretrained model
model = load_model('mnist-v1.keras')  # Replace with your model path

class DrawingApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Drawing App with Prediction")
        self.setGeometry(100, 100, 500, 650)

        # Canvas for drawing
        self.canvas = QImage(500, 500, QImage.Format_RGB32)
        self.canvas.fill(Qt.white)


        # Clear Canvas button
        self.clear_button = QPushButton("Clear Canvas", self)
        self.clear_button.clicked.connect(self.clear_canvas)
        self.clear_button.setStyleSheet("font-size: 16px; padding: 10px;")

        # Layout
        layout = QVBoxLayout()
        layout.addWidget(self.clear_button)
        self.setLayout(layout)

        # Drawing variables
        self.path = QPainterPath()
        self.drawing = False

    def paintEvent(self, event):
        # Draw the canvas
        painter = QPainter(self)
        painter.drawImage(0, 0, self.canvas)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = True
            self.path.moveTo(event.pos())

    def mouseMoveEvent(self, event):
        if self.drawing and event.buttons() == Qt.LeftButton:
            self.path.lineTo(event.pos())
            self.update_canvas()
            self.predict_drawing()
            

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = False

    def update_canvas(self):
        # Update the canvas with the current drawing
        painter = QPainter(self.canvas)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(QPen(Qt.black, 30, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
        painter.drawPath(self.path)
        self.update()

    def predict_drawing(self):
        # Convert the canvas to a grayscale numpy array
        image = self.canvas.scaled(28, 28)  # Resize to 28x28 (MNIST input size)
        image.invertPixels()
        image = image.convertToFormat(QImage.Format_Grayscale8)
        ptr = image.bits()
        ptr.setsize(image.byteCount())
        drawing_array = np.array(ptr).reshape(28, 28, 1)  # Reshape to (28, 28, 1)

        # Normalize the image (if required by the model)
        drawing_array = drawing_array.astype('float32') / 255.0

        # Add batch dimension
        drawing_array = np.expand_dims(drawing_array, axis=0)

        # Make a prediction
        prediction = model.predict(drawing_array)
        print(prediction)
        print(np.argmax(prediction))
        

        # Update the prediction label (not on the canvas)
        self.setWindowTitle("{}".format(1))

    def clear_canvas(self):
        # Clear the canvas
        self.canvas.fill(Qt.white)
        self.path = QPainterPath()
        self.update()

# Run the application
app = QApplication(sys.argv)
window = DrawingApp()
window.show()
sys.exit(app.exec_())