import sys
import numpy as np
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import QDialog, QVBoxLayout
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

class ThetaHistogramWindow(QDialog):
    def __init__(self, thetas, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Theta Angles Histogram")
        self.setGeometry(200, 200, 800, 600)
        
        # Create layout
        layout = QVBoxLayout()
        
        # Create matplotlib figure and canvas
        self.figure = Figure(figsize=(8, 6), dpi=100)
        self.canvas = FigureCanvas(self.figure)
        layout.addWidget(self.canvas)
        
        self.setLayout(layout)
        
        # Plot the histogram
        self.plot_histogram(thetas)
    
    def plot_histogram(self, thetas):
        if thetas is None or len(thetas) == 0:
            return
        # Clear any existing plots
        self.figure.clear()
        
        # Create subplot
        ax = self.figure.add_subplot(111)
        
        # Convert thetas from radians to degrees
        thetas_degrees = np.degrees(thetas)
        
        # Plot histogram
        ax.hist(thetas_degrees, bins=36, range=(0, 180), color='skyblue', edgecolor='black')
        ax.set_title('Histogram of Line Orientation Angles')
        ax.set_xlabel('Angle (degrees)')
        ax.set_ylabel('Frequency')
        ax.set_xlim(0, 180)
        
        # Redraw the canvas
        self.canvas.draw()