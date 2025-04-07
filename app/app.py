import sys
import os
import cv2
import numpy as np
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, QTimer
from components.RectangleSelector import RectangleSelector
from components.ImageProcessor import ImageProcessor as IP

class ImageProcessingApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Image Processing with ROI Selection")
        self.setGeometry(100, 100, 1200, 800)
        
        # Initialize instance variables
        self.image = None
        self.image_raw_h = None
        self.image_raw_w = None
        self.gray = None
        self.processed_image = None
        self.binary_image = None
        self.focus_mask = None
        self.focus_image = None
        self.contours = None
        
        # Create the main layout
        main_layout = QHBoxLayout()
        
        # Create left panel (controls)
        left_panel = QVBoxLayout()
        
        # Import button
        self.import_btn = QPushButton("Import Image")
        self.import_btn.clicked.connect(self.import_image)
        left_panel.addWidget(self.import_btn)
        
        # Processing parameters
        self.create_parameter_controls(left_panel)
        
        # Left panel container
        left_container = QWidget()
        left_container.setLayout(left_panel)
        left_container.setFixedWidth(300)
        
        # Right panel (image display)
        self.image_display = RectangleSelector()
        self.image_display.setAlignment(Qt.AlignCenter)
        self.image_display.setStyleSheet("QLabel { background-color: #f0f0f0; }")
        self.image_display.rectangle_updated.connect(self.auto_process)
        
        # Add panels to main layout
        main_layout.addWidget(left_container)
        main_layout.addWidget(self.image_display)
        
        # Central widget
        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)
        
    def create_parameter_controls(self, layout):
                
        # Adaptive Threshold parameters
        threshold_group = QGroupBox("Adaptive Threshold")
        threshold_layout = QVBoxLayout()
        
        block_size_label = QLabel("Block Size (3-51):")
        self.block_size_slider = QSlider(Qt.Horizontal)
        self.block_size_slider.setMinimum(1)
        self.block_size_slider.setMaximum(25)
        self.block_size_slider.setValue(5)
        self.block_size_slider.setTickPosition(QSlider.TicksBelow)
        self.block_size_slider.setTickInterval(2)
        self.block_size_slider.valueChanged.connect(self.auto_process)
        
        c_value_label = QLabel("C Value (0-20):")
        self.c_value_slider = QSlider(Qt.Horizontal)
        self.c_value_slider.setMinimum(0)
        self.c_value_slider.setMaximum(20)
        self.c_value_slider.setValue(2)
        self.c_value_slider.setTickPosition(QSlider.TicksBelow)
        self.c_value_slider.setTickInterval(1)
        self.c_value_slider.valueChanged.connect(self.auto_process)
        
        threshold_layout.addWidget(block_size_label)
        threshold_layout.addWidget(self.block_size_slider)
        threshold_layout.addWidget(c_value_label)
        threshold_layout.addWidget(self.c_value_slider)
        threshold_group.setLayout(threshold_layout)
        layout.addWidget(threshold_group)
        
        # Hough Transform parameters (threshold_sobel, threshold_hough, minLineLength, maxLineGap
        hough_group = QGroupBox("Hough Transform")
        hough_layout = QVBoxLayout()

        # Sobel Threshold
        sobel_label = QLabel("Sobel Threshold (0-255):")
        self.sobel_slider = QSlider(Qt.Horizontal)
        self.sobel_slider.setMinimum(0)
        self.sobel_slider.setMaximum(255)
        self.sobel_slider.setValue(100)
        self.sobel_slider.setTickPosition(QSlider.TicksBelow)
        self.sobel_slider.setTickInterval(10)
        self.sobel_slider.valueChanged.connect(self.auto_process)

        # Hough Threshold
        hough_label = QLabel("Hough Threshold (1-200):")
        self.hough_slider = QSlider(Qt.Horizontal)
        self.hough_slider.setMinimum(1)
        self.hough_slider.setMaximum(200)
        self.hough_slider.setValue(50)
        self.hough_slider.setTickPosition(QSlider.TicksBelow)
        self.hough_slider.setTickInterval(10)
        self.hough_slider.valueChanged.connect(self.auto_process)

        # Minimum Line Length
        min_line_length_label = QLabel("Min Line Length (1-500):")
        self.min_line_length_slider = QSlider(Qt.Horizontal)
        self.min_line_length_slider.setMinimum(1)
        self.min_line_length_slider.setMaximum(500)
        self.min_line_length_slider.setValue(50)
        self.min_line_length_slider.setTickPosition(QSlider.TicksBelow)
        self.min_line_length_slider.setTickInterval(10)
        self.min_line_length_slider.valueChanged.connect(self.auto_process)

        # Maximum Line Gap
        max_line_gap_label = QLabel("Max Line Gap (1-100):")
        self.max_line_gap_slider = QSlider(Qt.Horizontal)
        self.max_line_gap_slider.setMinimum(1)
        self.max_line_gap_slider.setMaximum(100)
        self.max_line_gap_slider.setValue(10)
        self.max_line_gap_slider.setTickPosition(QSlider.TicksBelow)
        self.max_line_gap_slider.setTickInterval(5)
        self.max_line_gap_slider.valueChanged.connect(self.auto_process)

        # Add widgets to layout
        hough_layout.addWidget(sobel_label)
        hough_layout.addWidget(self.sobel_slider)
        hough_layout.addWidget(hough_label)
        hough_layout.addWidget(self.hough_slider)
        hough_layout.addWidget(min_line_length_label)
        hough_layout.addWidget(self.min_line_length_slider)
        hough_layout.addWidget(max_line_gap_label)
        hough_layout.addWidget(self.max_line_gap_slider)
        hough_group.setLayout(hough_layout)
        layout.addWidget(hough_group)

        self._setup_circle_detection_controls(layout)

    def _setup_circle_detection_controls(self, layout):
        # Circle Detection parameters
        circle_group = QGroupBox("Circle Detection")
        circle_layout = QVBoxLayout()

        # Min Radius
        min_radius_label = QLabel("Min Radius (10-200):")
        self.min_radius_slider = QSlider(Qt.Horizontal)
        self.min_radius_slider.setMinimum(10)
        self.min_radius_slider.setMaximum(200)
        self.min_radius_slider.setValue(50)
        self.min_radius_slider.setTickPosition(QSlider.TicksBelow)
        self.min_radius_slider.setTickInterval(20)
        self.min_radius_slider.valueChanged.connect(self.auto_process)

        # Max Radius
        max_radius_label = QLabel("Max Radius (50-500):")
        self.max_radius_slider = QSlider(Qt.Horizontal)
        self.max_radius_slider.setMinimum(50)
        self.max_radius_slider.setMaximum(500)
        self.max_radius_slider.setValue(300)
        self.max_radius_slider.setTickPosition(QSlider.TicksBelow)
        self.max_radius_slider.setTickInterval(50)
        self.max_radius_slider.valueChanged.connect(self.auto_process)

        # Param1 (edge detection threshold)
        param1_label = QLabel("Param1 - Edge Threshold (10-200):")
        self.param1_slider = QSlider(Qt.Horizontal)
        self.param1_slider.setMinimum(10)
        self.param1_slider.setMaximum(200)
        self.param1_slider.setValue(100)
        self.param1_slider.setTickPosition(QSlider.TicksBelow)
        self.param1_slider.setTickInterval(20)
        self.param1_slider.valueChanged.connect(self.auto_process)

        # Param2 (circle detection threshold)
        param2_label = QLabel("Param2 - Circle Threshold (10-200):")
        self.param2_slider = QSlider(Qt.Horizontal)
        self.param2_slider.setMinimum(10)
        self.param2_slider.setMaximum(200)
        self.param2_slider.setValue(75)
        self.param2_slider.setTickPosition(QSlider.TicksBelow)
        self.param2_slider.setTickInterval(20)
        self.param2_slider.valueChanged.connect(self.auto_process)

        # DP (resolution ratio)
        dp_label = QLabel("DP - Resolution Ratio (1-10):")
        self.dp_slider = QSlider(Qt.Horizontal)
        self.dp_slider.setMinimum(1)
        self.dp_slider.setMaximum(10)
        self.dp_slider.setValue(1)
        self.dp_slider.setTickPosition(QSlider.TicksBelow)
        self.dp_slider.setTickInterval(1)
        self.dp_slider.valueChanged.connect(self.auto_process)

        # Add widgets to layout
        circle_layout.addWidget(min_radius_label)
        circle_layout.addWidget(self.min_radius_slider)
        circle_layout.addWidget(max_radius_label)
        circle_layout.addWidget(self.max_radius_slider)
        # circle_layout.addWidget(param1_label)
        # circle_layout.addWidget(self.param1_slider)
        circle_layout.addWidget(param2_label)
        circle_layout.addWidget(self.param2_slider)
        # circle_layout.addWidget(dp_label)
        # circle_layout.addWidget(self.dp_slider)
        circle_group.setLayout(circle_layout)
        layout.addWidget(circle_group)

    def import_image(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Image Files (*.png *.jpg *.jpeg *.bmp *.tif)")
        if file_path:
            self.image = cv2.imread(file_path)
            self.display_image(self.image)
            # Auto-process the image after importing
            self.auto_process()
            
    def display_image(self, img):
        if img is None:
            return

        # Convert to RGB for display
        if len(img.shape) == 3:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            # If grayscale, convert to RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            
        self.image_raw_h, self.image_raw_w = img_rgb.shape[:2]
        
        # Convert to QImage and then to QPixmap
        bytes_per_line = 3 * self.image_raw_w
        q_img = QImage(img_rgb.data, self.image_raw_w, self.image_raw_h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
        
        # Display in image label with appropriate scaling
        self.image_display.setPixmap(pixmap.scaled(
            self.image_display.width(), self.image_display.height(),
            Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def auto_process(self):
        # Wrapper method to trigger process_image with a slight delay to prevent multiple rapid calls
        if self.image is not None:
            # Use QTimer to prevent multiple rapid calls when multiple sliders are moved
            QTimer.singleShot(100, self.process_image)

    def process_image(self):
        if self.image is None:
            return
            
        # Get rectangle coordinates
        rect_coords = self.image_display.get_rectangle_coordinates(self.image_raw_h, self.image_raw_w)
        if rect_coords is None:
            print("No region selected")
            return
        
        # Get parameters from sliders
        block_size = self.block_size_slider.value() * 2 + 1  # Make sure it's odd
        c_value = self.c_value_slider.value()
        threshold_sobel = self.sobel_slider.value()
        threshold_hough = self.hough_slider.value()
        minLineLength = self.min_line_length_slider.value()
        maxLineGap = self.max_line_gap_slider.value()

        x, y, w, h = rect_coords
        cropped_image = self.image[y:y+h, x:x+w]

        # Process the cropped image
        cropped_processed_image = \
            IP.process(cropped_image, 
                block_size=block_size, 
                c_value=c_value,
                threshold_sobel=threshold_sobel,
                threshold_hough=threshold_hough,
                minLineLength=minLineLength,
                maxLineGap=maxLineGap
            )
        
        detected_circles = IP.detect_circles(
            self.image[y:y+h, x:x+w],
            dp=self.dp_slider.value(),
            minRadius=self.min_radius_slider.value(),
            maxRadius=self.max_radius_slider.value(),
            param1=self.param1_slider.value(),
            param2=self.param2_slider.value()
        )
        # Draw detected circles on the processed image
        print(detected_circles)
        if detected_circles is not None:
            for (cx, cy, r) in detected_circles:
                cv2.circle(cropped_processed_image, (int(cx), int(cy)), int(r), (0, 255, 0), 10)

        self.processed_image = self.image.copy()
        self.processed_image[y:y+h, x:x+w] = cropped_processed_image
        
        # Display the processed image
        self.display_image(self.processed_image)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ImageProcessingApp()
    window.show()
    sys.exit(app.exec_())