import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, QTimer
from components.RectangleSelector import RectangleSelector
from components.ImageProcessor import ImageProcessor as IP

IP.imshow_disable = True  # Disable imshow in ImageProcessor

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

        self.circle_real_area = 6*6*np.pi  # mm^2
        
        # Create the main layout
        main_layout = QHBoxLayout()
        
        # Create left panel (controls)
        left_panel = QVBoxLayout()
        
        # Import button
        self.import_btn = QPushButton("Import Image")
        self.import_btn.clicked.connect(self.import_image)
        left_panel.addWidget(self.import_btn)
        
        # display the Chattering Area
        self.true_area_label = QLabel("Chattering Area: N/A")
        self.true_area_label.setStyleSheet("font-size: 14px; font-weight: bold;")
        left_panel.addWidget(self.true_area_label)

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
        # Helper function to create sliders
        def add_slider(parent_layout, label_text, min_val, max_val, default_val, 
                       tick_interval=1, decimals=0):
            # Create a horizontal layout for the label and value
            label_layout = QHBoxLayout()
            
            # Create label with parameter name
            label = QLabel(label_text)
            label_layout.addWidget(label)
            
            # Create value label and add it to the right of the parameter name
            display_value = default_val if decimals == 0 else default_val/10**decimals
            value_label = QLabel(f"{display_value:.{decimals}f}")
            value_label.setAlignment(Qt.AlignRight)
            label_layout.addWidget(value_label)
            
            # Add the label layout to the parent layout
            parent_layout.addLayout(label_layout)
            
            # Create and configure the slider
            slider = QSlider(Qt.Horizontal)
            slider.setMinimum(min_val)
            slider.setMaximum(max_val)
            slider.setValue(default_val)
            slider.setTickPosition(QSlider.TicksBelow)
            slider.setTickInterval(tick_interval)
            slider.valueChanged.connect(self.auto_process)
            
            # Update value label when slider changes
            def update_label(value):
                display_value = value if decimals == 0 else value/10**decimals
                value_label.setText(f"{display_value:.{decimals}f}")
            
            slider.valueChanged.connect(update_label)
            
            parent_layout.addWidget(slider)
            return slider
                
        # Adaptive Threshold parameters
        threshold_group = QGroupBox("Adaptive Threshold")
        threshold_layout = QVBoxLayout()
        
        self.block_size_slider = add_slider(threshold_layout, "Block Size:", 
                                           1, 25, 5, tick_interval=2)
        
        threshold_group.setLayout(threshold_layout)
        layout.addWidget(threshold_group)

        # Gabor Filter parameters
        gabor_group = QGroupBox("Gabor Filter")
        gabor_layout = QVBoxLayout()

        self.ksize_slider = add_slider(gabor_layout, "Kernel Size:", 
                                      1, 15, 3, tick_interval=2)
        self.theta_slider = add_slider(gabor_layout, "Theta:", 
                                      0, 180, 0, tick_interval=15)
        self.lambda_slider = add_slider(gabor_layout, "Lambda:", 
                                       5, 20, 10, tick_interval=1)
        
        gabor_group.setLayout(gabor_layout)
        layout.addWidget(gabor_group)

        # Add blur parameters
        blur_group = QGroupBox("Blur")
        blur_layout = QVBoxLayout()
        self.blur_ksize_slider = add_slider(blur_layout, "Kernel Size:",
                                           1, 35, 15, tick_interval=2)
        
        blur_group.setLayout(blur_layout)
        layout.addWidget(blur_group)

        # Add circle detection parameters
        circle_detection_group = QGroupBox("Circle Detection")
        circle_detection_layout = QVBoxLayout()
        self.circle_min_radius_slider = add_slider(
            circle_detection_layout, "Minimum Circle Radius:", 
            1, 500, 50, tick_interval=10
        )
        self.circle_max_radius_slider = add_slider(
            circle_detection_layout, "Maximum Circle Radius:",
            1, 500, 300, tick_interval=10
        )
        self.circle_param2_slider = add_slider(
            circle_detection_layout, "Detection Threshold:",
            1, 300, 100, tick_interval=10
        )
        circle_detection_group.setLayout(circle_detection_layout)
        layout.addWidget(circle_detection_group)

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
    
    def calculate_true_area(self):
        if self.f_area is not None and self.c_area is not None:
            true_area = (self.f_area * self.circle_real_area) / self.c_area
            self.true_area_label.setText(f"Chattering Area: {true_area:.2f} mm^2")
            return true_area
        else:
            self.true_area_label.setText("Chattering Area: N/A")
            return None

    def auto_process(self):
        if self.image is not None:
            QTimer.singleShot(100, self.process_image)

    def process_image(self):
        if self.image is None:
            return
        
        self.processed_image = self.image.copy()

        self.f_area = None
        self.c_area = None
        
        # Get rectangle coordinates
        rect_coords = self.image_display.get_rectangle_coordinates(self.image_raw_h, self.image_raw_w)
        if rect_coords is not None:
            # Get parameters from sliders
            block_size = self.block_size_slider.value() * 2 + 1  # Make sure it's odd
            gabor_ksize = self.ksize_slider.value() * 2 + 1
            theta = self.theta_slider.value()
            lambd = self.lambda_slider.value()
            blur_ksize = self.blur_ksize_slider.value() * 2 + 1

            x, y, w, h = rect_coords
            cropped_image = self.image[y:y+h, x:x+w]

            # Process the cropped image
            max_contours = \
                IP.process(cropped_image, 
                    block_size=block_size, 
                    gabor_ksize=gabor_ksize,
                    theta=theta,
                    lambd=lambd,
                    blur_ksize=blur_ksize,
                )

            if max_contours is not None:
                processed_image_ = cropped_image.copy()
                cv2.drawContours(processed_image_, [max_contours], -1, (255, 0, 0), 3)
                self.f_area = cv2.contourArea(max_contours)
                self.processed_image[y:y+h, x:x+w] = processed_image_
                cx, cy, cw, ch = cv2.boundingRect(max_contours)
                text = f"Area: {int(self.f_area)} px"
                cv2.putText(self.processed_image, text, (int(x + cx + 0.2*cw), y + cy),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        
        else:
            print("No region selected")
        
        # Find circle
        circles = IP.detect_circles(
            self.image, 
            minRadius=self.circle_min_radius_slider.value(),
            maxRadius=self.circle_max_radius_slider.value(),
            param2=self.circle_param2_slider.value()
        )
        if circles is not None:
            for (x, y, r) in circles:
                x, y, r = int(x), int(y), int(r)
                cv2.circle(self.processed_image, (x, y), r, (0, 255, 0), 2)
                cv2.circle(self.processed_image, (x, y), 2, (0, 255, 0), 3)

                # write the area (px) of the circle at the left top corner of the circle
                self.c_area = np.pi * r**2
                text = f"Area: {int(self.c_area)} px"
                cv2.putText(self.processed_image, text, (int(x + 0.8*r), int(y - 0.8*r)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
        self.calculate_true_area()
        self.display_image(self.processed_image)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ImageProcessingApp()
    window.show()
    sys.exit(app.exec_())