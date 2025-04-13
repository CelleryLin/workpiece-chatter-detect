from PySide6.QtWidgets import QLabel, QSizePolicy
from PySide6.QtGui import QPainter, QPen
from PySide6.QtCore import Qt, QRect, Signal

class RectangleSelector(QLabel):
    rectangle_updated = Signal()
    def __init__(self, parent=None):
        super().__init__(parent)
        self.start_point = None
        self.end_point = None
        self.drawing = False
        self.selection_rect = QRect()  # Changed from rect to selection_rect to avoid confusion
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.start_point = event.pos()
            self.end_point = event.pos()
            self.drawing = True
            self.selection_rect = QRect(self.start_point, self.end_point)
            self.update()
            
    def mouseMoveEvent(self, event):
        if self.drawing:
            self.end_point = event.pos()
            self.selection_rect = QRect(self.start_point, self.end_point).normalized()
            self.update()
            
    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self.drawing:
            self.end_point = event.pos()
            self.selection_rect = QRect(self.start_point, self.end_point).normalized()
            self.drawing = False
            self.update()
            self.rectangle_updated.emit()
            
    def paintEvent(self, event):
        super().paintEvent(event)
        if self.start_point and self.end_point:
            painter = QPainter(self)
            pen = QPen(Qt.red, 2, Qt.SolidLine)
            painter.setPen(pen)
            painter.drawRect(self.selection_rect)
            
    def get_rectangle_coordinates(self, raw_h, raw_w):
        """Return the rectangle coordinates as (x, y, width, height)"""
        if not self.selection_rect.isValid():
            return None
        
        # Get the coordinates relative to the image
        if self.pixmap() is None:
            return None
            
        pixmap_rect = self.pixmap().rect()
        label_rect = self.rect()
        view_width = pixmap_rect.width()
        view_height = pixmap_rect.height()
        x_scale = raw_w / view_width
        y_scale = raw_h / view_height
        x_shift = (label_rect.width() - view_width) / 2
        y_shift = (label_rect.height() - view_height) / 2

        x = int((self.selection_rect.x() - x_shift) * x_scale)
        y = int((self.selection_rect.y() - y_shift) * y_scale)
        width = int(self.selection_rect.width() * x_scale)
        height = int(self.selection_rect.height() * y_scale)

        # Ensure the coordinates are within bounds
        x = max(0, min(x, raw_w - 1))
        y = max(0, min(y, raw_h - 1))
        width = max(1, min(width, raw_w - x))
        height = max(1, min(height, raw_h - y))
        # Ensure width and height are positive
        width = abs(width)
        height = abs(height)

        return x, y, width, height