from PySide6.QtWidgets import QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, QGraphicsTextItem
from PySide6.QtGui import QPainter, QFont
from PySide6.QtCore import Signal, QPointF, Qt


class ImageGraphicsView(QGraphicsView):
    """Custom QGraphicsView with pan/zoom and mouse tracking"""
    mouse_moved = Signal(int, int)  # x, y coordinates in image space
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Create scene
        self.scene = QGraphicsScene()
        self.setScene(self.scene)
        
        # Enable mouse tracking and interactions
        self.setMouseTracking(True)
        self.setDragMode(QGraphicsView.RubberBandDrag)
        self.setRenderHint(QPainter.Antialiasing)
        
        # Enable pan with middle mouse or dragging
        self.setDragMode(QGraphicsView.ScrollHandDrag)
        
        # Image item
        self.pixmap_item = None
        self.original_pixmap = None
        
        # Track last mouse position in scene coordinates
        self.last_scene_pos = QPointF()
        
    def setPixmap(self, pixmap, preserve_view=False):
        self.original_pixmap = pixmap
        
        # Save current transform if preserving view
        saved_transform = None
        if preserve_view and self.pixmap_item:
            saved_transform = self.transform()
        
        # Clear existing items
        self.scene.clear()
        
        # Add new pixmap
        self.pixmap_item = QGraphicsPixmapItem(pixmap)
        self.scene.addItem(self.pixmap_item)
        
        # Set scene rectangle to image bounds
        self.scene.setSceneRect(pixmap.rect())
        
        # Restore transform or fit to view
        if preserve_view and saved_transform:
            self.setTransform(saved_transform)
        else:
            # Fit the image to the view initially
            self.fitInView(self.pixmap_item, Qt.KeepAspectRatio)
        
    def mouseMoveEvent(self, event):
        """Handle mouse movement for coordinate tracking"""
        if self.pixmap_item and self.original_pixmap:
            # Convert viewport coordinates to scene coordinates
            scene_pos = self.mapToScene(event.pos())
            
            # Convert scene coordinates to image coordinates
            img_x = int(scene_pos.x())
            img_y = int(scene_pos.y())
            
            # Check bounds
            if (0 <= img_x < self.original_pixmap.width() and 
                0 <= img_y < self.original_pixmap.height()):
                self.mouse_moved.emit(img_x, img_y)
                
        super().mouseMoveEvent(event)
        
    def wheelEvent(self, event):
        """Handle mouse wheel for zooming"""
        # Always zoom with mouse wheel (no Ctrl required)
        zoom_factor = 1.15
        if event.angleDelta().y() > 0:
            # Zoom in
            self.scale(zoom_factor, zoom_factor)
        else:
            # Zoom out
            self.scale(1.0 / zoom_factor, 1.0 / zoom_factor)
        event.accept()
            
    def zoom_in(self):
        """Zoom in by 25%"""
        self.scale(1.25, 1.25)
        
    def zoom_out(self):
        """Zoom out by 25%"""
        self.scale(0.8, 0.8)
        
    def zoom_fit(self):
        """Fit image to window"""
        if self.pixmap_item:
            self.fitInView(self.pixmap_item, Qt.KeepAspectRatio)

    def clear(self):
        """Clear the view"""
        self.scene.clear()
        self.pixmap_item = None
        self.original_pixmap = None
        
    def clear_overlays(self):
        """Clear all overlay items (circles) while keeping the image"""
        if self.scene:
            # Find and remove all items except the pixmap item
            items_to_remove = []
            for item in self.scene.items():
                if item != self.pixmap_item:
                    items_to_remove.append(item)
            for item in items_to_remove:
                self.scene.removeItem(item)
        
    def show_message(self, message: str):
        """Show a text message in the view when no image is loaded"""
        # Clear existing items
        self.scene.clear()
        self.pixmap_item = None
        self.original_pixmap = None
        
        # Create text item
        text_item = QGraphicsTextItem(message)
        font = QFont()
        font.setPointSize(12)
        text_item.setFont(font)
        text_item.setDefaultTextColor(Qt.gray)
        
        # Center the text in the view
        text_rect = text_item.boundingRect()
        view_rect = self.viewport().rect()
        x = (view_rect.width() - text_rect.width()) / 2
        y = (view_rect.height() - text_rect.height()) / 2
        text_item.setPos(x, y)
        
        self.scene.addItem(text_item)
        self.scene.setSceneRect(view_rect)
