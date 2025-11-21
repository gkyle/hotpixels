"""Image Correction Tab for correcting images using hot pixel profiles."""

import os
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Tuple

import cv2
import numpy as np
import traceback as tb
import tempfile

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QFileDialog, QMessageBox,
    QSizePolicy
)
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtUiTools import QUiLoader
from PySide6.QtWidgets import QMessageBox
from PySide6.QtWidgets import QListWidgetItem as ListItem

from hotpixels.app import App
from hotpixels.profile import HotPixelProfile
from hotpixels.image import DNGImage
from .workers import CorrectionWorker, MultiImageLoadingWorker, TrainingDataWorker, CNNDetectionWorker
from .image_graphics_view import ImageGraphicsView
from .formatters import format_profile_summary, format_image_summary

if TYPE_CHECKING:
    from hotpixels.ui.main_window import HotPixelGUI


class ImageCorrectionTab(QWidget):
    """Tab for correcting images using hot pixel profiles"""

    def __init__(self, app: App):
        super().__init__()
        self.app = app  # Shared App instance
        self.image_paths: List[str] = []  # List of image file paths
        self.rgbImages: List[DNGImage] = []  # For display (process_rgb=True)
        self.rawImages: List[DNGImage] = []  # For correction (process_rgb=False)
        self.corrected_image_paths: List[str] = []  # Paths to corrected images
        self.corrected_rgb_images: List[DNGImage] = []  # Cached corrected RGB images for ROI display
        self.worker: Optional[CorrectionWorker] = None
        self.image_loading_worker: Optional[MultiImageLoadingWorker] = None  # Background image loader
        self.cnn_detection_worker: Optional[CNNDetectionWorker] = None  # Background CNN detection worker
        self.main_window: Optional['HotPixelGUI'] = None  # Reference to main window
        
        # Interactive mode state caching
        self.cached_original_raw: Optional[DNGImage] = None  # Uncorrected original for interactive preview
        self.cached_cnn_detections: List[Tuple[int, int, float]] = []  # CNN results with confidence
        self.cached_preview_image: Optional[DNGImage] = None  # Current preview with corrections applied
        
        # Current correction parameters
        self.current_deviation_threshold: Optional[float] = None  # From profile by default
        self.current_sensitivity: float = 0.9  # Default sensitivity (0.1 confidence threshold)
        
        # Enabled corrections flags
        self.subtract_noise_enabled: bool = True
        self.correct_profile_hotpixels_enabled: bool = True
        self.correct_cnn_hotpixels_enabled: bool = True
        
        # Debounce timers for slider updates
        self.deviation_threshold_timer: Optional[QTimer] = None
        self.sensitivity_timer: Optional[QTimer] = None
        
        # Hot pixel counts for display
        self.profile_hotpixel_count: int = 0
        self.cnn_hotpixel_count: int = 0
        
        self.load_ui()
        self.setup_connections()
        self.initialize_button_states()
        self.update_image_list_display()  # Initialize the file list display

    def set_main_window(self, main_window):
        """Set reference to the main window for shared profile access"""
        self.main_window = main_window
    
    def initialize_button_states(self):
        """Initialize button states."""
        # Initially disable both buttons until we have images and profile loaded
        if hasattr(self.ui, 'saveButton'):
            self.ui.saveButton.setEnabled(False)
        if hasattr(self.ui, 'batchProcessButton'):
            self.ui.batchProcessButton.setEnabled(False)

    def load_ui(self):
        """Load the UI from the .ui file"""
        loader = QUiLoader()
        ui_file = Path(__file__).parent / "image_correction_tab.ui"
        self.ui = loader.load(str(ui_file), self)

        # Set layout to contain the loaded UI
        layout = QVBoxLayout()
        layout.addWidget(self.ui)
        self.setLayout(layout)
        
        # Replace the originalImageLabel with ImageGraphicsView
        self.setup_image_view()

    def setup_image_view(self):
        """Replace originalImageLabel with ImageGraphicsView for pan/zoom support"""
        # Find the imageFrame within the splitter
        place_holder = self.ui.originalImageLabel
        image_frame = place_holder.parent()
        if image_frame:
            # Ensure imageFrame itself has an expanding size policy
            image_frame.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            
            layout = image_frame.layout()
            if layout:
                # Remove the existing originalImageLabel
                for i in reversed(range(layout.count())):
                    child = layout.itemAt(i).widget()
                    if child and hasattr(child, 'objectName') and child.objectName() == 'originalImageLabel':
                        child.setParent(None)
                        break
                
                # Create and add the ImageGraphicsView
                self.image_view = ImageGraphicsView()
                self.image_view.setObjectName('originalImageView')
                self.image_view.setMinimumSize(200, 200)
                # Set size policy to expand in both directions
                self.image_view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
                layout.addWidget(self.image_view, 1)  # Add stretch factor of 1
                
                # Connect mouse movement signal for ROI display
                self.image_view.mouseMoved.connect(self.on_image_mouse_moved)
                
                # Show initial message
                self.image_view.showMessage("No image loaded")
                
                # Store reference as originalImageLabel for compatibility
                self.ui.originalImageLabel = self.image_view

    def setup_connections(self):
        """Connect UI signals to slots"""
        self.ui.pushButton.clicked.connect(self.load_profile)
        self.ui.openOriginalButton.clicked.connect(self.select_image)
        self.ui.batchProcessButton.clicked.connect(self.start_batch_correction)
        self.ui.saveButton.clicked.connect(self.save_interactive_correction)
        self.ui.saveTrainingDataButton.clicked.connect(self.save_training_data)

        # Configure and connect sliders for interactive parameter adjustment
        if hasattr(self.ui, 'horizontalSlider_2'):  # Deviation threshold slider
            self.ui.horizontalSlider_2.setMinimum(10)  # 1.0σ * 10
            self.ui.horizontalSlider_2.setMaximum(250)  # 25.0σ * 10
            self.ui.horizontalSlider_2.setValue(50)  # Default 5.0σ
            self.ui.horizontalSlider_2.valueChanged.connect(self.on_deviation_threshold_slider_moved)
            
            # Create debounce timer for deviation threshold
            self.deviation_threshold_timer = QTimer()
            self.deviation_threshold_timer.setSingleShot(True)
            self.deviation_threshold_timer.timeout.connect(self.on_deviation_threshold_changed)
        
        if hasattr(self.ui, 'horizontalSlider'):  # CNN sensitivity slider
            self.ui.horizontalSlider.setMinimum(1)  # 0.01 sensitivity
            self.ui.horizontalSlider.setMaximum(99)  # 0.99 sensitivity
            self.ui.horizontalSlider.setValue(90)  # Default 0.90 sensitivity
            self.ui.horizontalSlider.valueChanged.connect(self.on_sensitivity_slider_moved)
            
            # Create debounce timer for sensitivity
            self.sensitivity_timer = QTimer()
            self.sensitivity_timer.setSingleShot(True)
            self.sensitivity_timer.timeout.connect(self.on_sensitivity_changed)

        # Connect checkboxes for enabling/disabling corrections (now with interactive preview)
        if hasattr(self.ui, 'subtractNoiseProfileCheckBox'):
            self.ui.subtractNoiseProfileCheckBox.setChecked(True)  # Match default enabled state
            self.ui.subtractNoiseProfileCheckBox.toggled.connect(self.on_subtract_noise_toggled)
        if hasattr(self.ui, 'removeHotPixelsCheckbox'):
            self.ui.removeHotPixelsCheckbox.setChecked(True)  # Match default enabled state
            self.ui.removeHotPixelsCheckbox.toggled.connect(self.on_remove_hotpixels_toggled)
        if hasattr(self.ui, 'checkBox_applyCNN'):
            self.ui.checkBox_applyCNN.setChecked(True)  # Match default enabled state
            self.ui.checkBox_applyCNN.toggled.connect(self.on_apply_cnn_toggled)
        
        # Connect corrections panel checkboxes
        if hasattr(self.ui, 'showCorrectedImageCheckBox'):
            self.ui.showCorrectedImageCheckBox.toggled.connect(self.on_show_corrected_image_toggled)
        if hasattr(self.ui, 'showRgbImageCheckBox'):
            self.ui.showRgbImageCheckBox.toggled.connect(self.on_show_rgb_image_toggled)
    
    def is_interactive_mode(self) -> bool:
        """Check if we're in interactive mode (single image) or batch mode (multiple images)."""
        return len(self.image_paths) == 1
    
    def start_batch_correction(self):
        """Start batch correction process for multiple images."""
        if not self.app.current_profile or not self.image_paths:
            QMessageBox.warning(self, "Missing Data", "Please load both a profile and images first.")
            return

        # Get common hot pixels from profile
        if self.app.current_profile._median_noise_frame is None:
            QMessageBox.warning(self, "No Hot Pixels", "The loaded profile contains no hot pixels to correct.")
            return
        
        # Start batch correction
        self._start_batch_correction()
    
    def save_interactive_correction(self):
        """Save the current corrected preview image (interactive mode only)."""
        if not self.cached_preview_image:
            QMessageBox.warning(self, "No Preview", "No corrected preview available to save.")
            return
        
        try:
            # Save the preview image
            filename = self.cached_preview_image.filename
            _, ext = os.path.splitext(filename)
            suffix = "_corrected"
            if self.subtract_noise_enabled:
                suffix += "_denoised"
            if self.correct_profile_hotpixels_enabled:
                suffix += "_hp"
            if self.correct_cnn_hotpixels_enabled:
                suffix += "_cnn"
            corrected_filename = filename.replace(ext, suffix + ext)
            
            self.cached_preview_image.save(corrected_filename)
            self.showStatusMessage(f"Saved: {os.path.basename(corrected_filename)}", 3000)
            
        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Failed to save corrected image:\n{str(e)}")
            tb.print_exc()
    
    def _start_interactive_preview(self):
        """Start interactive preview mode for single image."""
        self.showStatusMessage("Computing corrections preview...", 0)
        
        # Cache the original image
        self.cached_original_raw = self._copy_dng_image(self.rawImages[0])
        
        # Initialize deviation threshold from profile if not set
        if self.current_deviation_threshold is None:
            self.current_deviation_threshold = self.app.current_profile.deviation_threshold
            # Update slider to match
            if hasattr(self.ui, 'horizontalSlider_2'):
                self.ui.horizontalSlider_2.setValue(int(self.current_deviation_threshold * 10))
        
        # Run CNN detection if enabled (this is the slow part - run in background)
        if self.correct_cnn_hotpixels_enabled:
            self.ui.saveButton.setText("Running CNN...")
            self.ui.saveButton.setEnabled(False)
            self.ui.batchProcessButton.setEnabled(False)
            
            # Run detection in background worker
            detection_image = self._copy_dng_image(self.cached_original_raw)
            self.cnn_detection_worker = CNNDetectionWorker(self.app, detection_image)
            self.cnn_detection_worker.progress.connect(self.on_cnn_detection_progress)
            self.cnn_detection_worker.finished.connect(self.on_cnn_detection_finished)
            self.cnn_detection_worker.error.connect(self.on_cnn_detection_error)
            self.cnn_detection_worker.start()
        else:
            # No CNN detection needed, proceed immediately
            self.cached_cnn_detections = []
            self._finish_interactive_preview()
    
    def on_cnn_detection_progress(self, message: str):
        """Handle CNN detection progress updates"""
        self.showStatusMessage(message, 0)
    
    def on_cnn_detection_finished(self, detections: List[Tuple[int, int, float]]):
        """Handle CNN detection completion"""
        self.cached_cnn_detections = detections
        self.cnn_detection_worker = None
        self._finish_interactive_preview()
    
    def on_cnn_detection_error(self, error_message: str):
        """Handle CNN detection error"""
        QMessageBox.critical(self, "CNN Detection Error", f"Failed to run CNN detection:\n{error_message}")
        self.cnn_detection_worker = None
        # Continue without CNN detections
        self.cached_cnn_detections = []
        self._finish_interactive_preview()
    
    def _finish_interactive_preview(self):
        """Complete the interactive preview setup after CNN detection (if enabled)."""
        # Compute initial preview
        self.recompute_preview()
        
        # Update UI for interactive mode
        self.ui.saveButton.setEnabled(True)
        self.ui.saveButton.setText("Save")
        self.check_ready_state()  # Update button states
        
        # Enable parameter controls
        if hasattr(self.ui, 'horizontalSlider_2'):
            self.ui.horizontalSlider_2.setEnabled(True)
        if hasattr(self.ui, 'horizontalSlider'):
            self.ui.horizontalSlider.setEnabled(True)
        
        # Show corrected image checkbox
        if hasattr(self.ui, 'showCorrectedImageCheckBox'):
            self.ui.showCorrectedImageCheckBox.setEnabled(True)
            self.ui.showCorrectedImageCheckBox.setChecked(True)
        
        # Enable RGB checkbox
        if hasattr(self.ui, 'showRgbImageCheckBox'):
            self.ui.showRgbImageCheckBox.setEnabled(True)
        
        # Update labels with initial counts
        if hasattr(self.ui, 'label_2'):
            self.ui.label_2.setText(f"Deviation Threshold: {self.current_deviation_threshold:.1f}σ ({self.profile_hotpixel_count} pixels)")
        if hasattr(self.ui, 'label'):
            self.ui.label.setText(f"Sensitivity: {self.current_sensitivity:.2f} ({self.cnn_hotpixel_count} pixels)")
        
        self.showStatusMessage("Preview ready! Adjust parameters to see changes.", 5000)
        
        self.showStatusMessage("Preview ready! Adjust parameters to see changes.", 5000)
    
    def _start_batch_correction(self):
        """Start batch correction mode for multiple images."""
        # Check if we need to load all images (lazy loading optimization)
        if not self.rawImages or len([img for img in self.rawImages if img is not None]) < len(self.image_paths):
            self.showStatusMessage("Loading all images for correction...", 3000)

            # Load all images now for correction
            self.correction_loading_worker = MultiImageLoadingWorker(self.image_paths, load_all=True)
            self.correction_loading_worker.progress.connect(self.update_correction_loading_progress)
            self.correction_loading_worker.finished.connect(self.all_images_loaded_for_correction)
            self.correction_loading_worker.error.connect(self.correction_loading_error)

            # Disable button during loading
            self.ui.batchProcessButton.setEnabled(False)
            self.ui.batchProcessButton.setText("Loading images...")
            self.ui.saveButton.setEnabled(False)

            self.correction_loading_worker.start()
        else:
            # All images already loaded, proceed with correction
            self._start_correction()

    def update_correction_loading_progress(self, message: str):
        """Update button text with loading progress"""
        self.ui.batchProcessButton.setText(f"Loading: {message}")

    def all_images_loaded_for_correction(self, image_paths: List[str], rgb_images: List[DNGImage], raw_images: List[DNGImage]):
        """Handle completion of loading all images for correction"""
        # Update our image data
        self.image_paths = image_paths
        self.rgbImages = rgb_images
        self.rawImages = raw_images

        # Now start the actual correction
        self._start_correction()

    def correction_loading_error(self, error_message: str):
        """Handle error during image loading for correction"""
        QMessageBox.critical(self, "Loading Error", f"Failed to load images for correction:\n{error_message}")

        # Re-enable buttons
        self.check_ready_state()

    def _start_correction(self):
        """Start the actual correction process (helper method)"""
        # Check if noise profile subtraction is enabled
        subtract_noise = hasattr(
            self.ui, 'subtractNoiseProfileCheckBox') and self.ui.subtractNoiseProfileCheckBox.isChecked()
        apply_cnn = self.ui.checkBox_applyCNN.isChecked()

        # Start correction in background thread
        self.worker = CorrectionWorker(self.app, self.image_paths, self.app.current_profile,
                                       subtract_noise_profile=subtract_noise, 
                                       apply_residual_hotpixel_model=apply_cnn,
                                       cnn_sensitivity=self.current_sensitivity)
        self.worker.progress.connect(self.update_correction_progress)
        self.worker.finished.connect(self.correction_finished)
        self.worker.error.connect(self.correction_error)

        # Disable buttons during processing
        self.ui.batchProcessButton.setEnabled(False)
        self.ui.saveButton.setEnabled(False)

        self.worker.start()

    def update_correction_progress(self, message: str):
        """Update button text with progress message"""
        self.ui.batchProcessButton.setText(message)

    def correction_finished(self, corrected_paths: List[str], model_hot_pixels: List[List] = None):
        """Handle successful correction completion"""

        # Store corrected image paths for ROI comparison
        self.corrected_image_paths = corrected_paths
        
        # Store model hot pixels for overlay visualization
        self.model_hot_pixels = model_hot_pixels or []

        # Load and cache corrected RGB images for ROI display
        self.corrected_rgb_images = []
        try:
            for corrected_path in corrected_paths:
                corrected_rgb = DNGImage(corrected_path, process_rgb=True)
                corrected_rgb.white_balance()
                self.corrected_rgb_images.append(corrected_rgb)
            print(f"Cached {len(self.corrected_rgb_images)} corrected RGB images for ROI display")
        except Exception as e:
            print(f"Error caching corrected images: {e}")
            self.corrected_rgb_images = []

        hot_pixels = self.app.get_hot_pixels(self.app.current_profile)
        subtract_noise = hasattr(
            self.ui, 'subtractNoiseProfileCheckBox') and self.ui.subtractNoiseProfileCheckBox.isChecked()

        # Update UI with results
        result_text = "Batch correction completed successfully!\n"
        result_text += f"Images processed: {len(corrected_paths)}\n"
        result_text += f"Hot pixels corrected: {len(hot_pixels)}\n"
        if subtract_noise:
            result_text += "Noise profile subtracted: Yes\n"
        result_text += "\n"
        result_text += "Corrected files:\n"
        for corrected_path in corrected_paths:
            result_text += f"• {os.path.basename(corrected_path)}\n"

        # Update the results text area if it exists
        if hasattr(self.ui, 'resultsTextEdit'):
            self.ui.resultsTextEdit.setText(result_text)

        # Enable the "Show corrected image" checkbox now that we have corrected images
        if hasattr(self.ui, 'showCorrectedImageCheckBox'):
            self.ui.showCorrectedImageCheckBox.setEnabled(True)
            
        # Enable hot pixel visualization checkboxes
        if hasattr(self.ui, 'showDarkFrameHotpixelsCheckBox'):
            self.ui.showDarkFrameHotpixelsCheckBox.setEnabled(True)
        if hasattr(self.ui, 'showPredictedRandomHotpixelsCheckBox'):
            self.ui.showPredictedRandomHotpixelsCheckBox.setEnabled(True)

        # Re-enable buttons
        self.check_ready_state()
        
        self.showStatusMessage(
            f"Successfully corrected {len(corrected_paths)} image{'s' if len(corrected_paths) > 1 else ''}")

    def statusBar(self):
        """Get the status bar from the parent main window"""
        main_window = self.window()  # Get the top-level window
        if hasattr(main_window, 'statusBar'):
            return main_window.statusBar()
        return None

    def showStatusMessage(self, message: str, timeout: int = 5000):
        """Show a message in the status bar with optional timeout"""
        status_bar = self.statusBar()
        if status_bar:
            status_bar.showMessage(message, timeout)

    def correction_error(self, error_message: str):
        """Handle correction error"""
        QMessageBox.critical(self, "Correction Error", f"Batch correction failed:\n{error_message}")
        print(f"Correction error: {error_message}")

        # Re-enable button and clean up worker
        self.check_ready_state()
        self.worker = None
    
    def save_training_data(self):
        """Save training data from the corrected images"""
        # Check if we have the necessary data
        if not self.app.current_profile:
            QMessageBox.warning(self, "No Profile", "Please load a profile first.")
            return
        
        if not self.image_paths or not self.rawImages:
            QMessageBox.warning(self, "No Images", "Please load images first.")
            return
        
        if not self.corrected_image_paths or not len(self.corrected_image_paths):
            QMessageBox.warning(self, "No Corrected Images", 
                              "Please correct images first before saving training data.")
            return
        
        # Ensure all images are loaded (not just first one from lazy loading)
        if len([img for img in self.rawImages if img is not None]) < len(self.image_paths):
            QMessageBox.warning(self, "Images Not Loaded", 
                              "Not all images are loaded. Please run correction first to load all images.")
            return
        
        # Process each image pair
        self.showStatusMessage("Capturing training data...", 0)
        
        # Disable button during processing
        self.ui.saveTrainingDataButton.setEnabled(False)
        self.ui.saveTrainingDataButton.setText("Saving...")
        
        total_samples = 0
        
        try:
            for i, (uncorrected_path, corrected_path) in enumerate(zip(self.image_paths, self.corrected_image_paths)):
                self.showStatusMessage(f"Capturing training data from image {i+1}/{len(self.image_paths)}...", 0)
                
                # Load uncorrected image (raw)
                uncorrected_image = DNGImage(uncorrected_path, process_rgb=False)
                
                # Load corrected image (raw)
                corrected_image = DNGImage(corrected_path, process_rgb=False)
                
                # Start worker for this image pair
                self.training_worker = TrainingDataWorker(
                    self.app,
                    uncorrected_image,
                    corrected_image,
                    self.app.current_profile,
                    difference_threshold=255*25
                )
                
                self.training_worker.finished.connect(lambda samples: self.on_training_data_saved(samples, i, len(self.image_paths)))
                self.training_worker.error.connect(self.training_data_error)
                
                # Run synchronously for now (we could make this async for each image)
                self.training_worker.run()
                
            self.showStatusMessage("Training data saved successfully!", 5000)
            
            # Re-enable button
            self.ui.saveTrainingDataButton.setEnabled(True)
            self.ui.saveTrainingDataButton.setText("Save Training Data")
            
        except Exception as e:
            self.training_data_error(str(e))
    
    def on_training_data_saved(self, samples_captured: int, image_index: int, total_images: int):
        """Handle successful training data capture"""
        print(f"Captured {samples_captured} training samples from image {image_index+1}/{total_images}")
    
    def training_data_error(self, error_message: str):
        """Handle training data capture error"""
        QMessageBox.critical(self, "Training Data Error", 
                           f"Failed to capture training data:\n{error_message}")
        print(f"Training data error: {error_message}")
        
        # Re-enable button
        self.ui.saveTrainingDataButton.setEnabled(True)
        self.ui.saveTrainingDataButton.setText("Save Training Data")
        self.showStatusMessage("Training data capture failed", 5000)

    def load_profile(self):
        # Get the last used directory from preferences
        start_dir = ""
        if self.main_window and hasattr(self.main_window, 'preferences'):
            last_dir = self.main_window.preferences.get_last_directory("profile")
            if last_dir:
                start_dir = last_dir

        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Load Hot Pixel Profile",
            start_dir,
            "JSON Files (*.json);;All Files (*)"
        )

        if filename:
            if self.main_window:
                # Use consolidated profile loading
                self.main_window.load_profile_file(filename)
            else:
                # Fallback for standalone operation
                try:
                    profile = HotPixelProfile.load_from_file(filename)
                    self.app.current_profile = profile
                    self.update_profile_info()
                    self.check_ready_state()
                except Exception as e:
                    QMessageBox.critical(self, "Load Error", f"Failed to load profile:\n{str(e)}")

    def select_image(self):
        # Get the last used directory from preferences
        start_dir = ""
        if self.main_window and hasattr(self.main_window, 'preferences'):
            last_dir = self.main_window.preferences.get_last_directory("image")
            if last_dir:
                start_dir = last_dir

        filenames, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Images to Correct",
            start_dir,
            "DNG Files (*.dng);;All Files (*)"
        )

        if filenames:
            self.load_images(filenames)

    def load_images(self, filenames: List[str]):
        """Load multiple images programmatically with lazy loading optimization"""
        if not filenames:
            return

        # Save the directory to preferences
        if self.main_window and hasattr(self.main_window, 'preferences'):
            self.main_window.preferences.update_last_directory(filenames[0], "image")

        # Start multi-image loading in background thread with lazy loading (only first image)
        self.image_loading_worker = MultiImageLoadingWorker(filenames, load_all=False)
        self.image_loading_worker.progress.connect(self.update_image_loading_progress)
        self.image_loading_worker.finished.connect(self.multi_image_loading_finished)
        self.image_loading_worker.error.connect(self.image_loading_error)

        # Disable the open image button during loading
        self.ui.openOriginalButton.setEnabled(False)
        self.ui.openOriginalButton.setText("Loading...")

        self.image_loading_worker.start()

    def update_image_loading_progress(self, message: str):
        """Update UI with image loading progress"""
        # You could update a status label here if needed
        pass

    def multi_image_loading_finished(self, image_paths: List[str], rgb_images: List[DNGImage], raw_images: List[DNGImage]):
        """Handle successful multiple image loading completion"""
        # Set the loaded images
        self.image_paths = image_paths
        self.rgbImages = rgb_images
        self.rawImages = raw_images
        
        # Clear any previously cached data since we have new input images
        self.corrected_image_paths = []
        self.corrected_rgb_images = []
        self.model_hot_pixels = []
        self.cached_original_raw = None
        self.cached_cnn_detections = []
        self.cached_preview_image = None
        
        # Disable and uncheck the "Show corrected image" checkbox since we don't have corrected images
        if hasattr(self.ui, 'showCorrectedImageCheckBox'):
            self.ui.showCorrectedImageCheckBox.setEnabled(False)
            self.ui.showCorrectedImageCheckBox.setChecked(False)
            
        # Disable and uncheck hot pixel visualization checkboxes
        if hasattr(self.ui, 'showDarkFrameHotpixelsCheckBox'):
            self.ui.showDarkFrameHotpixelsCheckBox.setEnabled(False)
            self.ui.showDarkFrameHotpixelsCheckBox.setChecked(False)
        if hasattr(self.ui, 'showPredictedRandomHotpixelsCheckBox'):
            self.ui.showPredictedRandomHotpixelsCheckBox.setEnabled(False)
            self.ui.showPredictedRandomHotpixelsCheckBox.setChecked(False)

        # Update UI
        self.update_image_list_display()
        self.update_image_summary()
        self.display_original_image()
        self.check_ready_state()

        # Re-enable the open image button
        self.ui.openOriginalButton.setEnabled(True)
        self.ui.openOriginalButton.setText("Open Images")
        
        # Store flag to check if profile was auto-loaded
        profile_was_auto_loaded = False
        
        # Recommend matching profile if no profile is currently loaded
        if not self.app.current_profile:
            profile_was_auto_loaded = self.recommend_matching_profile()

        # Show status message about lazy loading
        if len(image_paths) > 1:
            loaded_count = len([img for img in rgb_images if img is not None])
            if loaded_count == 1:
                self.showStatusMessage(
                    f"Loaded first image for preview ({len(image_paths)} total selected). Additional images will be loaded during correction.", 8000)
            else:
                self.showStatusMessage(f"Loaded all {loaded_count} images successfully!", 3000)
        else:
            self.showStatusMessage("Image loaded successfully!", 3000)
            
            # Auto-run preview for single image if profile was loaded
            if profile_was_auto_loaded and self.app.current_profile and self.is_interactive_mode():
                self._start_interactive_preview()

        # Clean up worker
        self.image_loading_worker = None

    def image_loading_error(self, error_message: str):
        """Handle image loading error"""
        QMessageBox.critical(self, "Load Error", f"Failed to load image:\n{error_message}")

        # Re-enable the open image button
        self.ui.openOriginalButton.setEnabled(True)
        self.ui.openOriginalButton.setText("Open Image")

        # Clean up worker
        self.image_loading_worker = None

    def update_image_list_display(self):
        """Update the UI to show the list of loaded images with lazy loading status"""
        # Update the input files list widget
        if hasattr(self.ui, 'inputFilesListWidget'):
            self.ui.inputFilesListWidget.clear()

            if not self.image_paths:
                # Add placeholder text when no files are loaded
                item = ListItem("No images loaded. Click 'Open Images' to select files.")
                item.setFlags(item.flags() & ~Qt.ItemIsSelectable)  # Make non-selectable
                self.ui.inputFilesListWidget.addItem(item)
            else:
                # Add each image file path with loading status
                for i, image_path in enumerate(self.image_paths, 1):
                    filename = os.path.basename(image_path)

                    # Check if this image is loaded or just a placeholder
                    if (self.rgbImages and i <= len(self.rgbImages) and
                            self.rgbImages[i-1] is not None):
                        status = "✓ Loaded"
                        display_text = f"{i}. {filename} ({status})"
                    else:
                        status = "⏳ Will load during correction"
                        display_text = f"{i}. {filename} ({status})"

                    self.ui.inputFilesListWidget.addItem(display_text)

    def display_original_image(self):
        """Display the first loaded original or corrected image in the UI using ImageGraphicsView"""
        # Check if we should show corrected image
        show_corrected = (hasattr(self.ui, 'showCorrectedImageCheckBox') and 
                         self.ui.showCorrectedImageCheckBox.isChecked())
        
        # Check if we should show RGB version
        show_rgb = (hasattr(self.ui, 'showRgbImageCheckBox') and 
                   self.ui.showRgbImageCheckBox.isChecked())
        
        if show_corrected:
            # In interactive mode, use cached preview if available
            if self.cached_preview_image:
                try:
                    if show_rgb:
                        # Load RGB version from cached preview
                        with tempfile.NamedTemporaryFile(suffix='.dng', delete=False) as tmp:
                            tmp_path = tmp.name
                        
                        # Save the corrected raw to temp file
                        self.cached_preview_image.save(tmp_path)
                        
                        # Load as RGB
                        rgb_image = DNGImage(tmp_path, process_rgb=True)
                        rgb_image.white_balance()
                        rgb_data = rgb_image.get_data()
                        
                        # Clean up temp file
                        try:
                            os.unlink(tmp_path)
                        except:
                            pass
                        
                        pixmap = self.numpy_to_qpixmap(rgb_data)
                    else:
                        # Display the cached corrected raw image (grayscale)
                        raw_data = self.cached_preview_image.get_data()
                        pixmap = self.numpy_to_qpixmap(raw_data)
                    
                    if hasattr(self, 'image_view'):
                        # Preserve current pan/zoom state
                        self.image_view.setPixmap(pixmap, preserve_view=True)
                    return
                except Exception as e:
                    print(f"Error displaying cached preview: {e}")
                    tb.print_exc()
                    # Fall back to original image
            
            # Otherwise use batch-corrected images if available
            elif self.corrected_image_paths and len(self.corrected_image_paths) > 0:
                try:
                    if show_rgb:
                        # Load RGB version from corrected file
                        rgb_image = DNGImage(self.corrected_image_paths[0], process_rgb=True)
                        rgb_image.white_balance()
                        rgb_data = rgb_image.get_data()
                        pixmap = self.numpy_to_qpixmap(rgb_data)
                    else:
                        # Load raw grayscale version
                        raw_image = DNGImage(self.corrected_image_paths[0], process_rgb=False)
                        raw_data = raw_image.get_data()
                        pixmap = self.numpy_to_qpixmap(raw_data)
                    
                    if hasattr(self, 'image_view'):
                        # Preserve view for batch mode too
                        self.image_view.setPixmap(pixmap, preserve_view=True)
                    return
                except Exception as e:
                    print(f"Error displaying corrected image: {e}")
                    # Fall back to original image
        
        # Display original image (default behavior)
        if not self.rawImages or self.rawImages[0] is None:
            # Show a message when no image is loaded
            if hasattr(self, 'image_view'):
                self.image_view.showMessage("No images loaded")
            return

        try:
            if show_rgb:
                # Use RGB version if checkbox is checked
                if self.rgbImages and len(self.rgbImages) > 0:
                    # Load RGB image on-demand if not already loaded
                    if self.rgbImages[0] is None and self.image_paths:
                        print("Loading RGB image on-demand...")
                        rgb_image = DNGImage(self.image_paths[0], process_rgb=True)
                        rgb_image.white_balance()
                        self.rgbImages[0] = rgb_image
                    
                    if self.rgbImages[0] is not None:
                        rgb_data = self.rgbImages[0].get_data()
                        pixmap = self.numpy_to_qpixmap(rgb_data)
                    else:
                        # Fall back to raw if RGB loading failed
                        raw_data = self.rawImages[0].get_data()
                        pixmap = self.numpy_to_qpixmap(raw_data)
                else:
                    # Fall back to raw if RGB not available
                    raw_data = self.rawImages[0].get_data()
                    pixmap = self.numpy_to_qpixmap(raw_data)
            else:
                # Use raw grayscale version by default
                raw_data = self.rawImages[0].get_data()
                pixmap = self.numpy_to_qpixmap(raw_data)

            # Set the pixmap in the ImageGraphicsView and preserve pan/zoom state
            if hasattr(self, 'image_view'):
                self.image_view.setPixmap(pixmap, preserve_view=True)

        except Exception as e:
            # Clear the image view on error
            if hasattr(self, 'image_view'):
                self.image_view.clear()
            QMessageBox.critical(self, "Display Error", f"Failed to display image:\n{str(e)}")

    def numpy_to_qpixmap(self, img_array, brightness_factor=5.0, fixed_max=None):
        """Convert numpy array to QPixmap with brightness adjustment"""
        if len(img_array.shape) == 3:
            # RGB image
            # Normalize and apply brightness
            img_normalized = img_array.astype(np.float32)
            norm_max = fixed_max if fixed_max is not None else img_normalized.max()
            img_normalized = img_normalized / norm_max * 255 * brightness_factor
            img_normalized = np.clip(img_normalized, 0, 255).astype(np.uint8)
            
            height, width, channels = img_normalized.shape
            bytes_per_line = channels * width
            
            q_image = QImage(
                img_normalized.data,
                width,
                height,
                bytes_per_line,
                QImage.Format_RGB888
            )
        else:
            # Grayscale image
            img_normalized = img_array.astype(np.float32)
            norm_max = fixed_max if fixed_max is not None else img_normalized.max()
            img_normalized = img_normalized / norm_max * 255 * brightness_factor
            img_normalized = np.clip(img_normalized, 0, 255).astype(np.uint8)
            
            height, width = img_normalized.shape
            
            q_image = QImage(
                img_normalized.data,
                width,
                height,
                width,
                QImage.Format_Grayscale8
            )
            
        return QPixmap.fromImage(q_image)

    def check_ready_state(self):
        # Check if we have profile and at least some images (first one loaded for preview)
        ready = bool(self.app.current_profile and self.image_paths and self.rgbImages and self.rgbImages[0] is not None)
        
        if self.is_interactive_mode():
            # Interactive mode: Enable Save button if we have a preview, disable Batch Process
            save_ready = ready and self.cached_preview_image is not None
            self.ui.saveButton.setEnabled(save_ready)
            self.ui.saveButton.setText("Save")
            self.ui.batchProcessButton.setEnabled(False)
            self.ui.batchProcessButton.setText("Batch Process")
        else:
            # Batch mode: Enable Batch Process button, disable Save
            self.ui.saveButton.setEnabled(False)
            self.ui.saveButton.setText("Save")
            self.ui.batchProcessButton.setEnabled(ready)
            if ready:
                button_text = f"Batch Process ({len(self.image_paths)} images)"
            else:
                button_text = "Batch Process"
            self.ui.batchProcessButton.setText(button_text)
    
    def _copy_dng_image(self, dng_image: DNGImage) -> DNGImage:
        """Create a shallow copy of a DNGImage for preview modifications.
        
        This creates a new DNGImage instance that shares the same underlying
        DNG file handle but has an independent copy of the raw image data.
        This avoids re-reading the file from disk.
        """
        # Create new instance without going through full __init__
        copied_image = object.__new__(DNGImage)
        
        # Copy Image base class attributes
        copied_image.filename = dng_image.filename
        copied_image._exifread_tags = dng_image._exifread_tags  # Share lazy-loaded tags
        copied_image.sensor_temperature = dng_image.sensor_temperature
        copied_image.unique_id = dng_image.unique_id
        copied_image._exiftool_metadata = dng_image._exiftool_metadata  # Share lazy-loaded metadata
        
        # Share the DNG file handle (read-only operations)
        copied_image.dng = dng_image.dng
        copied_image.rgb = dng_image.rgb
        
        # Deep copy only the raw image data (what we actually modify)
        copied_image.raw_img = dng_image.raw_img.copy()
        
        return copied_image
    
    def recompute_preview(self):
        """Recompute the corrected preview based on current parameters and enabled corrections.
        
        This method is called when parameters change or corrections are toggled in interactive mode.
        It rebuilds the preview from cached data without saving to disk.
        """
        if not self.cached_original_raw or not self.app.current_profile:
            return
        
        # Start with a fresh copy of the original image
        preview_image = self._copy_dng_image(self.cached_original_raw)
        
        # Apply corrections in sequence based on enabled flags
        
        # 1. Dark frame subtraction
        if self.subtract_noise_enabled:
            self.app.subtract_dark_frame(preview_image, self.app.current_profile)
        
        # 2. Correct profile hot pixels with current deviation threshold
        if self.correct_profile_hotpixels_enabled:
            corrected_pixels = self.app.correct_hot_pixels(preview_image, self.app.current_profile, 
                                       deviation_threshold=self.current_deviation_threshold)
            self.profile_hotpixel_count = len(corrected_pixels)
        else:
            self.profile_hotpixel_count = 0
        
        # 3. Correct CNN-detected hot pixels with current sensitivity
        if self.correct_cnn_hotpixels_enabled and self.cached_cnn_detections:
            # Calculate how many pixels will be corrected at current sensitivity
            confidence_threshold = 1.0 - self.current_sensitivity
            filtered_detections = [(y, x, conf) for y, x, conf in self.cached_cnn_detections 
                                  if conf >= confidence_threshold]
            self.cnn_hotpixel_count = len(filtered_detections)
            
            self.app.apply_cnn_corrections(preview_image, self.cached_cnn_detections, 
                                          sensitivity=self.current_sensitivity)
        else:
            self.cnn_hotpixel_count = 0
        
        # Cache the preview image
        self.cached_preview_image = preview_image
        
        # For display purposes, we'll show the corrected raw image directly
        # Store it in corrected_rgb_images as a placeholder (we'll handle display differently)
        self.corrected_image_paths = [preview_image.filename]  # Use original path as reference
        
        # Update the display - display_original_image will check showCorrectedImageCheckBox
        # and use cached_preview_image if available
        self.display_original_image()
        
        # Update stats if available
        self.update_correction_stats()
    
    def update_correction_stats(self):
        """Update UI with correction statistics."""
        if not self.app.current_profile:
            return
        
        # Count hot pixels based on current settings
        profile_hot_pixels = []
        if self.correct_profile_hotpixels_enabled:
            profile_hot_pixels = self.app.get_hot_pixels(self.app.current_profile, 
                                                         deviation_threshold=self.current_deviation_threshold)
        
        cnn_hot_pixels = []
        if self.correct_cnn_hotpixels_enabled and self.cached_cnn_detections:
            confidence_threshold = 1.0 - self.current_sensitivity
            cnn_hot_pixels = [(y, x) for y, x, conf in self.cached_cnn_detections 
                             if conf >= confidence_threshold]
        
        # Update results text if available
        if hasattr(self.ui, 'resultsTextEdit'):
            result_text = "Correction Preview:\n"
            if self.subtract_noise_enabled:
                result_text += "✓ Dark frame subtraction enabled\n"
            else:
                result_text += "✗ Dark frame subtraction disabled\n"
            
            if self.correct_profile_hotpixels_enabled:
                result_text += f"✓ Profile hot pixels: {len(profile_hot_pixels)} (threshold: {self.current_deviation_threshold:.1f}σ)\n"
            else:
                result_text += "✗ Profile hot pixels disabled\n"
            
            if self.correct_cnn_hotpixels_enabled:
                result_text += f"✓ CNN hot pixels: {len(cnn_hot_pixels)} (sensitivity: {self.current_sensitivity:.2f})\n"
            else:
                result_text += "✗ CNN hot pixels disabled\n"
            
            self.ui.resultsTextEdit.setText(result_text)
    
    def on_deviation_threshold_slider_moved(self, value: int):
        """Handle deviation threshold slider movement (debounced)."""
        # Restart the debounce timer
        if self.deviation_threshold_timer:
            self.deviation_threshold_timer.stop()
            self.deviation_threshold_timer.start(200)  # 200ms debounce
    
    def on_deviation_threshold_changed(self):
        """Handle deviation threshold slider change after debounce."""
        # Get current slider value
        value = self.ui.horizontalSlider_2.value()
        
        # Convert slider value (10-250) to actual threshold (1.0-25.0)
        self.current_deviation_threshold = value / 10.0
        
        # Recompute preview if in interactive mode (this will update counts)
        if self.cached_original_raw:
            self.recompute_preview()
        
        # Update label with threshold and count
        if hasattr(self.ui, 'label_2'):
            self.ui.label_2.setText(f"Deviation Threshold: {self.current_deviation_threshold:.1f}σ ({self.profile_hotpixel_count} pixels)")
    
    def on_sensitivity_slider_moved(self, value: int):
        """Handle CNN sensitivity slider movement (debounced)."""
        # Restart the debounce timer
        if self.sensitivity_timer:
            self.sensitivity_timer.stop()
            self.sensitivity_timer.start(200)  # 200ms debounce
    
    def on_sensitivity_changed(self):
        """Handle CNN sensitivity slider change after debounce."""
        # Get current slider value
        value = self.ui.horizontalSlider.value()
        
        # Convert slider value (1-99) to actual sensitivity (0.01-0.99)
        self.current_sensitivity = value / 100.0
        
        # Recompute preview if in interactive mode (this will update counts)
        if self.cached_original_raw:
            self.recompute_preview()
        
        # Update label with sensitivity and count
        if hasattr(self.ui, 'label'):
            self.ui.label.setText(f"Sensitivity: {self.current_sensitivity:.2f} ({self.cnn_hotpixel_count} pixels)")
    
    def on_subtract_noise_toggled(self, checked: bool):
        """Handle dark frame subtraction checkbox toggle."""
        self.subtract_noise_enabled = checked
        
        # Recompute preview if in interactive mode
        if self.cached_original_raw:
            self.recompute_preview()
        else:
            self.check_ready_state()
    
    def on_remove_hotpixels_toggled(self, checked: bool):
        """Handle remove hot pixels checkbox toggle."""
        self.correct_profile_hotpixels_enabled = checked
        
        # Recompute preview if in interactive mode
        if self.cached_original_raw:
            self.recompute_preview()
        else:
            self.check_ready_state()
    
    def on_apply_cnn_toggled(self, checked: bool):
        """Handle CNN correction checkbox toggle."""
        self.correct_cnn_hotpixels_enabled = checked
        
        # Recompute preview if in interactive mode
        if self.cached_original_raw:
            self.recompute_preview()
        else:
            self.check_ready_state()

    def update_profile_info(self):
        styled_text = format_profile_summary(self.app.current_profile, "No profile loaded. Open a profile to see information.")
        self.ui.statisticsLabel.setText(styled_text)

    def update_image_summary(self):
        """Update the image summary display with current image information"""
        # Pass already-loaded image to avoid re-reading from disk
        first_image = self.rawImages[0] if self.rawImages and self.rawImages[0] else None
        styled_text = format_image_summary(self.image_paths, self.app.current_profile, first_image)
        if hasattr(self.ui, 'imageSummaryLabel'):
            self.ui.imageSummaryLabel.setText(styled_text)

    def open_corrected_image(self):
        if hasattr(self, 'corrected_image_path'):
            os.startfile(self.corrected_image_path)

    def on_dark_frame_hotpixels_toggled(self, checked: bool):
        """Handle Dark Frame Hotpixels checkbox toggle"""
        print(f"Dark Frame Hotpixels checkbox toggled: {checked}")
        self.update_hot_pixel_overlays()

    def on_dark_frame_subtraction_toggled(self, checked: bool):
        """Handle Dark Frame Subtraction checkbox toggle"""
        print(f"Dark Frame Subtraction checkbox toggled: {checked}")

    def on_predicted_random_hotpixels_toggled(self, checked: bool):
        """Handle Predicted Random Hotpixels checkbox toggle"""
        print(f"Predicted Random Hotpixels checkbox toggled: {checked}")
        
    def on_show_corrected_image_toggled(self, checked: bool):
        """Handle Show Corrected Image checkbox toggle"""
        print(f"Show Corrected Image checkbox toggled: {checked}")
        # Update the displayed image based on the checkbox state
        self.display_original_image()
    
    def on_show_rgb_image_toggled(self, checked: bool):
        """Handle Show RGB Image checkbox toggle"""
        print(f"Show RGB Image checkbox toggled: {checked}")
        # Update the displayed image based on the checkbox state
        self.display_original_image()

    def on_image_mouse_moved(self, x: int, y: int):
        """Handle mouse movement over the image - extract and display ROIs"""
        if not self.rgbImages or self.rgbImages[0] is None:
            return
            
        # Update position display
        if hasattr(self.ui, 'roiPositionLabel'):
            self.ui.roiPositionLabel.setText(f"Position: {x}, {y}")
        
        # Extract and display original ROI
        self.extract_and_display_roi(x, y)

    def extract_and_display_roi(self, center_x: int, center_y: int):
        """Extract 32x32 ROI around the given center point and display 4x scaled versions"""
        try:
            original_max = None
            
            # Extract ROI from original raw image (first loaded image)
            if self.rawImages and self.rawImages[0] is not None:
                original_raw_data = self.rawImages[0].get_data()
                original_roi = self.extract_roi_from_image(original_raw_data, center_x, center_y)
                
                # Scale up by 4x using nearest neighbor
                original_roi_scaled = cv2.resize(original_roi, (128, 128), interpolation=cv2.INTER_NEAREST)
                
                # Store the max from the full image for consistent normalization
                original_max = original_raw_data.max()
                
                # Convert to QPixmap and display with fixed max
                original_pixmap = self.numpy_to_qpixmap(original_roi_scaled, brightness_factor=5.0, fixed_max=original_max)
                if hasattr(self.ui, 'originalRoiLabel'):
                    self.ui.originalRoiLabel.setPixmap(original_pixmap)
            
            # Extract ROI from corrected image if available (use cached preview or batch corrected)
            corrected_data = None
            
            # In interactive mode, use cached preview if available
            if self.cached_preview_image:
                corrected_data = self.cached_preview_image.get_data()
            # Otherwise use batch-corrected image if available
            elif self.corrected_rgb_images and len(self.corrected_rgb_images) > 0:
                # Load the corrected image as raw (not RGB) for ROI comparison
                if self.corrected_image_paths and len(self.corrected_image_paths) > 0:
                    try:
                        corrected_raw = DNGImage(self.corrected_image_paths[0], process_rgb=False)
                        corrected_data = corrected_raw.get_data()
                    except Exception as e:
                        print(f"Error loading corrected raw image: {e}")
            
            if corrected_data is not None and original_max is not None:
                try:
                    corrected_roi = self.extract_roi_from_image(corrected_data, center_x, center_y)
                    
                    # Scale up by 4x using nearest neighbor
                    corrected_roi_scaled = cv2.resize(corrected_roi, (128, 128), interpolation=cv2.INTER_NEAREST)
                    
                    # Convert to QPixmap and display (use same max as original for comparable brightness)
                    corrected_pixmap = self.numpy_to_qpixmap(corrected_roi_scaled, brightness_factor=5.0, fixed_max=original_max)
                    if hasattr(self.ui, 'correctedRoiLabel'):
                        self.ui.correctedRoiLabel.setPixmap(corrected_pixmap)
                        
                except Exception as e:
                    print(f"Error extracting corrected image ROI: {e}")
                    tb.print_exc()
                    if hasattr(self.ui, 'correctedRoiLabel'):
                        self.ui.correctedRoiLabel.setText("Error extracting\ncorrected ROI")
            else:
                # No corrected image available
                if hasattr(self.ui, 'correctedRoiLabel'):
                    self.ui.correctedRoiLabel.setText("No corrected\nimage available")
                    
        except Exception as e:
            print(f"Error extracting ROI: {e}")
            tb.print_exc()

    def extract_roi_from_image(self, img_data: np.ndarray, center_x: int, center_y: int, roi_size: int = 32) -> np.ndarray:
        """Extract ROI from image data around the given center point"""
        height, width = img_data.shape[:2]
        half_size = roi_size // 2
        
        # Calculate ROI bounds
        x1 = max(0, center_x - half_size)
        y1 = max(0, center_y - half_size)
        x2 = min(width, center_x + half_size)
        y2 = min(height, center_y + half_size)
        
        # Extract ROI
        if len(img_data.shape) == 3:
            roi = img_data[y1:y2, x1:x2, :]
        else:
            roi = img_data[y1:y2, x1:x2]
        
        # Pad if ROI is smaller than expected (near edges)
        if len(img_data.shape) == 3:
            if roi.shape[:2] != (roi_size, roi_size):
                padded_roi = np.zeros((roi_size, roi_size, img_data.shape[2]), dtype=roi.dtype)
                roi_h, roi_w = roi.shape[:2]
                start_y = (roi_size - roi_h) // 2
                start_x = (roi_size - roi_w) // 2
                padded_roi[start_y:start_y+roi_h, start_x:start_x+roi_w, :] = roi
                roi = padded_roi
        else:
            if roi.shape != (roi_size, roi_size):
                padded_roi = np.zeros((roi_size, roi_size), dtype=roi.dtype)
                roi_h, roi_w = roi.shape
                start_y = (roi_size - roi_h) // 2
                start_x = (roi_size - roi_w) // 2
                padded_roi[start_y:start_y+roi_h, start_x:start_x+roi_w] = roi
                roi = padded_roi
        
        return roi
    
    def recommend_matching_profile(self) -> bool:
        """Recommend a matching profile from the library for the loaded image."""
        # Only recommend if we have images and no profile is loaded
        if not self.rawImages or not self.rawImages[0]:
            return False
        
        # Get the first image
        first_image = self.rawImages[0]
        
        # Find matching profile
        matching_profile_path = self.app.find_matching_profile(first_image)
        
        if matching_profile_path:
            # Automatically load the matching profile
            if self.main_window:
                profile_name = Path(matching_profile_path).name
                self.main_window.load_profile_file(matching_profile_path, show_success_message=True)
                print(f"Automatically loaded matching profile: {profile_name}")
                return True
        else:
            # Show warning that no matching profile was found
            QMessageBox.warning(
                self,
                "No Matching Profile",
                "No matching profile was found for this image.\n\n"
                "Hot pixel correction will not be available until you load or create a profile.",
                QMessageBox.Ok
            )
            return False
