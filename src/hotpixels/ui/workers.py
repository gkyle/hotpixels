"""Background worker threads for async operations."""

from typing import List
from pathlib import Path
import time

from PySide6.QtCore import QThread, Signal

from hotpixels.app import App
from hotpixels.profile import HotPixelProfile
from hotpixels.image import DNGImage


class AnalysisWorker(QThread):
    """Background worker for hot pixel analysis to keep GUI responsive"""
    
    progress = Signal(str)  # Progress message
    finished = Signal(object)  # Analysis result
    error = Signal(str)  # Error message
    
    def __init__(self, app: App, filenames: List[str], deviation_threshold: float):
        super().__init__()
        self.app = app
        self.filenames = filenames
        self.deviation_threshold = deviation_threshold
    
    def run(self):
        try:
            self.progress.emit("Starting analysis...")
            profile = self.app.create_hot_pixel_profile(
                self.filenames, 
                self.deviation_threshold
            )
            self.progress.emit("Analysis complete.")
            self.finished.emit(profile)
        except Exception as e:
            self.error.emit(str(e))


class CorrectionWorker(QThread):
    """Background worker for hot pixel correction"""
    
    progress = Signal(str)
    finished = Signal(list, list)  # List of corrected image paths, list of model hot pixel lists
    error = Signal(str)
    
    def __init__(self, app: App, image_paths: List[str], profile: HotPixelProfile, 
                 subtract_noise_profile: bool = False, apply_residual_hotpixel_model: bool = False,
                 cnn_sensitivity: float = 0.9):
        super().__init__()
        self.app = app
        self.image_paths = image_paths
        self.profile = profile
        self.subtract_noise_profile = subtract_noise_profile
        self.apply_residual_hotpixel_model = apply_residual_hotpixel_model
        self.cnn_sensitivity = cnn_sensitivity

    def run(self):
        try:
            corrected_paths = []
            model_hot_pixels = []
            
            for i, image_path in enumerate(self.image_paths):
                self.progress.emit(f"Processing {i+1}/{len(self.image_paths)}...")
                dngImage = DNGImage(image_path)
                
                if self.subtract_noise_profile:
                    self.app.subtract_dark_frame(dngImage, self.profile)
                
                self.app.correct_hot_pixels(dngImage, self.profile)

                if self.apply_residual_hotpixel_model:
                    # Reuse CNN detection logic (call directly since we're already in a background thread)
                    self.progress.emit(f"Running CNN detection on image {i+1}/{len(self.image_paths)}...")
                    detections = self.app.detect_residual_hotpixels_cnn(dngImage)
                    self.app.apply_cnn_corrections(dngImage, detections, sensitivity=self.cnn_sensitivity)
                    model_hot_pixels.append(detections)
                
                corrected_path = self.app.save_corrected_image(dngImage, self.subtract_noise_profile, self.apply_residual_hotpixel_model)
                corrected_paths.append(corrected_path)
            
            self.progress.emit("All corrections complete.")
            self.finished.emit(corrected_paths, model_hot_pixels)
        except Exception as e:
            self.error.emit(str(e))


class MultiImageLoadingWorker(QThread):
    """Background worker for loading multiple DNG images with lazy loading optimization"""
    
    progress = Signal(str)
    finished = Signal(list, list, list)  # Image paths, RGB images, Raw images
    error = Signal(str)
    
    def __init__(self, image_paths: List[str], load_all: bool = False):
        super().__init__()
        self.image_paths = image_paths
        self.load_all = load_all  # If False, only load first image for preview
    
    def run(self):
        try:
            rgb_images = []
            raw_images = []
            
            if self.load_all:
                # Load all images (used during correction)
                for i, image_path in enumerate(self.image_paths):
                    self.progress.emit(f"Loading image {i+1}/{len(self.image_paths)} for correction...")
                    rgb_image = DNGImage(image_path, process_rgb=True)
                    rgb_images.append(rgb_image)
                    
                    raw_image = DNGImage(image_path, process_rgb=False)
                    raw_images.append(raw_image)
                
                self.progress.emit("All images loaded successfully.")
            else:
                # Only load first image for preview (startup/open images optimization)
                if self.image_paths:
                    first_path = self.image_paths[0]
                    self.progress.emit("Loading first image for preview...")
                    
                    rgb_image = DNGImage(first_path, process_rgb=True)
                    rgb_image.white_balance()
                    rgb_images.append(rgb_image)
                    
                    raw_image = DNGImage(first_path, process_rgb=False)
                    raw_images.append(raw_image)
                    
                    # Add placeholders for remaining images (will be loaded on demand)
                    for i in range(1, len(self.image_paths)):
                        rgb_images.append(None)
                        raw_images.append(None)
                
                self.progress.emit(f"First image loaded successfully. ({len(self.image_paths)} total selected)")
            
            self.finished.emit(self.image_paths, rgb_images, raw_images)
        except Exception as e:
            self.error.emit(str(e))


class ProfileLoadingWorker(QThread):
    """Worker to load profile metadata asynchronously"""
    finished = Signal(list)  # Emits list of profile metadata
    progress = Signal(str)  # Progress messages
    error = Signal(str)  # Error messages

    def __init__(self, profiles_directory: Path):
        super().__init__()
        self.profiles_directory = profiles_directory

    def run(self):
        """Load all profile metadata from the profiles directory"""
        try:
            profile_data = []

            # Create profiles directory if it doesn't exist
            self.profiles_directory.mkdir(exist_ok=True)

            # Scan for .json profile files
            profile_files = list(self.profiles_directory.glob("*.json"))
            total_files = len(profile_files)

            self.progress.emit(f"Loading {total_files} profiles...")

            for i, profile_file in enumerate(profile_files):
                try:
                    start_time = time.time()
                    profile = HotPixelProfile.load_from_file(str(profile_file))

                    # Extract metadata for table display
                    metadata = {
                        'file_path': str(profile_file),
                        'file_name': profile_file.name,
                        'camera_id': profile.camera_metadata.camera_id if profile.camera_metadata else "Unknown",
                        'make': profile.camera_metadata.camera_make if profile.camera_metadata else "Unknown",
                        'model': profile.camera_metadata.camera_model if profile.camera_metadata else "Unknown",
                        'camera_uid': profile.camera_metadata.camera_uid if profile.camera_metadata and profile.camera_metadata.camera_uid else "N/A",
                        'shutter_speed': profile.camera_metadata.shutter_speed if profile.camera_metadata else "Unknown",
                        'iso': profile.camera_metadata.iso if profile.camera_metadata else "Unknown",
                        'sensor_temperature': profile.camera_metadata.sensor_temperature if profile.camera_metadata and profile.camera_metadata.sensor_temperature is not None else "N/A",
                        'date_created': profile.camera_metadata.date_created if profile.camera_metadata else "Unknown",
                        'profile': profile  # Keep reference to full profile for summary display
                    }

                    profile_data.append(metadata)

                except Exception as e:
                    print(f"Failed to load profile {profile_file}: {e}")
                    # Add a placeholder entry for corrupted profiles
                    metadata = {
                        'file_path': str(profile_file),
                        'file_name': profile_file.name,
                        'camera_id': "Error loading",
                        'make': "Error",
                        'model': "Error",
                        'camera_uid': "Error",
                        'shutter_speed': "Error",
                        'iso': "Error",
                        'sensor_temperature': "Error",
                        'date_created': "Error",
                        'profile': None
                    }
                    profile_data.append(metadata)

                # Update progress
                if total_files > 0:
                    progress_msg = f"Loaded {i+1}/{total_files} profiles..."
                    self.progress.emit(progress_msg)

            self.finished.emit(profile_data)

        except Exception as e:
            self.error.emit(str(e))


class CNNDetectionWorker(QThread):
    """Worker for running CNN hot pixel detection in background"""
    finished = Signal(list)  # Emits list of (y, x, confidence) tuples
    progress = Signal(str)  # Progress messages
    error = Signal(str)  # Error messages

    def __init__(self, app: App, dng_image):
        super().__init__()
        self.app = app
        self.dng_image = dng_image

    def run(self):
        """Run CNN detection in background"""
        try:
            self.progress.emit("Running CNN hot pixel detection...")
            
            detections = self.app.detect_residual_hotpixels_cnn(self.dng_image)
            
            self.progress.emit("CNN detection complete.")
            self.finished.emit(detections)
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.error.emit(str(e))


class TrainingDataWorker(QThread):
    """Worker to capture training data from uncorrected and corrected images"""
    finished = Signal(int)  # Emits number of samples captured
    progress = Signal(str)  # Progress messages
    error = Signal(str)  # Error messages

    def __init__(self, app: App, uncorrected_image, corrected_image, hot_pixel_profile, difference_threshold: float = 10.0):
        super().__init__()
        self.app = app
        self.uncorrected_image = uncorrected_image
        self.corrected_image = corrected_image
        self.hot_pixel_profile = hot_pixel_profile
        self.difference_threshold = difference_threshold

    def run(self):
        """Capture training data in background"""
        try:
            self.progress.emit("Capturing training data...")
            
            samples_captured = self.app.capture_training_data(
                self.uncorrected_image,
                self.corrected_image,
                self.hot_pixel_profile,
                self.difference_threshold
            )
            
            self.progress.emit("Training data capture complete.")
            self.finished.emit(samples_captured)
        except Exception as e:
            self.error.emit(str(e))
