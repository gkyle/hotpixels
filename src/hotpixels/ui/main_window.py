from pathlib import Path
from typing import List, Optional

from PySide6.QtWidgets import QMainWindow
from PySide6.QtCore import QTimer, QPoint
from PySide6.QtGui import QGuiApplication
from PySide6.QtUiTools import QUiLoader

from hotpixels.app import App
from hotpixels.profile import HotPixelProfile
from hotpixels.image import DNGImage
from hotpixels.preferences import get_preferences
from .workers import MultiImageLoadingWorker
from .profile_creation_tab import ProfileCreationTab
from .image_correction_tab import ImageCorrectionTab
from .profile_library_tab import ProfileLibraryTab


class HotPixelGUI(QMainWindow):
    """Main application window"""
    
    def __init__(self, profile_path=None, image_paths=None, darkframes_paths=None):
        super().__init__()
        self.app = App()  # Shared App instance for business logic
        self.image_loading_worker: Optional[MultiImageLoadingWorker] = None  # Background image loader
        self.preferences = get_preferences()  # Load user preferences
        
        self.load_ui()
        self.setup_tabs()
        
        # Set initial button styling
        self.profile_tab.update_select_files_button_style()
        
        # Load profile: command line arg takes precedence over preferences
        # But skip profile loading if darkframes or images are provided
        # (darkframes will create a new profile, images should get matching profile recommendation)
        if not darkframes_paths and not image_paths:
            profile_to_load = profile_path or self.preferences.get_valid_last_profile_path()
            if profile_to_load:
                self.load_startup_profile(profile_to_load)
        elif profile_path:
            # If a specific profile is provided via command line, always load it
            self.load_startup_profile(profile_path)
        
        # Load images if provided and switch to Correct Images tab
        if image_paths:
            self.load_startup_images(image_paths)
        
        # Load dark frames if provided and switch to Create Profile tab
        if darkframes_paths:
            self.load_startup_darkframes(darkframes_paths)

    def center(self):
        screen = QGuiApplication.primaryScreen().availableGeometry()
        window_size = self.geometry()
        x = (screen.width() - window_size.width()) // 2
        y = (screen.height() - window_size.height()) // 2
        self.move(QPoint(x, y))
    
    def load_ui(self):
        """Load the main window UI from the .ui file"""
        loader = QUiLoader()
        ui_file = Path(__file__).parent / "main_window.ui"
        self.ui = loader.load(str(ui_file))
        
        # Set the loaded UI as the central widget
        self.setCentralWidget(self.ui.centralwidget)
        self.setStatusBar(self.ui.statusbar)
        self.setWindowTitle(self.ui.windowTitle())
        self.resize(self.ui.size())

        screen_resolution = QGuiApplication.primaryScreen().availableGeometry()
        width = screen_resolution.width()
        height = screen_resolution.height()

        self.resize(width * 0.60, height * 0.70)

        # TODO: Remove hardcoded resize after testing
        self.resize(1920, 1080-24)
        print("Window size", self.size())

        self.center()
    
    def setup_tabs(self):
        """Setup the tab widgets"""
        # Clear existing tabs
        self.ui.tabWidget.clear()
        
        # Create and add our custom tabs
        self.profile_tab = ProfileCreationTab(self.app)
        self.correction_tab = ImageCorrectionTab(self.app)
        self.library_tab = ProfileLibraryTab()
        
        # Connect the tabs to the shared profile system
        self.profile_tab.set_main_window(self)
        self.correction_tab.set_main_window(self)
        self.library_tab.set_main_window(self)
        
        self.ui.tabWidget.addTab(self.library_tab, "Profile Library")
        self.ui.tabWidget.addTab(self.profile_tab, "Profile Details")
        self.ui.tabWidget.addTab(self.correction_tab, "Correct Images")
        
        # Default to Profile Library tab
        self.ui.tabWidget.setCurrentIndex(0)
    
    def set_shared_profile(self, profile: HotPixelProfile):
        """Set the shared profile and notify all tabs"""
        self.app.current_profile = profile
        
        # Update all tabs
        self.profile_tab.load_profile_data(profile)
        self.correction_tab.update_profile_info()
        self.correction_tab.update_image_summary()
        self.correction_tab.check_ready_state()

    def load_profile_file(self, profile_path: str, show_success_message: bool = True, switch_to_tab: int = None) -> bool:
        """Load a profile file for the whole application."""
        try:
            profile = HotPixelProfile.load_from_file(profile_path)

            # Use shared profile system
            self.set_shared_profile(profile)

            # Save the successfully loaded profile path to preferences
            self.preferences.update_last_profile_path(profile_path)

            # Show success message if requested
            if show_success_message:
                profile_name = Path(profile_path).name
                self.statusBar().showMessage(f"Profile loaded: {profile_name}", 5000)

            # Switch to specified tab if requested
            if switch_to_tab is not None:
                self.ui.tabWidget.setCurrentIndex(switch_to_tab)

            return True

        except Exception as e:
            # Show error message
            error_msg = f"Failed to load profile: {str(e)}"
            self.statusBar().showMessage(error_msg, 10000)
            print(f"Failed to load profile '{profile_path}': {e}")
            return False
    
    def load_startup_profile(self, profile_path):
        """Load a profile file at startup"""
        # Use consolidated profile loading (no success message for startup)
        if self.load_profile_file(profile_path, show_success_message=False):
            # After loading profile, ensure Profile Library tab is active and profile is selected
            self.ui.tabWidget.setCurrentIndex(0)
            # Select the profile in the library (profiles are now loaded immediately)
            self.library_tab.select_profile_by_path(profile_path)
    
    def load_startup_image(self, image_path):
        """Load an image file at startup and switch to Correct Images tab"""
        try:
            # Switch to the Correct Images tab first
            self.ui.tabWidget.setCurrentIndex(2)  # Index 2 is the Correct Images tab
            
            # Start image loading in background thread using MultiImageLoadingWorker
            self.image_loading_worker = MultiImageLoadingWorker([image_path], load_all=True)
            self.image_loading_worker.progress.connect(self.update_image_loading_progress)
            self.image_loading_worker.finished.connect(self.startup_image_loading_finished)
            self.image_loading_worker.error.connect(self.image_loading_error)
            
            # Show progress in status bar
            self.statusBar().showMessage("Loading image...", 0)  # 0 = no timeout
            
            self.image_loading_worker.start()
            
        except Exception as e:
            # Silently fail - no dialog as requested
            print(f"Failed to start image loading for '{image_path}': {e}")
    
    def update_image_loading_progress(self, message: str):
        """Update status bar with image loading progress"""
        self.statusBar().showMessage(message, 0)
    
    def startup_image_loading_finished(self, image_paths: List[str], rgb_images: List[DNGImage], raw_images: List[DNGImage]):
        """Handle successful startup image loading completion"""
        # Set the loaded images in the correction tab
        self.correction_tab.rgbImages = rgb_images
        self.correction_tab.rawImages = raw_images
        self.correction_tab.image_paths = image_paths
        self.correction_tab.display_original_image()
        self.correction_tab.check_ready_state()
        
        # Clear status bar
        self.statusBar().clearMessage()
        
        # Clean up worker
        self.image_loading_worker = None
    
    def image_loading_error(self, error_message: str):
        """Handle image loading error"""
        # Silently fail - no dialog as requested
        print(f"Failed to load startup image: {error_message}")
        
        # Clear status bar
        self.statusBar().clearMessage()
        
        # Clean up worker
        self.image_loading_worker = None
    
    def load_startup_images(self, image_paths):
        """Load multiple image files at startup and switch to Correct Images tab"""
        try:
            # Switch to the Correct Images tab first
            self.ui.tabWidget.setCurrentIndex(2)  # Index 2 is the Correct Images tab
            
            # Use QTimer to ensure UI is ready before starting
            QTimer.singleShot(100, lambda: self.correction_tab.load_images(image_paths))
            
        except Exception as e:
            # Silently fail - no dialog as requested
            print(f"Failed to start batch image loading: {e}")
    
    def load_startup_darkframes(self, darkframes_paths):
        """Load dark frame files at startup and switch to Create Profile tab"""
        try:
            # Switch to the Create Profile tab
            self.ui.tabWidget.setCurrentIndex(1)  # Index 1 is the Create Profile tab
            
            # Load the files using the same logic as select_files
            self.profile_tab.load_files(darkframes_paths)
            
        except Exception as e:
            # Silently fail - no dialog as requested
            print(f"Failed to load startup dark frames: {e}")
