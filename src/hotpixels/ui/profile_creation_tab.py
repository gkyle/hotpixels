"""Profile Creation Tab for creating hot pixel profiles from dark frames."""

import os
from pathlib import Path
from typing import TYPE_CHECKING, Optional

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QFileDialog, QMessageBox
)
from PySide6.QtCore import Qt
from PySide6.QtUiTools import QUiLoader

from hotpixels.app import App
from hotpixels.profile import HotPixelProfile
from .workers import AnalysisWorker
from .plot_widget import PlotWidget
from .formatters import format_profile_summary

if TYPE_CHECKING:
    from hotpixels.ui.main_window import HotPixelGUI


class ProfileCreationTab(QWidget):
    """Tab for creating hot pixel profiles from dark frames"""

    def __init__(self, app: App):
        super().__init__()
        self.app = app  # Shared App instance
        self.worker: Optional[AnalysisWorker] = None
        self.main_window: Optional['HotPixelGUI'] = None  # Reference to main window

        # Lazy rendering flags - track which tabs have been rendered
        self.rendered_tabs = {
            'statistics': False,
            'hot_pixel_map': False,
            'dark_frame_histogram': False,
            'deviation_threshold': False
        }
        self.load_ui()
        self.setup_connections()
        self.setup_plot_widget()
        self.setup_initial_state()

    def set_main_window(self, main_window):
        """Set reference to the main window for shared profile access"""
        self.main_window = main_window

    def showEvent(self, event):
        """Handle show event - check if graphs need updating when tab becomes visible"""
        super().showEvent(event)
        
        # If we have a profile loaded and graphs haven't been rendered yet, update them
        if self.app.current_profile and not all(self.rendered_tabs.values()):
            # Get the current sub-tab in the plot widget and render it if needed
            current_tab = self.ui.plotTabWidget.currentIndex()
            self.on_tab_changed(current_tab)

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

    def update_dark_frame_list_display(self):
        """Update the dark frame list display using QListWidget"""
        self.ui.fileListTextEdit.clear()

        if not hasattr(self, '_dark_frame_files') or not self._dark_frame_files:
            # Add placeholder text when no files are loaded
            from PySide6.QtWidgets import QListWidgetItem as ListItem
            item = ListItem("No dark frames selected. Click 'Select Dark Frame Images' to choose files.")
            item.setFlags(item.flags() & ~Qt.ItemIsSelectable)  # Make non-selectable
            self.ui.fileListTextEdit.addItem(item)
        else:
            # Add each dark frame file path
            for i, file_path in enumerate(self._dark_frame_files, 1):
                filename = os.path.basename(file_path)
                display_text = f"{i}. {filename}"
                self.ui.fileListTextEdit.addItem(display_text)

    def get_dark_frame_files(self):
        """Get list of dark frame files"""
        return getattr(self, '_dark_frame_files', [])

    def has_dark_frame_files(self):
        """Check if dark frame files are loaded"""
        return bool(getattr(self, '_dark_frame_files', []))

    def update_select_files_button_style(self):
        """Update the Select Dark Frames button style based on whether frames are loaded"""
        has_files = self.has_dark_frame_files()

        if has_files:
            # Normal button style when files are loaded
            self.ui.selectFilesButton.setStyleSheet("""
                QPushButton#selectFilesButton {
                    background-color: #f0f0f0;
                    color: black;
                    border: 1px solid #ccc;
                    border-radius: 4px;
                    padding: 4px 8px;
                }
                QPushButton#selectFilesButton:hover {
                    background-color: #e0e0e0;
                    border: 1px solid #999;
                }
                QPushButton#selectFilesButton:pressed {
                    background-color: #d0d0d0;
                }
            """)
        else:
            # Highlighted button style when no files are loaded (call to action)
            self.ui.selectFilesButton.setStyleSheet("""
                QPushButton#selectFilesButton {
                    background-color: #0078d4;
                    color: white;
                    border: 2px solid #106ebe;
                    border-radius: 4px;
                    font-weight: bold;
                    padding: 4px 8px;
                }
                QPushButton#selectFilesButton:hover {
                    background-color: #106ebe;
                }
                QPushButton#selectFilesButton:pressed {
                    background-color: #005a9e;
                }
            """)

    def load_ui(self):
        """Load the UI from the .ui file"""
        loader = QUiLoader()
        ui_file = Path(__file__).parent / "profile_creation_tab.ui"
        self.ui = loader.load(str(ui_file), self)

        # Set layout to contain the loaded UI
        layout = QVBoxLayout()
        layout.addWidget(self.ui)
        self.setLayout(layout)

    def setup_plot_widget(self):
        """Replace the placeholder plot widget with matplotlib widget inside scroll area"""
        # The plot widget is now inside a scroll area, so we need to access it differently
        plot_layout = self.ui.plotWidget.layout()
        if plot_layout:
            plot_layout.deleteLater()

        # Create plot widgets for each tab
        self.unified_plot_widget = PlotWidget(self.app)
        unified_layout = QVBoxLayout()
        unified_layout.addWidget(self.unified_plot_widget)
        self.ui.plotWidget.setLayout(unified_layout)

        self.map_plot_widget = PlotWidget(self.app)
        map_layout = QVBoxLayout()
        map_layout.addWidget(self.map_plot_widget)
        self.ui.mapWidget.setLayout(map_layout)

        self.deviation_plot_widget = PlotWidget(self.app)
        deviation_layout = QVBoxLayout()
        deviation_layout.addWidget(self.deviation_plot_widget)
        self.ui.deviationWidget.setLayout(deviation_layout)

        # Set the scroll area widgets
        self.ui.plotScrollArea.setWidget(self.ui.plotWidget)
        self.ui.mapScrollArea.setWidget(self.ui.mapWidget)
        self.ui.darkFrameScrollArea.setWidget(self.ui.darkFrameWidget)
        self.ui.deviationScrollArea.setWidget(self.ui.deviationWidget)

        # Configure splitter proportions (33% for statistics, 67% for plots)
        self.setup_splitter_proportions()

    def setup_splitter_proportions(self):
        """Configure the splitter to give statistics panel 33% and plots 67% of width"""
        # Set initial sizes - these will be proportional
        total_width = 1200  # Reasonable default width
        statistics_width = int(total_width * 0.33)
        plots_width = int(total_width * 0.67)

        self.ui.resultsSplitter.setSizes([statistics_width, plots_width])

        # Set stretch factors to maintain proportions when resizing
        self.ui.resultsSplitter.setStretchFactor(0, 1)  # Statistics panel
        self.ui.resultsSplitter.setStretchFactor(1, 2)  # Plots panel (2:1 ratio gives ~67%)

    def setup_connections(self):
        """Connect UI signals to slots"""
        self.ui.selectFilesButton.clicked.connect(self.select_files)
        self.ui.clearFilesButton.clicked.connect(self.clear_files)
        self.ui.analyzeButton.clicked.connect(self.analyze_frames)
        self.ui.plotTabWidget.currentChanged.connect(self.on_tab_changed)
        self.ui.saveButton.clicked.connect(self.save_profile)
        self.ui.openProfileButton.clicked.connect(self.open_profile)

        # Enable context menu for profile summary to edit camera ID
        self.ui.statisticsLabel.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.ui.statisticsLabel.customContextMenuRequested.connect(self.show_profile_context_menu)

        # Enable link clicking for camera ID editing
        self.ui.statisticsLabel.setOpenExternalLinks(False)  # Handle links internally
        self.ui.statisticsLabel.linkActivated.connect(self.handle_profile_link_click)

    def setup_initial_state(self):
        """Set up the initial state of UI elements"""
        # Initialize dark frame files list
        self._dark_frame_files = []
        self.update_dark_frame_list_display()

        # Disable tabs until analysis is complete
        self.ui.plotTabWidget.setEnabled(False)
        # Hide deviation tab initially
        self.update_deviation_tab_visibility()

    def select_files(self):
        # Get the last used directory from preferences
        start_dir = ""
        if self.main_window and hasattr(self.main_window, 'preferences'):
            last_dir = self.main_window.preferences.get_last_directory("darkframes")
            if last_dir:
                start_dir = last_dir

        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Dark Frame Images",
            start_dir,
            "DNG Files (*.dng);;All Files (*)"
        )

        if files:
            # Save the directory of the first selected file to preferences
            if self.main_window and hasattr(self.main_window, 'preferences'):
                self.main_window.preferences.update_last_directory(files[0], "darkframes")

            self.load_files(files)

    def load_files(self, files):
        """Load dark frame files programmatically (shared logic for GUI and startup loading)"""
        if files:
            self._dark_frame_files = files
            self.update_dark_frame_list_display()
            self.ui.analyzeButton.setEnabled(True)

            # Update button styling now that files are loaded
            self.update_select_files_button_style()

            # Automatically start analysis when files are selected
            self.analyze_frames()

    def clear_files(self):
        self._dark_frame_files = []
        self.update_dark_frame_list_display()
        self.ui.analyzeButton.setEnabled(False)

        # Update button styling now that files are cleared
        self.update_select_files_button_style()
        # self.ui.statusLabel.setText("Select dark frames to begin")

    def analyze_frames(self):
        files = self.get_dark_frame_files()
        if not files:
            return

        # Start analysis in background thread
        self.worker = AnalysisWorker(
            self.app,
            files,
            self.ui.deviationThresholdSpinBox.value()
        )

        self.worker.progress.connect(self.update_progress)
        self.worker.finished.connect(self.analysis_finished)
        self.worker.error.connect(self.analysis_error)
        self.worker.warning.connect(self.analysis_warning)

        self.ui.analyzeButton.setEnabled(False)
        self.ui.progressBar.setVisible(True)
        self.ui.progressBar.setRange(0, 0)  # Indeterminate progress

        self.worker.start()

    def update_progress(self, message: str):
        # self.ui.statusLabel.setText(message)
        pass

    def analysis_finished(self, profile: HotPixelProfile):
        # Use shared profile system if available
        if self.main_window:
            self.main_window.set_shared_profile(profile)
        else:
            # Fallback for standalone operation
            self.app.current_profile = profile

        self.ui.progressBar.setVisible(False)
        self.ui.analyzeButton.setEnabled(True)

        # Update statistics display
        self.update_statistics_display()

        # Enable UI elements
        self.ui.saveButton.setEnabled(True)
        self.ui.plotTabWidget.setEnabled(True)

        # Reset lazy rendering flags for new profile
        self.reset_rendered_tabs()

        # Render only the currently visible tab (usually the statistics tab after analysis)
        current_tab = self.ui.plotTabWidget.currentIndex()
        self.on_tab_changed(current_tab)

    def analysis_error(self, error_message: str):
        self.ui.progressBar.setVisible(False)
        self.ui.analyzeButton.setEnabled(True)
        # self.ui.statusLabel.setText(f"Error: {error_message}")

        QMessageBox.critical(self, "Analysis Error", f"Analysis failed:\n{error_message}")

    def analysis_warning(self, warnings: list, profile):
        """Handle profile creation warnings - let user decide whether to keep the profile"""
        self.ui.progressBar.setVisible(False)
        self.ui.analyzeButton.setEnabled(True)
        
        # Create warning message
        warning_text = "The following mismatches were detected in the dark frames:\n\n"
        warning_text += "\n\n".join(warnings)
        warning_text += "\n\nDo you want to keep this profile anyway?"
        
        # Show warning dialog with Yes/No buttons (No is default)
        reply = QMessageBox.question(
            self,
            "Profile Mismatch Warning",
            warning_text,
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No  # Default button is No
        )
        
        if reply == QMessageBox.Yes:
            self.analysis_finished(profile)
        else:
            pass  # Do nothing, just re-enable the analyze button

    def update_deviation_tab_visibility(self):
        """Show or hide the deviation threshold tab based on available data"""
        has_deviation_data = (self.app.current_profile and
                              hasattr(self.app.current_profile, 'deviation_threshold_comparisons') and
                              self.app.current_profile.deviation_threshold_comparisons)

        # Find the deviation tab index
        deviation_tab_index = -1
        for i in range(self.ui.plotTabWidget.count()):
            if self.ui.plotTabWidget.tabText(i) == "Hot Pixel Deviation":
                deviation_tab_index = i
                break

        if has_deviation_data:
            # Show the tab if it exists, or it should already be visible
            if deviation_tab_index >= 0:
                self.ui.plotTabWidget.setTabVisible(deviation_tab_index, True)
        else:
            # Hide the tab if it exists
            if deviation_tab_index >= 0:
                self.ui.plotTabWidget.setTabVisible(deviation_tab_index, False)

    def update_statistics_display(self):
        styled_text = format_profile_summary(self.app.current_profile, "No profile loaded")
        self.ui.statisticsLabel.setText(styled_text)

    def on_tab_changed(self, index):
        """Handle tab change to update the appropriate plot with lazy rendering"""
        if not self.app.current_profile:
            return

        if index == 0:  # Unified Analysis tab
            if not self.rendered_tabs['statistics']:
                self.plot_statistics()
                self.rendered_tabs['statistics'] = True
        elif index == 1:  # Hot Pixel Map tab
            if not self.rendered_tabs['hot_pixel_map']:
                self.plot_hot_pixel_map()
                self.rendered_tabs['hot_pixel_map'] = True
        elif index == 2:  # Hot Pixel Deviation tab
            if not self.rendered_tabs['deviation_threshold']:
                self.plot_deviation_threshold_comparison()
                self.rendered_tabs['deviation_threshold'] = True

    def plot_statistics(self):
        if self.app.current_profile:
            self.unified_plot_widget.plot_unified_hot_pixel_analysis(self.app.current_profile)

    def plot_hot_pixel_map(self):
        if self.app.current_profile:
            self.map_plot_widget.plot_hot_pixel_map(self.app.current_profile)

    def plot_deviation_threshold_comparison(self):
        if self.app.current_profile and hasattr(self.app.current_profile, 'deviation_threshold_comparisons'):
            self.deviation_plot_widget.plot_deviation_threshold_comparison(self.app.current_profile.deviation_threshold_comparisons)

    def reset_rendered_tabs(self):
        """Reset lazy rendering flags when a new profile is loaded"""
        self.rendered_tabs = {
            'statistics': False,
            'hot_pixel_map': False,
            'deviation_threshold': False
        }

    def save_profile(self):
        if not self.app.current_profile:
            return

        # Default to ./profiles directory
        profiles_dir = os.path.join(os.getcwd(), "profiles")

        # Create profiles directory if it doesn't exist
        if not os.path.exists(profiles_dir):
            os.makedirs(profiles_dir)

        # Generate suggested filename from profile metadata
        suggested_filename = self._generate_profile_filename()
        full_suggested_path = os.path.join(profiles_dir, suggested_filename)

        filename, _ = QFileDialog.getSaveFileName(
            self,
            "Save Hot Pixel Profile",
            full_suggested_path,  # Include suggested filename in the path
            "JSON Files (*.json);;All Files (*)"
        )

        if filename:
            # Ask user if they want to copy DNG files locally
            copy_dngs = self._ask_copy_dng_files()

            try:
                # Copy DNG files if requested
                if copy_dngs:
                    self._copy_dng_files_to_profile_directory(filename)

                self.app.current_profile.save_to_file(filename)

                # Save the successfully saved profile path to preferences
                if self.main_window and hasattr(self.main_window, 'preferences'):
                    self.main_window.preferences.update_last_profile_path(filename)

                # Update status bar with success message
                self.showStatusMessage(f"Profile saved successfully: {filename}", 5000)
            except Exception as e:
                self.showStatusMessage(f"Failed to save profile: {str(e)}", 10000)

    def _ask_copy_dng_files(self) -> bool:
        """Ask user if they want to copy DNG files to a local directory"""
        from PySide6.QtWidgets import QMessageBox

        reply = QMessageBox.question(
            self,
            "Copy DNG Files",
            "Would you like to copy the input DNG files to a local directory for this profile?\n\n"
            "This helps preserve the original files when they were created in temporary directories by Lightroom.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes
        )

        return reply == QMessageBox.StandardButton.Yes

    def _copy_dng_files_to_profile_directory(self, profile_filename: str):
        """Copy DNG files to a local directory structure"""
        import shutil

        # Create profile-specific directory
        base_name = os.path.splitext(os.path.basename(profile_filename))[0]
        profile_dir = os.path.join(os.path.dirname(profile_filename), f"{base_name}_files")
        dng_dir = os.path.join(profile_dir, "dng_files")

        if not os.path.exists(dng_dir):
            os.makedirs(dng_dir)

        # Copy DNG files and update paths in profile
        if self.app.current_profile and self.app.current_profile.frame_paths:
            for i, frame_path in enumerate(self.app.current_profile.frame_paths):
                if os.path.exists(frame_path):
                    # Copy file to local directory
                    filename = os.path.basename(frame_path)
                    local_path = os.path.join(dng_dir, filename)

                    shutil.copy2(frame_path, local_path)

                    # Update the path in the profile to use the local copy
                    self.app.current_profile.frame_paths[i] = local_path

            self.showStatusMessage(f"DNG files copied to: {dng_dir}", 5000)

    def _generate_profile_filename(self) -> str:
        """Generate a suggested filename based on profile metadata"""
        if not self.app.current_profile or not self.app.current_profile.camera_metadata:
            return "hot_pixel_profile.json"

        cam = self.app.current_profile.camera_metadata

        # Clean up camera_id (remove special characters, replace spaces with underscores)
        import re
        camera_id = cam.camera_id or "unknown_camera"
        camera_id = re.sub(r'[^\w\-_]', '_', camera_id)

        # Clean up shutter speed (remove / and special characters)
        shutter = cam.shutter_speed or "unknown_shutter"
        shutter = re.sub(r'[^\w]', '_', shutter)

        # Clean up ISO
        iso = cam.iso or "unknown_iso"
        iso = re.sub(r'[^\w]', '', iso)

        # Parse and format date from camera metadata
        try:
            if cam.date_created:
                print(f"Debug: Original date_created value: '{cam.date_created}'")

                # Try multiple date formats commonly used by cameras
                from datetime import datetime
                date_formats = [
                    "%Y-%m-%dT%H:%M:%SZ",           # ISO format with Z
                    "%Y-%m-%dT%H:%M:%S",            # ISO format without Z
                    "%Y-%m-%d %H:%M:%S",            # Space separated
                    "%Y:%m:%d %H:%M:%S",            # EXIF format (colon separated)
                    "%Y-%m-%d",                     # Date only
                    "%Y:%m:%d",                     # EXIF date only
                ]

                date_obj = None
                for fmt in date_formats:
                    try:
                        date_obj = datetime.strptime(cam.date_created, fmt)
                        print(f"Debug: Successfully parsed date with format: {fmt}")
                        break
                    except ValueError:
                        continue

                if date_obj:
                    date_str = date_obj.strftime("%Y%m%d")
                    print(f"Debug: Formatted date as: {date_str}")
                else:
                    # Try ISO format parsing as fallback
                    try:
                        date_obj = datetime.fromisoformat(cam.date_created.replace('Z', '+00:00'))
                        date_str = date_obj.strftime("%Y%m%d")
                        print(f"Debug: Parsed with fromisoformat: {date_str}")
                    except:
                        date_str = "unknown_date"
                        print(f"Debug: All date parsing failed, using: {date_str}")
            else:
                date_str = "unknown_date"
                print("Debug: No date_created field found")
        except Exception as e:
            date_str = "unknown_date"
            print(f"Debug: Date parsing exception: {e}")

        # Construct filename: camera_id_shutter_iso_date.json
        filename = f"{camera_id}_{shutter}_{iso}_{date_str}.json"

        # Ensure filename isn't too long (Windows has 260 char limit for full path)
        if len(filename) > 100:
            filename = f"{camera_id[:20]}_{iso}_{date_str}.json"

        return filename

    def edit_camera_id(self):
        """Allow user to edit the camera ID"""
        if not self.app.current_profile or not self.app.current_profile.camera_metadata:
            return

        from PySide6.QtWidgets import QInputDialog as InputDialog

        current_id = self.app.current_profile.camera_metadata.camera_id
        new_id, ok = InputDialog.getText(
            self,
            "Edit Camera ID",
            "Enter Camera ID:",
            text=current_id
        )

        if ok and new_id.strip():
            self.app.current_profile.camera_metadata.camera_id = new_id.strip()
            self.update_profile_summary()
            self.showStatusMessage("Camera ID updated", 3000)

    def update_profile_summary(self):
        """Update the profile summary display"""
        if hasattr(self, 'ui') and hasattr(self.ui, 'statisticsLabel'):
            styled_text = format_profile_summary(self.app.current_profile, "No profile loaded")
            self.ui.statisticsLabel.setText(styled_text)

    def show_profile_context_menu(self, position):
        """Show context menu for profile summary with camera ID editing option"""
        if not self.app.current_profile or not self.app.current_profile.camera_metadata:
            return

        from PySide6.QtWidgets import QMenu as Menu

        menu = Menu(self)
        edit_camera_id_action = menu.addAction("Edit Camera ID")
        edit_camera_id_action.triggered.connect(self.edit_camera_id)

        # Show menu at the clicked position
        menu.exec(self.ui.statisticsLabel.mapToGlobal(position))

    def handle_profile_link_click(self, link):
        """Handle clicks on links in the profile summary"""
        if link == "#edit_camera_id":
            self.edit_camera_id()

    def load_profile_data(self, profile: HotPixelProfile):
        """Load profile data into the UI - shared logic for opening and startup loading"""
        self.app.current_profile = profile

        # Populate file list from profile frame paths
        frame_paths = getattr(profile, 'frame_paths', None) or []
        if frame_paths and len(frame_paths) > 0:
            self._dark_frame_files = frame_paths
            self.update_dark_frame_list_display()
            self.ui.analyzeButton.setEnabled(True)

            # Update button styling now that files are loaded from profile
            self.update_select_files_button_style()

            # Set deviation threshold from the profile
            if self.app.current_profile.deviation_threshold is not None and self.app.current_profile.deviation_threshold > 0:
                self.ui.deviationThresholdSpinBox.setValue(self.app.current_profile.deviation_threshold)

        self.update_statistics_display()
        self.ui.saveButton.setEnabled(True)
        self.ui.plotTabWidget.setEnabled(True)

        # Update deviation tab visibility
        self.update_deviation_tab_visibility()

        # Reset lazy rendering flags for new profile
        self.reset_rendered_tabs()

        # Only render if the Create Profile tab is currently visible
        if self.main_window and hasattr(self.main_window.ui, 'tabWidget'):
            current_main_tab = self.main_window.ui.tabWidget.currentIndex()
            if current_main_tab == 0:  # Create Profile tab is visible
                current_tab = self.ui.plotTabWidget.currentIndex()
                self.on_tab_changed(current_tab)

    def open_profile(self):
        """Open and load a hot pixel profile from a JSON file"""
        # Get the last used directory from preferences
        start_dir = ""
        if self.main_window and hasattr(self.main_window, 'preferences'):
            last_dir = self.main_window.preferences.get_last_directory("profile")
            if last_dir:
                start_dir = last_dir

        filename, _ = QFileDialog.getOpenFileName(
            self,
            "Open Hot Pixel Profile",
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
                    self.load_profile_data(profile)
                    self.showStatusMessage(f"Profile loaded successfully: {filename}", 5000)
                except Exception as e:
                    self.showStatusMessage(f"Failed to open profile: {str(e)}", 10000)
