"""Profile Library Tab for browsing and managing saved hot pixel profiles."""

import os
from pathlib import Path
from datetime import datetime as dt
from typing import TYPE_CHECKING, Optional

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QTableWidgetItem, QPushButton, QHBoxLayout
)
from PySide6.QtCore import Qt
from PySide6 import QtCore, QtUiTools
from PySide6.QtWidgets import QMessageBox as MsgBox
from PySide6.QtWidgets import QHeaderView as HeaderView

from hotpixels.profile import HotPixelProfile
from .workers import ProfileLoadingWorker
from .formatters import format_profile_summary

if TYPE_CHECKING:
    from hotpixels.ui.main_window import HotPixelGUI


class ProfileLibraryTab(QWidget):
    """Tab for browsing and managing saved hot pixel profiles"""

    def __init__(self):
        super().__init__()
        self.profiles_directory = Path("./profiles")
        self.profile_data = []  # List of profile metadata dictionaries
        self.selected_profile_path = None
        self.pending_selection_path = None  # Profile to select after loading completes
        self.main_window: Optional['HotPixelGUI'] = None  # Reference to main window
        self.loading_worker = None  # Async loading worker
        self.load_ui()
        self.setup_connections()
        self.setup_table()
        
        # Load profiles immediately on init
        self.refresh_profiles()

    def set_main_window(self, main_window):
        """Set reference to the main window for shared profile access"""
        self.main_window = main_window

    def statusBar(self):
        """Get the status bar from the parent main window"""
        if self.main_window and hasattr(self.main_window, 'statusBar'):
            return self.main_window.statusBar()
        return None

    def showStatusMessage(self, message: str, timeout: int = 0):
        """Show a message in the status bar with optional timeout"""
        status_bar = self.statusBar()
        if status_bar:
            status_bar.showMessage(message, timeout)

    def load_ui(self):
        """Load the UI from the .ui file"""
        # Load the UI file
        ui_file_path = Path(__file__).parent / "profile_library_tab.ui"
        ui_file = QtCore.QFile(str(ui_file_path))
        ui_file.open(QtCore.QFile.ReadOnly)

        loader = QtUiTools.QUiLoader()
        self.ui = loader.load(ui_file, self)
        ui_file.close()

        # Set up layout to contain the loaded UI
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.ui)

    def setup_connections(self):
        """Set up signal connections"""
        self.ui.refreshButton.clicked.connect(self.refresh_profiles)
        self.ui.filterLineEdit.textChanged.connect(self.filter_profiles)
        self.ui.profileTableWidget.itemSelectionChanged.connect(self.on_profile_selection_changed)

    def setup_table(self):
        """Set up the profile table widget"""
        table = self.ui.profileTableWidget

        table.setColumnCount(9)

        # Set column headers
        headers = ["Camera ID", "Make", "Model", "Device ID", "Shutter Speed", "ISO", "Temperature", "Date Created", ""]
        table.setHorizontalHeaderLabels(headers)

        # Set column widths
        header = table.horizontalHeader()
        header.setSectionResizeMode(0, HeaderView.Stretch)  # Camera ID
        header.setSectionResizeMode(1, HeaderView.ResizeToContents)  # Make
        header.setSectionResizeMode(2, HeaderView.ResizeToContents)  # Model
        header.setSectionResizeMode(3, HeaderView.ResizeToContents)  # Device ID
        header.setSectionResizeMode(4, HeaderView.ResizeToContents)  # Shutter Speed
        header.setSectionResizeMode(5, HeaderView.ResizeToContents)  # ISO
        header.setSectionResizeMode(6, HeaderView.ResizeToContents)  # Temperature
        header.setSectionResizeMode(7, HeaderView.ResizeToContents)  # Date Created
        header.setSectionResizeMode(8, HeaderView.Fixed)  # Delete button
        table.setColumnWidth(8, 30)  # Fixed width for delete button

        # Enable sorting
        table.setSortingEnabled(True)
        
        # Set selection styling for better visibility
        table.setStyleSheet("""
            QTableWidget::item:selected {
                background-color: #0078D4;
                color: white;
            }
            QTableWidget::item:selected:focus {
                background-color: #0078D4;
                color: white;
            }
        """)
    
    def showEvent(self, event):
        """Called when the tab becomes visible - update profile summary and highlight current profile"""
        super().showEvent(event)
        self.update_profile_summary()
        
        # Highlight the current profile if one is loaded
        if self.main_window and self.main_window.app.current_profile_path:
            self.select_profile_by_path(self.main_window.app.current_profile_path)
    
    def update_profile_summary(self):
        """Update the profile summary display with the current profile"""
        if self.main_window and self.main_window.app.current_profile:
            self.display_profile_summary(self.main_window.app.current_profile)
        else:
            self.clear_profile_summary()

    def refresh_profiles(self):
        """Scan the profiles directory and load all profile metadata asynchronously"""
        # Don't start a new worker if one is already running
        if self.loading_worker and self.loading_worker.isRunning():
            return

        # Clear current data and table
        self.profile_data = []
        self.populate_table()  # Clear the table

        # Show loading status
        self.showStatusMessage("Loading profiles...", 0)

        # Disable refresh button during loading
        if hasattr(self.ui, 'refreshButton'):
            self.ui.refreshButton.setEnabled(False)

        # Start async loading
        self.loading_worker = ProfileLoadingWorker(self.profiles_directory)
        self.loading_worker.finished.connect(self.on_profiles_loaded)
        self.loading_worker.progress.connect(self.on_loading_progress)
        self.loading_worker.error.connect(self.on_loading_error)
        self.loading_worker.start()

    def on_profiles_loaded(self, profile_data):
        """Handle completion of async profile loading"""
        self.profile_data = profile_data
        self.populate_table()

        # Update status
        self.showStatusMessage(f"Found {len(self.profile_data)} profiles in library", 3000)

        # Re-enable refresh button
        if hasattr(self.ui, 'refreshButton'):
            self.ui.refreshButton.setEnabled(True)

        # Clean up worker
        if self.loading_worker:
            self.loading_worker.deleteLater()
            self.loading_worker = None
        
        # If there's a pending selection, select it now
        if self.pending_selection_path:
            self.select_profile_by_path(self.pending_selection_path)
            self.pending_selection_path = None
        else:
            # If no pending selection, just update the summary with current profile
            self.update_profile_summary()

    def on_loading_progress(self, message):
        """Handle progress updates from async profile loading"""
        self.showStatusMessage(message, 0)

    def on_loading_error(self, error_message):
        """Handle errors from async profile loading"""
        self.showStatusMessage(f"Error loading profiles: {error_message}", 10000)

        # Re-enable refresh button
        if hasattr(self.ui, 'refreshButton'):
            self.ui.refreshButton.setEnabled(True)

        # Clean up worker
        if self.loading_worker:
            self.loading_worker.deleteLater()
            self.loading_worker = None

    def populate_table(self, filtered_data=None):
        """Populate the table with profile data"""
        data = filtered_data if filtered_data is not None else self.profile_data

        table = self.ui.profileTableWidget
        table.setRowCount(len(data))

        for row, profile_info in enumerate(data):
            # Camera ID
            table.setItem(row, 0, QTableWidgetItem(profile_info['camera_id']))
            # Make
            table.setItem(row, 1, QTableWidgetItem(profile_info['make']))
            # Model
            table.setItem(row, 2, QTableWidgetItem(profile_info['model']))
            # Device ID
            device_id = profile_info.get('camera_uid', '') or 'N/A'
            table.setItem(row, 3, QTableWidgetItem(device_id))
            # Shutter Speed
            table.setItem(row, 4, QTableWidgetItem(str(profile_info['shutter_speed'])))
            # ISO
            table.setItem(row, 5, QTableWidgetItem(str(profile_info['iso'])))
            # Sensor Temperature
            table.setItem(row, 6, QTableWidgetItem(str(profile_info['sensor_temperature'])))
            # Date Created
            date_str = profile_info['date_created']
            if date_str != "Unknown":
                try:
                    # Try to format the date nicely
                    if 'T' in date_str:
                        # ISO format
                        date_obj = dt.fromisoformat(date_str.replace('Z', '+00:00'))
                        date_str = date_obj.strftime('%Y-%m-%d %H:%M')
                    elif ' ' in date_str:
                        # Already formatted
                        pass
                except:
                    # Keep original if parsing fails
                    pass
            table.setItem(row, 7, QTableWidgetItem(date_str))

            # Store the file path as user data for easy access
            table.item(row, 0).setData(Qt.UserRole, profile_info['file_path'])
            
            # Add delete button in a centered container
            delete_btn = QPushButton("✕")
            delete_btn.setFixedSize(25, 25)
            delete_btn.setStyleSheet("QPushButton { color: red; font-weight: bold; padding: 0px; margin: 0px; }")
            delete_btn.setToolTip("Delete this profile")
            delete_btn.clicked.connect(lambda checked, path=profile_info['file_path']: self.delete_profile(path))
            
            # Center the button in a container widget
            container = QWidget()
            container_layout = QHBoxLayout(container)
            container_layout.addWidget(delete_btn)
            container_layout.setAlignment(Qt.AlignCenter)
            container_layout.setContentsMargins(0, 0, 0, 0)
            container_layout.setSpacing(0)
            container.setFixedWidth(30)
            
            table.setCellWidget(row, 8, container)
        
        # Re-apply column width after all widgets are set
        table.setColumnWidth(8, 30)

    def filter_profiles(self, filter_text):
        """Filter the profiles based on the filter text"""
        if not filter_text.strip():
            # Show all profiles if filter is empty
            self.populate_table()
            return

        filter_text = filter_text.lower()
        filtered_data = []

        for profile_info in self.profile_data:
            # Check if filter text matches any of the displayed fields
            searchable_text = " ".join([
                profile_info['camera_id'].lower(),
                profile_info['make'].lower(),
                profile_info['model'].lower(),
                profile_info.get('camera_uid', '').lower(),
                str(profile_info['shutter_speed']).lower(),
                str(profile_info['iso']).lower(),
                str(profile_info['sensor_temperature']).lower(),
                str(profile_info['date_created']).lower()
            ])

            if filter_text in searchable_text:
                filtered_data.append(profile_info)

        self.populate_table(filtered_data)

    def on_profile_selection_changed(self):
        """Handle profile selection changes - automatically loads the selected profile"""
        table = self.ui.profileTableWidget
        selected_rows = table.selectionModel().selectedRows()

        if selected_rows:
            row = selected_rows[0].row()
            # Get the file path from the first column's user data
            file_path = table.item(row, 0).data(Qt.UserRole)
            self.selected_profile_path = file_path

            # Find the profile data
            profile_info = None
            for info in self.profile_data:
                if info['file_path'] == file_path:
                    profile_info = info
                    break

            if profile_info:
                self.display_profile_summary(profile_info['profile'])
                
                # Automatically load the selected profile
                if self.main_window:
                    self.main_window.load_profile_file(file_path, show_success_message=False)
        else:
            self.selected_profile_path = None
            self.clear_profile_summary()

    def display_profile_summary(self, profile: HotPixelProfile):
        """Display the profile summary in the summary area"""
        styled_text = format_profile_summary(profile, "No profile selected")
        self.ui.profileSummaryLabel.setText(styled_text)

    def clear_profile_summary(self):
        """Clear the profile summary display"""
        self.ui.profileSummaryLabel.setText("Select a profile from the library to view its summary.")
    
    def select_profile_by_path(self, profile_path: str):
        """Programmatically select a profile in the table by its file path"""
        # If profiles haven't loaded yet, store this as pending selection
        if not self.profile_data:
            self.pending_selection_path = profile_path
            return
        
        table = self.ui.profileTableWidget
        
        # Normalize the path for comparison (use absolute paths)
        target_path = str(Path(profile_path).resolve())
        
        # Find the row with this profile path
        for row in range(table.rowCount()):
            item = table.item(row, 0)
            if item:
                stored_path = str(Path(item.data(Qt.UserRole)).resolve())
                if stored_path == target_path:
                    # Select this row
                    table.selectRow(row)
                    # Scroll to ensure it's visible
                    table.scrollToItem(item)
                    return
        
        # If not found, clear selection
        table.clearSelection()

    def delete_profile(self, profile_path):
        """Delete a profile from disk with confirmation"""
        if not profile_path:
            return

        file_name = Path(profile_path).name

        # Ask for confirmation
        reply = MsgBox.question(
            self,
            "Confirm Deletion",
            f"Are you sure you want to delete the profile '{file_name}'?\n\nThis action cannot be undone.",
            MsgBox.Yes | MsgBox.No,
            MsgBox.No
        )

        if reply == MsgBox.Yes:
            try:
                os.remove(profile_path)
                self.showStatusMessage(f"Profile deleted: {file_name}", 3000)
                # Refresh the library to update the display
                self.refresh_profiles()
            except Exception as e:
                self.showStatusMessage(f"Failed to delete profile: {str(e)}", 10000)
