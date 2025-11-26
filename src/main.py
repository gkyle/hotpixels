#!/usr/bin/env python3
import sys
import argparse

from PySide6.QtWidgets import QApplication

# Import all UI components from the ui package
from hotpixels.ui import HotPixelGUI


def main(profile_path=None, image_paths=None, darkframes_paths=None):
    """Main application entry point"""
    app = QApplication(sys.argv)
    
    # Set application style
    app.setStyle('Fusion')
    
    # Create and show main window
    window = HotPixelGUI(profile_path=profile_path, image_paths=image_paths, darkframes_paths=darkframes_paths)
    window.show()
    
    # Run application
    sys.exit(app.exec())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='HotPixels')
    parser.add_argument('--profile', type=str, help='Path to a profile JSON file to load at startup')
    parser.add_argument('--image', type=str, nargs='+', help='Paths to image files to load at startup')
    parser.add_argument('--darkframes', type=str, nargs='+', help='Paths to dark frame files to load at startup')
    
    args = parser.parse_args()
    main(profile_path=args.profile, image_paths=args.image, darkframes_paths=args.darkframes)
