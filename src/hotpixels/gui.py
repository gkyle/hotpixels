import sys
import os
import argparse

# Add the src directory to the path so we can import hotpixels modules
src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

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
    parser = argparse.ArgumentParser(description='Hot Pixel Analysis GUI')
    parser.add_argument('--profile', type=str, help='Path to a profile JSON file to load at startup')
    parser.add_argument('--image', type=str, nargs='+', help='Paths to image files to load at startup')
    parser.add_argument('--darkframes', type=str, nargs='+', help='Paths to dark frame files to load at startup')
    
    args = parser.parse_args()
    main(profile_path=args.profile, image_paths=args.image, darkframes_paths=args.darkframes)
