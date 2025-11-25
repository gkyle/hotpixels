from pathlib import Path
import platform
import exifread
import dngio
import numpy as np
# exiftool provide deeper access to MakerNotes and XMPPrivateData fields. It relies on an external executable. For now, treat as optional and use exifread for standard tags.
import exiftool

MOSAIC_MAP = {
    0: "R",
    1: "G",
    2: "B",
}


class Image:
    def __init__(self, filename: str):
        self.filename = filename
        self._exifread_tags = None  # Lazy-loaded
        self.sensor_temperature = None
        self.unique_id = None
        self._exiftool_metadata = None  # Lazy-loaded

    def _get_exifread_tags(self):
        """Lazily load exifread tags when needed."""
        if self._exifread_tags is None:
            with open(self.filename, "rb") as f:
                self._exifread_tags = exifread.process_file(f)
        return self._exifread_tags

    def _get_exiftool_metadata(self):
        """Lazily load exiftool metadata when needed."""
        if self._exiftool_metadata is None:
            try:
                # Go up from src/hotpixels/image.py to repository root
                repo_root = Path(__file__).parent.parent.parent
                
                # Determine the correct exiftool executable based on platform
                if platform.system() == "Windows":
                    exiftool_path = str(repo_root / "vendor" / "exiftool.exe")
                else:  # macOS, Linux, or other Unix-like systems
                    exiftool_path = str(repo_root / "vendor" / "exiftool")
                
                with exiftool.ExifToolHelper(executable=exiftool_path) as et:
                    self._exiftool_metadata = et.get_metadata([self.filename])[0]
            except Exception as e:
                self._exiftool_metadata = {}  # Empty dict to avoid re-trying
                print(f"Warning: Could not read metadata with exiftool: {e}")
        return self._exiftool_metadata
        
    def get_camera_model(self) -> str:
        return str(self._get_exifread_tags().get("Image Model", "Unknown"))
    
    def get_camera_make(self) -> str:
        return str(self._get_exifread_tags().get("Image Make", "Unknown"))
    
    def get_shutter_speed(self) -> str:
        return str(self._get_exifread_tags().get("EXIF ExposureTime", "Unknown"))
    
    def get_iso(self) -> str:
        return str(self._get_exifread_tags().get("EXIF ISOSpeedRatings", "Unknown"))
    
    def get_resolution(self) -> tuple[int, int]:
        tags = self._get_exifread_tags()
        width = int(str(tags.get("EXIF SubIFD0 ImageWidth", 0)))
        height = int(str(tags.get("EXIF SubIFD0 ImageLength", 0)))
        return (width, height)
    
    def get_date_created(self) -> str:
        return str(self._get_exifread_tags().get("EXIF DateTimeOriginal", "Unknown"))
    
    def get_sensor_temperature(self) -> float | None:
        if self.sensor_temperature is not None:
            return self.sensor_temperature
        
        possible_tags = [
            "MakerNotes:SensorTemperature",
            "MakerNotes:CameraTemperature",
        ]

        exiftool_metadata = self._get_exiftool_metadata()
        if exiftool_metadata:
            for tag in possible_tags:
                temp_raw = exiftool_metadata.get(tag, None)
                if temp_raw:
                    # Parse temperature - it might be a string like "32 C" or just a number
                    temp = parseTemperature(temp_raw)
                    if temp is not None:
                        self.sensor_temperature = temp
                        return temp
        
        return None
    
    def get_unique_id(self) -> str | None:
        if self.unique_id is not None:
            return self.unique_id
        
        possible_tags = [
            "MakerNotes:InternalSerialNumber",
            "MakerNotes:SerialNumber",
        ]

        exiftool_metadata = self._get_exiftool_metadata()
        if exiftool_metadata:
            for tag in possible_tags:
                serial_raw = exiftool_metadata.get(tag, None)
                if serial_raw:
                    uid = parseSerialNumber(serial_raw)
                    if uid:
                        self.unique_id = uid
                        return uid
                    
        return None
    
class DNGImage(Image):
    def __init__(self, filename: str, process_rgb: bool = False, debug: bool = False):
        super().__init__(filename)
        self.dng = dngio.DNG(filename, process_rgb, debug)
        self.raw_img = self.dng.readRawData()
        self.rgb = process_rgb

    def get_data(self):
        return self.raw_img

    def white_balance(self, r_scale=None, g_scale=None, b_scale=None):
        """Apply white balance to the RGB image using provided or calculated scaling factors."""
        if not self.rgb:
            raise ValueError("White balance can only be applied to RGB images")
        
        # If no scales are provided calculate gray world scales
        if r_scale is None or g_scale is None or b_scale is None:
            r_mean = np.mean(self.raw_img[:, :, 0])
            g_mean = np.mean(self.raw_img[:, :, 1])
            b_mean = np.mean(self.raw_img[:, :, 2])
            gray = (r_mean + g_mean + b_mean) / 3

            # Calculate scaling factors
            r_scale = gray / r_mean
            g_scale = gray / g_mean
            b_scale = gray / b_mean

        # Apply white balance with proper dtype handling
        original_dtype = self.raw_img.dtype
        
        # Convert to float for calculations to avoid overflow
        img_float = self.raw_img.astype(np.float64)
        img_float[:, :, 0] *= r_scale
        img_float[:, :, 1] *= g_scale
        img_float[:, :, 2] *= b_scale

        # Clip to valid range and convert back to original dtype
        if original_dtype == np.uint16:
            self.raw_img = np.clip(img_float, 0, 65535).astype(original_dtype)
        elif original_dtype == np.uint8:
            self.raw_img = np.clip(img_float, 0, 255).astype(original_dtype)
        else:
            self.raw_img = np.clip(img_float, 0, img_float.max()).astype(original_dtype)

    # Replace raw pixels and save to new DNG file
    def save(self, output_dng_path: str):
        return self.dng.replaceRawData(self.raw_img, output_dng_path)

    def get_bayer_pattern(self) -> str:
        """Get the Bayer pattern string like "RGGB"."""
        mosaic = self.dng.getMosaic()
        bayer_pattern = ""
        for val in mosaic.flatten().tolist():
            bayer_pattern += MOSAIC_MAP[val]
        return bayer_pattern

    def subtract_dark_frame(self, dark_noise_frame):
        """Subtract a dark noise frame from the raw image with normalization."""
        # Normalize to dark frame floor
        dark_mean = np.mean(dark_noise_frame)
        norm_dark_noise_frame = dark_noise_frame - dark_mean
        # No value in norm_dark_noise_frame should be <0
        norm_dark_noise_frame = np.maximum(norm_dark_noise_frame, 0)

        corrected_img = self.raw_img.astype(np.int32) - norm_dark_noise_frame.astype(np.int32)
        corrected_img = np.clip(corrected_img, 0, 65535).astype(np.uint16)

        self.raw_img = corrected_img
        return corrected_img

    def correct_hot_pixels(self, hot_pixels, radius=4):
        """Correct hot pixels in the raw image using neighboring pixels of the same Bayer color."""
        raw_img = self.raw_img
        bayer_pattern = self.get_bayer_pattern()

        rows, cols = raw_img.shape
        corrected_img = raw_img.copy()

        for pixel in hot_pixels:
            y, x, _ = pixel

            # Get area around bad pixel
            y1 = max(0, y - radius)
            y2 = min(rows, y + radius)
            x1 = max(0, x - radius)
            x2 = min(cols, x + radius)
            area = corrected_img[y1:y2, x1:x2]

            # Get pixel values from neighbors with same bayer color
            pixel_bayer_color = bayer_pattern[(y % 2) * 2 + (x % 2)]
            neighbor_values = []
            for dy in range(area.shape[0]):
                for dx in range(area.shape[1]):
                    neighbor_bayer_color = bayer_pattern[
                        ((y1 + dy) % 2) * 2 + ((x1 + dx) % 2)
                    ]
                    is_hot_pixel = (y1 + dy, x1 + dx) == (y, x)
                    if neighbor_bayer_color == pixel_bayer_color and not is_hot_pixel:
                        neighbor_values.append(area[dy, dx])

            # Replace hot pixel with mean of neighbor values
            if neighbor_values:
                corrected_img[y, x] = int(np.mean(neighbor_values, axis=0))

        self.raw_img = corrected_img

def parseSerialNumber(raw: str) -> str:
    """Convert space delimited byte string to hex string."""
    try:
        # Convert the space-separated bytes to hex string
        if isinstance(raw, str) and " " in raw:
            bytes_list = [int(b) for b in raw.split()]
            str_hex = ''.join(f'{b:02x}' for b in bytes_list)
            return str_hex
        else:
            return raw

    except ValueError:
        return ''

def parseTemperature(raw) -> float | None:
    """Parse temperature value from EXIF data.
    
    Args:
        raw: Temperature value which may be:
            - A string like "32 C"
            - A numeric value
            - A multi-value string like "42 42 0" (takes first value)
        
    Returns:
        Temperature as float in Celsius, or None if parsing fails
    """
    try:
        if isinstance(raw, (int, float)):
            return float(raw)
        
        if isinstance(raw, str):
            # Remove common suffixes and whitespace
            temp_str = raw.strip().upper()
            temp_str = temp_str.replace(' C', '').replace('C', '').replace('°', '').strip()
            
            # Handle multi-value format like "42 42 0" - take first value
            if ' ' in temp_str:
                parts = temp_str.split()
                if parts:
                    return float(parts[0])
            
            return float(temp_str)
        
        return None
    except (ValueError, AttributeError):
        return None