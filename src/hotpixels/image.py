import exifread
import dngio
import numpy as np

from hotpixels.dng_vendor_data import DNGVendorData

MOSAIC_MAP = {
    0: "R",
    1: "G",
    2: "B",
}


class Image:
    def __init__(self, filename: str):
        self.filename = filename
        self.tags = self.get_exif_tags()
        self.sensor_temperature = None
        self.unique_id = None

    def get_exif_tags(self):
        with open(self.filename, "rb") as f:
            tags = exifread.process_file(f)
            return tags
        
    def get_camera_model(self) -> str:
        return str(self.tags.get("Image Model", "Unknown"))
    
    def get_camera_make(self) -> str:
        return str(self.tags.get("Image Make", "Unknown"))
    
    def get_shutter_speed(self) -> str:
        return str(self.tags.get("EXIF ExposureTime", "Unknown"))
    
    def get_iso(self) -> str:
        return str(self.tags.get("EXIF ISOSpeedRatings", "Unknown"))
    
    def get_resolution(self) -> tuple[int, int]:
        width = int(str(self.tags.get("EXIF SubIFD0 ImageWidth", 0)))
        height = int(str(self.tags.get("EXIF SubIFD0 ImageLength", 0)))
        return (width, height)
    
    def get_date_created(self) -> str:
        return str(self.tags.get("EXIF DateTimeOriginal", "Unknown"))
    
    def get_sensor_temperature(self) -> float | None:
        if self.sensor_temperature is not None:
            return self.sensor_temperature
        
        camera_make = self.get_camera_make()
        if DNGVendorData.is_device_supported(camera_make):
            temp = DNGVendorData.get_temperature(self.filename)
            self.sensor_temperature = temp
            return temp
        else:
            return None
    
    def get_unique_id(self) -> str | None:
        if self.unique_id is not None:
            return self.unique_id
        
        camera_make = self.get_camera_make()
        if DNGVendorData.is_device_supported(camera_make):
            uid = DNGVendorData.get_unique_id(self.filename)
            self.unique_id = uid
            return uid
        else:
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

    # Returns Bayer pattern string like "RGGB"
    def get_bayer_pattern(self) -> str:
        mosaic = self.dng.getMosaic()
        bayer_pattern = ""
        for val in mosaic.flatten().tolist():
            bayer_pattern += MOSAIC_MAP[val]
        return bayer_pattern

    def subtract_dark_frame(self, dark_noise_frame):
        # Normalize to dark frame floor
        dark_mean = np.mean(dark_noise_frame)
        norm_dark_noise_frame = dark_noise_frame - dark_mean
        # No value in norm_dark_noise_frame should be <0
        norm_dark_noise_frame = np.maximum(norm_dark_noise_frame, 0)

        corrected_img = self.raw_img.astype(np.int32) - norm_dark_noise_frame.astype(np.int32)
        corrected_img = np.clip(corrected_img, 0, 65535).astype(np.uint16)
        return corrected_img

    def correct_hot_pixels(self, hot_pixels, radius=4):
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
