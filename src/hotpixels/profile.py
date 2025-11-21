from dataclasses import dataclass, field
from dataclasses_json import dataclass_json
from typing import List, Tuple, Optional
import numpy as np
import json
import os
import cv2

from hotpixels.image import DNGImage


@dataclass_json
@dataclass
class CameraMetaData:
    camera_make: str
    camera_model: str
    shutter_speed: str
    iso: str
    image_resolution: Tuple[int, int]
    bayer_pattern: str
    date_created: str
    camera_id: str
    camera_uid: str = field(default="")
    sensor_temperature: float | None = field(default=None)

    @staticmethod
    def from_dng_image(dng_image: DNGImage) -> 'CameraMetaData':
        # Use last 4 characters from uid
        camera_uid = dng_image.get_unique_id()
        if camera_uid is not None:
            camera_id = "_".join([dng_image.get_camera_make(), dng_image.get_camera_model(), camera_uid[-4:]])
        else:
            camera_id = "_".join([dng_image.get_camera_make(), dng_image.get_camera_model()])

        return CameraMetaData(
            camera_make=dng_image.get_camera_make(),
            camera_model=dng_image.get_camera_model(),
            shutter_speed=dng_image.get_shutter_speed(),
            iso=dng_image.get_iso(),
            image_resolution=dng_image.get_resolution(),
            bayer_pattern=dng_image.get_bayer_pattern(),
            date_created=dng_image.get_date_created(),
            camera_id=camera_id,
            camera_uid=camera_uid,
            sensor_temperature=dng_image.get_sensor_temperature()
        )


@dataclass_json
@dataclass
class HotPixelFrameStats:
    file_path: str = ""
    mean_value: float = 0.0
    deviation_threshold: float = 0.0
    count_pixels: int = 0
    count_hot_pixels: int = 0
    hot_pixels: List[Tuple[int, int, float]] = field(default_factory=list)  # (y, x, value)

    @staticmethod
    def from_dng_image(dng_image: DNGImage, deviation_threshold: float) -> 'HotPixelFrameStats':
        raw_data = dng_image.get_data()
        mean_value = float(np.mean(raw_data))

        threshold = mean_value + deviation_threshold * np.std(raw_data)
        hot_pixel_coords = np.argwhere(raw_data > threshold)
        hot_pixels = [(int(y), int(x), float(raw_data[y, x])) for y, x in hot_pixel_coords]

        count_pixels = raw_data.size
        count_hot_pixels = len(hot_pixels)

        return HotPixelFrameStats(
            file_path=dng_image.filename,
            mean_value=mean_value,
            deviation_threshold=deviation_threshold,
            count_pixels=count_pixels,
            count_hot_pixels=count_hot_pixels,
            hot_pixels=hot_pixels,
        )


@dataclass_json
@dataclass
class HotPixelCommonStats:
    count_total_frames: int = 0
    count_hot_pixels: int = 0
    fraction_hot_pixels: float = 0.0
    fraction_common_hot_pixels: float = 0.0
    mean_count_hot_pixels: float = 0.0
    count_distinct_hot_pixels: int = 0


@dataclass_json
@dataclass
class HotPixelProfile:
    camera_metadata: Optional[CameraMetaData] = None
    common_statistics: Optional[HotPixelCommonStats] = None
    deviation_threshold: Optional[float] = None
    frame_paths: List[str] = field(default_factory=list)
    mean_values: List[float] = field(default_factory=list)
    hot_pixel_counts: List[int] = field(default_factory=list)

    median_noise_path: Optional[str] = None
    mean_noise_path: Optional[str] = None

    def __post_init__(self):
        # Initialize numpy arrays
        self._median_noise_frame = None
        self._mean_noise_frame = None
        
        # Ensure frame_paths is never None
        if self.frame_paths is None:
            self.frame_paths = []

        # Load from paths if they exist
        if self.median_noise_path is not None:
            self._median_noise_frame = cv2.imread(self.median_noise_path, cv2.IMREAD_UNCHANGED)

        if self.mean_noise_path is not None:
            self._mean_noise_frame = cv2.imread(self.mean_noise_path, cv2.IMREAD_UNCHANGED)

    @staticmethod
    def from_dark_frames(dark_frames_paths: List[str], deviation_threshold: int) -> 'HotPixelProfile':
        # Camera Info
        reference_dng = DNGImage(dark_frames_paths[0])
        camera_metadata = CameraMetaData.from_dng_image(reference_dng)

        dng_imgs = [DNGImage(path) for path in dark_frames_paths]
        
        # Validate that all dark frames have matching properties
        ref_make = reference_dng.get_camera_make()
        ref_model = reference_dng.get_camera_model()
        ref_uid = reference_dng.get_unique_id()
        ref_iso = reference_dng.get_iso()
        ref_shutter = reference_dng.get_shutter_speed()
        ref_temp = reference_dng.get_sensor_temperature()
        
        for i, dng_img in enumerate(dng_imgs[1:], start=1):  # Skip first image (reference)
            frame_path = dark_frames_paths[i]
            
            # Check camera make
            if dng_img.get_camera_make() != ref_make:
                raise ValueError(
                    f"Camera make mismatch in dark frame {i} ({frame_path}):\n"
                    f"  Expected: {ref_make}\n"
                    f"  Got: {dng_img.get_camera_make()}"
                )
            
            # Check camera model
            if dng_img.get_camera_model() != ref_model:
                raise ValueError(
                    f"Camera model mismatch in dark frame {i} ({frame_path}):\n"
                    f"  Expected: {ref_model}\n"
                    f"  Got: {dng_img.get_camera_model()}"
                )
            
            # Check camera UID (if present in reference)
            if ref_uid is not None:
                frame_uid = dng_img.get_unique_id()
                if frame_uid != ref_uid:
                    raise ValueError(
                        f"Camera UID (Device ID) mismatch in dark frame {i} ({frame_path}):\n"
                        f"  Expected: {ref_uid}\n"
                        f"  Got: {frame_uid if frame_uid else 'None'}"
                    )
            
            # Check ISO
            if dng_img.get_iso() != ref_iso:
                raise ValueError(
                    f"ISO mismatch in dark frame {i} ({frame_path}):\n"
                    f"  Expected: {ref_iso}\n"
                    f"  Got: {dng_img.get_iso()}"
                )
            
            # Check shutter speed (within 5% tolerance)
            frame_shutter_str = dng_img.get_shutter_speed()
            if frame_shutter_str is not None and ref_shutter is not None:
                # Parse shutter speed strings to floats
                def parse_shutter(s):
                    if '/' in s:
                        parts = s.split('/')
                        return float(parts[0]) / float(parts[1]) if len(parts) == 2 else float(s)
                    return float(s)
                
                try:
                    frame_shutter = parse_shutter(frame_shutter_str)
                    ref_shutter_float = parse_shutter(ref_shutter)
                    
                    shutter_diff = abs(frame_shutter - ref_shutter_float)
                    max_shutter = max(abs(ref_shutter_float), abs(frame_shutter))
                    if max_shutter > 0 and (shutter_diff / max_shutter) > 0.05:
                        raise ValueError(
                            f"Shutter speed mismatch in dark frame {i} ({frame_path}):\n"
                            f"  Expected: {ref_shutter}s\n"
                            f"  Got: {frame_shutter_str}s\n"
                            f"  Difference: {shutter_diff:.4f}s (>{5}% of max)"
                        )
                except (ValueError, ZeroDivisionError):
                    # If parsing fails, fall back to exact string match
                    if frame_shutter_str != ref_shutter:
                        raise ValueError(
                            f"Shutter speed mismatch in dark frame {i} ({frame_path}):\n"
                            f"  Expected: {ref_shutter}\n"
                            f"  Got: {frame_shutter_str}"
                        )
            
            # Check temperature (within 10% tolerance if both present)
            if ref_temp is not None:
                frame_temp = dng_img.get_sensor_temperature()
                if frame_temp is not None:
                    temp_diff = abs(frame_temp - ref_temp)
                    max_temp = max(abs(ref_temp), abs(frame_temp))
                    if max_temp > 0 and (temp_diff / max_temp) > 0.10:
                        raise ValueError(
                            f"Temperature mismatch in dark frame {i} ({frame_path}):\n"
                            f"  Expected: {ref_temp}°C\n"
                            f"  Got: {frame_temp}°C\n"
                            f"  Difference: {temp_diff:.2f}°C (>{10}% of max)"
                        )

        # Dark Frames
        median_noise_frame, mean_noise_frame = generate_dark_frames(dng_imgs)

        # Common Hot Pixels
        # hot_pixels = find_hot_pixels(median_noise_frame, reference_dng.get_bayer_pattern(),
        #                              deviation_threshold=deviation_threshold)
        threshold = np.mean(median_noise_frame) + deviation_threshold * np.std(median_noise_frame)
        hot_pixel_coords = np.argwhere(median_noise_frame > threshold)
        hot_pixels = [(int(y), int(x), float(median_noise_frame[y, x])) for y, x in hot_pixel_coords]

        # Individual Dark Frame Stats
        frame_stats = []
        for dark_dng in dng_imgs:
            frame_stats.append(HotPixelFrameStats.from_dng_image(dark_dng, deviation_threshold))

        # For each frame, compute the fraction of frame-hot-pixels that are in hot_pixels
        hot_pixels_coords_set = set((y, x) for y, x, _ in hot_pixels)
        fraction_common_hot_pixels_list = []
        for frame_stat in frame_stats:
            frame_hot_pixels_set = set((y, x) for y, x, _ in frame_stat.hot_pixels)
            count_in_common = len(frame_hot_pixels_set & hot_pixels_coords_set)
            fraction_common_hot_pixels_list.append(
                count_in_common/len(frame_stat.hot_pixels) if len(frame_stat.hot_pixels) > 0 else 0.0)
        fraction_common_hot_pixels = np.mean(
            fraction_common_hot_pixels_list) if fraction_common_hot_pixels_list else 0.0

        mean_count_hot_pixels = np.mean([fs.count_hot_pixels for fs in frame_stats])

        # Sum of distinct hot_pixels across frames
        distinct_hot_pixels = set()
        for frame_stat in frame_stats:
            for y, x, _ in frame_stat.hot_pixels:
                distinct_hot_pixels.add((y, x))

        common_stats = HotPixelCommonStats(
            count_total_frames=len(dark_frames_paths),
            count_hot_pixels=len(hot_pixels),
            fraction_hot_pixels=len(hot_pixels)/(median_noise_frame.size) if median_noise_frame is not None else 0.0,
            fraction_common_hot_pixels=fraction_common_hot_pixels,
            mean_count_hot_pixels=mean_count_hot_pixels,
            count_distinct_hot_pixels=len(distinct_hot_pixels),
            # hot_pixels=hot_pixels
        )

        profile = HotPixelProfile(
            camera_metadata=camera_metadata,
            common_statistics=common_stats,
            deviation_threshold=deviation_threshold,
            frame_paths=dark_frames_paths,
            mean_values=[fs.mean_value for fs in frame_stats],
            hot_pixel_counts=[fs.count_hot_pixels for fs in frame_stats],
        )

        # Set the numpy arrays directly
        profile._median_noise_frame = median_noise_frame
        profile._mean_noise_frame = mean_noise_frame
        return profile

    def to_dict(self):
        """Manually serialize to dict to avoid dataclass_json reflection issues"""
        data = {
            'camera_metadata': self.camera_metadata.to_dict() if self.camera_metadata else None,
            'common_statistics': self.common_statistics.to_dict() if self.common_statistics else None,
            'deviation_threshold': self.deviation_threshold,
            'frame_paths': self.frame_paths,
            'median_noise_path': self.median_noise_path,
            'mean_noise_path': self.mean_noise_path,
            'mean_values': self.mean_values,
            'hot_pixel_counts': self.hot_pixel_counts,
        }
        return data

    def save_to_file(self, filename: str) -> None:
        """Save the profile to a JSON file with noise profile as sidecar image"""
        # Generate noise profile sidecar filename
        base_name = os.path.splitext(filename)[0]
        
        # Create dir if it doesn't exist
        os.makedirs(f"{base_name}_files", exist_ok=True)

        # Save noise profile as image if we have the data
        if hasattr(self, '_median_noise_frame') and self._median_noise_frame is not None:
            median_noise_filename = f"{base_name}_files/median_noise.tif"
            cv2.imwrite(median_noise_filename, self._median_noise_frame, [
                        cv2.IMWRITE_TIFF_COMPRESSION, cv2.IMWRITE_TIFF_COMPRESSION_LZW])
            self.median_noise_path = median_noise_filename  # Save full path

        if hasattr(self, '_mean_noise_frame') and self._mean_noise_frame is not None:
            mean_noise_filename = f"{base_name}_files/mean_noise.tif"
            cv2.imwrite(mean_noise_filename, self._mean_noise_frame, [
                        cv2.IMWRITE_TIFF_COMPRESSION, cv2.IMWRITE_TIFF_COMPRESSION_LZW])
            self.mean_noise_path = mean_noise_filename  # Save full path
        
        # Save the JSON profile (paths only, not the numpy arrays)
        with open(filename, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load_from_file(cls, filename: str) -> 'HotPixelProfile':
        """Load a profile from a JSON file"""
        with open(filename, 'r') as f:
            json_data = json.load(f)
        
        # Use dataclasses_json for deserialization
        return cls.from_dict(json_data)        


def generate_dark_frames(dark_frames: List[DNGImage]):
    if len(dark_frames) == 0:
        return None

    reference_dark_dng = dark_frames[0]
    reference_raw_data = reference_dark_dng.get_data()
    dark_frames_data = np.zeros((len(dark_frames),) + reference_raw_data.shape, dtype=reference_raw_data.dtype)
    dark_frames_data[0] = reference_raw_data

    for i, dark_dng in enumerate(dark_frames):
        if i == 0:
            continue
        dark_frames_data[i] = dark_dng.get_data()

    median_noise_frame = np.median(dark_frames_data, axis=0)
    print("Unified Median Noise Frame - Min / Max:", np.min(median_noise_frame), np.max(median_noise_frame))
    print("Shape:", median_noise_frame.shape, median_noise_frame.dtype)
    mean_noise_frame = np.mean(dark_frames_data, axis=0)
    print("Unified Mean Noise Frame - Min / Max:", np.min(mean_noise_frame), np.max(mean_noise_frame))

    return median_noise_frame, mean_noise_frame
