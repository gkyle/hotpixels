import cv2
import numpy as np
import os
import torch
from typing import List, Tuple, Optional
from pathlib import Path

from hotpixels.gpu import GPUInfo
from hotpixels.image import DNGImage
from hotpixels.profile import HotPixelProfile
from hotpixels.hot_pixel_segmentation_cnn import HotPixelSegmentationCNN


class App:
    def __init__(self):
        self.current_profile: HotPixelProfile = None
        self.current_profile_path: str = None
        self.profiles_directory = Path("./profiles")
        self.gpuInfo = GPUInfo()
    
    def create_hot_pixel_profile(self, filenames, deviation_threshold) -> HotPixelProfile:
        return HotPixelProfile.from_dark_frames(filenames, deviation_threshold)
    
    def save_corrected_image(self, dng_image: DNGImage, subtract_noise_profile: bool, apply_residual_hotpixel_model: bool) -> str:
        filename = dng_image.filename
        _, ext = os.path.splitext(filename)
        suffix = "_corrected"
        if subtract_noise_profile:
            suffix += "_denoised"
        if apply_residual_hotpixel_model:
            suffix += "_cnn"
        corrected_filename = filename.replace(ext, suffix + ext)

        dng_image.save(corrected_filename)
        return corrected_filename

    def subtract_dark_frame(self, dng_image: DNGImage, hot_pixel_profile: HotPixelProfile):
        dng_image.subtract_dark_frame(hot_pixel_profile.get_mean_noise_frame())
        
    def get_hot_pixels(self, hot_pixel_profile: HotPixelProfile, deviation_threshold: Optional[float] = None) -> List[Tuple[int, int, float]]:
        """Get hot pixels from profile with optional custom deviation threshold.
        """
        median_dark_frame = hot_pixel_profile.get_median_noise_frame()
        if deviation_threshold is None:
            deviation_threshold = hot_pixel_profile.deviation_threshold
        threshold = np.mean(median_dark_frame) + deviation_threshold * np.std(median_dark_frame)
        hot_pixel_coords = np.argwhere(median_dark_frame > threshold)
        return [(y, x, median_dark_frame[y, x]) for y, x in hot_pixel_coords]

    def correct_hot_pixels(self, dng_image: DNGImage, hot_pixel_profile: HotPixelProfile, deviation_threshold: Optional[float] = None) -> List[Tuple[int, int, float]]:
        """Correct hot pixels using profile with optional custom deviation threshold.
        """
        hot_pixels = self.get_hot_pixels(hot_pixel_profile, deviation_threshold)
        dng_image.correct_hot_pixels(hot_pixels)
        return hot_pixels

    def detect_residual_hotpixels_cnn(self, dng_image: DNGImage, batch_size: int = None, progress_callback=None) -> List[Tuple[int, int, float]]:
        """Detect residual hot pixels using CNN model"""

        # Create model and load trained weights
        use_gpu = self.gpuInfo.get_gpu_present()
        device = self.gpuInfo.get_gpu_names()[0][0] if use_gpu else 'cpu'
        
        # Adaptive batch size: CPU benefits from larger batches for this workload
        if batch_size is None:
            batch_size = 32 if use_gpu else 16  # 16 on CPU for good speed/memory balance
        
        # CPU optimizations
        if not use_gpu:
            # Set thread count for optimal CPU performance
            torch.set_num_threads(torch.get_num_threads())
            torch.set_grad_enabled(False)
        
        model = HotPixelSegmentationCNN().to(device)
        
        # Load appropriate model: FP32 for CPU, FP16 for GPU
        model_path = './models/hotpixel_cnn_syn_fp32.pt' if not use_gpu else './models/hotpixel_cnn_syn_fp16.pt'
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        normalized_raw_image = dng_image.raw_img.astype(np.float32) / np.max(dng_image.raw_img)

        # Convert numpy array to tensor (grayscale, no channel dimension yet)
        tensor_raw_image = torch.from_numpy(normalized_raw_image).float()  # H,W
        print(f"Tensor shape: {tensor_raw_image.shape}")

        tile_size = 128
        stride = int(tile_size * 0.8)

        # Process image in tiles with batching
        output_mask = torch.zeros_like(tensor_raw_image)
        
        # Pre-compute all tile positions
        tile_positions = []
        for y in range(0, tensor_raw_image.shape[0], stride):
            for x in range(0, tensor_raw_image.shape[1], stride):
                tile_positions.append((y, x))
        
        total_tiles = len(tile_positions)
        print(f"Processing {total_tiles} tiles in batches of {batch_size}...")
        
        # Process tiles in batches
        for batch_start in range(0, total_tiles, batch_size):
            batch_end = min(batch_start + batch_size, total_tiles)
            batch_tiles = []
            batch_metadata = []  # Store (y, x, original_h, original_w) for each tile
            
            # Prepare batch
            for i in range(batch_start, batch_end):
                y, x = tile_positions[i]
                
                # Extract tile
                tile = tensor_raw_image[y:y+tile_size, x:x+tile_size]
                # Normalize tile
                tile_max = tile.max()
                if tile_max > 0:
                    tile = tile / tile_max
                original_h, original_w = tile.shape
                
                # Pad if needed
                if tile.shape[0] < tile_size or tile.shape[1] < tile_size:
                    pad_h = tile_size - tile.shape[0]
                    pad_w = tile_size - tile.shape[1]
                    tile = torch.nn.functional.pad(tile, (0, pad_w, 0, pad_h), mode='constant', value=0)
                
                # Add channel dimension: (1, H, W)
                tile = tile.unsqueeze(0)
                batch_tiles.append(tile)
                batch_metadata.append((y, x, original_h, original_w))
            
            # Stack tiles into batch: (batch_size, 1, H, W)
            batch_tensor = torch.stack(batch_tiles).to(device)
            
            # Run inference on entire batch
            with torch.no_grad():
                batch_output = model(batch_tensor)
            
            # Move batch output back to CPU and unpack
            batch_output = batch_output.cpu()
            
            # Place each tile output back into the output mask
            for idx, (y, x, original_h, original_w) in enumerate(batch_metadata):
                output_tile = batch_output[idx, 0]  # Remove batch and channel dims
                output_mask[y:y+original_h, x:x+original_w] = output_tile[:original_h, :original_w]
            
            # Progress update
            if progress_callback:
                progress_callback(batch_end, total_tiles, f"CNN: Processing tiles {batch_end}/{total_tiles}")
            elif (batch_start // batch_size) % 10 == 0 or batch_end == total_tiles:
                print(f"Processed {batch_end}/{total_tiles} tiles...")

        print(f"\nProcessed {total_tiles} tiles total")

        # Convert output mask to numpy and extract all detections with their confidence values
        output_mask_np = output_mask.numpy()
        
        # Get all pixel coordinates
        y_coords, x_coords = np.meshgrid(np.arange(output_mask_np.shape[0]), 
                                         np.arange(output_mask_np.shape[1]), 
                                         indexing='ij')
        
        # Flatten and create list of (y, x, confidence) tuples
        detections = [(int(y), int(x), float(conf)) 
                     for y, x, conf in zip(y_coords.flatten(), 
                                           x_coords.flatten(), 
                                           output_mask_np.flatten())
                     if conf > 0.01]  # Only include detections with confidence > 1%
        
        print(f"Total detections above 1% confidence: {len(detections)}")
        
        return detections
    
    def apply_cnn_corrections(self, dng_image: DNGImage, cnn_detections: List[Tuple[int, int, float]], 
                             sensitivity: float = 0.9):
        """Apply CNN hot pixel corrections based on sensitivity threshold.
        """
        confidence_threshold = 1.0 - sensitivity
        
        # Filter detections by confidence threshold
        hot_pixels_to_correct = [(y, x, 0) for y, x, conf in cnn_detections 
                                if conf >= confidence_threshold]
        
        print(f"Correcting {len(hot_pixels_to_correct)} hot pixels with sensitivity {sensitivity:.2f} (confidence >= {confidence_threshold:.2f})")
        
        dng_image.correct_hot_pixels(hot_pixels_to_correct)
       
    def load_profile(self, profile_path: str) -> Optional[HotPixelProfile]:
        """Load a profile from a file path"""
        try:
            profile = HotPixelProfile.load_from_file(profile_path)
            self.current_profile = profile
            self.current_profile_path = profile_path
            return profile
        except Exception as e:
            print(f"Failed to load profile from {profile_path}: {e}")
            return None
    
    def scan_profiles(self) -> List[dict]:
        """Scan the profiles directory and return metadata for all profiles"""
        profile_data = []
        
        # Create profiles directory if it doesn't exist
        self.profiles_directory.mkdir(exist_ok=True)
        
        # Scan for .json profile files
        profile_files = list(self.profiles_directory.glob("*.json"))
        
        for profile_file in profile_files:
            try:
                # Use fast metadata-only loading to avoid loading large noise frame files
                metadata = HotPixelProfile.load_metadata_only(str(profile_file))
                profile_data.append(metadata)
                
            except Exception as e:
                print(f"Failed to load profile {profile_file}: {e}")
        
        return profile_data
    
    def find_matching_profile(self, image: DNGImage) -> Optional[str]:
        """
        Find the best matching profile for the given image.
        
        Matching criteria:
        - Must match: make, model, camera_uid, resolution
        - Should be close: shutter_speed, iso, temperature (within 10%)
        
        Returns:
            Path to the best matching profile, or None if no suitable match found
        """
        # Get image metadata
        image_make = image.get_camera_make()
        image_model = image.get_camera_model()
        image_uid = image.get_unique_id()
        image_resolution = image.get_resolution()
        image_shutter_speed = self._parse_shutter_speed(image.get_shutter_speed())
        image_iso = self._parse_iso(image.get_iso())
        image_temperature = image.get_sensor_temperature()
        
        # Scan available profiles
        profiles = self.scan_profiles()
        
        # Filter for exact matches on required fields
        candidates = []
        for profile in profiles:
            # Must match make, model, and resolution
            if profile['camera_make'] != image_make:
                continue
            if profile['camera_model'] != image_model:
                continue
            if profile['image_resolution'] != image_resolution:
                continue
            
            # Must match camera_uid (if available)
            if image_uid and profile['camera_uid']:
                if profile['camera_uid'] != image_uid:
                    continue
            
            candidates.append(profile)
        
        if not candidates:
            return None
        
        # Score candidates based on proximity of variable parameters
        best_profile = None
        best_score = float('inf')
        
        for profile in candidates:
            score = 0.0
            
            # Score shutter speed difference (lower is better)
            profile_shutter = self._parse_shutter_speed(profile['shutter_speed'])
            if image_shutter_speed is not None and profile_shutter is not None:
                shutter_diff = abs(image_shutter_speed - profile_shutter) / max(image_shutter_speed, profile_shutter)
                if shutter_diff > 0.20:  # More than 20% different
                    continue  # Skip this profile
                score += shutter_diff
            
            # Score ISO difference (lower is better)
            profile_iso = self._parse_iso(profile['iso'])
            if image_iso is not None and profile_iso is not None:
                iso_diff = abs(image_iso - profile_iso) / max(image_iso, profile_iso)
                if iso_diff > 0.20:  # More than 20% different
                    continue  # Skip this profile
                score += iso_diff
            
            # Score temperature difference (lower is better)
            profile_temp = profile['sensor_temperature']
            if image_temperature is not None and profile_temp is not None:
                temp_diff = abs(image_temperature - profile_temp) / max(abs(image_temperature), abs(profile_temp), 1.0)
                if temp_diff > 0.20:  # More than 20% different
                    continue  # Skip this profile
                score += temp_diff
            
            # Update best profile if this one is better
            if score < best_score:
                best_score = score
                best_profile = profile
        
        return best_profile['file_path'] if best_profile else None
    
    def _parse_shutter_speed(self, shutter_str: Optional[str]) -> Optional[float]:
        """Parse shutter speed string to float (in seconds)"""
        if not shutter_str:
            return None
        
        try:
            # Handle fractional strings like "1/100"
            if '/' in shutter_str:
                parts = shutter_str.split('/')
                if len(parts) == 2:
                    return float(parts[0]) / float(parts[1])
            # Handle decimal strings
            return float(shutter_str)
        except:
            return None
    
    def _parse_iso(self, iso_str: Optional[str]) -> Optional[int]:
        """Parse ISO string to integer"""
        if not iso_str:
            return None
        
        try:
            return int(iso_str)
        except:
            return None
    
    def capture_training_data(
        self,
        uncorrected_image: DNGImage,
        corrected_image: DNGImage,
        hot_pixel_profile: HotPixelProfile,
        difference_threshold: float = 10.0,
        roi_size: int = 128,
        jitter: int = 8,
        output_dir: str = "data/training"
    ) -> int:
        """Capture training data for hot pixel detection by comparing uncorrected and corrected images."""
        
        # Create output directory if it doesn't exist
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get raw data from both images
        uncorrected_data = uncorrected_image.raw_img
        corrected_data = corrected_image.raw_img
        
        # Get hot pixels from profile
        hot_pixels = self.get_hot_pixels(hot_pixel_profile)
        
        # Get base filename for naming output files
        base_filename = Path(uncorrected_image.filename).stem
        
        # Track number of samples captured and which hot pixels we've already processed
        samples_captured = 0
        half_roi = roi_size // 2
        processed_hot_pixels = set()  # Track hot pixels already included in an ROI
        
        # Process each hot pixel
        for idx, (y, x, profile_value) in enumerate(hot_pixels):
            # Skip if this hot pixel was already included in a previous ROI
            if (y, x) in processed_hot_pixels:
                continue
            
            # Calculate difference between uncorrected and corrected
            uncorrected_val = float(uncorrected_data[y, x])
            corrected_val = float(corrected_data[y, x])
            difference = abs(uncorrected_val - corrected_val)
            
            # Skip if difference is below threshold
            if difference < difference_threshold:
                continue
            
            # Add random jitter to avoid always centering the hot pixel
            jitter_y = np.random.randint(-jitter, jitter + 1)
            jitter_x = np.random.randint(-jitter, jitter + 1)
            
            center_y = y + jitter_y
            center_x = x + jitter_x
            
            # Check if ROI is within image bounds
            height, width = uncorrected_data.shape[:2]
            y_min = max(0, center_y - half_roi)
            y_max = min(height, center_y + half_roi)
            x_min = max(0, center_x - half_roi)
            x_max = min(width, center_x + half_roi)
            
            # Skip if ROI would be too small (near edges)
            if (y_max - y_min) < roi_size or (x_max - x_min) < roi_size:
                continue
            
            # Extract ROI from uncorrected image (centered at hot pixel + jitter)
            roi = uncorrected_data[y_min:y_max, x_min:x_max].copy()
            
            # Ensure ROI is exactly roi_size x roi_size (pad if needed at edges)
            if roi.shape[0] != roi_size or roi.shape[1] != roi_size:
                padded_roi = np.zeros((roi_size, roi_size), dtype=roi.dtype)
                roi_h, roi_w = roi.shape[:2]
                start_y = (roi_size - roi_h) // 2
                start_x = (roi_size - roi_w) // 2
                padded_roi[start_y:start_y+roi_h, start_x:start_x+roi_w] = roi
                roi = padded_roi
            
            # Find all hot pixels within this ROI that meet the threshold
            hot_pixels_in_roi = []
            for hp_y, hp_x, hp_val in hot_pixels:
                # Check if this hot pixel is within the ROI bounds
                if y_min <= hp_y < y_max and x_min <= hp_x < x_max:
                    # Check if it meets the difference threshold
                    hp_uncorrected = float(uncorrected_data[hp_y, hp_x])
                    hp_corrected = float(corrected_data[hp_y, hp_x])
                    hp_diff = abs(hp_uncorrected - hp_corrected)
                    
                    if hp_diff >= difference_threshold:
                        # Convert to ROI-relative coordinates
                        rel_y = hp_y - y_min
                        rel_x = hp_x - x_min
                        hot_pixels_in_roi.append((rel_x, rel_y))
                        # Mark this hot pixel as processed so it won't be used again
                        processed_hot_pixels.add((hp_y, hp_x))
            
            # Skip if no hot pixels found in ROI (shouldn't happen, but be safe)
            if not hot_pixels_in_roi:
                continue
            
            # Create label text: "0 x y" for each hot pixel in ROI
            label_parts = []
            for hp_x, hp_y in hot_pixels_in_roi:
                label_parts.append(f'0 {hp_x} {hp_y}\n')
            
            # Generate output filenames
            output_base = f"{base_filename}_{idx:04d}_{x}_{y}"
            roi_filename = output_dir / f"{output_base}.tif"
            label_filename = output_dir / f"{output_base}.txt"
            
            # Save ROI as TIF
            cv2.imwrite(str(roi_filename), roi)
            
            # Save label as text file
            with open(label_filename, 'w') as f:
                for label_text in label_parts:
                    f.write(label_text)
            
            samples_captured += 1
        
        # Capture negative examples (background regions without hot pixels)
        num_negative_samples = samples_captured // 2
        negative_samples_captured = 0
        max_attempts = num_negative_samples * 10  # Limit attempts to avoid infinite loop
        attempts = 0
        
        # Create a set of hot pixel coordinates for fast lookup
        hot_pixel_coords_set = set((y, x) for y, x, _ in hot_pixels)
        
        height, width = uncorrected_data.shape[:2]
        
        while negative_samples_captured < num_negative_samples and attempts < max_attempts:
            attempts += 1
            
            # Randomly select a center point for the ROI
            center_y = np.random.randint(half_roi, height - half_roi)
            center_x = np.random.randint(half_roi, width - half_roi)
            
            # Define ROI bounds
            y_min = center_y - half_roi
            y_max = center_y + half_roi
            x_min = center_x - half_roi
            x_max = center_x + half_roi
            
            # Check if this ROI contains any hot pixels
            contains_hot_pixel = False
            for hp_y, hp_x in hot_pixel_coords_set:
                if y_min <= hp_y < y_max and x_min <= hp_x < x_max:
                    contains_hot_pixel = True
                    break
            
            # Skip if this ROI contains a hot pixel
            if contains_hot_pixel:
                continue
            
            # Extract ROI from uncorrected image
            roi = uncorrected_data[y_min:y_max, x_min:x_max].copy()
            
            # Ensure ROI is exactly roi_size x roi_size
            if roi.shape[0] != roi_size or roi.shape[1] != roi_size:
                continue  # Skip if size doesn't match (shouldn't happen with our bounds check)
            
            # Generate output filenames with "background" prefix
            output_base = f"background_{base_filename}_{negative_samples_captured:04d}_{center_x}_{center_y}"
            roi_filename = output_dir / f"{output_base}.tif"
            label_filename = output_dir / f"{output_base}.txt"
            
            # Save ROI as TIF
            cv2.imwrite(str(roi_filename), roi)
            
            # Save empty label file (no hot pixels)
            with open(label_filename, 'w') as f:
                pass  # Empty file for negative examples
            
            negative_samples_captured += 1
        
        print(f"Captured {samples_captured} positive training samples to {output_dir}")
        print(f"Captured {negative_samples_captured} negative (background) training samples to {output_dir}")
        return samples_captured



    
