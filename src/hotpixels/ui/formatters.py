from typing import Optional, TYPE_CHECKING

from hotpixels.profile import HotPixelProfile

if TYPE_CHECKING:
    from hotpixels.image import DNGImage


def format_profile_summary(profile: Optional[HotPixelProfile], no_profile_message: str = "No profile loaded", show_camera_id_edit: bool = False) -> str:
    """Format a hot pixel profile for display in the GUI."""
    if not profile:
        return no_profile_message
    
    # Build formatted text content with improved styling
    text_lines = []
    
    # Profile Summary - Camera and capture info
    if profile.camera_metadata:
        cam = profile.camera_metadata
        # Camera ID as the top item with change link
        camera_id_display = cam.camera_id if cam.camera_id and cam.camera_id.strip() else "<i style='color: #888;'>Not set</i>"
        camera_id_line = f"<b>Camera ID:</b> {camera_id_display} <a href='#edit_camera_id' style='color: #0066cc; text-decoration: none; font-size: 10px; background-color: #f0f0f0; padding: 2px 4px; border-radius: 3px; border: 1px solid #ccc;'>[change]</a>"
        
        text_lines.extend([
            "<b style='color: #2b5aa0; font-size: 14px;'>Camera Information</b>",
            camera_id_line,
            f"<b>Make:</b> {cam.camera_make}",
            f"<b>Model:</b> {cam.camera_model}",
            f"<b>Device UID:</b> {cam.camera_uid if cam.camera_uid else 'N/A'}",
            f"<b>Shutter Speed:</b> {cam.shutter_speed}",
            f"<b>ISO:</b> {cam.iso}",
            f"<b>Sensor Temperature:</b> {cam.sensor_temperature if cam.sensor_temperature is not None else 'N/A'} °C",
            f"<b>Date Created:</b> {cam.date_created}",
            ""  # Empty line for spacing
        ])
    
    # Frame count and analysis summary
    frame_paths = getattr(profile, 'frame_paths', None) or []
    if len(frame_paths) > 0:
        # Get deviation threshold from the profile
        deviation_threshold = profile.deviation_threshold if profile.deviation_threshold is not None else "N/A"
        
        text_lines.extend([
            "<b style='color: #2b5aa0; font-size: 14px;'>Analysis Summary</b>",
            f"<b>Input Frames:</b> {len(frame_paths)}",
            f"<b>Deviation Threshold:</b> {deviation_threshold}",
        ])
    
    # Hot pixel statistics from common analysis
    if profile.common_statistics:
        cs = profile.common_statistics
        
        text_lines.extend([
            "",
            "<b style='color: #2b5aa0; font-size: 14px;'>Hot Pixel Statistics</b>",
            f"<b>Detected Hot Pixels:</b> {cs.count_hot_pixels:,}",
            f"<b>Distinct Hot Pixels (all frames):</b> {cs.count_distinct_hot_pixels:,}",
            f"<b>Average Count of Hot Pixels/Frame:</b> {cs.mean_count_hot_pixels:.1f}",
        ])
    
    # Join all lines with HTML line breaks and apply overall styling
    formatted_text = "<br>".join(text_lines)
    styled_text = f"<div style='font-family: Segoe UI, Arial, sans-serif; font-size: 12px; line-height: 1.4;'>{formatted_text}</div>"
    return styled_text


def format_image_summary(image_files: list[str], profile: Optional[HotPixelProfile] = None, first_image: Optional['DNGImage'] = None) -> str:
    """Format image information for display in the GUI."""
    if not image_files:
        return "No images loaded. Select dark frames to see image information."
    
    try:
        # Load the first image to get metadata if not provided
        from hotpixels.image import DNGImage
        if first_image is None:
            first_image = DNGImage(image_files[0])
        
        # Build formatted text content with improved styling
        text_lines = []

        # Profile Compatibility Check
        if profile and profile.camera_metadata:
            cam = profile.camera_metadata
            image_make = first_image.get_camera_make()
            image_model = first_image.get_camera_model()
            image_uid = first_image.get_unique_id()
            image_shutter = first_image.get_shutter_speed()
            image_iso = first_image.get_iso()
            resolution = first_image.get_resolution()
            image_temp = first_image.get_sensor_temperature()
            
            # Parse values for comparison
            def parse_shutter_speed(shutter_str):
                if not shutter_str:
                    return None
                try:
                    if '/' in shutter_str:
                        parts = shutter_str.split('/')
                        if len(parts) == 2:
                            return float(parts[0]) / float(parts[1])
                    return float(shutter_str)
                except:
                    return None
            
            def parse_iso(iso_str):
                if not iso_str:
                    return None
                try:
                    return int(iso_str)
                except:
                    return None
            
            profile_shutter = parse_shutter_speed(cam.shutter_speed)
            image_shutter_val = parse_shutter_speed(image_shutter)
            profile_iso_val = parse_iso(cam.iso)
            image_iso_val = parse_iso(image_iso)
            
            # Check compatibility (using same logic as app.py)
            make_match = cam.camera_make == image_make
            model_match = cam.camera_model == image_model
            uid_match = cam.camera_uid == image_uid if cam.camera_uid and image_uid else None
            resolution_match = cam.image_resolution == first_image.get_resolution()
            
            # Check shutter speed with 20% tolerance
            if profile_shutter is not None and image_shutter_val is not None:
                shutter_diff = abs(profile_shutter - image_shutter_val) / max(profile_shutter, image_shutter_val)
                shutter_match = shutter_diff <= 0.20
            else:
                shutter_match = cam.shutter_speed == image_shutter
            
            # Check ISO with 20% tolerance
            if profile_iso_val is not None and image_iso_val is not None:
                iso_diff = abs(profile_iso_val - image_iso_val) / max(profile_iso_val, image_iso_val)
                iso_match = iso_diff <= 0.20
            else:
                iso_match = cam.iso == image_iso
            
            # Check temperature with 10% tolerance
            if cam.sensor_temperature is not None and image_temp is not None:
                temp_diff = abs(cam.sensor_temperature - image_temp) / max(abs(cam.sensor_temperature), abs(image_temp), 1.0)
                temp_match = temp_diff <= 0.10
            else:
                temp_match = cam.sensor_temperature == image_temp if cam.sensor_temperature is not None and image_temp is not None else None
            
            text_lines.extend([
                "<b style='color: #2b5aa0; font-size: 14px;'>Profile Compatibility</b>",
            ])
            
            def format_match(label, profile_val, image_val, is_match):
                status_color = "#4CAF50" if is_match else "#F44336"  # Green for match, red for mismatch
                status_icon = "✓" if is_match else "✗"
                return f"<b>{label}:</b> <span style='color: {status_color};'>{status_icon}</span> Profile: {profile_val} | Image: {image_val}"
            
            text_lines.extend([
                format_match("Make", cam.camera_make, image_make, make_match),
                format_match("Model", cam.camera_model, image_model, model_match),
                format_match("Device ID", cam.camera_uid if cam.camera_uid else "N/A", image_uid if image_uid else "N/A", uid_match) if uid_match is not None else "",
                format_match("Resolution", cam.image_resolution, resolution, resolution_match),
                format_match("Shutter Speed", cam.shutter_speed, image_shutter, shutter_match),
                format_match("ISO", cam.iso, image_iso, iso_match),
            ])
            
            # Add sensor temperature if available
            if temp_match is not None:
                temp_display_profile = f"{cam.sensor_temperature}°C" if cam.sensor_temperature is not None else "N/A"
                temp_display_image = f"{image_temp}°C" if image_temp is not None else "N/A"
                text_lines.append(format_match("Sensor Temperature", temp_display_profile, temp_display_image, temp_match))
            
            # Overall compatibility summary
            all_match = make_match and model_match and shutter_match and iso_match and (temp_match is None or temp_match) and (uid_match is None or uid_match)
            if all_match:
                text_lines.append("<span style='color: #4CAF50; font-weight: bold;'>✓ Images are fully compatible with the current profile</span>")
            else:
                text_lines.append("<span style='color: #F44336; font-weight: bold;'>⚠ Images have different settings than the current profile</span>")
        
        # Join all lines with HTML line breaks and apply overall styling
        formatted_text = "<br>".join(text_lines)
        styled_text = f"<div style='font-family: Segoe UI, Arial, sans-serif; font-size: 12px; line-height: 1.4;'>{formatted_text}</div>"
        return styled_text
        
    except Exception as e:
        return f"Error reading image metadata: {str(e)}"
