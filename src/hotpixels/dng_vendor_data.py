# Utility to extract vendor-specific data from DNG images.
#
# This data originates from EXIF MakerNotes, but is located in a DNGPrivate field after Adobe DNG conversion.
#
# Parsing these fields requires vendor-specific implementations for each field, which are handling well by exiftool. This module ports implementations from exiftool, but perhaps it would be better to install exiftool as a dependency for this project.

import struct


class DNGVendorData:

    SUPPORTED_DEVICES = ['sony', 'olympus', 'om digital']

    def __init__(self):
        pass

    @staticmethod
    def is_device_supported(camera_make: str) -> bool:
        camera_make_lower = camera_make.lower()
        for device in DNGVendorData.SUPPORTED_DEVICES:
            if device in camera_make_lower:
                return True
        return False

    @staticmethod
    def get_temperature(img_path):
        try:
            # First try with exifread
            with open(img_path, 'rb') as f:
                # Try manual search for DNG files
                result = find_temperature_tag(img_path)
                if not result or result[0] is None:
                    return None
                
                offset, length, tag_type = result
                f.seek(offset)
                data = f.read(length)
                
                if tag_type == 'olympus':
                    # Olympus data is plain text - 3 signed int16s
                    values = struct.unpack('<hhh', data)
                    # Return first value (the actual temperature)
                    return values[0]
                elif tag_type == 'sony':
                    # Sony data is encrypted
                    deciphered_data = decipher_sony_data(data)
                    if len(deciphered_data) >= 6:
                        temp_test = deciphered_data[0x04]
                        if temp_test != 0 and temp_test < 130:
                            temperature_c = struct.unpack('b', deciphered_data[0x05:0x06])[0]
                            return temperature_c
            
                return None
                
        except Exception:
            return None
        
    @staticmethod
    def get_unique_id(img_path):
        try:
            with open(img_path, 'rb') as f:
                # Search for unique ID tags
                result = find_uniqueid_tag(img_path)
                if not result or result[0] is None:
                    return None
                
                offset, length, tag_type = result
                f.seek(offset)
                data = f.read(length)
                
                if tag_type == 'olympus':
                    # Olympus SerialNumber is ASCII text, possibly with internal structure
                    # Format: "         \x000100S0121\x00BJRA22254   "
                    # We want the last non-empty part after splitting by null bytes
                    try:
                        # Split by null bytes and get non-empty parts
                        parts = [p.strip() for p in data.split(b'\x00') if p.strip()]
                        # Return the last non-empty part (the actual serial number)
                        serial = parts[-1].decode('ascii') if parts else None
                        return serial if serial else None
                    except:
                        return None
                elif tag_type == 'sony':
                    # Sony InternalSerialNumber is at offset 0x0088 within encrypted MakerNotes (tag 0x9050)
                    deciphered_data = decipher_sony_data(data)
                    if len(deciphered_data) >= 0x0088 + 6:
                        try:
                            # Extract 6 bytes at offset 0x0088
                            serial_bytes = deciphered_data[0x0088:0x0088+6]
                            # Convert to hex string
                            serial = ''.join(f'{b:02x}' for b in serial_bytes)
                            return serial if serial else None
                        except:
                            return None
            
                return None
                
        except Exception:
            return None


def decipher_sony_data(data):
    """
    Decipher Sony encrypted MakerNote data.
    Based on exiftool's Sony.pm Decipher function.
    The cipher is: $c = ($b*$b*$b) % 249
    """
    # Translation table for deciphering (reverse mapping)
    # This maps enciphered bytes (0x02-0xf7) to their original values
    decipher_table = bytes.maketrans(
        b'\x08\x1b\x40\x7d\xd8\x5e\x0e\xe7\x04\x56\xea\xcd\x05\x8a\x70\xb6'
        b'\x69\x88\x20\x30\xbe\xd7\x81\xbb\x92\x0c\x28\xec\x6c\xa0\x95\x51'
        b'\xd3\x2f\x5d\x6a\x5c\x39\x07\xc5\x87\x4c\x1a\xf0\xe2\xef\x24\x79'
        b'\x02\xb7\xac\xe0\x60\x2b\x47\xba\x91\xcb\x75\x8e\x23\x33\xc4\xe3'
        b'\x96\xdc\xc2\x4e\x7f\x62\xf6\x4f\x65\x45\xee\x74\xcf\x13\x38\x4b'
        b'\x52\x53\x54\x5b\x6e\x93\xd0\x32\xb1\x61\x41\x57\xa9\x44\x27\x58'
        b'\xdd\xc3\x10\xbc\xdb\x73\x83\x18\x31\xd4\x15\xe5\x5f\x7b\x46\xbf'
        b'\xf3\xe8\xa4\x2d\x82\xb0\xbd\xaf\x8c\x5a\x1f\xda\x9f\x6d\x4a\x3c'
        b'\x49\x77\xcc\x55\x11\x06\x3a\xb3\x7e\x9a\x14\xe4\x25\xc8\xe1\x76'
        b'\x86\x1e\x3d\xe9\x36\x1c\xa1\xd2\xb5\x50\xa2\xb8\x98\x48\xc7\x29'
        b'\x66\x8b\x9e\xa5\xa6\xa7\xae\xc1\xe6\x2a\x85\x0b\xb4\x94\xaa\x03'
        b'\x97\x7a\xab\x37\x1d\x63\x16\x35\xc6\xd6\x6b\x84\x2e\x68\x3f\xb2'
        b'\xce\x99\x19\x4d\x42\xf7\x80\xd5\x0a\x17\x09\xdf\xad\x72\x34\xf2'
        b'\xc0\x9d\x8f\x9c\xca\x26\xa8\x64\x59\x8d\x0d\xd1\xed\x67\x3e\x78'
        b'\x22\x3b\xc9\xd9\x71\x90\x43\x89\x6f\xf4\x2c\x0f\xa3\xf5\x12\xeb'
        b'\x9b\x21\x7c\xb9\xde\xf1',
        b'\x02\x03\x04\x05\x06\x07\x08\x09\x0a\x0b\x0c\x0d\x0e\x0f\x10\x11'
        b'\x12\x13\x14\x15\x16\x17\x18\x19\x1a\x1b\x1c\x1d\x1e\x1f\x20\x21'
        b'\x22\x23\x24\x25\x26\x27\x28\x29\x2a\x2b\x2c\x2d\x2e\x2f\x30\x31'
        b'\x32\x33\x34\x35\x36\x37\x38\x39\x3a\x3b\x3c\x3d\x3e\x3f\x40\x41'
        b'\x42\x43\x44\x45\x46\x47\x48\x49\x4a\x4b\x4c\x4d\x4e\x4f\x50\x51'
        b'\x52\x53\x54\x55\x56\x57\x58\x59\x5a\x5b\x5c\x5d\x5e\x5f\x60\x61'
        b'\x62\x63\x64\x65\x66\x67\x68\x69\x6a\x6b\x6c\x6d\x6e\x6f\x70\x71'
        b'\x72\x73\x74\x75\x76\x77\x78\x79\x7a\x7b\x7c\x7d\x7e\x7f\x80\x81'
        b'\x82\x83\x84\x85\x86\x87\x88\x89\x8a\x8b\x8c\x8d\x8e\x8f\x90\x91'
        b'\x92\x93\x94\x95\x96\x97\x98\x99\x9a\x9b\x9c\x9d\x9e\x9f\xa0\xa1'
        b'\xa2\xa3\xa4\xa5\xa6\xa7\xa8\xa9\xaa\xab\xac\xad\xae\xaf\xb0\xb1'
        b'\xb2\xb3\xb4\xb5\xb6\xb7\xb8\xb9\xba\xbb\xbc\xbd\xbe\xbf\xc0\xc1'
        b'\xc2\xc3\xc4\xc5\xc6\xc7\xc8\xc9\xca\xcb\xcc\xcd\xce\xcf\xd0\xd1'
        b'\xd2\xd3\xd4\xd5\xd6\xd7\xd8\xd9\xda\xdb\xdc\xdd\xde\xdf\xe0\xe1'
        b'\xe2\xe3\xe4\xe5\xe6\xe7\xe8\xe9\xea\xeb\xec\xed\xee\xef\xf0\xf1'
        b'\xf2\xf3\xf4\xf5\xf6\xf7'
    )
    
    # Decipher the data
    deciphered = bytearray(data)
    for i in range(len(deciphered)):
        if 0x02 <= deciphered[i] <= 0xf7:
            deciphered[i] = decipher_table[deciphered[i]]
    
    return bytes(deciphered)


def find_temperature_tag(file_path, camera_make=None):
    """
    Manually search for temperature tags in TIFF/DNG file structure.
    - Sony: tag 0x9403 (CameraTemperature) - encrypted, 1000 bytes
    - Olympus: tag 0x1500 (SensorTemperature) - plain text, 3 int16s
    Handles both raw files and Adobe DNG files with DNGAdobeData tag.
    Returns (offset, length, tag_type) tuple or None if not found.
    """
    
    def search_ifd(f, ifd_offset, endian, visited=None):
        if visited is None:
            visited = set()
        
        if ifd_offset == 0 or ifd_offset in visited:
            return None
        
        visited.add(ifd_offset)
        
        try:
            f.seek(ifd_offset)
            num_entries = struct.unpack(endian + 'H', f.read(2))[0]
            
            sub_ifds = []
            
            for _ in range(num_entries):
                entry_data = f.read(12)
                tag = struct.unpack(endian + 'H', entry_data[0:2])[0]
                field_type = struct.unpack(endian + 'H', entry_data[2:4])[0]
                count = struct.unpack(endian + 'I', entry_data[4:8])[0]
                value_offset = struct.unpack(endian + 'I', entry_data[8:12])[0]
                
                if tag == 0x9403 and count > 4:  # Found it directly!
                    return (value_offset, count)
                
                # Check for DNGAdobeData tag (0xc634)
                if tag == 0xc634 and count > 100:
                    # Read full Adobe MakerNote data
                    f.seek(value_offset)
                    adobe_data = f.read(count)
                    
                    # Adobe MakerNote structure (see exiftool DNG.pm ProcessAdobeMakN):
                    # - First 6 bytes: "Adobe\0"
                    # - Next 6 bytes: "MakN\0\0" (or similar)  
                    # - Remaining: The actual MakerNote data
                    # 
                    # The MakerNote data starts with:
                    # - Bytes 0-1: Byte order ('II' or 'MM')
                    # - Bytes 2-3: TIFF version (0x002a or 0x2a00)
                    # - Bytes 4-7: Original position (big-endian uint32)
                    #              This is where the maker notes were in the original file
                    #              Offsets in the IFD are relative to this original position
                    
                    # Find where the MakerNote TIFF structure starts (look for 'II' or 'MM')
                    makn_start = -1
                    for i in range(min(20, len(adobe_data) - 8)):
                        if adobe_data[i:i+2] in (b'II', b'MM'):
                            makn_start = i
                            break
                    
                    if makn_start < 0:
                        continue
                    
                    # Read the originalPos (bytes 2-5 after byte order, as big-endian)
                    # According to exiftool, this is always read as big-endian
                    original_pos = struct.unpack('>I', adobe_data[makn_start+2:makn_start+6])[0]
                    
                    # Brute force search for temperature tags in little-endian format
                    # IFD entries are 12 bytes: tag(2) + type(2) + count(4) + value/offset(4)
                    
                    # Sony tag 0x9403 (little-endian: 03 94)
                    for i in range(len(adobe_data) - 12):
                        if adobe_data[i:i+2] == b'\x03\x94':  # Tag 0x9403
                            # Read the full IFD entry
                            tag_num = struct.unpack('<H', adobe_data[i:i+2])[0]
                            field_type = struct.unpack('<H', adobe_data[i+2:i+4])[0]
                            field_count = struct.unpack('<I', adobe_data[i+4:i+8])[0]
                            field_offset = struct.unpack('<I', adobe_data[i+8:i+12])[0]
                            
                            if field_count == 1000:  # This is the one we want!
                                # The offset is relative to the original position
                                # The original_pos is where the MakerNote TIFF header was in the original file
                                # We add 6 bytes to account for the Adobe MakerNote wrapper structure
                                makn_file_offset = value_offset + makn_start
                                abs_tag_offset = makn_file_offset - original_pos + field_offset + 6
                                
                                return (abs_tag_offset, field_count, 'sony')
                    
                    # Olympus tag 0x1500 (little-endian: 00 15)
                    for i in range(len(adobe_data) - 12):
                        if adobe_data[i:i+2] == b'\x00\x15':  # Tag 0x1500
                            tag_num = struct.unpack('<H', adobe_data[i:i+2])[0]
                            field_type = struct.unpack('<H', adobe_data[i+2:i+4])[0]
                            field_count = struct.unpack('<I', adobe_data[i+4:i+8])[0]
                            field_offset = struct.unpack('<I', adobe_data[i+8:i+12])[0]
                            
                            # Type 8 = SSHORT (signed int16), count 3
                            if field_type == 8 and field_count == 3:
                                # For Olympus, there's a consistent 20-byte offset adjustment
                                # The actual data is 20 bytes after the offset value indicates
                                data_pos_in_adobe = field_offset + 20
                                abs_tag_offset = value_offset + data_pos_in_adobe
                                
                                return (abs_tag_offset, 6, 'olympus')  # 3 int16s = 6 bytes
                
                # Check for sub-IFD tags
                if tag in [0x8769, 0x8825, 0x927c, 0x014a] and count > 0:
                    sub_ifds.append(value_offset)
            
            # Check next IFD in chain
            next_ifd = struct.unpack(endian + 'I', f.read(4))[0]
            if next_ifd != 0:
                result = search_ifd(f, next_ifd, endian, visited)
                if result:
                    return result
            
            # Search sub-IFDs
            for sub_ifd_offset in sub_ifds:
                result = search_ifd(f, sub_ifd_offset, endian, visited)
                if result:
                    return result
        except Exception as e:
            pass
        
        return None
    
    with open(file_path, 'rb') as f:
        # Read TIFF header
        header = f.read(4)
        if header[:2] == b'II':  # Little-endian
            endian = '<'
        elif header[:2] == b'MM':  # Big-endian
            endian = '>'
        else:
            return None
        
        # Read first IFD offset
        ifd_offset = struct.unpack(endian + 'I', f.read(4))[0]
        
        return search_ifd(f, ifd_offset, endian)


def find_uniqueid_tag(file_path, camera_make=None):
    """
    Manually search for unique ID tags in TIFF/DNG file structure.
    - Sony: tag 0x9050 (encrypted MakerNotes block containing InternalSerialNumber at offset 0x0088)
    - Olympus: tag 0x0101 (SerialNumber) - plain text, ASCII string
    Handles both raw files and Adobe DNG files with DNGAdobeData tag.
    Returns (offset, length, tag_type) tuple or None if not found.
    """
    
    def search_ifd(f, ifd_offset, endian, visited=None):
        if visited is None:
            visited = set()
        
        if ifd_offset == 0 or ifd_offset in visited:
            return None
        
        visited.add(ifd_offset)
        
        try:
            f.seek(ifd_offset)
            num_entries = struct.unpack(endian + 'H', f.read(2))[0]
            
            sub_ifds = []
            
            for _ in range(num_entries):
                entry_data = f.read(12)
                tag = struct.unpack(endian + 'H', entry_data[0:2])[0]
                field_type = struct.unpack(endian + 'H', entry_data[2:4])[0]
                count = struct.unpack(endian + 'I', entry_data[4:8])[0]
                value_offset = struct.unpack(endian + 'I', entry_data[8:12])[0]
                
                # Sony tag 0x9050 - encrypted MakerNotes containing InternalSerialNumber
                if tag == 0x9050 and count > 4:
                    return (value_offset, count, 'sony')
                
                # Olympus SerialNumber (0x0101) in main EXIF
                if tag == 0x0101 and field_type == 2:  # Type 2 = ASCII
                    if count <= 4:
                        # Value is stored inline in the value_offset field
                        return (ifd_offset + 8 + (_ * 12) + 8, count, 'olympus')
                    else:
                        return (value_offset, count, 'olympus')
                
                # Check for MakerNote tag (0x927c) - direct MakerNotes
                if tag == 0x927c and count > 100:
                    # Read MakerNote data
                    f.seek(value_offset)
                    makn_data = f.read(min(count, 100000))  # Limit read size
                    
                    # Search for Olympus SerialNumber (0x101) or InternalSerialNumber (0x102)
                    for i in range(len(makn_data) - 12):
                        if makn_data[i:i+2] in (b'\x01\x01', b'\x02\x01'):  # Tags 0x0101 or 0x0102
                            field_type_mk = struct.unpack('<H', makn_data[i+2:i+4])[0]
                            field_count_mk = struct.unpack('<I', makn_data[i+4:i+8])[0]
                            field_offset_mk = struct.unpack('<I', makn_data[i+8:i+12])[0]
                            
                            if field_type_mk == 2:  # Type 2 = ASCII string
                                # For Olympus, data is at an offset within MakerNotes
                                abs_tag_offset = value_offset + field_offset_mk
                                return (abs_tag_offset, field_count_mk, 'olympus')
                
                # Check for DNGAdobeData tag (0xc634)
                if tag == 0xc634 and count > 100:
                    # Read full Adobe MakerNote data
                    f.seek(value_offset)
                    adobe_data = f.read(count)
                    
                    # Check if this is Olympus MakerNotes (has "OLYMPUS\x00" or "OLYMP\x00\x01" or "OM SYSTEM" signature)
                    is_olympus = (b'OLYMPUS\x00' in adobe_data[:100] or 
                                  b'OLYMP\x00\x01' in adobe_data[:100] or 
                                  b'OLYMP\x00\x02' in adobe_data[:100] or
                                  b'OM SYSTEM' in adobe_data[:100])
                    
                    # Find where the MakerNote TIFF structure starts (look for 'II' or 'MM')
                    makn_start = -1
                    for i in range(min(20, len(adobe_data) - 8)):
                        if adobe_data[i:i+2] in (b'II', b'MM'):
                            makn_start = i
                            break
                    
                    if makn_start < 0:
                        continue
                    
                    # Read the originalPos (bytes 2-5 after byte order, as big-endian)
                    original_pos = struct.unpack('>I', adobe_data[makn_start+2:makn_start+6])[0]
                    
                    # Handle Olympus MakerNotes differently from Sony
                    if is_olympus:
                        # Olympus SerialNumber (tag 0x0101 or 0x0102) in MakerNotes
                        # Search for both little-endian and big-endian
                        for i in range(len(adobe_data) - 12):
                            tag_bytes = adobe_data[i:i+2]
                            if tag_bytes in [b'\x01\x01', b'\x02\x01']:  # Tags 0x0101, 0x0102
                                field_type = struct.unpack('<H', adobe_data[i+2:i+4])[0]
                                field_count = struct.unpack('<I', adobe_data[i+4:i+8])[0]
                                field_offset = struct.unpack('<I', adobe_data[i+8:i+12])[0]
                                
                                # Type 2 = ASCII string
                                if field_type == 2 and field_count >= 8:  # At least 8 chars for a serial
                                    # For Olympus, the offset might be relative to different bases
                                    # Try multiple offset interpretations
                                    for base_adjustment in [0, 20, 12, -8]:
                                        data_pos_in_adobe = field_offset + base_adjustment
                                        if 0 <= data_pos_in_adobe < len(adobe_data) - field_count:
                                            test_data = adobe_data[data_pos_in_adobe:data_pos_in_adobe+min(field_count, 32)]
                                            # Check if it looks like ASCII
                                            if all(32 <= b < 127 or b == 0 for b in test_data[:16]):
                                                abs_tag_offset = value_offset + data_pos_in_adobe
                                                return (abs_tag_offset, field_count, 'olympus')
                    else:
                        # Sony tag 0x9050 (little-endian: 50 90) - Encrypted MakerNotes containing InternalSerialNumber
                        for i in range(len(adobe_data) - 12):
                            if adobe_data[i:i+2] == b'\x50\x90':  # Tag 0x9050
                                # Read the full IFD entry
                                tag_num = struct.unpack('<H', adobe_data[i:i+2])[0]
                                field_type = struct.unpack('<H', adobe_data[i+2:i+4])[0]
                                field_count = struct.unpack('<I', adobe_data[i+4:i+8])[0]
                                field_offset = struct.unpack('<I', adobe_data[i+8:i+12])[0]
                                
                                if field_count > 100:  # Should be around 944 bytes for newer cameras
                                    # Data is at an offset
                                    makn_file_offset = value_offset + makn_start
                                    abs_tag_offset = makn_file_offset - original_pos + field_offset + 6
                                    
                                    return (abs_tag_offset, field_count, 'sony')
                
                # Check for sub-IFD tags
                if tag in [0x8769, 0x8825, 0x927c, 0x014a] and count > 0:
                    sub_ifds.append(value_offset)
            
            # Check next IFD in chain
            next_ifd = struct.unpack(endian + 'I', f.read(4))[0]
            if next_ifd != 0:
                result = search_ifd(f, next_ifd, endian, visited)
                if result:
                    return result
            
            # Search sub-IFDs
            for sub_ifd_offset in sub_ifds:
                result = search_ifd(f, sub_ifd_offset, endian, visited)
                if result:
                    return result
        except Exception as e:
            pass
        
        return None
    
    with open(file_path, 'rb') as f:
        # Read TIFF header
        header = f.read(4)
        if header[:2] == b'II':  # Little-endian
            endian = '<'
        elif header[:2] == b'MM':  # Big-endian
            endian = '>'
        else:
            return None
        
        # Read first IFD offset
        ifd_offset = struct.unpack(endian + 'I', f.read(4))[0]
        
        return search_ifd(f, ifd_offset, endian)
